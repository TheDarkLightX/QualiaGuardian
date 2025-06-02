"""
Sensor for Mutation Score and EMT Gain.
"""
import json
import os
import logging
import subprocess # Added for running mutmut
import re # Added for parsing mutmut output
from typing import Tuple, Dict, Any, Optional, List # Added List

logger = logging.getLogger(__name__)

GUARDIAN_STATE_DIR = ".guardian"
MUTATION_STATE_FILE = "mutation_state.json"

def _get_state_file_path(project_path: str) -> str:
    """Gets the absolute path to the mutation state file."""
    return os.path.join(project_path, GUARDIAN_STATE_DIR, MUTATION_STATE_FILE)

def _load_last_mutation_score(project_path: str) -> Optional[float]:
    """Loads the last mutation score from the state file."""
    state_file = _get_state_file_path(project_path)
    if not os.path.exists(state_file):
        return None
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
        return data.get("last_mutation_score")
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load mutation state from {state_file}: {e}")
        return None

def _save_mutation_score(project_path: str, mutation_score: float) -> None:
    """Saves the current mutation score to the state file."""
    state_file = _get_state_file_path(project_path)
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    data = {"last_mutation_score": mutation_score}
    try:
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        logger.warning(f"Could not save mutation state to {state_file}: {e}")


def get_mutation_score_data(
    config: Dict[str, Any],
    project_path: str,
    test_targets: Optional[List[str]] = None
) -> Tuple[float, int, int]:
    """
    Runs mutmut and parses its output to get mutation score data.
    
    Args:
        config: Sensor configuration (e.g., path to mutation tool, arguments).
        project_path: The root path of the project being analyzed.
        test_targets: Optional list of specific test files or node IDs to run.

    Returns:
        A tuple: (mutation_score_percentage, total_mutants, killed_mutants).
                 Returns (0.0, 0, 0) if data cannot be obtained.
    """
    if test_targets:
        logger.info(f"Attempting to run mutmut for project: {project_path}, targeting tests: {test_targets}")
    else:
        logger.info(f"Attempting to run mutmut for project: {project_path} (all tests)")

    paths_to_mutate = config.get("mutmut_paths_to_mutate", None) # e.g., "src/" or "./your_package"
    base_runner_args = config.get("mutmut_runner_args", "pytest") # e.g., "pytest -x --assert=plain"
    mutmut_executable = config.get("mutmut_executable", "mutmut") # Allows specifying path to mutmut

    if not paths_to_mutate:
        logger.error("`mutmut_paths_to_mutate` not specified in sensor config. Cannot run mutation testing.")
        _save_mutation_score(project_path, 0.0) # Save 0 to avoid stale data issues
        return 0.0, 0, 0

    # Ensure paths_to_mutate is a list for the command
    if isinstance(paths_to_mutate, str):
        paths_to_mutate_list = [paths_to_mutate]
    elif isinstance(paths_to_mutate, list):
        paths_to_mutate_list = paths_to_mutate
    else:
        logger.error("`mutmut_paths_to_mutate` must be a string or a list of strings.")
        _save_mutation_score(project_path, 0.0)
        return 0.0, 0, 0

    # Construct the final runner command string
    final_runner_command = base_runner_args
    if test_targets:
        final_runner_command += " " + " ".join(test_targets)

    # Create a temporary mutmut_config.ini
    config_content = "[mutmut]\n"
    # paths_to_mutate_list should contain strings. Join them if it's a list for the config.
    # Mutmut config expects a single string for paths_to_mutate, space or comma separated.
    paths_to_mutate_str = paths_to_mutate_list[0] if isinstance(paths_to_mutate_list, list) and len(paths_to_mutate_list) == 1 else " ".join(paths_to_mutate_list)
    
    config_content += f"paths_to_mutate = {paths_to_mutate_str}\n"
    config_content += f"test_command = {final_runner_command}\n"
    # Add other default mutmut settings if necessary, e.g., backup = false
    config_content += "backup = false\n"


    temp_config_path = os.path.join(project_path, "mutmut_config.ini.guardian_tmp")
    
    command = [mutmut_executable, "run"]
    # No need to pass paths or runner via CLI if using config file.
    # Add other mutmut options from config if needed, e.g., --disable-stdin, --simple-output
    # Forcing mutmut to use our temp config if one already exists could be done by
    # temporarily renaming an existing one, or hoping mutmut prioritizes .guardian_tmp
    # Simpler: mutmut usually looks for `mutmut_config.ini` or `pyproject.toml`.
    # We will rely on it picking up our temp file if no other is present, or if it's named `mutmut_config.ini`.
    # Let's name it `mutmut_config.ini` directly for higher chance of being picked up,
    # and handle existing file by backing it up.

    actual_config_file_path = os.path.join(project_path, "mutmut_config.ini")
    backup_config_path = os.path.join(project_path, "mutmut_config.ini.guardian_bak")
    existing_config_renamed = False


    killed_mutants = 0
    total_mutants = 0
    mutation_score_percentage = 0.0

    try:
        # Backup existing mutmut_config.ini if it exists
        if os.path.exists(actual_config_file_path):
            os.rename(actual_config_file_path, backup_config_path)
            existing_config_renamed = True
            logger.debug(f"Backed up existing {actual_config_file_path} to {backup_config_path}")

        with open(actual_config_file_path, 'w') as f:
            f.write(config_content)
        logger.debug(f"Created temporary mutmut config at {actual_config_file_path} with content:\n{config_content}")

        logger.info(f"Executing mutmut command: {' '.join(command)} (using generated config)")
        # Run mutmut in the project_path directory
        process = subprocess.run(command, capture_output=True, text=True, check=False, cwd=project_path)
        
        if process.returncode != 0 and process.returncode != 1: # 0=all good, 1=mutants found/run
             # Mutmut exits with 0 if no mutants found/run, 1 if mutants were run (even if all survived).
             # Non-zero other than 1 might indicate actual errors.
            logger.warning(f"Mutmut exited with code {process.returncode}. Output:\n{process.stdout}\nStderr:\n{process.stderr}")
        else:
            logger.info(f"Mutmut run completed (exit code {process.returncode}). Output:\n{process.stdout}")

        # Parse mutmut stdout for summary
        # Example output lines:
        # "killed: 75"
        # "timeout: 0"
        # "suspicious: 0"
        # "survived: 25"
        # "skipped: 0"
        
        output_lines = process.stdout.splitlines()
        # Initialize counts
        killed_mutants = 0
        survived_mutants = 0
        timeout_mutants = 0
        suspicious_mutants = 0
        
        # Regex to capture the new summary line format, e.g., "🎉 4 🫥 0  ⏰ 0  🤔 0  🙁 2  🔇 0"
        # It might appear on a line that starts with progress characters like ⠹ X/Y
        summary_regex = re.compile(
            r"🎉\s*(?P<killed>\d+)\s*"       # Killed (🎉)
            r"🫥\s*(?P<skipped_internally>\d+)\s*"  # Not sure what this is, maybe skipped by mutmut's internal logic
            r"⏰\s*(?P<timeout>\d+)\s*"      # Timeout (⏰)
            r"🤔\s*(?P<suspicious>\d+)\s*"   # Suspicious (🤔)
            r"🙁\s*(?P<survived>\d+)\s*"     # Survived (🙁)
            r"🔇\s*(?P<no_coverage>\d+)"     # No coverage (🔇)
        )

        parsed_summary_new_format = False
        for line in reversed(output_lines): # Check from the end, summary is usually last
            match = summary_regex.search(line)
            if match:
                data = match.groupdict()
                killed_mutants = int(data.get("killed", 0))
                survived_mutants = int(data.get("survived", 0))
                timeout_mutants = int(data.get("timeout", 0))
                suspicious_mutants = int(data.get("suspicious", 0))
                parsed_summary_new_format = True
                logger.info(f"Parsed mutmut new summary: Killed={killed_mutants}, Survived={survived_mutants}, Timeout={timeout_mutants}, Suspicious={suspicious_mutants}")
                break
        
        if not parsed_summary_new_format:
            logger.warning("Could not parse mutmut summary using new emoji-based regex. Attempting old format.")
            # Fallback to old parsing logic if new one fails
            summary_counts_old = {}
            for line in output_lines: # No need to reverse for old format, it was simpler
                match_old = re.match(r"(\w+):\s*(\d+)", line.strip())
                if match_old:
                    key, value = match_old.groups()
                    summary_counts_old[key.lower()] = int(value)
            
            if summary_counts_old:
                logger.info(f"Parsed mutmut old summary: {summary_counts_old}")
                killed_mutants = summary_counts_old.get("killed", 0)
                survived_mutants = summary_counts_old.get("survived", 0)
                timeout_mutants = summary_counts_old.get("timeout", 0)
                suspicious_mutants = summary_counts_old.get("suspicious", 0) # Might not exist in old format
            else:
                logger.warning("Could not parse mutmut summary using old format either.")

        # Standard mutation score: (killed + timeout) / (killed + timeout + survived + suspicious)
        # Treat suspicious mutants as not killed for a conservative score.
        # Timeouts are generally considered killed.
        
        numerator = killed_mutants + timeout_mutants
        denominator = killed_mutants + timeout_mutants + survived_mutants + suspicious_mutants

        if denominator > 0:
            mutation_score_percentage = numerator / denominator
        else:
            mutation_score_percentage = 0.0
            # Log if no scorable mutants were found, and neither parsing method yielded counts
            if not parsed_summary_new_format and not summary_counts_old:
                 logger.warning("No scorable mutants found and summary parsing failed for both formats.")

        total_mutants = denominator # For reporting purposes, this is the effective total considered for score
        logger.info(f"Final parsed counts: Killed={killed_mutants}, Survived={survived_mutants}, Timeout={timeout_mutants}, Suspicious={suspicious_mutants}")
        logger.info(f"Calculated mutation score: {mutation_score_percentage:.4f} ({numerator}/{denominator})")

    except FileNotFoundError:
        logger.error(f"'{mutmut_executable}' command not found. Please ensure mutmut is installed and in PATH.")
        # Save 0 to avoid stale data issues if mutmut can't run
        _save_mutation_score(project_path, 0.0)
        return 0.0, 0, 0
    except Exception as e:
        logger.error(f"Error running or parsing mutmut: {e}", exc_info=True)
        # Save 0 to avoid stale data issues
        _save_mutation_score(project_path, 0.0)
        return 0.0, 0, 0
    finally:
        # Clean up temporary config file
        if os.path.exists(actual_config_file_path): # Check if it's our temp file
            try:
                with open(actual_config_file_path, 'r') as f:
                    # A simple check to see if it's likely our generated file
                    # This is not foolproof but better than blindly deleting.
                    # A more robust way would be to check content or a unique marker.
                    # For now, if it was created by this function, it will be deleted.
                    # The risk is if another process created a mutmut_config.ini in between.
                    # Given the short duration, this risk is low.
                    pass # Assuming if it exists, it's ours to delete
                os.remove(actual_config_file_path)
                logger.debug(f"Removed temporary mutmut config: {actual_config_file_path}")
            except Exception as e_cleanup:
                logger.warning(f"Could not remove temporary mutmut config {actual_config_file_path}: {e_cleanup}")
        
        # Restore backup if one was made
        if existing_config_renamed and os.path.exists(backup_config_path):
            os.rename(backup_config_path, actual_config_file_path)
            logger.debug(f"Restored original mutmut_config.ini from {backup_config_path}")


    # Save this current score for next run's EMT gain calculation
    _save_mutation_score(project_path, mutation_score_percentage)
    
    return mutation_score_percentage, total_mutants, killed_mutants


def get_emt_gain(current_mutation_score: float, project_path: str, config: Dict[str, Any]) -> float:
    """
    Calculates the EMT (Evolutionary Mutation Testing) Gain.
    EMT Gain = Current MS - Initial MS (last recorded MS).

    Args:
        current_mutation_score: The mutation score from the current run.
        project_path: The root path of the project being analyzed.
        config: Sensor configuration (e.g., baseline MS if no history).

    Returns:
        The calculated EMT gain. Defaults to 0.0 if no previous score.
    """
    last_ms = _load_last_mutation_score(project_path)
    
    initial_ms_source = "previous_run"

    if last_ms is None:
        # Fallback to a configured baseline if no history, or default to current_ms for 0 gain
        last_ms = config.get("baseline_mutation_score", current_mutation_score) 
        initial_ms_source = "configured_baseline_or_current"
        if last_ms == current_mutation_score and "baseline_mutation_score" not in config:
            initial_ms_source = "current_ms_as_baseline (no history/config)"


    emt_gain = current_mutation_score - last_ms
    logger.info(f"EMT Gain calculated: {emt_gain:.4f} (current: {current_mutation_score:.4f}, initial ({initial_ms_source}): {last_ms:.4f})")
    return emt_gain

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Simulate a project path
    test_project_path = os.path.join(os.getcwd(), ".tmp_guardian_tests")
    if not os.path.exists(test_project_path):
        os.makedirs(test_project_path)
    
    print(f"Testing mutation sensor with project path: {test_project_path}")

    # First run (no history)
    print("\n--- First Run ---")
    cfg1 = {"simulated_killed_mutants": 70, "simulated_total_mutants": 100}
    ms1, total1, killed1 = get_mutation_score_data(cfg1, test_project_path)
    gain1 = get_emt_gain(ms1, test_project_path, cfg1) # EMT gain should be 0 or based on baseline

    # Second run
    print("\n--- Second Run ---")
    cfg2 = {"simulated_killed_mutants": 75, "simulated_total_mutants": 100}
    ms2, total2, killed2 = get_mutation_score_data(cfg2, test_project_path)
    gain2 = get_emt_gain(ms2, test_project_path, cfg2)

    # Third run with a different baseline if history was cleared
    print("\n--- Third Run (simulating cleared history, using baseline) ---")
    # Clear state for this test
    state_file_path = _get_state_file_path(test_project_path)
    if os.path.exists(state_file_path):
        os.remove(state_file_path)
    
    cfg3 = {"simulated_killed_mutants": 80, "simulated_total_mutants": 100, "baseline_mutation_score": 0.65}
    ms3, total3, killed3 = get_mutation_score_data(cfg3, test_project_path) # This will save 0.80
    # For EMT gain, we need to call _load_last_mutation_score *before* get_mutation_score_data saves the new one
    # Or, the EMT gain calculation should be more tightly coupled if it's always based on the *immediately* preceding run.
    # The current design: get_mutation_score_data saves its result. get_emt_gain loads what was saved *before* this current run.
    # Let's simulate this properly:
    
    # Simulate state before run 3
    _save_mutation_score(test_project_path, 0.78) # Say previous run was 0.78
    
    print("\n--- Fourth Run (with pre-saved state for EMT gain) ---")
    cfg4 = {"simulated_killed_mutants": 82, "simulated_total_mutants": 100}
    # EMT gain will use the 0.78 loaded by _load_last_mutation_score
    gain4_before_save = get_emt_gain(0.82, test_project_path, cfg4) # current_ms is 0.82
    print(f"Calculated gain4 (before saving 0.82): {gain4_before_save}")
    # Now, get_mutation_score_data would run and save 0.82
    ms4, _, _ = get_mutation_score_data(cfg4, test_project_path)
    print(f"MS4 (saved): {ms4}")

    # Clean up test state file
    if os.path.exists(state_file_path):
        # os.remove(state_file_path)
        # pass # Keep it for manual inspection
        print(f"\nState file at: {state_file_path}")