"""
Sensor for Test Suite Flakiness Rate.
"""
import os
import logging
import json
import random # For simulating flakiness
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Placeholder for CI API interaction and local caching of test history
CI_HISTORY_CACHE_DIR = ".guardian"
FLAKINESS_CACHE_FILE = "flakiness_history.json"

def _get_flakiness_cache_path(project_path: str) -> str:
    """Gets the absolute path to the flakiness cache file."""
    return os.path.join(project_path, CI_HISTORY_CACHE_DIR, FLAKINESS_CACHE_FILE)

def _load_flakiness_history(project_path: str) -> Dict[str, Dict[str, int]]:
    """
    Loads flakiness history (per test ID: {'runs': X, 'fails': Y}).
    """
    cache_file = _get_flakiness_cache_path(project_path)
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load flakiness history from {cache_file}: {e}")
        return {}

def _save_flakiness_history(project_path: str, history: Dict[str, Dict[str, int]]) -> None:
    """Saves the updated flakiness history."""
    cache_file = _get_flakiness_cache_path(project_path)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    try:
        with open(cache_file, 'w') as f:
            json.dump(history, f, indent=2)
    except IOError as e:
        logger.warning(f"Could not save flakiness history to {cache_file}: {e}")


def _simulate_fetch_ci_job_logs(ci_platform: Optional[str], project_identifier: str, num_runs_to_fetch: int) -> List[Dict[str, Any]]:
    """
    Placeholder for fetching CI job summaries from a CI platform's API.
    Each job summary should contain a list of tests and their outcomes.
    """
    logger.info(f"Simulating fetch of last {num_runs_to_fetch} CI job logs for '{project_identifier}' from '{ci_platform}'...")
    
    simulated_runs = []
    test_names = [f"test_case_{i}" for i in range(1, 6)] # 5 example test cases

    for run_id in range(num_runs_to_fetch):
        run_results: Dict[str, List[Dict[str, str]]] = {"tests": []}
        for test_name in test_names:
            # Simulate some flakiness: test_case_3 fails 20% of the time
            outcome = "passed"
            if test_name == "test_case_3" and random.random() < 0.20:
                outcome = "failed"
            elif test_name == "test_case_5" and random.random() < 0.05: # Very rarely flaky
                outcome = "failed"
            
            run_results["tests"].append({"name": test_name, "outcome": outcome})
        simulated_runs.append({"run_id": run_id + 1, "results": run_results})
    
    return simulated_runs


def get_suite_flakiness_rate(
    project_path: str, # Used for caching path
    ci_platform: Optional[str], # e.g., "gh-actions", "gitlab"
    project_identifier: str, # e.g., "org/repo" or project ID for CI API
    config: Dict[str, Any]
) -> float:
    """
    Calculates the overall test suite flakiness rate.
    Flakiness = (Total flaky fails for a test / Total runs for that test)
    Suite Flakiness = Mean of individual test flakiness rates.

    Args:
        project_path: Root path of the project (for caching).
        ci_platform: Identifier for the CI platform to use for fetching logs.
        project_identifier: Specific identifier for the project on the CI platform.
        config: Sensor configuration, e.g.,
                `num_ci_runs_for_flakiness` (default 20),
                `default_flakiness_if_no_history` (default 0.0).

    Returns:
        The calculated suite flakiness rate (0.0 to 1.0).
    """
    logger.info("Calculating suite flakiness rate...")
    
    num_runs_to_check = config.get("num_ci_runs_for_flakiness", 20)
    default_rate = config.get("default_flakiness_if_no_history", 0.0)

    if not ci_platform:
        logger.warning("CI platform not specified. Cannot calculate flakiness from CI logs.")
        return default_rate

    # This is where real CI API calls would happen.
    # For now, we simulate it.
    # In a real scenario, this would also interact with a local cache to avoid re-fetching.
    # The cache would store per-test run/fail counts over time.
    
    # For this placeholder, we'll do a simplified simulation without persistent caching logic here,
    # but the _load/_save functions are provided as a template.
    # The simulation will generate fresh "history" each time.
    
    simulated_job_logs = _simulate_fetch_ci_job_logs(ci_platform, project_identifier, num_runs_to_check)

    if not simulated_job_logs:
        logger.warning("No CI job logs fetched/simulated. Using default flakiness rate.")
        return default_rate

    test_stats: Dict[str, Dict[str, int]] = {} # test_id -> {"runs": X, "fails": Y}

    for job_log in simulated_job_logs:
        for test_result in job_log.get("results", {}).get("tests", []):
            test_name = test_result.get("name")
            outcome = test_result.get("outcome")
            if not test_name:
                continue

            if test_name not in test_stats:
                test_stats[test_name] = {"runs": 0, "fails": 0}
            
            test_stats[test_name]["runs"] += 1
            if outcome == "failed":
                test_stats[test_name]["fails"] += 1
    
    if not test_stats:
        logger.info("No individual test results found in CI logs to calculate flakiness.")
        return default_rate

    individual_flakiness_rates: List[float] = []
    for test_name, stats in test_stats.items():
        if stats["runs"] > 0:
            flakiness = stats["fails"] / stats["runs"]
            individual_flakiness_rates.append(flakiness)
            logger.debug(f"Test '{test_name}': Flakiness = {flakiness:.2f} ({stats['fails']}/{stats['runs']})")
        else:
            logger.debug(f"Test '{test_name}': No runs recorded.")

    if not individual_flakiness_rates:
        suite_flakiness = default_rate
    else:
        suite_flakiness = sum(individual_flakiness_rates) / len(individual_flakiness_rates)
        
    logger.info(f"Calculated Suite Flakiness Rate: {suite_flakiness:.3f} (from {len(individual_flakiness_rates)} tests over {num_runs_to_check} simulated runs)")
    return suite_flakiness


# Example usage
if __name__ == "__main__":
    dummy_project_path = os.path.join(os.getcwd(), ".tmp_guardian_tests")
    os.makedirs(dummy_project_path, exist_ok=True)

    print(f"Testing Flakiness sensor, project path for cache: {dummy_project_path}")
    
    sensor_config = {
        "num_ci_runs_for_flakiness": 20,
        "default_flakiness_if_no_history": 0.01
    }

    # Test case 1: Simulate with a CI platform
    print("\n--- Test Case 1: Simulate with CI platform ---")
    flake_rate1 = get_suite_flakiness_rate(
        project_path=dummy_project_path,
        ci_platform="github-actions", 
        project_identifier="my-org/my-repo", 
        config=sensor_config
    )
    print(f"Calculated Suite Flakiness Rate (Test 1): {flake_rate1:.3f}")

    # Test case 2: No CI platform specified
    print("\n--- Test Case 2: No CI platform ---")
    flake_rate2 = get_suite_flakiness_rate(
        project_path=dummy_project_path,
        ci_platform=None, 
        project_identifier="my-org/my-repo", 
        config=sensor_config
    )
    print(f"Calculated Suite Flakiness Rate (Test 2 - No CI Platform): {flake_rate2:.3f}")
    
    # Test case 3: Fewer runs
    print("\n--- Test Case 3: Fewer CI runs ---")
    sensor_config_few_runs = {**sensor_config, "num_ci_runs_for_flakiness": 5}
    flake_rate3 = get_suite_flakiness_rate(
        project_path=dummy_project_path,
        ci_platform="gitlab-ci", 
        project_identifier="group/project", 
        config=sensor_config_few_runs
    )
    print(f"Calculated Suite Flakiness Rate (Test 3 - 5 runs): {flake_rate3:.3f}")

    print(f"\nDummy cache files might be in: {os.path.join(dummy_project_path, CI_HISTORY_CACHE_DIR)} (can be manually inspected/deleted)")