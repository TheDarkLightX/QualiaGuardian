import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import shutil # For cleanup in __main__ if needed, not directly in evaluate_subset

from ..test_execution.pytest_runner import run_pytest
# SmartMutator and Mutant are no longer directly used here
# from ..evolution.smart_mutator import SmartMutator, Mutant
from ..sensors import mutation as mutation_sensor # Import the mutation sensor

logger = logging.getLogger(__name__)

def evaluate_subset(
    project_path: Path,
    selected_tests: List[Path],
    # mutant_cache: Optional[Dict[Path, List[Mutant]]] = None, # No longer needed, mutmut handles its own cache
    # config: Optional[QualityConfig] = None # Future
) -> float:
    """
    Evaluates a given subset of tests by performing mutation testing on the
    project's source code using the mutation sensor and returns the mutation score.

    Args:
        project_path: The root path of the project.
        selected_tests: A list of Path objects representing the test files
                        or specific test node IDs. Paths are resolved to be
                        absolute or relative to project_path for pytest.

    Returns:
        A float mutation score for the subset (killed_mutants / tested_mutants),
        or 0.0 if no tests/mutants.
    """
    logger.info(f"Evaluating subset of {len(selected_tests)} tests: {selected_tests} for project: {project_path} using mutation sensor.")

    # The mutant_cache is handled by mutmut internally, not passed here.

    if not selected_tests:
        logger.info("No tests selected, mutation score is 0.")
        return 0.0

    # Ensure test paths are absolute or correctly relative for pytest
    test_target_strings: List[str] = [
        str(p.resolve() if p.is_absolute() else (project_path / p).resolve())
        for p in selected_tests
    ]

    # 1. Initial pytest run on original code (optional, but good for baseline)
    #    This is useful to ensure the test subset itself is valid before mutation.
    #    The mutation_sensor also does its own "clean run" internally if needed by mutmut.
    initial_pytest_results = run_pytest(target_path=str(project_path), test_targets=test_target_strings)
    if not initial_pytest_results["success"] and initial_pytest_results["exit_code"] != 5: # 5 = no tests collected
        logger.warning(
            f"Initial pytest run for subset {test_target_strings} failed (exit code {initial_pytest_results['exit_code']}). "
            f"Mutation score for subset will be 0."
        )
        return 0.0
    logger.info(f"Initial pytest run on original code for subset {test_target_strings} successful or no tests found.")

    # 2. Configure and run mutation sensor
    #    Determine paths to mutate. For now, assume 'src' or the project root if 'src' doesn't exist.
    #    A more robust solution would get this from a project config or allow specification.
    src_dir = project_path / "src"
    paths_to_mutate_config = str(src_dir) if src_dir.is_dir() else str(project_path)
    
    mutation_sensor_config = {
        "mutmut_paths_to_mutate": paths_to_mutate_config,
        "mutmut_runner_args": "pytest" # Base runner, test_targets will be appended by the sensor
    }

    mutation_score, total_mutants, killed_mutants = mutation_sensor.get_mutation_score_data(
        config=mutation_sensor_config,
        project_path=str(project_path), # sensor expects string
        test_targets=test_target_strings # Pass the specific tests to run against mutants
    )

    logger.info(
        f"Mutation sensor for subset {test_target_strings}: Score={mutation_score:.4f}, "
        f"Killed={killed_mutants}, TotalApplicable={total_mutants}"
    )
    
    return mutation_score

if __name__ == '__main__':
    # Example usage (requires a dummy project and tests)
    # Ensure this example is updated to reflect actual project structure if used for testing.
    # When running this script directly for testing, use:
    # python -m guardian_ai_tool.guardian.core.api
    # from the project root directory (/home/trevormoc/Downloads/Qualia).
    logging.basicConfig(
        level=logging.DEBUG, # Use DEBUG for more verbose output during local testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a dummy project structure for testing
    # Relative to the script's location or an absolute path
    current_file_dir = Path(__file__).parent # .../guardian_ai_tool/guardian/core
    # The dummy project will be created relative to where the script is,
    # or use an absolute path if preferred for consistency.
    # Let's make it in the Qualia directory, sibling to guardian_ai_tool
    project_base_dir = Path(__file__).resolve().parent.parent.parent.parent # Should be /home/trevormoc/Downloads/Qualia
    dummy_project_root = project_base_dir / "temp_guardian_project_for_api_test"
    
    # Clean up previous dummy project if it exists
    if dummy_project_root.exists():
        shutil.rmtree(dummy_project_root)
        logger.info(f"Cleaned up existing dummy project at: {dummy_project_root.resolve()}")

    dummy_src_dir = dummy_project_root / "src"
    dummy_tests_dir = dummy_project_root / "tests"
    dummy_src_dir.mkdir(parents=True, exist_ok=True)
    dummy_tests_dir.mkdir(parents=True, exist_ok=True)

    # Dummy source file
    dummy_source_content = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def check_value(x):
    if x > 10:
        return True
    return False
"""
    with open(dummy_src_dir / "calculator.py", "w") as f:
        f.write(dummy_source_content)
    
    # Dummy test files
    test_file_1_content = """
import sys
from pathlib import Path
# Adjust path to import from dummy_src_dir
project_root = Path(__file__).parent.parent # Should be dummy_project_root
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from calculator import add, subtract, check_value

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 2) == 3
"""
    test_file_2_content = """
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from calculator import check_value

def test_check_value_positive():
    assert check_value(15) is True

def test_check_value_negative():
    assert check_value(5) is False
"""
    with open(dummy_tests_dir / "test_calculator_part1.py", "w") as f:
        f.write(test_file_1_content)
    with open(dummy_tests_dir / "test_calculator_part2.py", "w") as f:
        f.write(test_file_2_content)

    logger.info(f"Created dummy project at: {dummy_project_root.resolve()}")

    # Test cases
    # Mutant cache is no longer passed to evaluate_subset as mutmut handles its own.

    all_dummy_tests = [
        dummy_tests_dir / "test_calculator_part1.py",
        dummy_tests_dir / "test_calculator_part2.py"
    ]
    
    logger.info(f"\n--- Running evaluate_subset with all tests ---")
    score_all = evaluate_subset(
        project_path=dummy_project_root,
        selected_tests=all_dummy_tests
    )
    logger.info(f"Score for all tests: {score_all:.4f}")

    logger.info(f"\n--- Running evaluate_subset with only part1 tests ---")
    subset1 = [dummy_tests_dir / "test_calculator_part1.py"]
    score1 = evaluate_subset(
        project_path=dummy_project_root,
        selected_tests=subset1
    )
    logger.info(f"Score for subset 1 ({[str(p.name) for p in subset1]}): {score1:.4f}")
    
    logger.info(f"\n--- Running evaluate_subset with an empty test list ---")
    score_empty = evaluate_subset(
        project_path=dummy_project_root,
        selected_tests=[]
    )
    logger.info(f"Score for empty subset: {score_empty:.4f}")

    # Clean up
    # shutil.rmtree(dummy_project_root)
    # logger.info(f"Cleaned up dummy project: {dummy_project_root.resolve()}")
    logger.info(f"Dummy project left at {dummy_project_root.resolve()} for inspection.")