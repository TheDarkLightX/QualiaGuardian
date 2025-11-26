import logging
import subprocess
import os
import sys
import time
from pathlib import Path

from typing import List, Optional # Added

logger = logging.getLogger(__name__)

def run_pytest(target_path: str, test_targets: Optional[List[str]] = None):
    """
    Runs pytest on the specified target path, optionally filtering by specific test targets.

    Args:
        target_path (str): The path to the directory or file to run pytest on.
        test_targets (Optional[List[str]]): A list of specific test targets
            (e.g., file paths, node IDs like 'path/to/test.py::test_name')
            to run. If None, pytest runs against target_path normally.

    Returns:
        dict: A dictionary containing pytest execution results, including:
              'success' (bool), 'exit_code' (int), 'stdout' (str),
              'stderr' (str), 'duration_seconds' (float),
              and potentially parsed metrics like 'tests_collected', 'tests_passed', etc. (for future parsing).
    """
    results = {
        "success": False,
        "exit_code": -1,
        "stdout": "",
        "stderr": "",
        "duration_seconds": 0.0,
        "summary": {} # For parsed numbers like collected, passed, failed
    }

    if not os.path.exists(target_path):
        results["stderr"] = f"Error: Target path for pytest does not exist: {target_path}"
        return results

    # Determine pytest executable (try from venv first)
    venv_bin_path = Path(sys.executable).parent
    pytest_candidate = venv_bin_path / "pytest"
    if pytest_candidate.exists():
        pytest_executable = str(pytest_candidate)
    else:
        pytest_executable = "pytest" # Fallback to PATH
        logger.warning("Could not find pytest executable in %s; falling back to PATH lookup.", venv_bin_path)

    cmd = [pytest_executable]
    if test_targets:
        cmd.extend(test_targets) # Add specific test targets
    else:
        cmd.append(target_path) # Default to running on the whole target_path
    
    logger.info("Running pytest command: %s", " ".join(cmd))
    start_time = time.time()
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to handle non-zero exits
        results["stdout"] = process.stdout
        results["stderr"] = process.stderr
        results["exit_code"] = process.returncode
        
        # Pytest exit codes:
        # 0: All tests passed
        # 1: Tests were collected and run but some tests failed
        # 2: Test execution was interrupted by the user
        # 3: Internal error happened while executing tests
        # 4: pytest command line usage error
        # 5: No tests were collected
        
        # Pytest exit code 0 means all tests passed.
        # Pytest exit code 5 means no tests were collected.
        # For the 'success' flag of the runner, we consider these successful executions of pytest.
        # Other codes (1=failures, 2=interrupted, 3=internal error, 4=usage error) mean not successful.
        if process.returncode in (0, 5):
            results["success"] = True
        else:
            results["success"] = False # Explicitly false for other cases, including test failures (exit code 1)
            
            # Basic parsing of summary line (e.g., "=== 1 passed, 1 failed, 1 skipped in 0.01s ===")
            # This will be made more robust later, possibly with pytest-json-report
            # For now, just a placeholder for summary parsing
            if process.stdout:
                logger.debug("Pytest stdout tail:\n%s", process.stdout[-500:])
            if process.stderr:
                logger.debug("Pytest stderr:\n%s", process.stderr)


    except FileNotFoundError:
        results["stderr"] = f"Error: '{pytest_executable}' command not found. Make sure pytest is installed and in your PATH."
        results["exit_code"] = -1 # Indicate command not found
    except Exception as e:
        results["stderr"] = f"An unexpected error occurred while running pytest: {e}"
        results["exit_code"] = -2 # Indicate unexpected error
        logger.exception("Unexpected error while running pytest.")
        
    end_time = time.time()
    results["duration_seconds"] = round(end_time - start_time, 3)
    
    return results

if __name__ == '__main__':
    # Create a dummy test file for direct execution testing
    dummy_test_content_pass = """
import pytest

def test_example_pass():
    assert 1 == 1

def test_another_pass():
    assert True
"""
    dummy_test_content_fail = """
import pytest

def test_example_fail():
    assert 1 == 0

def test_another_pass_in_fail_file():
    assert "a" == "a"
"""
    dummy_test_file_pass = "dummy_pytest_pass.py"
    dummy_test_file_fail = "dummy_pytest_fail.py"

    with open(dummy_test_file_pass, "w") as f:
        f.write(dummy_test_content_pass)
    
    with open(dummy_test_file_fail, "w") as f:
        f.write(dummy_test_content_fail)

    print(f"--- Testing {dummy_test_file_pass} ---")
    pass_results = run_pytest(dummy_test_file_pass)
    print(f"Success: {pass_results['success']}")
    print(f"Exit Code: {pass_results['exit_code']}")
    print(f"Duration: {pass_results['duration_seconds']}s")
    # print(f"Stdout:\n{pass_results['stdout']}")
    # print(f"Stderr:\n{pass_results['stderr']}")
    print("-" * 30)

    print(f"--- Testing {dummy_test_file_fail} ---")
    fail_results = run_pytest(dummy_test_file_fail)
    print(f"Success: {fail_results['success']}") # Should be True as tests ran
    print(f"Exit Code: {fail_results['exit_code']}") # Should be 1
    print(f"Duration: {fail_results['duration_seconds']}s")
    # print(f"Stdout:\n{fail_results['stdout']}")
    # print(f"Stderr:\n{fail_results['stderr']}")
    print("-" * 30)
    
    print(f"--- Testing non_existent_file.py ---")
    non_existent_results = run_pytest("non_existent_file.py")
    print(f"Success: {non_existent_results['success']}")
    print(f"Exit Code: {non_existent_results['exit_code']}")
    print(f"Stderr:\n{non_existent_results['stderr']}")
    print("-" * 30)

    # Clean up dummy files
    os.remove(dummy_test_file_pass)
    os.remove(dummy_test_file_fail)