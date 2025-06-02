"""
Sensor for Test Speed (Median Test Time).
"""
import os
import logging
import json
import statistics # For median calculation
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def _parse_pytest_reportlog(reportlog_path: str, outcome_filter: Optional[List[str]] = None) -> List[float]:
    """
    Parses a pytest-reportlog JSON file to get individual test durations.
    Each line in the reportlog is a JSON object. We are interested in items with
    `"when": "call"`, a "duration" field, and matching the outcome_filter.
    
    Args:
        reportlog_path: Path to the pytest reportlog file.
        outcome_filter: Optional list of outcomes to include (e.g., ["passed", "failed"]).
                        If None or empty, defaults to ["passed", "failed", "skipped"].
    """
    if not outcome_filter: # Default filter if none provided
        outcome_filter = ["passed", "failed", "skipped"]
    test_durations_ms: List[float] = []
    if not os.path.exists(reportlog_path):
        logger.warning(f"Pytest reportlog file not found: {reportlog_path}")
        return test_durations_ms

    logger.info(f"Parsing pytest-reportlog for test durations: {reportlog_path}")
    try:
        with open(reportlog_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    log_entry = json.loads(line.strip())
                    # We are interested in the 'call' phase of test execution
                    if log_entry.get("when") == "call" and "duration" in log_entry and log_entry.get("outcome") in outcome_filter:
                        # Duration is in seconds, convert to milliseconds
                        duration_ms = log_entry["duration"] * 1000.0
                        test_durations_ms.append(duration_ms)
                except json.JSONDecodeError:
                    logger.warning(f"Malformed JSON in {reportlog_path} at line {line_num + 1}: {line.strip()}")
                except KeyError:
                    # Some entries might not be test items or lack duration
                    pass
        
        if not test_durations_ms:
            logger.info(f"No valid test durations found in reportlog: {reportlog_path}")

    except IOError as e:
        logger.error(f"Could not read pytest reportlog {reportlog_path}: {e}")
        # Return empty list, let caller handle default

    logger.debug(f"Extracted test durations (ms) from {reportlog_path}: {test_durations_ms}")
    return test_durations_ms


def get_median_test_time_ms(
    reportlog_path: Optional[str], # Path to a file like pytest-reportlog.jsonl
    config: Dict[str, Any]
) -> float:
    """
    Calculates the median test execution time in milliseconds
    by parsing test execution reports (e.g., pytest-reportlog).

    Args:
        reportlog_path: Path to the test execution report log (e.g., JSON lines format).
        config: Sensor configuration (e.g., default values if report is missing).

    Returns:
        The median test execution time in milliseconds.
        Defaults to a value from config or a hardcoded default if data cannot be obtained.
    """
    logger.info("Calculating median test time...")
    
    default_median_time = config.get("default_median_test_time_ms", 100.0)

    if not reportlog_path:
        logger.warning("Test report log path not provided for speed analysis.")
        return default_median_time

    outcome_filter_from_config = config.get("speed_sensor_outcome_filter", ["passed", "failed", "skipped"])
    test_durations = _parse_pytest_reportlog(reportlog_path, outcome_filter=outcome_filter_from_config)

    if not test_durations:
        logger.warning("No test durations extracted from the report log.")
        return default_median_time
    
    median_time_ms = statistics.median(test_durations)
    
    logger.info(f"Median Test Time: {median_time_ms:.2f} ms (from {len(test_durations)} tests)")
    return median_time_ms

# Example usage
if __name__ == "__main__":
    dummy_project_path = os.path.join(os.getcwd(), ".tmp_guardian_tests")
    os.makedirs(dummy_project_path, exist_ok=True)

    # Create dummy pytest-reportlog.jsonl file
    dummy_reportlog_path = os.path.join(dummy_project_path, "reportlog.jsonl")
    report_entries = [
        {"nodeid": "test_one.py::test_fast", "when": "call", "outcome": "passed", "duration": 0.05}, # 50ms
        {"nodeid": "test_one.py::test_medium", "when": "call", "outcome": "passed", "duration": 0.12}, # 120ms
        {"nodeid": "test_two.py::test_slow", "when": "call", "outcome": "failed", "duration": 0.30}, # 300ms
        {"nodeid": "test_two.py::test_another", "when": "call", "outcome": "passed", "duration": 0.08}, # 80ms
        {"nodeid": "test_setup", "when": "setup", "outcome": "passed", "duration": 0.01}, # Not a 'call' for test item
        {"nodeid": "test_one.py::test_skipped", "when": "call", "outcome": "skipped", "duration": 0.001} # 1ms
    ]
    with open(dummy_reportlog_path, "w") as f:
        for entry in report_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Testing Speed sensor with dummy reportlog in: {dummy_reportlog_path}")
    
    # Test case 1: File present
    print("\n--- Test Case 1: Reportlog present ---")
    median_time1 = get_median_test_time_ms(dummy_reportlog_path, {})
    # Durations: 50, 120, 300, 80, 1. Sorted: 1, 50, 80, 120, 300. Median: 80
    print(f"Calculated Median Test Time (Test 1): {median_time1} ms")

    # Test case 2: Missing reportlog
    print("\n--- Test Case 2: Missing reportlog ---")
    median_time2 = get_median_test_time_ms(None, {"default_median_test_time_ms": 150.0})
    print(f"Calculated Median Test Time (Test 2 - No reportlog): {median_time2} ms")

    # Test case 3: Empty reportlog
    print("\n--- Test Case 3: Empty reportlog ---")
    empty_reportlog_path = os.path.join(dummy_project_path, "empty_reportlog.jsonl")
    with open(empty_reportlog_path, "w") as f:
        pass # Create empty file
    median_time3 = get_median_test_time_ms(empty_reportlog_path, {"default_median_test_time_ms": 123.0})
    print(f"Calculated Median Test Time (Test 3 - Empty reportlog): {median_time3} ms")
    
    print(f"\nDummy files in: {dummy_project_path} (can be manually inspected/deleted)")