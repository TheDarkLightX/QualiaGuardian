"""
Sensor for Behaviour Coverage.
"""
import os
import logging
import yaml # Requires PyYAML
from typing import Dict, Any, Set, List, Optional

logger = logging.getLogger(__name__)

def _parse_lcov_placeholder(lcov_file_path: str) -> Set[str]:
    """
    Placeholder for parsing an LCOV file to get covered function/method names.
    In a real implementation, this would parse the LCOV format.
    """
    covered_elements = set()
    if not os.path.exists(lcov_file_path):
        logger.warning(f"LCOV file not found: {lcov_file_path}")
        return covered_elements
    
    logger.info(f"Simulating LCOV parsing for: {lcov_file_path}")
    # Example: SF:source/module/file.py
    # FN:10,my_function
    # FNDA:1,my_function (1 execution)
    # This would extract 'my_function' from 'source/module/file.py'
    
    # Dummy implementation:
    # Assume the LCOV file lists one covered function per line for simplicity
    try:
        with open(lcov_file_path, 'r') as f:
            for line in f:
                if line.startswith("FN:") and "," in line: # Simplified LCOV line
                    try:
                        func_name = line.split(",")[1].strip()
                        covered_elements.add(func_name)
                    except IndexError:
                        pass # Ignore malformed lines
        if not covered_elements: # Add some defaults if file was empty or not parsable by dummy
             covered_elements.update(["critical_function_one", "utility_function_a"])
    except IOError as e:
        logger.warning(f"Could not read LCOV file {lcov_file_path}: {e}")
        # Add some defaults if file could not be read
        covered_elements.update(["critical_function_one_fallback", "utility_function_a_fallback"])

    logger.debug(f"Simulated covered elements from LCOV: {covered_elements}")
    return covered_elements

def _load_critical_behaviors_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Loads critical behaviors from a YAML manifest.
    Each behavior can be linked to functions/methods.
    """
    if not os.path.exists(manifest_path):
        logger.warning(f"Critical behaviors manifest not found: {manifest_path}")
        return []
        
    logger.info(f"Loading critical behaviors manifest: {manifest_path}")
    try:
        with open(manifest_path, 'r') as f:
            manifest_data = yaml.safe_load(f)
        
        if not isinstance(manifest_data, dict) or "critical_behaviors" not in manifest_data:
            logger.warning(f"Manifest {manifest_path} is not in expected format (missing 'critical_behaviors' key).")
            return []
        
        behaviors = manifest_data.get("critical_behaviors", [])
        if not isinstance(behaviors, list):
            logger.warning(f"'critical_behaviors' in {manifest_path} is not a list.")
            return []
            
        logger.debug(f"Loaded critical behaviors: {behaviors}")
        return behaviors
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML manifest {manifest_path}: {e}")
        return []
    except IOError as e:
        logger.error(f"Could not read manifest file {manifest_path}: {e}")
        return []


def get_behaviour_coverage_ratio(
    coverage_file_path: Optional[str], 
    critical_behaviors_manifest_path: Optional[str],
    config: Dict[str, Any]
) -> float:
    """
    Calculates the ratio of covered critical behaviors.
    B' = Covered_Critical_Behaviors / Total_Critical_Behaviors.

    Args:
        coverage_file_path: Path to the LCOV (or similar) coverage data file.
        critical_behaviors_manifest_path: Path to the YAML manifest defining critical behaviors
                                          and their associated functions/methods.
        config: Sensor configuration (e.g., default values if files are missing).

    Returns:
        The behavior coverage ratio (0.0 to 1.0).
    """
    logger.info("Calculating behaviour coverage ratio...")

    if not coverage_file_path:
        logger.warning("Coverage file path not provided for behaviour coverage calculation.")
        # Fallback to a default or error value based on policy
        return config.get("default_behaviour_coverage_if_no_lcov", 0.0)

    if not critical_behaviors_manifest_path:
        logger.warning("Critical behaviors manifest path not provided.")
        # Fallback: if no manifest, assume 0% coverage of critical items or 100% if no critical items defined (ambiguous)
        # Let's assume 0% if manifest is key.
        return config.get("default_behaviour_coverage_if_no_manifest", 0.0)

    covered_functions = _parse_lcov_placeholder(coverage_file_path)
    critical_behaviors_data = _load_critical_behaviors_manifest(critical_behaviors_manifest_path)

    if not critical_behaviors_data:
        logger.info("No critical behaviors defined in the manifest. Coverage ratio is undefined or 1.0 by convention.")
        # Depending on policy: if no critical items, is coverage 100% or 0% or N/A?
        # For bE-TES, if B' is a factor, 0 might be safer if it means "no critical coverage measured".
        return config.get("behaviour_coverage_if_no_critical_items", 1.0) # Defaulting to 1.0 if no critical items to cover

    total_critical_behaviors = len(critical_behaviors_data)
    covered_critical_behaviors_count = 0

    for behavior in critical_behaviors_data:
        behavior_name = behavior.get("name", "Unnamed Behavior")
        linked_functions = behavior.get("linked_functions", [])
        if not isinstance(linked_functions, list):
            logger.warning(f"Behavior '{behavior_name}' has malformed 'linked_functions'. Skipping.")
            continue

        # A behavior is covered if AT LEAST ONE of its linked functions is covered.
        # More complex logic (e.g., ALL linked functions must be covered) could be implemented.
        is_behavior_covered = False
        for func_identifier in linked_functions:
            # func_identifier could be "module.submodule.func_name" or just "func_name"
            # The LCOV parser needs to provide identifiers in a comparable format.
            # For this placeholder, we assume direct name matching.
            if func_identifier in covered_functions:
                is_behavior_covered = True
                break
        
        if is_behavior_covered:
            covered_critical_behaviors_count += 1
            logger.debug(f"Critical behavior '{behavior_name}' is COVERED.")
        else:
            logger.debug(f"Critical behavior '{behavior_name}' is NOT COVERED.")


    if total_critical_behaviors == 0: # Should have been caught by `if not critical_behaviors_data`
        ratio = 1.0 # Or 0.0, based on interpretation.
    else:
        ratio = covered_critical_behaviors_count / total_critical_behaviors
    
    logger.info(f"Behaviour Coverage Ratio: {ratio:.2f} ({covered_critical_behaviors_count}/{total_critical_behaviors} critical behaviors)")
    return ratio

# Example usage
if __name__ == "__main__":
    dummy_project_path = os.path.join(os.getcwd(), ".tmp_guardian_tests")
    os.makedirs(dummy_project_path, exist_ok=True)

    # Create dummy LCOV file
    dummy_lcov_path = os.path.join(dummy_project_path, "coverage.lcov")
    with open(dummy_lcov_path, "w") as f:
        f.write("SF:source/service.py\n")
        f.write("FN:10,critical_function_one\n") # Covered
        f.write("FNDA:1,critical_function_one\n")
        f.write("FN:20,utility_function_b\n") # Not critical, but covered
        f.write("FNDA:1,utility_function_b\n")
        f.write("FN:30,critical_function_two\n") # Critical, but NOT marked as covered by FNDA
        # No FNDA for critical_function_two

    # Create dummy manifest file
    dummy_manifest_path = os.path.join(dummy_project_path, "critical_behaviors.yml")
    manifest_content = {
        "critical_behaviors": [
            {"name": "User Login", "linked_functions": ["critical_function_one", "authenticate_user"]},
            {"name": "Process Payment", "linked_functions": ["critical_function_two", "charge_card"]},
            {"name": "Data Export", "linked_functions": ["export_user_data"]}, # This one won't be found in LCOV
        ]
    }
    with open(dummy_manifest_path, "w") as f:
        yaml.dump(manifest_content, f)

    print(f"Testing Behaviour Coverage sensor with dummy files in: {dummy_project_path}")
    
    # Test case 1: All files present
    print("\n--- Test Case 1: All files ---")
    bc_ratio1 = get_behaviour_coverage_ratio(dummy_lcov_path, dummy_manifest_path, {})
    # Expected: critical_function_one is covered -> "User Login" is covered.
    # critical_function_two is NOT covered (no FNDA) -> "Process Payment" is NOT covered.
    # "Data Export" is not covered.
    # So, 1 out of 3 critical behaviors covered = 0.333
    print(f"Calculated Behaviour Coverage Ratio (Test 1): {bc_ratio1}")

    # Test case 2: Missing LCOV
    print("\n--- Test Case 2: Missing LCOV ---")
    bc_ratio2 = get_behaviour_coverage_ratio(None, dummy_manifest_path, {"default_behaviour_coverage_if_no_lcov": 0.1})
    print(f"Calculated Behaviour Coverage Ratio (Test 2 - No LCOV): {bc_ratio2}")
    
    # Test case 3: Missing Manifest
    print("\n--- Test Case 3: Missing Manifest ---")
    bc_ratio3 = get_behaviour_coverage_ratio(dummy_lcov_path, None, {"default_behaviour_coverage_if_no_manifest": 0.05})
    print(f"Calculated Behaviour Coverage Ratio (Test 3 - No Manifest): {bc_ratio3}")

    # Test case 4: Empty Manifest
    print("\n--- Test Case 4: Empty Manifest ---")
    empty_manifest_path = os.path.join(dummy_project_path, "empty_behaviors.yml")
    with open(empty_manifest_path, "w") as f:
        yaml.dump({"critical_behaviors": []}, f)
    bc_ratio4 = get_behaviour_coverage_ratio(dummy_lcov_path, empty_manifest_path, {"behaviour_coverage_if_no_critical_items": 0.99})
    print(f"Calculated Behaviour Coverage Ratio (Test 4 - Empty Manifest): {bc_ratio4}")

    print(f"\nDummy files in: {dummy_project_path} (can be manually inspected/deleted)")