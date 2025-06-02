"""
Sensor for Assertion IQ.
"""
import os
import ast
import logging
import re # For parsing pragma
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

PRAGMA_RE = re.compile(r"#\s*pragma\s+iq\s*=\s*([0-5](?:\.\d+)?)", re.IGNORECASE)

# Simplified heuristic for Assertion IQ (1-5 rubric)
# In a real implementation, this would be more sophisticated.
def _estimate_assertion_iq_for_test_function(node: ast.FunctionDef, file_content_lines: List[str]) -> float:
    """
    Estimates an IQ score for a single test function based on its assertions
    and pragma overrides, attempting to map to the 1-5 rubric.
    """
    # 1. Check for Pragma override
    # Check line of function definition and the line before it.
    # ast.FunctionDef.lineno is 1-indexed.
    lines_to_check_for_pragma = []
    if node.lineno > 1: # Check line before def
        lines_to_check_for_pragma.append(file_content_lines[node.lineno - 2])
    lines_to_check_for_pragma.append(file_content_lines[node.lineno - 1]) # Check line of def

    for line_content in lines_to_check_for_pragma:
        match = PRAGMA_RE.search(line_content)
        if match:
            try:
                pragma_iq = float(match.group(1))
                if 1.0 <= pragma_iq <= 5.0:
                    logger.debug(f"Assertion IQ for '{node.name}' overridden by pragma to: {pragma_iq}")
                    return pragma_iq
                else:
                    logger.warning(f"Invalid pragma IQ value '{match.group(1)}' for '{node.name}'. Must be 1.0-5.0. Ignoring pragma.")
            except ValueError:
                logger.warning(f"Could not parse pragma IQ value '{match.group(1)}' for '{node.name}'. Ignoring pragma.")
    
    # 2. If no pragma, proceed with AST-based heuristic
    assertions_details = [] # Store type and complexity of each assertion
    
    # Check for Hypothesis decorators for Level 5
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            if decorator.func.attr == 'given' and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == 'hypothesis':
                return 5.0 # Property-based test
        elif isinstance(decorator, ast.Name) and decorator.id == 'given': # Simpler @given from hypothesis
             # Could be hypothesis, but less certain without full import analysis here
             # For now, let's assume it is if 'hypothesis' is imported in the file (complex to check here)
             # A simpler heuristic: if @given is present, lean towards high score.
             pass # We'll check assertion counts too. If it has asserts, it's likely a 5.

    # Walk through the function body to find assertions
    for sub_node in ast.walk(node):
        assertion_type = None
        is_complex_expr = False

        if isinstance(sub_node, ast.Assert): # Direct assert statement
            assertion_type = "direct_assert"
            if any(isinstance(child, (ast.BoolOp, ast.Compare, ast.Call)) for child in ast.iter_child_nodes(sub_node.test)):
                is_complex_expr = True
            assertions_details.append({"type": assertion_type, "complex_expr": is_complex_expr})

        elif isinstance(sub_node, ast.Call): # unittest-style self.assertX or pytest.raises
            func_node = sub_node.func
            method_name = None
            if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name) and func_node.value.id == 'self':
                method_name = func_node.attr # e.g., 'assertEqual', 'assertTrue'
            elif isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name) and func_node.value.id == 'pytest' and func_node.attr == 'raises':
                 method_name = 'pytest.raises'


            if method_name:
                assertion_type = method_name
                # Check complexity of arguments for some assertion types
                if method_name in ["assertEqual", "assertNotEqual", "assertAlmostEqual", "assertNotAlmostEqual", "assertGreater", "assertLess", "assertIn", "assertNotIn"]:
                    if len(sub_node.args) >= 2: # Typically two main args, plus optional msg
                        # Check if args involve calls or complex structures
                        for arg_node in sub_node.args[:2]: # Check first two main args
                            if isinstance(arg_node, ast.Call) or \
                               any(isinstance(child, (ast.BoolOp, ast.Compare, ast.Call)) for child in ast.walk(arg_node)):
                                is_complex_expr = True
                                break
                elif method_name in ["assertTrue", "assertFalse", "assertIsNone", "assertIsNotNone"]:
                     if len(sub_node.args) >= 1:
                         arg_node = sub_node.args[0]
                         if isinstance(arg_node, ast.Call) or \
                            any(isinstance(child, (ast.BoolOp, ast.Compare, ast.Call)) for child in ast.walk(arg_node)):
                            is_complex_expr = True
                elif method_name == "pytest.raises": # This is a strong indicator
                    is_complex_expr = True # Context manager for exceptions is good

                assertions_details.append({"type": assertion_type, "complex_expr": is_complex_expr})

    num_assertions = len(assertions_details)

    if num_assertions == 0:
        # Check if @given was present without assertions (might be an incomplete test)
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                if decorator.func.attr == 'given' and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == 'hypothesis':
                    return 2.0 # Property-based test setup but no explicit assert found in AST, score low.
        return 1.0 # No assertions

    # --- Scoring Logic based on Rubric ---
    # Level 5: Property-based (already returned if @hypothesis.given found)
    # For now, we don't have a strong heuristic for metamorphic from AST alone.

    # Level 4: State Invariant / Contract
    # Heuristic: multiple assertions (e.g., > 2) AND at least one involves complex expression or specific types
    is_level_4_candidate = False
    if num_assertions > 2:
        count_complex_or_specific = 0
        for detail in assertions_details:
            if detail["complex_expr"]:
                count_complex_or_specific += 1
            elif detail["type"] in ["assertRaises", "pytest.raises", "assertDictContainsSubset", "assertListEqual", "assertSetEqual", "assertDictEqual"]: # Examples
                count_complex_or_specific +=1
        if count_complex_or_specific >=1: # or maybe > num_assertions / 2
            is_level_4_candidate = True
    
    if is_level_4_candidate: # Could be refined further
        # If it also had a @given decorator but no explicit return yet, it's a 5
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                if decorator.func.attr == 'given' and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == 'hypothesis':
                    return 5.0
        return 4.0

    # Level 3: Value Range / Multi-field / Collections / Exceptions
    is_level_3_candidate = False
    if num_assertions > 1: # More than one assertion often implies checking multiple aspects
        is_level_3_candidate = True
    else: # Single assertion
        detail = assertions_details[0]
        if detail["complex_expr"]: # e.g. assert x > 0 and x < 10
            is_level_3_candidate = True
        elif detail["type"] in ["assertRaises", "pytest.raises", "assertIn", "assertNotIn", "assertGreater", "assertLess", "assertIsInstance", "assertRegex", "assertListEqual", "assertSetEqual", "assertDictEqual"]:
            is_level_3_candidate = True
            
    if is_level_3_candidate:
        return 3.0

    # Level 2: Simple Equality / Truthy
    # If it's not 1, 3, 4, or 5, it defaults to 2 if assertions exist.
    return 2.0


def get_mean_assertion_iq(test_root_path: str, config: Dict[str, Any]) -> float:
    """
    Calculates the mean Assertion IQ (1-5 rubric score) across all test files.
    This is a placeholder and simulates static analysis of test files.

    Args:
        test_root_path: The root directory containing test files.
        config: Sensor configuration (not used in this placeholder).

    Returns:
        The mean Assertion IQ score (1.0 to 5.0). Defaults to 3.0 if no tests found.
    """
    logger.info(f"Calculating Mean Assertion IQ for test files in: {test_root_path}")
    
    total_iq_score = 0.0
    test_functions_found = 0

    if not os.path.isdir(test_root_path):
        logger.warning(f"Test root path does not exist or is not a directory: {test_root_path}")
        return 3.0 # Default

    for root, _, files in os.walk(test_root_path):
        if any(skip_dir in root for skip_dir in ['.venv', '.git', '__pycache__', '.pytest_cache']):
            continue
        for file_name in files:
            if file_name.startswith("test_") and file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_content_lines = content.splitlines() # Read content into lines
                    tree = ast.parse(content, filename=file_path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and (node.name.startswith("test_") or any(isinstance(d, ast.Name) and "pytest.mark" in d.id for d in node.decorator_list)):
                            # Could also check for class Test... method names
                            iq = _estimate_assertion_iq_for_test_function(node, file_content_lines) # Pass lines
                            total_iq_score += iq
                            test_functions_found += 1
                            logger.debug(f"File: {file_name}, Test: {node.name}, Estimated IQ: {iq}")
                except Exception as e:
                    logger.warning(f"Could not parse or analyze {file_path} for Assertion IQ: {e}")

    if test_functions_found == 0:
        logger.warning("No test functions found for Assertion IQ analysis.")
        # Use a default from config if specified, otherwise a fallback default.
        mean_iq = config.get("default_assertion_iq_if_no_tests", 3.0)
    else:
        mean_iq = total_iq_score / test_functions_found
    
    mean_iq = max(1.0, min(mean_iq, 5.0)) # Ensure it's within 1-5 range
    logger.info(f"Calculated Mean Assertion IQ: {mean_iq:.2f} (from {test_functions_found} test functions)")
    return mean_iq


# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Create dummy test files for testing
    dummy_test_dir = os.path.join(os.getcwd(), ".tmp_guardian_tests", "tests_for_iq")
    os.makedirs(dummy_test_dir, exist_ok=True)

    test_file_1_content = """
import unittest

class MyTests(unittest.TestCase):
    def test_simple_equality(self):
        a = 1
        b = 1
        self.assertEqual(a, b) # IQ ~2

    def test_no_asserts(self): # IQ 1
        pass
"""
    test_file_2_content = """
import pytest

def test_multiple_asserts_and_call(): # IQ ~3-4
    x = 10
    assert x > 5
    assert x < 15
    assert isinstance(x, int)
    
@pytest.mark.slow
def test_decorated_no_asserts(): # IQ 1
    # some setup
    pass
"""
    with open(os.path.join(dummy_test_dir, "test_one.py"), "w") as f:
        f.write(test_file_1_content)
    with open(os.path.join(dummy_test_dir, "test_two.py"), "w") as f:
        f.write(test_file_2_content)

    print(f"Testing Assertion IQ sensor with dummy tests in: {dummy_test_dir}")
    mean_iq_result = get_mean_assertion_iq(dummy_test_dir, {})
    print(f"Calculated Mean Assertion IQ: {mean_iq_result}")

    # Clean up dummy files
    # import shutil
    # shutil.rmtree(os.path.join(os.getcwd(), ".tmp_guardian_tests"))
    print(f"\nDummy test files in: {dummy_test_dir} (can be manually inspected/deleted)")