"""
Unit tests for the Assertion IQ sensor.
"""
import unittest
import ast
import tempfile
import os
from guardian.sensors.assertion_iq import get_mean_assertion_iq, _estimate_assertion_iq_for_test_function

class TestAssertionIQSensor(unittest.TestCase):

    def test_estimate_assertion_iq_with_pragma_override(self):
        """Test pragma override for assertion IQ."""
        code_with_pragma = """
# pragma iq=4.5
def test_something_with_override(self):
    assert True
"""
        node = ast.parse(code_with_pragma).body[0]
        self.assertIsInstance(node, ast.FunctionDef)
        lines = code_with_pragma.splitlines()
        iq = _estimate_assertion_iq_for_test_function(node, lines)
        self.assertAlmostEqual(iq, 4.5)

    def test_estimate_assertion_iq_with_pragma_on_def_line(self):
        """Test pragma override on the same line as function definition."""
        code_with_pragma = """
def test_another_override(self): # pragma iq=2.5
    assert 1 == 1
"""
        node = ast.parse(code_with_pragma).body[0]
        self.assertIsInstance(node, ast.FunctionDef)
        lines = code_with_pragma.splitlines()
        # Note: AST lineno is 1-indexed and points to the 'def' keyword.
        # The pragma is on the same line.
        iq = _estimate_assertion_iq_for_test_function(node, lines)
        self.assertAlmostEqual(iq, 2.5)
        
    def test_estimate_assertion_iq_invalid_pragma_value(self):
        """Test invalid pragma values are ignored."""
        code_invalid_pragma = """
# pragma iq=6.0 
def test_invalid_high(self):
    assert True # Should default to IQ 2.0
"""
        node_high = ast.parse(code_invalid_pragma).body[0]
        lines_high = code_invalid_pragma.splitlines()
        iq_high = _estimate_assertion_iq_for_test_function(node_high, lines_high)
        self.assertAlmostEqual(iq_high, 2.0) # Default for simple assert

        code_invalid_pragma_low = """
# pragma iq=0.5 
def test_invalid_low(self):
    assert True # Should default to IQ 2.0
"""
        node_low = ast.parse(code_invalid_pragma_low).body[0]
        lines_low = code_invalid_pragma_low.splitlines()
        iq_low = _estimate_assertion_iq_for_test_function(node_low, lines_low)
        self.assertAlmostEqual(iq_low, 2.0)

        code_invalid_pragma_text = """
# pragma iq=foo 
def test_invalid_text(self):
    assert True # Should default to IQ 2.0
"""
        node_text = ast.parse(code_invalid_pragma_text).body[0]
        lines_text = code_invalid_pragma_text.splitlines()
        iq_text = _estimate_assertion_iq_for_test_function(node_text, lines_text)
        self.assertAlmostEqual(iq_text, 2.0)


    def test_get_mean_assertion_iq_with_pragmas(self):
        """Test mean IQ calculation considering pragma overrides in files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file_1_content = """
# pragma iq=5.0
def test_file1_func1(self):
    pass # IQ overridden to 5.0

def test_file1_func2(self): # pragma iq=4.0
    assert 1 == 1 # IQ overridden to 4.0
"""
            test_file_2_content = """
def test_file2_func1(self):
    assert True # Default IQ ~2.0
"""
            with open(os.path.join(tmpdir, "test_one.py"), "w") as f:
                f.write(test_file_1_content)
            with open(os.path.join(tmpdir, "test_two.py"), "w") as f:
                f.write(test_file_2_content)

            # Expected: (5.0 + 4.0 + 2.0) / 3 = 11.0 / 3 = 3.666...
            mean_iq = get_mean_assertion_iq(tmpdir, {})
            self.assertAlmostEqual(mean_iq, 11.0 / 3.0, places=3)

    def test_get_mean_assertion_iq_no_tests_found(self):
        """Test mean IQ when no test functions are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an empty file
            with open(os.path.join(tmpdir, "test_empty.py"), "w") as f:
                f.write("pass")
            
            # Test with default_assertion_iq_if_no_tests from config
            mean_iq_config = get_mean_assertion_iq(tmpdir, {"default_assertion_iq_if_no_tests": 2.5})
            self.assertAlmostEqual(mean_iq_config, 2.5)

            # Test with hardcoded default in function
            mean_iq_default = get_mean_assertion_iq(tmpdir, {})
            self.assertAlmostEqual(mean_iq_default, 3.0) # Default in function is 3.0

    def test_get_mean_assertion_iq_nonexistent_path(self):
        """Test mean IQ with a non-existent test root path."""
        mean_iq = get_mean_assertion_iq("/non/existent/path/for/testing", {})
        self.assertAlmostEqual(mean_iq, 3.0) # Should return default

if __name__ == "__main__":
    unittest.main()