import unittest
import tempfile
import os
import math
from pathlib import Path

from guardian.sensors.shannon_entropy_sensor import ShannonEntropySensor, PYTHON_KEYWORDS_OPERATORS

class TestShannonEntropySensor(unittest.TestCase):

    def setUp(self):
        self.sensor = ShannonEntropySensor()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = Path(self.temp_dir.name) / "test_file.py"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_to_test_file(self, content: str):
        with open(self.test_file_path, "wb") as f: # Write as bytes
            f.write(content.encode('utf-8'))

    def test_calculate_entropy_empty_file(self):
        self._write_to_test_file("")
        entropy = self.sensor.calculate_entropy_for_file(str(self.test_file_path))
        self.assertEqual(entropy, 0.0, "Entropy of an empty file should be 0.0")

    def test_calculate_entropy_no_relevant_tokens(self):
        content = "variable_name = 123\n# A simple comment\n"
        self._write_to_test_file(content)
        entropy = self.sensor.calculate_entropy_for_file(str(self.test_file_path))
        self.assertEqual(entropy, 0.0, "Entropy should be 0.0 if no relevant tokens are present")

    def test_calculate_entropy_single_token_type(self):
        content = "if if if if" # Only 'if' keyword
        # Relevant tokens: 'if' (4 times)
        # P(if) = 1.0
        # Entropy = -1.0 * log2(1.0) = 0
        self._write_to_test_file(content)
        entropy = self.sensor.calculate_entropy_for_file(str(self.test_file_path))
        self.assertAlmostEqual(entropy, 0.0, places=6, msg="Entropy of single token type should be 0")

    def test_calculate_entropy_two_equally_probable_tokens(self):
        content = "if else if else" # 'if': 2, 'else': 2. Total relevant: 4
        # P(if) = 0.5, P(else) = 0.5
        # Entropy = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = - (0.5 * -1 + 0.5 * -1) = 1.0
        self._write_to_test_file(content)
        entropy = self.sensor.calculate_entropy_for_file(str(self.test_file_path))
        self.assertAlmostEqual(entropy, 1.0, places=6, msg="Entropy for two equally probable tokens should be 1.0")

    def test_calculate_entropy_simple_function(self):
        content = """
def simple_function(a, b):
    if a > b: # if, >, 
        return a + b # return, +, 
    else: # else
        return a - b # return, -
"""
        # Relevant tokens (approximate count, depends on exact PYTHON_KEYWORDS_OPERATORS):
        # def: 1, (: 1, ): 1, if: 1, >: 1, return: 2, +: 1, else: 1, -: 1
        # Total relevant tokens: 10
        # Counts: def:1, (:1, ):1, if:1, >:1, return:2, +:1, else:1, -:1
        # P(def)=0.1, P(()=0.1, P())=0.1, P(if)=0.1, P(>)=0.1, P(return)=0.2, P(+)=0.1, P(else)=0.1, P(-)=0.1
        # Entropy = - (7 * (0.1 * log2(0.1)) + 1 * (0.2 * log2(0.2)))
        #         = - (7 * (0.1 * -3.3219) + 1 * (0.2 * -2.3219))
        #         = - (7 * -0.33219 + -0.46438)
        #         = - (-2.32533 - 0.46438) = - (-2.78971) = 2.78971
        # This is a manual calculation, actual result depends on the exact token set and tokenizer behavior.
        self._write_to_test_file(content)
        entropy = self.sensor.calculate_entropy_for_file(str(self.test_file_path))
        self.assertIsNotNone(entropy)
        self.assertGreater(entropy, 2.5, "Entropy for a simple function should be positive and non-trivial")
        self.assertLess(entropy, 3.5, "Entropy for a simple function should be within expected range")


    def test_file_not_found(self):
        entropy = self.sensor.calculate_entropy_for_file("non_existent_file_for_test.py")
        self.assertIsNone(entropy, "Should return None for a non-existent file")

    def test_custom_token_set(self):
        custom_tokens = {'def', 'return'}
        sensor_custom = ShannonEntropySensor(relevant_tokens=custom_tokens)
        content = """
def my_func(): # def
    x = 1
    return x   # return
# if else pass
"""
        # Relevant: def, return. Total: 2
        # P(def)=0.5, P(return)=0.5. Entropy = 1.0
        self._write_to_test_file(content)
        entropy = sensor_custom.calculate_entropy_for_file(str(self.test_file_path))
        self.assertAlmostEqual(entropy, 1.0, places=6)

    def test_tokenization_error(self):
        # Create content that will cause a tokenize.TokenError (e.g., unmatched parenthesis at EOF)
        # However, generate_tokens is quite robust. Let's try invalid UTF-8 sequence.
        # This might raise UnicodeDecodeError before TokenError, handled by general Exception.
        # A more direct way to cause TokenError is harder without deep tokenizer internals.
        # The sensor's current error handling for tokenization is broad.
        # For now, test that it returns None on general processing error.
        invalid_content_bytes = b"def func():\n\tpass\n\xff # Invalid UTF-8 byte"
        entropy = self.sensor.calculate_entropy_for_content(invalid_content_bytes)
        # Depending on where the error is caught, it might be None.
        # The current code catches general Exception for tokenization issues.
        self.assertIsNone(entropy, "Entropy should be None if tokenization fails")


if __name__ == '__main__':
    unittest.main()