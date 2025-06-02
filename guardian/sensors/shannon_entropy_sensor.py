"""
Sensor for calculating Shannon Entropy of code, as a measure of complexity.
"""
import tokenize
import math
import logging
from collections import Counter
from io import BytesIO
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Define a set of Python keywords and common operators to consider for entropy calculation.
# This set can be expanded or refined.
PYTHON_KEYWORDS_OPERATORS: Set[str] = {
    # Keywords
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
    'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
    'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
    'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
    # Common Operators (represented by their token names or string values)
    '+', '-', '*', '**', '/', '//', '%', '@', '<<', '>>', '&', '|', '^', '~',
    ':=', '<', '>', '<=', '>=', '==', '!=', '(', ')', '[', ']', '{', '}', ',',
    ':', '.', ';', '=', '->', '+=', '-=', '*=', '/=', '//=', '%=', '@=', '&=',
    '|=', '^=', '>>=', '<<=', '**='
    # Token names for operators can also be used, e.g., token.PLUS, token.MINUS
    # For simplicity, using string representations that tokenize.generate_tokens produces for common ops.
}

class ShannonEntropySensor:
    """
    Calculates the Shannon entropy of tokens in a Python source file.
    A higher entropy can indicate more complex or less predictable code structure.
    """

    def __init__(self, relevant_tokens: Optional[Set[str]] = None):
        """
        Initializes the sensor.

        Args:
            relevant_tokens: A set of token strings to consider for entropy calculation.
                             If None, defaults to PYTHON_KEYWORDS_OPERATORS.
        """
        self.relevant_tokens = relevant_tokens or PYTHON_KEYWORDS_OPERATORS

    def calculate_entropy_for_file(self, file_path: str) -> Optional[float]:
        """
        Calculates the Shannon entropy for the relevant tokens in a given source file.

        Args:
            file_path: Path to the Python source file.

        Returns:
            The calculated Shannon entropy (H_code) as a float, or None if an error occurs
            or no relevant tokens are found.
        """
        try:
            with open(file_path, 'rb') as f: # tokenize needs bytes
                return self.calculate_entropy_for_content(f.read())
        except FileNotFoundError:
            logger.error(f"ShannonEntropySensor: File not found at {file_path}")
            return None
        except Exception as e:
            logger.error(f"ShannonEntropySensor: Error reading file {file_path}: {e}")
            return None

    def calculate_entropy_for_content(self, content_bytes: bytes) -> Optional[float]:
        """
        Calculates Shannon entropy for the relevant tokens in the given byte content.

        Args:
            content_bytes: Byte string of the Python source code.

        Returns:
            The calculated Shannon entropy (H_code) as a float, or None if an error occurs
            or no relevant tokens are found.
        """
        tokens = []
        try:
            # Convert bytes to a file-like object for generate_tokens
            content_io = BytesIO(content_bytes)
            for token_info in tokenize.generate_tokens(content_io.readline):
                # We are interested in the string value of keywords and operators
                if token_info.string in self.relevant_tokens:
                    tokens.append(token_info.string)
                # Additionally, capture token names for some types if not covered by string
                elif tokenize.tok_name[token_info.type] == 'OP' and token_info.string in self.relevant_tokens:
                     tokens.append(token_info.string) # Already covered if string is in relevant_tokens
                elif tokenize.tok_name[token_info.type] == 'KEYWORD' and token_info.string in self.relevant_tokens:
                     tokens.append(token_info.string) # Already covered

        except tokenize.TokenError as e:
            logger.error(f"ShannonEntropySensor: Tokenization error: {e}")
            return None
        except Exception as e:
            logger.error(f"ShannonEntropySensor: Error processing content: {e}")
            return None

        if not tokens:
            logger.warning("ShannonEntropySensor: No relevant tokens found in the content.")
            return 0.0 # Or None, depending on how we want to treat empty/irrelevant files

        counts = Counter(tokens)
        total_tokens = len(tokens)
        entropy = 0.0

        for token_type in counts:
            probability = counts[token_type] / total_tokens
            if probability > 0: # Avoid math.log(0)
                entropy -= probability * math.log2(probability)
        
        logger.debug(f"Calculated Shannon entropy: {entropy:.4f} from {total_tokens} relevant tokens.")
        return entropy

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    sensor = ShannonEntropySensor()

    # Create a dummy Python file for testing
    dummy_file_path = "dummy_test_entropy_file.py"
    dummy_content_simple = """
def simple_function(a, b):
    if a > b:
        return a + b
    else:
        return a - b
"""
    dummy_content_complex = """
import os
import sys

class MyClass:
    def __init__(self, x):
        self.x = x

    async def complex_method(self, y, z=10):
        if self.x > y and (z % 2 == 0 or not self.x < 0):
            try:
                for i in range(y):
                    self.x += (i ** 2) / max(1, z)
                    await asyncio.sleep(0.01)
                return self.x
            except Exception as e:
                print(f"Error: {e}")
                raise
        elif y < 5:
            return None
        else:
            return self.x - y * z
"""
    with open(dummy_file_path, "wb") as f: # Write as bytes
        f.write(dummy_content_simple.encode('utf-8'))
    
    entropy_simple = sensor.calculate_entropy_for_file(dummy_file_path)
    if entropy_simple is not None:
        logger.info(f"Shannon Entropy for simple dummy file: {entropy_simple:.4f}")

    with open(dummy_file_path, "wb") as f: # Write as bytes
        f.write(dummy_content_complex.encode('utf-8'))

    entropy_complex = sensor.calculate_entropy_for_file(dummy_file_path)
    if entropy_complex is not None:
        logger.info(f"Shannon Entropy for complex dummy file: {entropy_complex:.4f}")

    # Test with non-existent file
    sensor.calculate_entropy_for_file("non_existent_file.py")

    # Test with empty content
    empty_entropy = sensor.calculate_entropy_for_content(b"")
    logger.info(f"Shannon Entropy for empty content: {empty_entropy}")
    
    # Test with content having no relevant tokens
    irrelevant_entropy = sensor.calculate_entropy_for_content(b"my_variable_name = 123\n# This is a comment")
    logger.info(f"Shannon Entropy for irrelevant content: {irrelevant_entropy}")


    # Clean up dummy file
    import os
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)