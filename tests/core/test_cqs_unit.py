"""
Unit Tests for Code Quality Score (CQS)

Tests individual components and functions in isolation.
"""

import unittest
import ast
import sys
import os

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from guardian.core.cqs import CQSCalculator, CQSComponents


class TestCQSCalculator(unittest.TestCase):
    """Unit tests for CQS Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = CQSCalculator()
    
    def test_calculate_simple_code(self):
        """Test CQS calculation for simple, clean code."""
        code = """
def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a + b
"""
        result = self.calc.calculate_from_code(code)
        
        self.assertIsInstance(result, CQSComponents)
        self.assertGreater(result.cqs_score, 0.0)
        self.assertLessEqual(result.cqs_score, 1.0)
        self.assertGreater(result.readability_score, 0.7)  # Good code should be readable
        self.assertGreater(result.simplicity_score, 0.7)  # Simple code
    
    def test_calculate_complex_code(self):
        """Test CQS calculation for complex, messy code."""
        code = """
def calc(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x * y * z
            else:
                return 0
        else:
            return 0
    else:
        return 0
"""
        result = self.calc.calculate_from_code(code)
        
        self.assertIsInstance(result, CQSComponents)
        self.assertLess(result.cqs_score, 0.8)  # Complex code should score lower
        self.assertLess(result.simplicity_score, 0.7)  # Not simple
    
    def test_readability_naming(self):
        """Test naming quality assessment."""
        # Good naming
        good_code = """
def calculate_total_price(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
        good_result = self.calc.calculate_from_code(good_code)
        
        # Bad naming
        bad_code = """
def calc(x):
    t = 0
    for i in x:
        t += i
    return t
"""
        bad_result = self.calc.calculate_from_code(bad_code)
        
        self.assertGreater(good_result.naming_quality, bad_result.naming_quality)
    
    def test_simplicity_function_size(self):
        """Test function size scoring."""
        # Small function
        small_code = """
def add(a, b):
    return a + b
"""
        small_result = self.calc.calculate_from_code(small_code)
        
        # Large function (simulated)
        large_code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            if data[i] < 100:
                result.append(data[i] * 2)
            else:
                result.append(data[i])
        else:
            result.append(0)
    return result
"""
        large_result = self.calc.calculate_from_code(large_code)
        
        self.assertGreater(small_result.function_size_score, large_result.function_size_score)
    
    def test_complexity_scoring(self):
        """Test complexity scoring."""
        # Low complexity
        simple_code = """
def is_positive(n):
    return n > 0
"""
        simple_result = self.calc.calculate_from_code(simple_code)
        
        # High complexity
        complex_code = """
def process(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                if x + y > z:
                    if x + z > y:
                        if y + z > x:
                            return True
    return False
"""
        complex_result = self.calc.calculate_from_code(complex_code)
        
        self.assertGreater(simple_result.complexity_score, complex_result.complexity_score)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty code
        empty_result = self.calc.calculate_from_code("")
        self.assertIsInstance(empty_result, CQSComponents)
        
        # Single line
        single_result = self.calc.calculate_from_code("x = 1")
        self.assertIsInstance(single_result, CQSComponents)
        
        # Syntax error
        error_result = self.calc.calculate_from_code("def invalid syntax")
        self.assertIsInstance(error_result, CQSComponents)
        self.assertGreater(len(error_result.insights), 0)
    
    def test_insights_generation(self):
        """Test that insights are generated."""
        code = """
def bad_function(x):
    if x > 0:
        if x < 100:
            if x % 2 == 0:
                return x * 2
            else:
                return x
        else:
            return 0
    else:
        return 0
"""
        result = self.calc.calculate_from_code(code)
        
        self.assertIsInstance(result.insights, list)
        self.assertGreater(len(result.insights), 0)
    
    def test_suggestions_generation(self):
        """Test that suggestions are generated."""
        code = """
def calc(x):
    return x * 2
"""
        result = self.calc.calculate_from_code(code)
        
        self.assertIsInstance(result.improvement_suggestions, list)


class TestCQSComponents(unittest.TestCase):
    """Unit tests for CQS Components."""
    
    def test_component_initialization(self):
        """Test component initialization."""
        components = CQSComponents()
        
        self.assertEqual(components.readability_score, 0.0)
        self.assertEqual(components.simplicity_score, 0.0)
        self.assertEqual(components.maintainability_score, 0.0)
        self.assertEqual(components.clarity_score, 0.0)
        self.assertEqual(components.cqs_score, 0.0)
        self.assertEqual(len(components.insights), 0)
        self.assertEqual(len(components.improvement_suggestions), 0)


if __name__ == '__main__':
    unittest.main()
