"""
Unit Tests for Enhanced CQS

Tests semantic analysis, pattern recognition, and code smell detection.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from guardian.core.cqs_enhanced import EnhancedCQSCalculator, EnhancedCQSComponents


class TestEnhancedCQS(unittest.TestCase):
    """Unit tests for Enhanced CQS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = EnhancedCQSCalculator()
    
    def test_semantic_clarity(self):
        """Test semantic clarity calculation."""
        # Good: name matches behavior
        good_code = """
def calculate_sum(a, b):
    return a + b
"""
        good_result = self.calc.calculate_enhanced(good_code)
        
        # Bad: name doesn't match (says calculate but doesn't return)
        bad_code = """
def calculate_sum(a, b):
    print(a + b)
"""
        bad_result = self.calc.calculate_enhanced(bad_code)
        
        self.assertGreater(good_result.semantic_clarity, bad_result.semantic_clarity)
    
    def test_pattern_quality(self):
        """Test pattern quality detection."""
        # Code with quality patterns
        good_code = """
def process_data(data: list) -> list:
    \"\"\"Process data.\"\"\"
    if not data:
        return []
    return [x * 2 for x in data if x > 0]
"""
        good_result = self.calc.calculate_enhanced(good_code)
        
        # Code without patterns
        bad_code = """
def proc(d):
    r = []
    for i in range(len(d)):
        if d[i] > 0:
            r.append(d[i] * 2)
    return r
"""
        bad_result = self.calc.calculate_enhanced(bad_code)
        
        self.assertGreater(good_result.pattern_quality, bad_result.pattern_quality)
        self.assertGreater(len(good_result.quality_patterns), len(bad_result.quality_patterns))
    
    def test_code_smell_detection(self):
        """Test code smell detection."""
        # Code with smells
        smelly_code = """
def process(x, y, z, a, b, c, d, e, f):
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        return x + y + z + a + b + c + d + e + f
    return 0
"""
        result = self.calc.calculate_enhanced(smelly_code)
        
        self.assertGreater(len(result.code_smells), 0)
        self.assertIn("Deep nesting", str(result.code_smells))
    
    def test_testability_scoring(self):
        """Test testability calculation."""
        # Testable: pure function
        testable_code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        testable_result = self.calc.calculate_enhanced(testable_code)
        
        # Not testable: many parameters, side effects
        untestable_code = """
def process(a, b, c, d, e, f, g):
    global state
    state = a + b
    print(c)
    return d
"""
        untestable_result = self.calc.calculate_enhanced(untestable_code)
        
        self.assertGreater(testable_result.testability, untestable_result.testability)
    
    def test_enhanced_score_calculation(self):
        """Test enhanced CQS score calculation."""
        code = """
def calculate(x: int) -> int:
    \"\"\"Calculate value.\"\"\"
    return x * 2
"""
        result = self.calc.calculate_enhanced(code)
        
        self.assertGreater(result.enhanced_cqs_score, 0.0)
        self.assertLessEqual(result.enhanced_cqs_score, 1.0)
        # Enhanced should be different from base (includes more factors)
        self.assertNotEqual(result.enhanced_cqs_score, result.cqs_score)
    
    def test_quality_tier_determination(self):
        """Test quality tier determination."""
        # Excellent code
        excellent_code = """
def calculate_total(items: list[float]) -> float:
    \"\"\"Calculate total of items.\"\"\"
    return sum(items)
"""
        excellent_result = self.calc.calculate_enhanced(excellent_code)
        self.assertEqual(excellent_result.quality_tier, "excellent" if excellent_result.enhanced_cqs_score >= 0.9 else "good")
        
        # Poor code
        poor_code = """
def calc(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x * y * z
"""
        poor_result = self.calc.calculate_enhanced(poor_code)
        self.assertIn(poor_result.quality_tier, ["fair", "poor"])
    
    def test_improvement_priorities(self):
        """Test improvement priority generation."""
        code = """
def calc(x):
    if x > 0:
        if x < 100:
            return x * 2
"""
        result = self.calc.calculate_enhanced(code)
        
        self.assertIsInstance(result.improvement_priorities, list)
        self.assertGreater(len(result.improvement_priorities), 0)
    
    def test_generated_improvements(self):
        """Test generated improvement suggestions."""
        code = """
def process(x):
    if x > 0:
        if x < 100:
            return x * 2
"""
        result = self.calc.calculate_enhanced(code)
        
        self.assertIsInstance(result.generated_improvements, list)
        # Should have suggestions for code with issues
        if result.enhanced_cqs_score < 0.8:
            self.assertGreater(len(result.generated_improvements), 0)


if __name__ == '__main__':
    unittest.main()
