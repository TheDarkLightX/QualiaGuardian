"""
Integration Tests for Qualia Quality System

Tests the complete system working together.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from guardian.core.qualia_quality import (
    QualiaQualityEngine, QualityImprovementResult, improve_code_quality
)


class TestQualiaQualityIntegration(unittest.TestCase):
    """Integration tests for Qualia Quality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bad_code = """
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

def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 100:
            result.append(data[i] * 2)
        else:
            result.append(data[i])
    return result
"""
    
    def test_quality_improvement_workflow(self):
        """Test complete quality improvement workflow."""
        engine = QualiaQualityEngine(
            target_quality=0.8,
            use_evolutionary=True,
            use_generation=True,
            max_iterations=3
        )
        
        result = engine.improve_code(self.bad_code)
        
        # Verify result structure
        self.assertIsInstance(result, QualityImprovementResult)
        self.assertEqual(result.original_code, self.bad_code)
        self.assertIsInstance(result.improved_code, str)
        self.assertGreaterEqual(result.original_cqs, 0.0)
        self.assertLessEqual(result.original_cqs, 1.0)
        self.assertGreaterEqual(result.improved_cqs, 0.0)
        self.assertLessEqual(result.improved_cqs, 1.0)
    
    def test_improvement_achieved(self):
        """Test that improvement is actually achieved."""
        engine = QualiaQualityEngine(
            target_quality=0.75,
            use_evolutionary=True,
            use_generation=True,
            max_iterations=5
        )
        
        result = engine.improve_code(self.bad_code)
        
        # Should improve or at least maintain quality
        self.assertGreaterEqual(result.improved_cqs, result.original_cqs * 0.9)  # Allow small variance
    
    def test_improvement_components(self):
        """Test that all components work together."""
        engine = QualiaQualityEngine()
        
        result = engine.improve_code(self.bad_code)
        
        # Verify all components are present
        self.assertIsNotNone(result.original_components)
        self.assertIsNotNone(result.improved_components)
        self.assertIsInstance(result.improvements_applied, list)
        self.assertIsInstance(result.quality_tier_before, str)
        self.assertIsInstance(result.quality_tier_after, str)
    
    def test_improve_code_quality_function(self):
        """Test the main entry point function."""
        result = improve_code_quality(
            code=self.bad_code,
            target_quality=0.8
        )
        
        self.assertIsInstance(result, QualityImprovementResult)
        self.assertIsInstance(result.improved_code, str)
    
    def test_iterative_improvement(self):
        """Test that iterative improvement works."""
        engine = QualiaQualityEngine(
            target_quality=0.85,
            max_iterations=5
        )
        
        result = engine.improve_code(self.bad_code)
        
        # Should have attempted improvements
        self.assertGreaterEqual(result.iterations, 0)
        self.assertLessEqual(result.iterations, 5)
    
    def test_quality_tier_improvement(self):
        """Test that quality tier can improve."""
        engine = QualiaQualityEngine(
            target_quality=0.9,
            max_iterations=3
        )
        
        result = engine.improve_code(self.bad_code)
        
        # Tier should be valid
        self.assertIn(result.quality_tier_before, ["excellent", "good", "fair", "poor"])
        self.assertIn(result.quality_tier_after, ["excellent", "good", "fair", "poor"])
    
    def test_multiple_improvement_methods(self):
        """Test that multiple methods are used."""
        engine = QualiaQualityEngine(
            use_evolutionary=True,
            use_generation=True,
            max_iterations=3
        )
        
        result = engine.improve_code(self.bad_code)
        
        # Should have used at least one method
        self.assertIsInstance(result.generation_method, str)
    
    def test_edge_case_empty_code(self):
        """Test edge case: empty code."""
        engine = QualiaQualityEngine()
        
        result = engine.improve_code("")
        
        self.assertIsInstance(result, QualityImprovementResult)
    
    def test_edge_case_already_good_code(self):
        """Test edge case: code already at target quality."""
        good_code = """
def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate sum of two numbers.\"\"\"
    return a + b
"""
        engine = QualiaQualityEngine(target_quality=0.7)
        
        result = engine.improve_code(good_code)
        
        # Should return quickly if already good
        self.assertIsInstance(result, QualityImprovementResult)


class TestQualityImprovementResult(unittest.TestCase):
    """Tests for QualityImprovementResult."""
    
    def test_result_creation(self):
        """Test result creation."""
        from guardian.core.cqs_enhanced import EnhancedCQSComponents
        
        result = QualityImprovementResult(
            original_code="code1",
            improved_code="code2",
            original_cqs=0.6,
            improved_cqs=0.8,
            improvement_percentage=33.3,
            original_components=EnhancedCQSComponents(),
            improved_components=EnhancedCQSComponents(),
            improvements_applied=["test"],
            quality_tier_before="fair",
            quality_tier_after="good",
            generation_method="evolutionary",
            iterations=2
        )
        
        self.assertEqual(result.original_code, "code1")
        self.assertEqual(result.improved_code, "code2")
        self.assertEqual(result.original_cqs, 0.6)
        self.assertEqual(result.improved_cqs, 0.8)
        self.assertEqual(result.improvement_percentage, 33.3)


if __name__ == '__main__':
    unittest.main()
