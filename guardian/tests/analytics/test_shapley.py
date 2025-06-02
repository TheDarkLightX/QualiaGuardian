import unittest
from unittest.mock import MagicMock
from pathlib import Path
from typing import List, Dict

from guardian_ai_tool.guardian.analytics.shapley import calculate_shapley_values, TestId

class TestShapleyValues(unittest.TestCase):

    def test_calculate_shapley_values_simple_case(self):
        """Test with a simple 2-test case with predictable contributions."""
        test_a = Path("test_a.py")
        test_b = Path("test_b.py")
        all_tests: List[TestId] = [test_a, test_b]

        # Mock metric evaluator:
        # score({}) = 0
        # score({a}) = 0.6
        # score({b}) = 0.3
        # score({a, b}) = 0.8
        # Expected Shapley for A: ( (0.6-0) + (0.8-0.3) ) / 2 = (0.6 + 0.5) / 2 = 0.55
        # Expected Shapley for B: ( (0.3-0) + (0.8-0.6) ) / 2 = (0.3 + 0.2) / 2 = 0.25
        def mock_evaluator(subset: List[TestId]) -> float:
            if not subset:
                return 0.0
            if len(subset) == 1:
                if subset[0] == test_a:
                    return 0.6
                if subset[0] == test_b:
                    return 0.3
            if len(subset) == 2: # {a, b}
                return 0.8
            return 0.0

        # Use enough permutations for a deterministic result in this simple case,
        # or mock random.shuffle if needed for perfect determinism with low permutations.
        # For this direct calculation, permutations don't strictly matter if all permutations are implicitly covered.
        # The actual implementation uses sampling, so for testing exact values,
        # we might need to mock the sampling or use a very high number of permutations.
        # Let's use a reasonable number for a functional test.
        shapley_values = calculate_shapley_values(all_tests, mock_evaluator, num_permutations=100)

        self.assertIn(test_a, shapley_values)
        self.assertIn(test_b, shapley_values)
        # Due to sampling, we check for approximate values or relative order.
        # For a more precise check, we'd need to control the permutations or mock random.shuffle.
        # For now, let's check if A > B, which should hold.
        self.assertGreater(shapley_values[test_a], shapley_values[test_b])
        # Approximate check (can be flaky with low permutations)
        self.assertAlmostEqual(shapley_values[test_a], 0.55, delta=0.15) # Increased delta for sampling
        self.assertAlmostEqual(shapley_values[test_b], 0.25, delta=0.15) # Increased delta for sampling


    def test_calculate_shapley_values_single_test(self):
        """Test with a single test."""
        test_a = Path("test_a.py")
        all_tests: List[TestId] = [test_a]

        def mock_evaluator(subset: List[TestId]) -> float:
            if subset and subset[0] == test_a:
                return 0.7
            return 0.0

        shapley_values = calculate_shapley_values(all_tests, mock_evaluator, num_permutations=10)
        self.assertEqual(len(shapley_values), 1)
        self.assertIn(test_a, shapley_values)
        self.assertAlmostEqual(shapley_values[test_a], 0.7)

    def test_calculate_shapley_values_empty_set(self):
        """Test with an empty set of tests."""
        all_tests: List[TestId] = []
        mock_evaluator = MagicMock(return_value=0.0)
        
        shapley_values = calculate_shapley_values(all_tests, mock_evaluator, num_permutations=10)
        self.assertEqual(len(shapley_values), 0)

    def test_calculate_shapley_values_zero_contribution(self):
        """Test where one test adds no value."""
        test_a = Path("test_a.py")
        test_b = Path("test_b.py") # test_b adds no value
        all_tests: List[TestId] = [test_a, test_b]

        def mock_evaluator(subset: List[TestId]) -> float:
            if not subset: return 0.0
            if test_a in subset: return 0.9 # Score is always 0.9 if A is present
            return 0.0 # Otherwise 0

        # Expected: A gets 0.9, B gets 0.0
        shapley_values = calculate_shapley_values(all_tests, mock_evaluator, num_permutations=50)
        self.assertAlmostEqual(shapley_values[test_a], 0.9, delta=0.1)
        self.assertAlmostEqual(shapley_values[test_b], 0.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()