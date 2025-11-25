"""
Tests for numerical stability and edge cases in bE-TES calculation.
"""
import unittest
import math
import numpy as np
from guardian.core.betes import BETESCalculator
from guardian.core.etes import BETESSettingsV31, BETESWeights


class TestBETESNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases."""
    
    def test_very_small_factors(self):
        """Test with very small normalized factors (potential underflow)."""
        calc = BETESCalculator()
        # Create scenario where factors are very small
        # This tests geometric mean with near-zero values
        result = calc.calculate(0.6, -1.0, 1.0, 0.001, 10000, 0.0)
        # Should not crash or produce NaN
        self.assertFalse(math.isnan(result.betes_score))
        self.assertFalse(math.isinf(result.betes_score))
        self.assertGreaterEqual(result.betes_score, 0.0)
    
    def test_very_large_test_time(self):
        """Test with very large test execution time."""
        calc = BETESCalculator()
        result = calc.calculate(0.8, 0.1, 3.0, 0.7, 1e6, 0.05)  # 1 million ms
        # Should handle gracefully
        self.assertGreaterEqual(result.norm_speed_factor, 0.0)
        self.assertLessEqual(result.norm_speed_factor, 1.0)
        self.assertFalse(math.isnan(result.norm_speed_factor))
    
    def test_sigmoid_extreme_values(self):
        """Test sigmoid with extreme input values."""
        settings = BETESSettingsV31(smooth_m=True, k_m=100.0)  # Very steep
        calc = BETESCalculator(settings_v3_1=settings)
        
        # Very far from center
        result1 = calc.calculate(-100.0, 0.1, 3.0, 0.7, 100, 0.05)
        result2 = calc.calculate(100.0, 0.1, 3.0, 0.7, 100, 0.05)
        
        # Should be near 0 and 1 respectively, but not NaN/Inf
        self.assertFalse(math.isnan(result1.norm_mutation_score))
        self.assertFalse(math.isnan(result2.norm_mutation_score))
        self.assertLess(result1.norm_mutation_score, 0.01)
        self.assertGreater(result2.norm_mutation_score, 0.99)
    
    def test_all_weights_zero(self):
        """Test geometric mean with all weights zero (edge case)."""
        import math
        weights = BETESWeights(w_m=0.0, w_e=0.0, w_a=0.0, w_b=0.0, w_s=0.0)
        calc = BETESCalculator(weights=weights)
        result = calc.calculate(0.8, 0.1, 3.0, 0.7, 100, 0.05)
        # Mathematically undefined - should return NaN
        self.assertTrue(math.isnan(result.geometric_mean_g))
    
    def test_mixed_zero_weights(self):
        """Test with some weights zero."""
        weights = BETESWeights(w_m=1.0, w_e=0.0, w_a=1.0, w_b=0.0, w_s=1.0)
        calc = BETESCalculator(weights=weights)
        result = calc.calculate(0.8, 0.1, 3.0, 0.7, 100, 0.05)
        # Should calculate correctly (zero weights mean those factors don't contribute)
        self.assertFalse(math.isnan(result.geometric_mean_g))
        self.assertGreaterEqual(result.geometric_mean_g, 0.0)
    
    def test_zero_mutation_score(self):
        """Test with zero mutation score."""
        calc = BETESCalculator()
        result = calc.calculate(0.0, 0.1, 3.0, 0.7, 100, 0.05)
        # Should result in zero geometric mean and thus zero score
        self.assertEqual(result.geometric_mean_g, 0.0)
        self.assertEqual(result.betes_score, 0.0)
    
    def test_boundary_speed_threshold(self):
        """Test speed factor at boundary (100ms)."""
        calc = BETESCalculator()
        # Just below threshold
        result1 = calc.calculate(0.8, 0.1, 3.0, 0.7, 99.999, 0.05)
        # At threshold
        result2 = calc.calculate(0.8, 0.1, 3.0, 0.7, 100.0, 0.05)
        # Just above threshold
        result3 = calc.calculate(0.8, 0.1, 3.0, 0.7, 100.001, 0.05)
        
        # All should be valid
        self.assertEqual(result1.norm_speed_factor, 1.0)
        self.assertEqual(result2.norm_speed_factor, 1.0)
        # Note: There may be a discontinuity here - this test documents it
        self.assertGreaterEqual(result3.norm_speed_factor, 0.0)
        self.assertLessEqual(result3.norm_speed_factor, 1.0)
    
    def test_negative_inputs_clamped(self):
        """Test that negative inputs are properly clamped."""
        calc = BETESCalculator()
        # Negative values should be clamped
        result = calc.calculate(-1.0, -2.0, 0.0, -0.5, -10, -0.1)
        # All normalized values should be in [0, 1]
        self.assertGreaterEqual(result.norm_mutation_score, 0.0)
        self.assertGreaterEqual(result.norm_emt_gain, 0.0)
        self.assertGreaterEqual(result.norm_assertion_iq, 0.0)
        self.assertGreaterEqual(result.norm_behaviour_coverage, 0.0)
        self.assertGreaterEqual(result.norm_speed_factor, 0.0)
    
    def test_assertion_iq_out_of_range(self):
        """Test assertion IQ outside [1, 5] range."""
        calc = BETESCalculator()
        # Below minimum
        result1 = calc.calculate(0.8, 0.1, 0.5, 0.7, 100, 0.05)
        # Above maximum
        result2 = calc.calculate(0.8, 0.1, 10.0, 0.7, 100, 0.05)
        
        # Should be clamped to [0, 1]
        self.assertGreaterEqual(result1.norm_assertion_iq, 0.0)
        self.assertLessEqual(result1.norm_assertion_iq, 1.0)
        self.assertGreaterEqual(result2.norm_assertion_iq, 0.0)
        self.assertLessEqual(result2.norm_assertion_iq, 1.0)
        # Below min should give 0, above max should give 1
        self.assertEqual(result1.norm_assertion_iq, 0.0)
        self.assertEqual(result2.norm_assertion_iq, 1.0)
    
    def test_flakiness_out_of_range(self):
        """Test flakiness outside [0, 1] range."""
        calc = BETESCalculator()
        # Negative flakiness
        result1 = calc.calculate(0.8, 0.1, 3.0, 0.7, 100, -0.1)
        # Flakiness > 1
        result2 = calc.calculate(0.8, 0.1, 3.0, 0.7, 100, 1.5)
        
        # Trust coefficient should be clamped
        self.assertGreaterEqual(result1.trust_coefficient_t, 0.0)
        self.assertLessEqual(result1.trust_coefficient_t, 1.0)
        self.assertGreaterEqual(result2.trust_coefficient_t, 0.0)
        self.assertLessEqual(result2.trust_coefficient_t, 1.0)
        # Negative should give T > 1 (clamped to 1), >1 should give T < 0 (clamped to 0)
        self.assertEqual(result1.trust_coefficient_t, 1.0)
        self.assertEqual(result2.trust_coefficient_t, 0.0)


if __name__ == '__main__':
    unittest.main()
