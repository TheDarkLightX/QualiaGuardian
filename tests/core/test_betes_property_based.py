"""
Property-based tests for bE-TES calculator using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs.
"""
import unittest
import math
from hypothesis import given, strategies as st, assume, settings
from guardian.core.betes import BETESCalculator, BETESComponents
from guardian.core.etes import BETESSettingsV31, BETESWeights


class TestBETESPropertyBased(unittest.TestCase):
    """Property-based tests for bE-TES calculation."""
    
    def setUp(self):
        self.calculator = BETESCalculator()
    
    @given(
        mutation=st.floats(min_value=0.0, max_value=1.0),
        emt=st.floats(min_value=-1.0, max_value=1.0),
        iq=st.floats(min_value=1.0, max_value=5.0),
        coverage=st.floats(min_value=0.0, max_value=1.0),
        time=st.floats(min_value=0.0, max_value=10000.0),
        flakiness=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)  # Reduced for CI speed
    def test_boundedness_property(self, mutation, emt, iq, coverage, time, flakiness):
        """Property: All normalized values and final score are in [0, 1]"""
        result = self.calculator.calculate(mutation, emt, iq, coverage, time, flakiness)
        
        # Final score bounded
        self.assertGreaterEqual(result.betes_score, 0.0)
        self.assertLessEqual(result.betes_score, 1.0)
        
        # All normalized factors bounded
        self.assertGreaterEqual(result.norm_mutation_score, 0.0)
        self.assertLessEqual(result.norm_mutation_score, 1.0)
        self.assertGreaterEqual(result.norm_emt_gain, 0.0)
        self.assertLessEqual(result.norm_emt_gain, 1.0)
        self.assertGreaterEqual(result.norm_assertion_iq, 0.0)
        self.assertLessEqual(result.norm_assertion_iq, 1.0)
        self.assertGreaterEqual(result.norm_behaviour_coverage, 0.0)
        self.assertLessEqual(result.norm_behaviour_coverage, 1.0)
        self.assertGreaterEqual(result.norm_speed_factor, 0.0)
        self.assertLessEqual(result.norm_speed_factor, 1.0)
        self.assertGreaterEqual(result.trust_coefficient_t, 0.0)
        self.assertLessEqual(result.trust_coefficient_t, 1.0)
        self.assertGreaterEqual(result.geometric_mean_g, 0.0)
        self.assertLessEqual(result.geometric_mean_g, 1.0)
    
    @given(
        base_mutation=st.floats(min_value=0.6, max_value=0.95),
        delta=st.floats(min_value=0.0, max_value=0.1)
    )
    @settings(max_examples=50)
    def test_mutation_score_monotonicity(self, base_mutation, delta):
        """Property: Higher mutation score → higher or equal normalized score"""
        assume(base_mutation + delta <= 1.0)
        
        result1 = self.calculator.calculate(base_mutation, 0.1, 3.0, 0.7, 100, 0.05)
        result2 = self.calculator.calculate(base_mutation + delta, 0.1, 3.0, 0.7, 100, 0.05)
        
        self.assertGreaterEqual(result2.norm_mutation_score, result1.norm_mutation_score)
    
    @given(
        base_flakiness=st.floats(min_value=0.0, max_value=0.9),
        delta=st.floats(min_value=0.0, max_value=0.1)
    )
    @settings(max_examples=50)
    def test_flakiness_antimonotonicity(self, base_flakiness, delta):
        """Property: Higher flakiness → lower or equal trust coefficient"""
        assume(base_flakiness + delta <= 1.0)
        
        result1 = self.calculator.calculate(0.8, 0.1, 3.0, 0.7, 100, base_flakiness)
        result2 = self.calculator.calculate(0.8, 0.1, 3.0, 0.7, 100, base_flakiness + delta)
        
        self.assertLessEqual(result2.trust_coefficient_t, result1.trust_coefficient_t)
        self.assertLessEqual(result2.betes_score, result1.betes_score)
    
    @given(
        time1=st.floats(min_value=0.0, max_value=10000.0),
        time2=st.floats(min_value=0.0, max_value=10000.0)
    )
    @settings(max_examples=50)
    def test_speed_factor_monotonicity(self, time1, time2):
        """Property: Higher test time → lower or equal speed factor"""
        assume(time1 < time2)
        
        result1 = self.calculator.calculate(0.8, 0.1, 3.0, 0.7, time1, 0.05)
        result2 = self.calculator.calculate(0.8, 0.1, 3.0, 0.7, time2, 0.05)
        
        self.assertLessEqual(result2.norm_speed_factor, result1.norm_speed_factor)
    
    @given(
        iq1=st.floats(min_value=1.0, max_value=5.0),
        iq2=st.floats(min_value=1.0, max_value=5.0)
    )
    @settings(max_examples=50)
    def test_assertion_iq_monotonicity(self, iq1, iq2):
        """Property: Higher assertion IQ → higher or equal normalized IQ"""
        assume(iq1 < iq2)
        
        result1 = self.calculator.calculate(0.8, 0.1, iq1, 0.7, 100, 0.05)
        result2 = self.calculator.calculate(0.8, 0.1, iq2, 0.7, 100, 0.05)
        
        self.assertGreaterEqual(result2.norm_assertion_iq, result1.norm_assertion_iq)
    
    @given(
        coverage1=st.floats(min_value=0.0, max_value=1.0),
        coverage2=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_coverage_monotonicity(self, coverage1, coverage2):
        """Property: Higher coverage → higher or equal normalized coverage"""
        assume(coverage1 < coverage2)
        
        result1 = self.calculator.calculate(0.8, 0.1, 3.0, coverage1, 100, 0.05)
        result2 = self.calculator.calculate(0.8, 0.1, 3.0, coverage2, 100, 0.05)
        
        self.assertGreaterEqual(result2.norm_behaviour_coverage, result1.norm_behaviour_coverage)
    
    def test_zero_flakiness_gives_full_trust(self):
        """Property: Zero flakiness → trust coefficient = 1.0"""
        result = self.calculator.calculate(0.8, 0.1, 3.0, 0.7, 100, 0.0)
        self.assertEqual(result.trust_coefficient_t, 1.0)
    
    def test_max_flakiness_gives_zero_trust(self):
        """Property: Flakiness = 1.0 → trust coefficient = 0.0"""
        result = self.calculator.calculate(0.8, 0.1, 3.0, 0.7, 100, 1.0)
        self.assertEqual(result.trust_coefficient_t, 0.0)
        self.assertEqual(result.betes_score, 0.0)
    
    @given(
        mutation=st.floats(min_value=0.0, max_value=1.0),
        emt=st.floats(min_value=-1.0, max_value=1.0),
        iq=st.floats(min_value=1.0, max_value=5.0),
        coverage=st.floats(min_value=0.0, max_value=1.0),
        time=st.floats(min_value=0.0, max_value=10000.0)
    )
    @settings(max_examples=50)
    def test_zero_flakiness_preserves_geometric_mean(self, mutation, emt, iq, coverage, time):
        """Property: With zero flakiness, bE-TES = geometric mean"""
        result = self.calculator.calculate(mutation, emt, iq, coverage, time, 0.0)
        self.assertAlmostEqual(result.betes_score, result.geometric_mean_g, places=10)
    
    @given(
        mutation=st.floats(min_value=0.0, max_value=1.0),
        emt=st.floats(min_value=-1.0, max_value=1.0),
        iq=st.floats(min_value=1.0, max_value=5.0),
        coverage=st.floats(min_value=0.0, max_value=1.0),
        time=st.floats(min_value=0.0, max_value=10000.0),
        flakiness=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_final_score_formula(self, mutation, emt, iq, coverage, time, flakiness):
        """Property: bE-TES = G * T (geometric mean * trust coefficient)"""
        result = self.calculator.calculate(mutation, emt, iq, coverage, time, flakiness)
        expected = result.geometric_mean_g * result.trust_coefficient_t
        self.assertAlmostEqual(result.betes_score, expected, places=10)


class TestBETESSigmoidProperties(unittest.TestCase):
    """Property-based tests for sigmoid normalization."""
    
    @given(
        value=st.floats(min_value=-10.0, max_value=10.0),
        k=st.floats(min_value=0.1, max_value=50.0),
        center=st.floats(min_value=-5.0, max_value=5.0)
    )
    @settings(max_examples=50)
    def test_sigmoid_boundedness(self, value, k, center):
        """Property: Sigmoid output is always in [0, 1]"""
        settings = BETESSettingsV31(smooth_m=True, k_m=k)
        calc = BETESCalculator(settings_v3_1=settings)
        # Use mutation score normalization with sigmoid
        result = calc.calculate(value, 0.1, 3.0, 0.7, 100, 0.05)
        self.assertGreaterEqual(result.norm_mutation_score, 0.0)
        self.assertLessEqual(result.norm_mutation_score, 1.0)
    
    @given(
        value1=st.floats(min_value=-10.0, max_value=10.0),
        value2=st.floats(min_value=-10.0, max_value=10.0),
        k=st.floats(min_value=0.1, max_value=50.0),
        center=st.floats(min_value=-5.0, max_value=5.0)
    )
    @settings(max_examples=50)
    def test_sigmoid_monotonicity(self, value1, value2, k, center):
        """Property: Sigmoid is monotonic (higher input → higher output)"""
        assume(value1 < value2)
        settings = BETESSettingsV31(smooth_m=True, k_m=k)
        calc = BETESCalculator(settings_v3_1=settings)
        
        result1 = calc.calculate(value1, 0.1, 3.0, 0.7, 100, 0.05)
        result2 = calc.calculate(value2, 0.1, 3.0, 0.7, 100, 0.05)
        
        self.assertGreaterEqual(result2.norm_mutation_score, result1.norm_mutation_score)
    
    def test_sigmoid_center_property(self):
        """Property: At center point, sigmoid = 0.5"""
        settings = BETESSettingsV31(smooth_m=True, k_m=14.0)
        calc = BETESCalculator(settings_v3_1=settings)
        # Center is 0.775 for mutation score
        result = calc.calculate(0.775, 0.1, 3.0, 0.7, 100, 0.05)
        self.assertAlmostEqual(result.norm_mutation_score, 0.5, places=3)


if __name__ == '__main__':
    unittest.main()
