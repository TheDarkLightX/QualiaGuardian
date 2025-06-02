"""
Unit tests for the Architecture Violation Score sensor.
"""
import unittest
from unittest.mock import patch
from guardian.sensors.arch import get_raw_architectural_violation_score

class TestArchSensor(unittest.TestCase):

    def test_get_raw_architectural_violation_score_placeholder(self):
        """Test the placeholder architecture violation sensor."""
        with patch('guardian.sensors.arch.logger') as mock_logger:
            score = get_raw_architectural_violation_score("/fake/project/path", {})
            self.assertIsInstance(score, float)
            self.assertTrue(0.0 <= score <= 1.0, "Score should be between 0.0 and 1.0")
            mock_logger.info.assert_any_call("Simulating architectural violation analysis for project at: /fake/project/path")
            
            # Check default calculation (5 violations / 20 threshold = 0.25)
            self.assertAlmostEqual(score, 5.0 / 20.0)

    def test_get_raw_architectural_violation_score_with_config_threshold(self):
        """Test the sensor with a custom normalization threshold from config."""
        config = {"arch_max_violations_threshold": 10.0}
        with patch('guardian.sensors.arch.logger') as mock_logger:
            score = get_raw_architectural_violation_score("/fake/project/path", config)
            self.assertIsInstance(score, float)
            self.assertTrue(0.0 <= score <= 1.0)
            # Default internal simulated violations is 5. So, 5 / 10 = 0.5
            self.assertAlmostEqual(score, 5.0 / 10.0)

    def test_get_raw_architectural_violation_score_capping_at_one(self):
        """Test that the score is capped at 1.0 if violations exceed threshold."""
        # Simulate more violations internally for this test case by mocking the internal value
        # or by setting a very low threshold.
        config = {"arch_max_violations_threshold": 3.0} # Threshold is 3, simulated violations is 5
        with patch('guardian.sensors.arch.logger') as mock_logger:
            score = get_raw_architectural_violation_score("/fake/project/path", config)
            self.assertIsInstance(score, float)
            self.assertAlmostEqual(score, 1.0, "Score should be capped at 1.0")

    def test_get_raw_architectural_violation_score_zero_threshold_config(self):
        """Test behavior with zero or negative threshold in config (should use default)."""
        config_zero = {"arch_max_violations_threshold": 0}
        with patch('guardian.sensors.arch.logger') as mock_logger:
            score_zero = get_raw_architectural_violation_score("/fake/project/path", config_zero)
            self.assertAlmostEqual(score_zero, 5.0 / 20.0) # Should use default 20
            mock_logger.info.assert_any_call(
                f"Simulated raw architectural violation score: {5.0/20.0:.3f} "
                f"(based on 5 violations / 20.0 threshold)" # Default threshold used
            )

        config_neg = {"arch_max_violations_threshold": -5}
        with patch('guardian.sensors.arch.logger') as mock_logger:
            score_neg = get_raw_architectural_violation_score("/fake/project/path", config_neg)
            self.assertAlmostEqual(score_neg, 5.0 / 20.0) # Should use default 20


if __name__ == "__main__":
    unittest.main()