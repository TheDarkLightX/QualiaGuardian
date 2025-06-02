"""
Unit tests for the CHS (Code Health Score) sub-metrics sensor.
"""
import unittest
from unittest.mock import patch
from guardian.sensors.chs import get_raw_chs_sub_metrics

class TestCHSSensor(unittest.TestCase):

    def test_get_raw_chs_sub_metrics_python_placeholder(self):
        """Test the placeholder CHS sensor for Python projects."""
        # Since it's a placeholder, we mainly check the structure of the output.
        with patch('guardian.sensors.chs.logger') as mock_logger:
            metrics = get_raw_chs_sub_metrics("/fake/project/path", "python", {})
            self.assertIsInstance(metrics, dict)
            # Check for expected placeholder keys for Python
            self.assertIn("cyclomatic_complexity", metrics)
            self.assertIn("duplication_percentage", metrics)
            self.assertIsInstance(metrics["cyclomatic_complexity"], (int, float))
            self.assertIsInstance(metrics["duplication_percentage"], (int, float))
            mock_logger.info.assert_any_call("Simulating CHS sub-metric collection for project at: /fake/project/path (Language: python)")

    def test_get_raw_chs_sub_metrics_javascript_placeholder(self):
        """Test the placeholder CHS sensor for JavaScript projects."""
        with patch('guardian.sensors.chs.logger') as mock_logger:
            metrics = get_raw_chs_sub_metrics("/fake/project/path", "javascript", {})
            self.assertIsInstance(metrics, dict)
            # Check for expected placeholder keys for JavaScript
            self.assertIn("cyclomatic_complexity", metrics)
            self.assertIn("duplication_percentage", metrics)
            mock_logger.info.assert_any_call("Simulating CHS sub-metric collection for project at: /fake/project/path (Language: javascript)")

    def test_get_raw_chs_sub_metrics_unsupported_language(self):
        """Test CHS sensor with an unsupported language."""
        with patch('guardian.sensors.chs.logger') as mock_logger:
            metrics = get_raw_chs_sub_metrics("/fake/project/path", "java", {})
            self.assertIsInstance(metrics, dict)
            self.assertEqual(len(metrics), 0) # Expect empty dict for unsupported
            mock_logger.warning.assert_called_with("CHS sub-metric collection not implemented for language: java. Returning empty metrics.")

    def test_get_raw_chs_sub_metrics_with_config(self):
        """Test that the sensor function accepts a config argument."""
        # This test doesn't check config usage as it's a placeholder,
        # but ensures the interface is met.
        metrics = get_raw_chs_sub_metrics("/fake/project/path", "python", {"some_config_key": "value"})
        self.assertIsInstance(metrics, dict)
        self.assertIn("cyclomatic_complexity", metrics)


if __name__ == "__main__":
    unittest.main()