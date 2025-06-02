"""
Unit tests for the OSQI (Overall Software Quality Index) v1.0 calculator.
"""
import unittest
import yaml
import os
import math
from guardian.core.osqi import (
    OSQICalculator, 
    OSQIResult, 
    OSQIRawPillarsInput, 
    OSQINormalizedPillars,
    classify_osqi
)
from guardian.core.etes import OSQIWeightsConfig # Using the config version from etes.py

# Dummy chs_thresholds.yml content for testing
DUMMY_CHS_THRESHOLDS_CONTENT = {
    "python": {
        "cyclomatic_complexity": {"ideal_max": 5, "acceptable_max": 10, "poor_min": 15},
        "duplication_percentage": {"ideal_max": 3.0, "acceptable_max": 10.0, "poor_min": 25.0},
        "cognitive_complexity": {"ideal_max": 8, "acceptable_max": 15, "poor_min": 25} # Added for testing
    },
    "javascript": {
        "cyclomatic_complexity": {"ideal_max": 7, "acceptable_max": 12, "poor_min": 20}
    }
}
DUMMY_CHS_THRESHOLDS_FILE = "dummy_test_chs_thresholds.yml"

DUMMY_RISK_DEFINITIONS = {
    "standard_saas": {"min_score": 0.70},
    "financial": {"min_score": 0.80},
    "medical": {"min_score": 0.88}
}

class TestOSQICalculator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a dummy chs_thresholds.yml for the calculator to use
        with open(DUMMY_CHS_THRESHOLDS_FILE, 'w') as f:
            yaml.dump(DUMMY_CHS_THRESHOLDS_CONTENT, f)

    @classmethod
    def tearDownClass(cls):
        # Clean up the dummy file
        if os.path.exists(DUMMY_CHS_THRESHOLDS_FILE):
            os.remove(DUMMY_CHS_THRESHOLDS_FILE)

    def test_normalize_chs_sub_metric_lower_is_better(self):
        calc = OSQICalculator(OSQIWeightsConfig(), chs_thresholds_path=DUMMY_CHS_THRESHOLDS_FILE)
        
        # Test cyclomatic_complexity for python
        self.assertAlmostEqual(calc._normalize_chs_sub_metric("cyclomatic_complexity", 3, "python"), 1.0) # Ideal
        self.assertAlmostEqual(calc._normalize_chs_sub_metric("cyclomatic_complexity", 5, "python"), 1.0) # Ideal edge
        self.assertAlmostEqual(calc._normalize_chs_sub_metric("cyclomatic_complexity", 7.5, "python"), 0.75) # Mid between ideal and acceptable
        self.assertAlmostEqual(calc._normalize_chs_sub_metric("cyclomatic_complexity", 10, "python"), 0.5) # Acceptable
        self.assertAlmostEqual(calc._normalize_chs_sub_metric("cyclomatic_complexity", 12.5, "python"), 0.25) # Mid between acceptable and poor
        self.assertAlmostEqual(calc._normalize_chs_sub_metric("cyclomatic_complexity", 15, "python"), 0.0) # Poor edge
        self.assertAlmostEqual(calc._normalize_chs_sub_metric("cyclomatic_complexity", 20, "python"), 0.0) # Poor

    def test_calculate_code_health_score_c_hs(self):
        calc = OSQICalculator(OSQIWeightsConfig(), chs_thresholds_path=DUMMY_CHS_THRESHOLDS_FILE)
        raw_metrics = {
            "cyclomatic_complexity": 7,  # Normalized: ~0.8 (between 1.0 at 5 and 0.5 at 10) -> (1 - 0.5 * (7-5)/(10-5)) = 1 - 0.5 * 2/5 = 0.8
            "duplication_percentage": 5.0 # Normalized: ~0.857 (between 1.0 at 3 and 0.5 at 10) -> (1 - 0.5 * (5-3)/(10-3)) = 1 - 0.5 * 2/7 = ~0.857
        }
        # Expected normalized: CC=0.8, Dup=0.85714...
        # Geometric mean: (0.8 * 0.85714)^(1/2) = (0.685712)^(1/2) = ~0.828
        expected_cc_norm = 1.0 - (1.0 - 0.5) * (7 - 5) / (10 - 5) # 0.8
        expected_dup_norm = 1.0 - (1.0 - 0.5) * (5 - 3) / (10 - 3) # ~0.85714
        expected_chs = (expected_cc_norm * expected_dup_norm) ** 0.5
        
        c_hs = calc._calculate_code_health_score_c_hs(raw_metrics, "python")
        self.assertAlmostEqual(c_hs, expected_chs, places=5)

        # Test with one metric being zero
        raw_metrics_one_zero = {"cyclomatic_complexity": 20} # Normalized CC = 0
        c_hs_zero = calc._calculate_code_health_score_c_hs(raw_metrics_one_zero, "python")
        self.assertAlmostEqual(c_hs_zero, 0.0)
        
        # Test with no metrics
        c_hs_empty = calc._calculate_code_health_score_c_hs({}, "python")
        self.assertAlmostEqual(c_hs_empty, 0.0)

    def test_osqi_calculation_all_good(self):
        calc = OSQICalculator(OSQIWeightsConfig(), chs_thresholds_path=DUMMY_CHS_THRESHOLDS_FILE)
        inputs = OSQIRawPillarsInput(
            betes_score=0.9,
            raw_code_health_sub_metrics={"cyclomatic_complexity": 4, "duplication_percentage": 2.0}, # CHS should be 1.0
            raw_weighted_vulnerability_density=0.05, # Sec_S = 0.95
            raw_architectural_violation_score=0.02  # Arch_S = 0.98
        )
        result = calc.calculate(inputs, "python")
        
        self.assertAlmostEqual(result.normalized_pillars.betes_score, 0.9)
        self.assertAlmostEqual(result.normalized_pillars.code_health_score_c_hs, 1.0)
        self.assertAlmostEqual(result.normalized_pillars.security_score_sec_s, 0.95)
        self.assertAlmostEqual(result.normalized_pillars.architecture_score_arch_s, 0.98)

        # Expected OSQI: (0.9^2 * 1.0^1 * 0.95^1.5 * 0.98^1)^(1/(2+1+1.5+1))
        # (0.81 * 1.0 * 0.9259 * 0.98)^(1/5.5) = (0.7353)^(1/5.5) = ~0.945
        expected_osqi = ( (0.9**2) * (1.0**1) * (0.95**1.5) * (0.98**1) )**(1/5.5)
        self.assertAlmostEqual(result.osqi_score, expected_osqi, places=4)

    def test_osqi_calculation_one_pillar_zero(self):
        calc = OSQICalculator(OSQIWeightsConfig(), chs_thresholds_path=DUMMY_CHS_THRESHOLDS_FILE)
        inputs = OSQIRawPillarsInput(
            betes_score=0.9,
            raw_code_health_sub_metrics={"cyclomatic_complexity": 4, "duplication_percentage": 2.0}, # CHS = 1.0
            raw_weighted_vulnerability_density=1.0, # Sec_S = 0.0
            raw_architectural_violation_score=0.02  # Arch_S = 0.98
        )
        result = calc.calculate(inputs, "python")
        self.assertAlmostEqual(result.osqi_score, 0.0)

    def test_osqi_calculation_zero_betes(self):
        calc = OSQICalculator(OSQIWeightsConfig(), chs_thresholds_path=DUMMY_CHS_THRESHOLDS_FILE)
        inputs = OSQIRawPillarsInput(
            betes_score=0.0, # bE-TES is zero
            raw_code_health_sub_metrics={"cyclomatic_complexity": 4, "duplication_percentage": 2.0},
            raw_weighted_vulnerability_density=0.1,
            raw_architectural_violation_score=0.1
        )
        result = calc.calculate(inputs, "python")
        self.assertAlmostEqual(result.osqi_score, 0.0)


class TestOSQIClassification(unittest.TestCase):
    def test_classify_osqi_pass(self):
        classification = classify_osqi(0.75, "standard_saas", DUMMY_RISK_DEFINITIONS)
        self.assertEqual(classification["verdict"], "PASS")
        self.assertEqual(classification["metric_name"], "OSQI") # Check default metric name

    def test_classify_osqi_fail(self):
        classification = classify_osqi(0.65, "standard_saas", DUMMY_RISK_DEFINITIONS)
        self.assertEqual(classification["verdict"], "FAIL")

    def test_classify_osqi_custom_metric_name(self):
        classification = classify_osqi(0.85, "financial", DUMMY_RISK_DEFINITIONS, metric_name="CustomQualityIndex")
        self.assertEqual(classification["verdict"], "PASS")
        self.assertIn("CustomQualityIndex score", classification["message"])

    def test_classify_osqi_unknown_risk_class(self):
        classification = classify_osqi(0.8, "non_existent_class", DUMMY_RISK_DEFINITIONS)
        self.assertEqual(classification["verdict"], "UNKNOWN")
        self.assertIn("not found", classification["error"])

    def test_classify_osqi_no_risk_class_provided(self):
        classification = classify_osqi(0.8, None, DUMMY_RISK_DEFINITIONS)
        self.assertNotIn("verdict", classification) # No verdict if no risk class
        self.assertEqual(classification["score"], 0.8)


if __name__ == "__main__":
    unittest.main()