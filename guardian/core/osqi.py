"""
OSQI: Overall Software Quality Index v1.0

Core implementation of the OSQI scoring system.
"""
import math
import logging
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Assuming QualityConfig and its nested configs are in .etes or a renamed quality_config.py
# For OSQICalculator, we'll directly use OSQIWeightsConfig from there.
# For standalone use, we define OSQIWeights here.
from .etes import OSQIWeightsConfig # To be used by OSQICalculator

logger = logging.getLogger(__name__)

@dataclass
class OSQIWeights:
    """
    Weights for the OSQI components.
    Matches OSQIWeightsConfig but can be used independently if needed.
    """
    w_test: float = 2.0  # Weight for bE-TES score
    w_code: float = 1.0  # Weight for Code Health Score (C_HS)
    w_sec: float = 1.5   # Weight for Security Score (Sec_S)
    w_arch: float = 1.0  # Weight for Architecture Score (Arch_S)

@dataclass
class OSQIRawPillarsInput:
    """Raw inputs required to calculate the OSQI score."""
    betes_score: float = 0.0  # Already normalized (0-1)
    raw_code_health_sub_metrics: Dict[str, float] = field(default_factory=dict)
    raw_weighted_vulnerability_density: float = 0.0 # Normalized (0-1)
    raw_architectural_violation_score: float = 0.0 # Normalized (0-1)

@dataclass
class OSQINormalizedPillars:
    """The four core OSQI pillars, all normalized to a 0-1 scale."""
    betes_score: float = 0.0
    code_health_score_c_hs: float = 0.0
    security_score_sec_s: float = 0.0
    architecture_score_arch_s: float = 0.0

@dataclass
class OSQIResult:
    """Container for all OSQI calculation inputs, intermediate values, and final score."""
    raw_pillars_input: Optional[OSQIRawPillarsInput] = None
    normalized_pillars: Optional[OSQINormalizedPillars] = None
    applied_weights: Optional[OSQIWeightsConfig] = None # Using config type from .etes
    osqi_score: float = 0.0
    calculation_time_s: float = 0.0
    insights: List[str] = field(default_factory=list)


class OSQICalculator:
    """Calculates the Overall Software Quality Index (OSQI v1.0)."""

    def __init__(self, 
                 osqi_weights: OSQIWeightsConfig, 
                 chs_thresholds_path: str = "guardian_ai_tool/config/chs_thresholds.yml"):
        """
        Initializes the OSQI calculator.

        Args:
            osqi_weights: Configuration for OSQI component weights.
            chs_thresholds_path: Path to the YAML file containing thresholds for CHS sub-metrics.
        """
        self.weights = osqi_weights
        self.chs_thresholds_path = chs_thresholds_path
        self.chs_thresholds: Dict[str, Any] = {}
        self._load_chs_thresholds()

    def _load_chs_thresholds(self):
        try:
            with open(self.chs_thresholds_path, 'r', encoding='utf-8') as f:
                self.chs_thresholds = yaml.safe_load(f)
            if not self.chs_thresholds:
                self.chs_thresholds = {}
                logger.warning(f"CHS thresholds file '{self.chs_thresholds_path}' is empty or invalid. CHS normalization may be impaired.")
        except FileNotFoundError:
            logger.error(f"CHS thresholds file not found: {self.chs_thresholds_path}. CHS normalization will use defaults or fail.")
            self.chs_thresholds = {} # Ensure it's an empty dict
        except yaml.YAMLError as e:
            logger.error(f"Error parsing CHS thresholds YAML file '{self.chs_thresholds_path}': {e}. CHS normalization will use defaults or fail.")
            self.chs_thresholds = {}

    def _normalize_chs_sub_metric(self, metric_name: str, raw_value: float, language: str) -> float:
        """
        Normalizes a single raw CHS sub-metric to a 0-1 scale using defined thresholds.
        Lower raw values are generally better, unless 'higher_is_better' is true for the metric.
        """
        lang_thresholds = self.chs_thresholds.get(language.lower(), {})
        metric_config = lang_thresholds.get(metric_name, {})

        ideal_max = metric_config.get("ideal_max") # Value at or below which score is 1.0 (lower is better)
        acceptable_max = metric_config.get("acceptable_max") # Value at which score is ~0.5
        poor_min = metric_config.get("poor_min") # Value at or above which score is 0.0

        # Support for metrics where higher is better
        higher_is_better = metric_config.get("higher_is_better", False)
        if higher_is_better:
            ideal_min_raw = metric_config.get("ideal_min_raw") # Value at or above which score is 1.0
            acceptable_min_raw = metric_config.get("acceptable_min_raw") # Value at which score is ~0.5
            poor_max_raw = metric_config.get("poor_max_raw") # Value at or below which score is 0.0
            
            if ideal_min_raw is None or poor_max_raw is None:
                logger.warning(f"CHS sub-metric '{metric_name}' for language '{language}' is 'higher_is_better' but missing thresholds. Defaulting to 0.5.")
                return 0.5
            
            if raw_value >= ideal_min_raw: return 1.0
            if raw_value <= poor_max_raw: return 0.0
            
            # Interpolate: use acceptable_min_raw if available, else linear between poor and ideal
            mid_raw = acceptable_min_raw if acceptable_min_raw is not None else (ideal_min_raw + poor_max_raw) / 2.0
            mid_score = 0.5

            if raw_value >= mid_raw: # Between mid and ideal
                if (ideal_min_raw - mid_raw) == 0: return 1.0 # Avoid division by zero
                return mid_score + (1.0 - mid_score) * (raw_value - mid_raw) / (ideal_min_raw - mid_raw)
            else: # Between poor and mid
                if (mid_raw - poor_max_raw) == 0: return 0.0 # Avoid division by zero
                return 0.0 + mid_score * (raw_value - poor_max_raw) / (mid_raw - poor_max_raw)

        # Default: lower is better
        if ideal_max is None or poor_min is None:
            logger.warning(f"CHS sub-metric '{metric_name}' for language '{language}' missing thresholds. Defaulting to 0.5.")
            return 0.5 # Fallback if thresholds are not defined

        if raw_value <= ideal_max: return 1.0
        if raw_value >= poor_min: return 0.0

        # Interpolate: use acceptable_max if available, else linear between ideal and poor
        mid_raw = acceptable_max if acceptable_max is not None else (ideal_max + poor_min) / 2.0
        mid_score = 0.5
        
        # Score decreases as raw_value increases
        if raw_value <= mid_raw: # Between ideal and mid
            if (mid_raw - ideal_max) == 0: return 1.0
            return 1.0 - (1.0 - mid_score) * (raw_value - ideal_max) / (mid_raw - ideal_max)
        else: # Between mid and poor
            if (poor_min - mid_raw) == 0: return 0.0
            return mid_score - mid_score * (raw_value - mid_raw) / (poor_min - mid_raw)


    def _calculate_code_health_score_c_hs(self, raw_sub_metrics: Dict[str, float], language: str) -> float:
        """Calculates the Code Health Score (C_HS) from normalized sub-metrics."""
        if not raw_sub_metrics:
            logger.warning("No raw CHS sub-metrics provided. C_HS will be 0.")
            return 0.0

        normalized_sub_scores = []
        for name, raw_val in raw_sub_metrics.items():
            norm_score = self._normalize_chs_sub_metric(name, raw_val, language)
            normalized_sub_scores.append(max(0.0, min(norm_score, 1.0))) # Clip to 0-1

        if not normalized_sub_scores: # Should not happen if raw_sub_metrics was not empty
            return 0.0

        # Geometric mean of normalized scores
        # If any normalized score is 0, the geometric mean is 0.
        if any(math.isclose(s, 0.0) for s in normalized_sub_scores):
            return 0.0
        
        product = 1.0
        for score in normalized_sub_scores:
            product *= score
        
        c_hs = product ** (1.0 / len(normalized_sub_scores))
        return max(0.0, min(c_hs, 1.0)) # Ensure 0-1

    def calculate(self, inputs: OSQIRawPillarsInput, project_language: str) -> OSQIResult:
        """
        Calculates the OSQI score and its components.

        Args:
            inputs: An OSQIRawPillarsInput object containing all necessary raw values.
            project_language: The primary programming language of the project (e.g., "python").

        Returns:
            An OSQIResult object.
        """
        import time # Local import for calculation time
        start_time = time.monotonic()

        # 1. Calculate/Normalize individual pillars
        betes_score = max(0.0, min(1.0, inputs.betes_score))

        c_hs = self._calculate_code_health_score_c_hs(
            inputs.raw_code_health_sub_metrics,
            project_language
        )
        
        sec_s = max(0.0, min(1.0, 1.0 - inputs.raw_weighted_vulnerability_density))
        
        arch_s = max(0.0, min(1.0, 1.0 - inputs.raw_architectural_violation_score))

        normalized_pillars = OSQINormalizedPillars(
            betes_score=betes_score,
            code_health_score_c_hs=c_hs,
            security_score_sec_s=sec_s,
            architecture_score_arch_s=arch_s
        )

        # 2. Calculate final OSQI (Weighted Geometric Mean)
        pillars = [
            normalized_pillars.betes_score,
            normalized_pillars.code_health_score_c_hs,
            normalized_pillars.security_score_sec_s,
            normalized_pillars.architecture_score_arch_s
        ]
        current_weights = [
            self.weights.w_test,
            self.weights.w_code,
            self.weights.w_sec,
            self.weights.w_arch
        ]
        sum_of_weights = sum(current_weights)
        
        weighted_product = 1.0
        has_zero_pillar_with_weight = False
        for pillar_val, weight_val in zip(pillars, current_weights):
            if math.isclose(pillar_val, 0.0) and weight_val > 0.0:
                has_zero_pillar_with_weight = True
                break
            if pillar_val > 0.0 or math.isclose(weight_val, 0.0): # only raise to power if base > 0 or weight is 0
                weighted_product *= (pillar_val ** weight_val)
            # Negative base with fractional exponent is not an issue here as pillars are 0-1

        final_osqi_score = 0.0
        if not has_zero_pillar_with_weight and sum_of_weights > 0:
            final_osqi_score = weighted_product ** (1.0 / sum_of_weights)
        
        final_osqi_score = max(0.0, min(1.0, final_osqi_score)) # Ensure 0-1

        end_time = time.monotonic()
        
        return OSQIResult(
            raw_pillars_input=inputs,
            normalized_pillars=normalized_pillars,
            applied_weights=self.weights,
            osqi_score=final_osqi_score,
            calculation_time_s=(end_time - start_time)
        )


def classify_osqi(
    score: float,
    risk_class_name: Optional[str],
    risk_definitions: Dict[str, Dict[str, Any]],
    metric_name: str = "OSQI" # Default metric name for messages
) -> Dict[str, Any]:
    """
    Evaluates the OSQI score against a defined risk class and its threshold.
    Uses the 'min_score' key from risk_definitions.

    Args:
        score: The calculated OSQI score (0.0 to 1.0).
        risk_class_name: The name of the risk class to evaluate against.
        risk_definitions: A dictionary loaded from risk_classes.yml.
        metric_name: The name of the metric being classified.

    Returns:
        A dictionary with classification details.
    """
    # This function can reuse the logic from betes.classify_betes
    # For now, let's duplicate and adapt slightly.
    # Ideally, factor out the common classification logic.
    from .betes import classify_betes as classify_metric_score 
    # Re-using the same classification logic, just passing "OSQI" as metric name
    return classify_metric_score(score, risk_class_name, risk_definitions, metric_name=metric_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger.info("Testing OSQI Calculator...")

    # Dummy config and inputs
    test_osqi_weights = OSQIWeightsConfig(w_test=2, w_code=1, w_sec=1.5, w_arch=1)
    
    # Create a dummy chs_thresholds.yml for testing
    dummy_chs_threshold_file = "dummy_chs_thresholds.yml"
    dummy_chs_content = {
        "python": {
            "cyclomatic_complexity": {"ideal_max": 5, "acceptable_max": 10, "poor_min": 15},
            "duplication_percentage": {"ideal_max": 3.0, "acceptable_max": 10.0, "poor_min": 25.0}
        }
    }
    with open(dummy_chs_threshold_file, 'w') as f:
        yaml.dump(dummy_chs_content, f)

    calculator = OSQICalculator(osqi_weights=test_osqi_weights, chs_thresholds_path=dummy_chs_threshold_file)

    test_inputs = OSQIRawPillarsInput(
        betes_score=0.85,
        raw_code_health_sub_metrics={
            "cyclomatic_complexity": 7,  # Should be > 0.5, < 1.0
            "duplication_percentage": 5.0 # Should be > 0.5, < 1.0
        },
        raw_weighted_vulnerability_density=0.1, # Sec_S should be 0.9
        raw_architectural_violation_score=0.05  # Arch_S should be 0.95
    )

    result = calculator.calculate(test_inputs, project_language="python")
    
    logger.info(f"OSQI Calculation Result:")
    logger.info(f"  Raw Inputs: {result.raw_pillars_input}")
    logger.info(f"  Normalized Pillars: {result.normalized_pillars}")
    logger.info(f"  Applied Weights: {result.applied_weights}")
    logger.info(f"  Final OSQI Score: {result.osqi_score:.4f}")
    logger.info(f"  Calculation Time: {result.calculation_time_s:.6f}s")

    # Test classification
    dummy_risk_defs = {
        "standard_saas": {"min_score": 0.70},
        "financial": {"min_score": 0.80}
    }
    classification_standard = classify_osqi(result.osqi_score, "standard_saas", dummy_risk_defs)
    logger.info(f"Classification (Standard SaaS): {classification_standard}")
    
    classification_financial = classify_osqi(result.osqi_score, "financial", dummy_risk_defs)
    logger.info(f"Classification (Financial): {classification_financial}")

    # Test case: one pillar is zero
    test_inputs_zero_sec = OSQIRawPillarsInput(
        betes_score=0.85,
        raw_code_health_sub_metrics={"cyclomatic_complexity": 7, "duplication_percentage": 5.0},
        raw_weighted_vulnerability_density=1.0, # Sec_S will be 0
        raw_architectural_violation_score=0.05
    )
    result_zero_sec = calculator.calculate(test_inputs_zero_sec, project_language="python")
    logger.info(f"OSQI with Zero Security Pillar: {result_zero_sec.osqi_score:.4f}")
    assert math.isclose(result_zero_sec.osqi_score, 0.0), "OSQI should be 0 if a weighted pillar is 0"

    # Clean up dummy file
    import os
    if os.path.exists(dummy_chs_threshold_file):
        os.remove(dummy_chs_threshold_file)