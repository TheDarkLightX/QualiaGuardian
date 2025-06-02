"""
bE-TES: Bounded Evolutionary Test Effectiveness Score v3.1

Core implementation of the bE-TES scoring system, incorporating
optional sigmoid normalization for M' and E', and revised S' calculation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
# Assuming BETESSettingsV31 and BETESWeights will be in .etes or a renamed quality_config.py
from .etes import BETESSettingsV31, BETESWeights


@dataclass
class BETESComponents:
    """Container for bE-TES score components and results."""
    # Raw input values
    raw_mutation_score: float = 0.0
    raw_emt_gain: float = 0.0 # This is E_raw = Final_MS - Initial_MS
    raw_assertion_iq: float = 0.0 # Mean rubric score (1-5)
    raw_behaviour_coverage: float = 0.0 # Ratio: Covered_Critical / Total_Critical
    raw_median_test_time_ms: float = 0.0
    raw_flakiness_rate: float = 0.0

    # Normalized factor values (0-1)
    norm_mutation_score: float = 0.0    # M'
    norm_emt_gain: float = 0.0          # E'
    norm_assertion_iq: float = 0.0      # A'
    norm_behaviour_coverage: float = 0.0 # B'
    norm_speed_factor: float = 0.0      # S'

    # Intermediate calculations
    geometric_mean_g: float = 0.0 # G
    trust_coefficient_t: float = 0.0 # T

    # Final score
    betes_score: float = 0.0

    # Metadata
    calculation_time_s: float = 0.0
    applied_weights: Optional[BETESWeights] = None
    insights: List[str] = field(default_factory=list)


class BETESCalculator:
    """Calculates the Bounded Evolutionary Test Effectiveness Score (bE-TES v3.1)."""

    def __init__(self,
                 weights: Optional[BETESWeights] = None,
                 settings_v3_1: Optional[BETESSettingsV31] = None):
        """
        Initializes the calculator with specific component weights and v3.1 settings.
        If no weights are provided, default weights (all 1.0) will be used.
        If no v3.1 settings are provided, default v3.1 settings will be used.
        """
        self.weights = weights if weights is not None else BETESWeights()
        self.settings_v3_1 = settings_v3_1 if settings_v3_1 is not None else BETESSettingsV31()

    def calculate(
        self,
        raw_mutation_score: float,
        raw_emt_gain: float, # This is E_raw = Final_MS - Initial_MS
        raw_assertion_iq: float, # Mean rubric score (1-5)
        raw_behaviour_coverage: float, # Ratio: Covered_Critical / Total_Critical
        raw_median_test_time_ms: float,
        raw_flakiness_rate: float
    ) -> BETESComponents:
        """
        Calculates the bE-TES v3.1 score and its components.

        Args:
            raw_mutation_score: The raw mutation score (e.g., 0.0 to 1.0).
            raw_emt_gain: The raw EMT gain (Final MS - Initial MS).
            raw_assertion_iq: The mean assertion IQ rubric score (1.0 to 5.0).
            raw_behaviour_coverage: The raw behavior coverage ratio (e.g., 0.0 to 1.0).
            raw_median_test_time_ms: The median test execution time in milliseconds.
            raw_flakiness_rate: The test suite flakiness rate (0.0 to 1.0).

        Returns:
            A BETESComponents object populated with all raw, normalized, intermediate,
            and final score values.
        """
        import time # Add import for calculation time
        start_time = time.monotonic()

        components = BETESComponents(
            raw_mutation_score=raw_mutation_score,
            raw_emt_gain=raw_emt_gain,
            raw_assertion_iq=raw_assertion_iq,
            raw_behaviour_coverage=raw_behaviour_coverage,
            raw_median_test_time_ms=raw_median_test_time_ms,
            raw_flakiness_rate=raw_flakiness_rate,
            applied_weights=self.weights
        )

        # 1. Normalize factors
        # M' (Mutation Score)
        if self.settings_v3_1.smooth_m:
            # Sigmoid normalization: 1 / (1 + exp(-k_m * (M - 0.775)))
            # k_m from settings, center is 0.775
            try:
                val = -self.settings_v3_1.k_m * (raw_mutation_score - 0.775)
                components.norm_mutation_score = 1 / (1 + math.exp(val))
            except OverflowError: # math.exp can overflow
                 components.norm_mutation_score = 0.0 if val > 0 else 1.0 # exp(large_positive) -> inf, exp(large_negative) -> 0
        else:
            # Min-max normalization: minmax(M, 0.6, 0.95)
            if (0.95 - 0.6) == 0: # Avoid division by zero if lo == hi
                 components.norm_mutation_score = 0.0 if raw_mutation_score < 0.6 else 1.0
            else:
                components.norm_mutation_score = max(0.0, min(1.0, (raw_mutation_score - 0.6) / (0.95 - 0.6)))
        components.norm_mutation_score = max(0.0, min(1.0, components.norm_mutation_score)) # Ensure 0-1

        # E' (EMT Gain)
        if self.settings_v3_1.smooth_e:
            # Sigmoid normalization: 1 / (1 + exp(-k_e * (E_raw - 0.125)))
            # k_e from settings, center is 0.125
            try:
                val = -self.settings_v3_1.k_e * (raw_emt_gain - 0.125)
                components.norm_emt_gain = 1 / (1 + math.exp(val))
            except OverflowError:
                components.norm_emt_gain = 0.0 if val > 0 else 1.0
        else:
            # Clip normalization: clip(E_raw / 0.25, 0, 1)
            components.norm_emt_gain = max(0.0, min(1.0, raw_emt_gain / 0.25 if 0.25 != 0 else (0.0 if raw_emt_gain < 0 else float('inf')) ))
        components.norm_emt_gain = max(0.0, min(1.0, components.norm_emt_gain)) # Ensure 0-1

        # A' (Assertion IQ)
        # A_raw is raw_assertion_iq (1-5)
        components.norm_assertion_iq = max(0.0, min(1.0, (raw_assertion_iq - 1.0) / 4.0 if 4.0 !=0 else 0.0))

        # B' = Covered_Critical_Behaviors / Total_Critical_Behaviors
        # raw_behaviour_coverage is already this ratio
        components.norm_behaviour_coverage = max(0.0, min(1.0, raw_behaviour_coverage))

        # S' (Speed Factor) - Piece-wise normalization
        if raw_median_test_time_ms <= 0:
            components.norm_speed_factor = 0.0
        elif raw_median_test_time_ms <= 100:
            components.norm_speed_factor = 1.0
        else: # raw_median_test_time_ms > 100
            # Ensure (raw_median_test_time_ms / 100.0) is > 0 for log10
            # This condition is met since raw_median_test_time_ms > 100
            try:
                log_input = raw_median_test_time_ms / 100.0
                if log_input <= 0: # Should not happen given outer condition, but safeguard
                    components.norm_speed_factor = 0.0
                else:
                    log_val = math.log10(log_input)
                    # Denominator (1.0 + log_val) can be <= 0 if log_val <= -1
                    # This happens if log_input <= 0.1, i.e., raw_median_test_time_ms <= 10
                    # But this case is covered by raw_median_test_time_ms <= 100 yielding 1.0.
                    # If raw_median_test_time_ms is very large, log_val is large positive, denominator is large positive, S' approaches 0.
                    denominator = 1.0 + log_val
                    if denominator <= 1e-9: # Avoid division by zero or very small numbers leading to huge S'
                        components.norm_speed_factor = 0.0 # Effectively, if time is excessively large
                    else:
                        components.norm_speed_factor = 1.0 / denominator
            except ValueError: # math.log10 domain error, though unlikely with checks
                 components.norm_speed_factor = 0.0
        components.norm_speed_factor = max(0.0, min(1.0, components.norm_speed_factor)) # Ensure 0-1


        # 2. Calculate G (Weighted Geometric Mean)
        # G = (M'^(w_M) * E'^(w_E) * A'^(w_A) * B'^(w_B) * S'^(w_S))^(1 / sum(w))
        
        factors = [
            components.norm_mutation_score,
            components.norm_emt_gain,
            components.norm_assertion_iq,
            components.norm_behaviour_coverage,
            components.norm_speed_factor
        ]
        
        current_weights = [
            self.weights.w_m,
            self.weights.w_e,
            self.weights.w_a,
            self.weights.w_b,
            self.weights.w_s
        ]

        sum_of_weights = sum(current_weights)
        
        weighted_product = 1.0
        # Handle cases where a factor is 0, which would make the geometric mean 0
        # unless its corresponding weight is 0.
        # If any factor is 0 and its weight is > 0, G is 0.
        # If a factor is 0 and its weight is 0, it doesn't contribute.
        can_calculate_product = True
        for factor_val, weight_val in zip(factors, current_weights):
            if factor_val == 0.0 and weight_val > 0.0:
                weighted_product = 0.0
                can_calculate_product = False
                break
            if factor_val > 0.0 or weight_val == 0.0: # only raise to power if base > 0 or weight is 0 (x^0=1)
                 weighted_product *= (factor_val ** weight_val)
            elif factor_val < 0.0 and weight_val % 1 != 0: # negative base with fractional exponent
                # This case should ideally not happen with normalized factors (0-1)
                # but as a safeguard:
                weighted_product = 0.0 # Or handle as an error/undefined
                can_calculate_product = False
                break


        if not can_calculate_product or weighted_product == 0.0:
            components.geometric_mean_g = 0.0
        elif sum_of_weights == 0:
            # Undefined or could be 1.0 if all factors were 1.0.
            # For safety, if sum_of_weights is 0, G is typically considered undefined or 0.
            components.geometric_mean_g = 0.0
        else:
            components.geometric_mean_g = weighted_product ** (1.0 / sum_of_weights)
        
        # 3. Calculate T (Trust Coefficient)
        # T = 1 - flakiness_rate
        components.trust_coefficient_t = max(0.0, min(1.0, 1.0 - raw_flakiness_rate))

        # 4. Calculate final bE-TES score
        # bE-TES = G * T
        components.betes_score = components.geometric_mean_g * components.trust_coefficient_t
        
        # Ensure final score is also within 0-1
        components.betes_score = max(0.0, min(1.0, components.betes_score))

        end_time = time.monotonic()
        components.calculation_time_s = end_time - start_time
        
        # Placeholder for insights - can be added later
        # components.insights = self._generate_insights(components)

        return components


def classify_betes(
    score: float,
    risk_class_name: Optional[str],
    risk_definitions: Dict[str, Dict[str, Any]],
    metric_name: str = "bE-TES"
) -> Dict[str, Any]:
    """
    Evaluates a given score against a defined risk class and its threshold.

    Args:
        score: The calculated score (0.0 to 1.0).
        risk_class_name: The name of the risk class to evaluate against (e.g., "standard_saas").
                         If None, no specific classification is performed, only the score is returned.
        risk_definitions: A dictionary loaded from risk_classes.yml, where keys are
                          risk class names and values are their definitions (e.g., {"min_score": 0.75}).
        metric_name: The name of the metric being classified (e.g., "bE-TES", "OSQI").

    Returns:
        A dictionary containing the score, and if a risk_class_name is provided,
        it includes the risk class, its threshold, a pass/fail verdict, and a message.
        Example:
            {
                "score": 0.82,
                "risk_class": "standard_saas",
                "threshold": 0.75,
                "verdict": "PASS",
                "message": "bE-TES score meets the threshold for Standard SaaS."
            }
        If risk_class_name is None:
            {"score": 0.82}
    """
    result: Dict[str, Any] = {"score": round(score, 3)} # Round score for presentation

    if not risk_class_name:
        return result # No classification requested

    if not risk_definitions:
        result["error"] = "Risk definitions not provided."
        result["risk_class"] = risk_class_name
        result["verdict"] = "UNKNOWN"
        return result

    risk_class_info = risk_definitions.get(risk_class_name)

    if not risk_class_info:
        result["error"] = f"Risk class '{risk_class_name}' not found in definitions."
        result["risk_class"] = risk_class_name
        result["verdict"] = "UNKNOWN"
        return result

    min_score_threshold = risk_class_info.get("min_score")

    if min_score_threshold is None:
        result["error"] = f"No 'min_score' threshold defined for risk class '{risk_class_name}'."
        result["risk_class"] = risk_class_name
        result["verdict"] = "UNKNOWN"
        return result
    
    try:
        min_score_threshold = float(min_score_threshold)
    except ValueError:
        result["error"] = f"'min_score' for risk class '{risk_class_name}' is not a valid number."
        result["risk_class"] = risk_class_name
        result["verdict"] = "UNKNOWN"
        return result

    result["risk_class"] = risk_class_name
    result["threshold"] = round(min_score_threshold, 3)

    if score >= min_score_threshold:
        result["verdict"] = "PASS"
        result["message"] = f"{metric_name} score {result['score']:.3f} meets or exceeds the threshold of {result['threshold']:.3f} for risk class '{risk_class_name}'."
    else:
        result["verdict"] = "FAIL"
        result["message"] = f"{metric_name} score {result['score']:.3f} is below the threshold of {result['threshold']:.3f} for risk class '{risk_class_name}'."
        
    return result