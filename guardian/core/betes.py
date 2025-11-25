"""
bE-TES: Bounded Evolutionary Test Effectiveness Score v3.1

Core implementation of the bE-TES scoring system, incorporating
optional sigmoid normalization for M' and E', and revised S' calculation.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from .etes import BETESSettingsV31, BETESWeights

# Constants for normalization
MUTATION_SCORE_MIN = 0.6
MUTATION_SCORE_MAX = 0.95
MUTATION_SCORE_CENTER = 0.775
EMT_GAIN_CENTER = 0.125
EMT_GAIN_DIVISOR = 0.25
ASSERTION_IQ_MIN = 1.0
ASSERTION_IQ_RANGE = 4.0
SPEED_THRESHOLD_MS = 100.0
SPEED_DIVISOR = 100.0
EPSILON = 1e-9


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

    @staticmethod
    def _normalize_mutation_score(raw_score: float, settings: BETESSettingsV31) -> float:
        """Normalize mutation score (M') using sigmoid or min-max normalization."""
        if settings.smooth_m:
            return BETESCalculator._sigmoid_normalize(
                raw_score, settings.k_m, MUTATION_SCORE_CENTER
            )
        return BETESCalculator._minmax_normalize(
            raw_score, MUTATION_SCORE_MIN, MUTATION_SCORE_MAX
        )

    @staticmethod
    def _normalize_emt_gain(raw_gain: float, settings: BETESSettingsV31) -> float:
        """Normalize EMT gain (E') using sigmoid or clip normalization."""
        if settings.smooth_e:
            return BETESCalculator._sigmoid_normalize(
                raw_gain, settings.k_e, EMT_GAIN_CENTER
            )
        return BETESCalculator._clip_normalize(raw_gain, EMT_GAIN_DIVISOR)

    @staticmethod
    def _normalize_assertion_iq(raw_iq: float) -> float:
        """Normalize assertion IQ (A') from 1-5 scale to 0-1."""
        if ASSERTION_IQ_RANGE == 0:
            return 0.0
        normalized = (raw_iq - ASSERTION_IQ_MIN) / ASSERTION_IQ_RANGE
        return BETESCalculator._clamp(normalized)

    @staticmethod
    def _normalize_speed_factor(raw_time_ms: float) -> float:
        """
        Normalize speed factor (S') using piece-wise function.
        
        For t <= 100ms: S' = 1.0
        For t > 100ms: S' = 1 / (1 + log10(t/100))
        
        Note: At t=100, log10(100/100)=0, so S'=1/(1+0)=1.0 (continuous).
        """
        if raw_time_ms <= 0:
            return 0.0
        if raw_time_ms <= SPEED_THRESHOLD_MS:
            return 1.0
        
        try:
            log_input = raw_time_ms / SPEED_DIVISOR
            if log_input <= 0:
                return 0.0
            
            log_val = math.log10(log_input)
            # At threshold (100ms): log10(1) = 0, so 1/(1+0) = 1.0 (continuous)
            denominator = 1.0 + log_val
            
            if denominator <= EPSILON:
                return 0.0
            
            return BETESCalculator._clamp(1.0 / denominator)
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _sigmoid_normalize(value: float, k: float, center: float) -> float:
        """
        Apply sigmoid normalization: 1 / (1 + exp(-k * (value - center))).
        
        Uses numerically stable implementation to prevent overflow.
        """
        try:
            exponent = -k * (value - center)
            # Clamp exponent to prevent overflow (exp(±700) is near machine limits)
            exponent = max(-700.0, min(700.0, exponent))
            
            # Use stable sigmoid: for x > 0, use 1/(1+exp(-x)); for x <= 0, use exp(x)/(1+exp(x))
            if exponent > 0:
                exp_val = math.exp(-exponent)  # exp(-x) where x>0, so < 1
                result = 1.0 / (1.0 + exp_val)
            else:
                exp_val = math.exp(exponent)  # exp(x) where x<=0, so <= 1
                result = exp_val / (1.0 + exp_val)
            
            return BETESCalculator._clamp(result)
        except (OverflowError, ValueError):
            # Fallback to asymptotic behavior
            if value < center:
                return 0.0
            else:
                return 1.0

    @staticmethod
    def _minmax_normalize(value: float, min_val: float, max_val: float) -> float:
        """Apply min-max normalization: (value - min) / (max - min)."""
        range_size = max_val - min_val
        if range_size == 0:
            return 1.0 if value >= min_val else 0.0
        
        normalized = (value - min_val) / range_size
        return BETESCalculator._clamp(normalized)

    @staticmethod
    def _clip_normalize(value: float, divisor: float) -> float:
        """Apply clip normalization: clip(value / divisor, 0, 1)."""
        if divisor == 0:
            return 0.0 if value < 0 else 1.0
        
        normalized = value / divisor
        return BETESCalculator._clamp(normalized)

    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to [min_val, max_val] range."""
        return max(min_val, min(max_val, value))

    @staticmethod
    def _calculate_weighted_geometric_mean(
        factors: List[float], weights: List[float]
    ) -> float:
        """
        Calculate weighted geometric mean: (∏(factor_i^weight_i))^(1/∑weights).
        
        Uses log-space arithmetic for numerical stability to prevent underflow.
        """
        if not factors or not weights or len(factors) != len(weights):
            return 0.0

        sum_of_weights = sum(weights)
        if sum_of_weights == 0:
            # Mathematically undefined - all weights zero
            return float('nan')

        # Check for zero factors with non-zero weights
        for factor, weight in zip(factors, weights):
            if factor == 0.0 and weight > 0.0:
                return 0.0
            if factor < 0.0:
                # Negative values not allowed (would require complex numbers)
                return float('nan')

        # Use log-space to prevent underflow/overflow
        # G = exp(∑(w_i * log(factor_i)) / ∑w_i)
        log_sum = 0.0
        for factor, weight in zip(factors, weights):
            if factor > 0.0 and weight > 0.0:
                log_sum += weight * math.log(factor)
            # If weight == 0, skip (factor^0 = 1, log(1) = 0, doesn't contribute)

        if log_sum == float('-inf'):
            return 0.0

        try:
            return math.exp(log_sum / sum_of_weights)
        except (OverflowError, ValueError):
            # Should not happen with log-space, but handle gracefully
            return 0.0

    def calculate(
        self,
        raw_mutation_score: float,
        raw_emt_gain: float,
        raw_assertion_iq: float,
        raw_behaviour_coverage: float,
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

        # Normalize all factors
        components.norm_mutation_score = self._normalize_mutation_score(
            raw_mutation_score, self.settings_v3_1
        )
        components.norm_emt_gain = self._normalize_emt_gain(
            raw_emt_gain, self.settings_v3_1
        )
        components.norm_assertion_iq = self._normalize_assertion_iq(raw_assertion_iq)
        components.norm_behaviour_coverage = self._clamp(raw_behaviour_coverage)
        components.norm_speed_factor = self._normalize_speed_factor(raw_median_test_time_ms)

        # Calculate weighted geometric mean
        factors = [
            components.norm_mutation_score,
            components.norm_emt_gain,
            components.norm_assertion_iq,
            components.norm_behaviour_coverage,
            components.norm_speed_factor
        ]
        weights = [
            self.weights.w_m,
            self.weights.w_e,
            self.weights.w_a,
            self.weights.w_b,
            self.weights.w_s
        ]
        components.geometric_mean_g = self._calculate_weighted_geometric_mean(factors, weights)

        # Calculate trust coefficient
        components.trust_coefficient_t = self._clamp(1.0 - raw_flakiness_rate)

        # Calculate final score
        components.betes_score = self._clamp(
            components.geometric_mean_g * components.trust_coefficient_t
        )

        components.calculation_time_s = time.monotonic() - start_time
        return components


def classify_betes(
    score: float,
    risk_class_name: Optional[str],
    risk_definitions: Dict[str, Dict[str, Any]],
    metric_name: str = "bE-TES"
) -> Dict[str, Any]:
    """
    Evaluates a bE-TES score against a defined risk class and its threshold.

    Args:
        score: The calculated bE-TES score (0.0 to 1.0).
        risk_class_name: The name of the risk class to evaluate against.
        risk_definitions: A dictionary loaded from risk_classes.yml.
        metric_name: The name of the metric being classified (default: "bE-TES").

    Returns:
        A dictionary containing classification results.
    """
    from .classification import classify_metric_score
    return classify_metric_score(score, risk_class_name, risk_definitions, metric_name)