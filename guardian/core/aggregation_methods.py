"""
Advanced Aggregation Methods for Quality Metrics

Implements multiple aggregation strategies beyond geometric mean:
- Harmonic mean
- Power means (generalized means)
- Choquet integral (for non-additive aggregation)
- Ordered Weighted Average (OWA)
- Ensemble aggregation
"""

import math
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Enumeration of aggregation methods."""
    GEOMETRIC = "geometric"
    ARITHMETIC = "arithmetic"
    HARMONIC = "harmonic"
    POWER = "power"
    CHOQUET = "choquet"
    OWA = "owa"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


@dataclass
class AggregationResult:
    """Result of aggregation operation."""
    value: float
    method: AggregationMethod
    parameters: Dict
    component_contributions: Optional[List[float]] = None


class AdvancedAggregator:
    """
    Advanced aggregation methods for quality metrics.
    
    Provides multiple aggregation strategies to handle different scenarios:
    - Geometric mean: When factors are independent (current default)
    - Harmonic mean: When factors are complementary (all must be high)
    - Power mean: Configurable sensitivity to extreme values
    - Choquet integral: For non-additive aggregation with interactions
    - OWA: For risk-averse or risk-seeking aggregation
    """
    
    @staticmethod
    def weighted_geometric_mean(
        values: List[float],
        weights: Optional[List[float]] = None,
        epsilon: float = 1e-9
    ) -> float:
        """
        Calculate weighted geometric mean: (∏(x_i^w_i))^(1/∑w_i)
        
        Uses log-space arithmetic for numerical stability to prevent underflow.
        Returns 0.0 if any factor with non-zero weight is zero.
        
        Args:
            values: List of values to aggregate (must be positive)
            weights: Optional weights (default: equal weights)
            epsilon: Small value to prevent log(0)
            
        Returns:
            Weighted geometric mean
        """
        if not values:
            return 0.0
        
        if weights is None:
            weights = [1.0] * len(values)
        
        if len(weights) != len(values):
            raise ValueError("Weights must match values length")
        
        # Check for zeros with non-zero weights
        for val, weight in zip(values, weights):
            if val <= 0 and weight > 0:
                return 0.0
        
        # Use log-space for numerical stability
        sum_weights = sum(weights)
        if sum_weights == 0:
            return 0.0
        
        log_sum = 0.0
        for val, weight in zip(values, weights):
            if val > 0 and weight > 0:
                log_sum += weight * math.log(max(val, epsilon))
        
        return math.exp(log_sum / sum_weights)
    
    @staticmethod
    def weighted_arithmetic_mean(
        values: List[float],
        weights: Optional[List[float]] = None
    ) -> float:
        """
        Calculate weighted arithmetic mean: (∑w_i * x_i) / ∑w_i
        
        Args:
            values: List of values to aggregate
            weights: Optional weights (default: equal weights)
            
        Returns:
            Weighted arithmetic mean
        """
        if not values:
            return 0.0
        
        if weights is None:
            weights = [1.0] * len(values)
        
        if len(weights) != len(values):
            raise ValueError("Weights must match values length")
        
        sum_weights = sum(weights)
        if sum_weights == 0:
            return 0.0
        
        weighted_sum = sum(val * weight for val, weight in zip(values, weights))
        return weighted_sum / sum_weights
    
    @staticmethod
    def weighted_harmonic_mean(
        values: List[float],
        weights: Optional[List[float]] = None,
        epsilon: float = 1e-9
    ) -> float:
        """
        Calculate weighted harmonic mean: (∑w_i) / (∑w_i/x_i)
        
        Harmonic mean is appropriate when factors are complementary
        (all must be high for good overall score).
        
        Args:
            values: List of values to aggregate
            weights: Optional weights (default: equal weights)
            epsilon: Small value to prevent division by zero
            
        Returns:
            Weighted harmonic mean
        """
        if not values:
            return 0.0
        
        if weights is None:
            weights = [1.0] * len(values)
        
        if len(weights) != len(values):
            raise ValueError("Weights must match values length")
        
        # Check for zeros
        for val in values:
            if val <= 0:
                return 0.0
        
        sum_weights = sum(weights)
        if sum_weights == 0:
            return 0.0
        
        reciprocal_sum = sum(weight / max(val, epsilon) for val, weight in zip(values, weights))
        return sum_weights / reciprocal_sum if reciprocal_sum > 0 else 0.0
    
    @staticmethod
    def power_mean(
        values: List[float],
        power: float,
        weights: Optional[List[float]] = None,
        epsilon: float = 1e-9
    ) -> float:
        """
        Calculate power mean (generalized mean): ((∑w_i * x_i^p) / ∑w_i)^(1/p)
        
        Special cases:
        - p = 1: Arithmetic mean
        - p → 0: Geometric mean
        - p = -1: Harmonic mean
        - p → ∞: Maximum
        - p → -∞: Minimum
        
        Args:
            values: List of values to aggregate
            power: Power parameter (p)
            weights: Optional weights (default: equal weights)
            epsilon: Small value for numerical stability
            
        Returns:
            Power mean
        """
        if not values:
            return 0.0
        
        if weights is None:
            weights = [1.0] * len(values)
        
        if len(weights) != len(values):
            raise ValueError("Weights must match values length")
        
        sum_weights = sum(weights)
        if sum_weights == 0:
            return 0.0
        
        # Handle special cases
        if abs(power) < epsilon:
            # Geometric mean
            return AdvancedAggregator.weighted_geometric_mean(values, weights, epsilon)
        elif power == 1.0:
            # Arithmetic mean
            return AdvancedAggregator.weighted_arithmetic_mean(values, weights)
        elif power == -1.0:
            # Harmonic mean
            return AdvancedAggregator.weighted_harmonic_mean(values, weights, epsilon)
        elif power > 100:
            # Maximum
            return max(values)
        elif power < -100:
            # Minimum
            return min(values)
        
        # General case
        if power > 0:
            # Positive power: handle zeros
            for val in values:
                if val <= 0:
                    return 0.0
            powered_sum = sum(weight * (val ** power) for val, weight in zip(values, weights))
            return (powered_sum / sum_weights) ** (1.0 / power)
        else:
            # Negative power: handle zeros
            for val in values:
                if val <= 0:
                    return 0.0
            powered_sum = sum(weight * (val ** power) for val, weight in zip(values, weights))
            return (powered_sum / sum_weights) ** (1.0 / power)
    
    @staticmethod
    def ordered_weighted_average(
        values: List[float],
        owa_weights: List[float],
        descending: bool = True
    ) -> float:
        """
        Calculate Ordered Weighted Average (OWA).
        
        OWA sorts values and applies weights to sorted positions.
        - Risk-averse: Higher weights on lower values
        - Risk-seeking: Higher weights on higher values
        
        Args:
            values: List of values to aggregate
            owa_weights: Weights for sorted positions (must sum to 1)
            descending: Sort descending (True) or ascending (False)
            
        Returns:
            OWA value
        """
        if not values:
            return 0.0
        
        if len(owa_weights) != len(values):
            raise ValueError("OWA weights must match values length")
        
        if abs(sum(owa_weights) - 1.0) > 1e-6:
            raise ValueError("OWA weights must sum to 1.0")
        
        # Sort values
        sorted_values = sorted(values, reverse=descending)
        
        # Apply weights
        owa_value = sum(val * weight for val, weight in zip(sorted_values, owa_weights))
        return owa_value
    
    @staticmethod
    def choquet_integral(
        values: List[float],
        fuzzy_measure: Dict[Tuple[int, ...], float],
        weights: Optional[List[float]] = None
    ) -> float:
        """
        Calculate Choquet integral for non-additive aggregation.
        
        Choquet integral handles interactions between factors.
        Requires a fuzzy measure (capacity) that defines the value
        of each subset of factors.
        
        Args:
            values: List of values to aggregate
            fuzzy_measure: Dictionary mapping subsets (tuples) to their measure values
            weights: Optional weights (not used in Choquet, but kept for interface)
            
        Returns:
            Choquet integral value
        """
        if not values:
            return 0.0
        
        n = len(values)
        
        # Sort indices by value (descending)
        indexed_values = [(i, values[i]) for i in range(n)]
        indexed_values.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate Choquet integral
        choquet_value = 0.0
        for i, (idx, val) in enumerate(indexed_values):
            # Get subset of indices with values >= current value
            subset = tuple(sorted([idx for idx, _ in indexed_values[:i+1]]))
            
            # Get measure of this subset
            if subset in fuzzy_measure:
                measure = fuzzy_measure[subset]
            else:
                # Default: use additive measure if not specified
                measure = len(subset) / n
            
            # Get measure of previous subset
            if i > 0:
                prev_subset = tuple(sorted([idx for idx, _ in indexed_values[:i]]))
                if prev_subset in fuzzy_measure:
                    prev_measure = fuzzy_measure[prev_subset]
                else:
                    prev_measure = len(prev_subset) / n
            else:
                prev_measure = 0.0
            
            # Add contribution
            choquet_value += val * (measure - prev_measure)
        
        return choquet_value
    
    @staticmethod
    def ensemble_aggregate(
        values: List[float],
        weights: Optional[List[float]] = None,
        methods: Optional[List[AggregationMethod]] = None,
        method_weights: Optional[List[float]] = None,
        power_mean_p: float = 0.0,
        owa_weights: Optional[List[float]] = None,
        choquet_measure: Optional[Dict[Tuple[int, ...], float]] = None
    ) -> AggregationResult:
        """
        Ensemble aggregation using multiple methods.
        
        Args:
            values: List of values to aggregate
            weights: Weights for individual values
            methods: List of aggregation methods to use (default: all)
            method_weights: Weights for each method in ensemble
            power_mean_p: Power parameter for power mean
            owa_weights: Weights for OWA (if using OWA)
            choquet_measure: Fuzzy measure for Choquet (if using Choquet)
            
        Returns:
            AggregationResult with ensemble value
        """
        if methods is None:
            methods = [
                AggregationMethod.GEOMETRIC,
                AggregationMethod.ARITHMETIC,
                AggregationMethod.HARMONIC
            ]
        
        if method_weights is None:
            method_weights = [1.0 / len(methods)] * len(methods)
        
        if len(method_weights) != len(methods):
            raise ValueError("Method weights must match methods length")
        
        # Calculate each method
        method_results = []
        for method in methods:
            if method == AggregationMethod.GEOMETRIC:
                result = AdvancedAggregator.weighted_geometric_mean(values, weights)
            elif method == AggregationMethod.ARITHMETIC:
                result = AdvancedAggregator.weighted_arithmetic_mean(values, weights)
            elif method == AggregationMethod.HARMONIC:
                result = AdvancedAggregator.weighted_harmonic_mean(values, weights)
            elif method == AggregationMethod.POWER:
                result = AdvancedAggregator.power_mean(values, power_mean_p, weights)
            elif method == AggregationMethod.OWA:
                if owa_weights is None:
                    # Default: equal weights
                    owa_weights = [1.0 / len(values)] * len(values)
                result = AdvancedAggregator.ordered_weighted_average(values, owa_weights)
            elif method == AggregationMethod.CHOQUET:
                if choquet_measure is None:
                    # Default: additive measure
                    choquet_measure = {tuple(range(len(values))): 1.0}
                result = AdvancedAggregator.choquet_integral(values, choquet_measure, weights)
            elif method == AggregationMethod.MIN:
                result = min(values) if values else 0.0
            elif method == AggregationMethod.MAX:
                result = max(values) if values else 0.0
            elif method == AggregationMethod.MEDIAN:
                result = np.median(values) if values else 0.0
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
            
            method_results.append(result)
        
        # Weighted average of methods
        ensemble_value = sum(result * weight for result, weight in zip(method_results, method_weights))
        
        return AggregationResult(
            value=ensemble_value,
            method=AggregationMethod.GEOMETRIC,  # Mark as ensemble
            parameters={
                "methods": [m.value for m in methods],
                "method_weights": method_weights,
                "component_scores": method_results
            },
            component_contributions=method_results
        )


def select_optimal_aggregation(
    values: List[float],
    weights: Optional[List[float]] = None,
    risk_preference: str = "balanced"
) -> Tuple[AggregationMethod, float]:
    """
    Select optimal aggregation method based on data characteristics.
    
    Args:
        values: List of values to aggregate
        weights: Optional weights
        risk_preference: "risk_averse", "balanced", or "risk_seeking"
        
    Returns:
        Tuple of (recommended_method, score)
    """
    if not values:
        return AggregationMethod.GEOMETRIC, 0.0
    
    # Calculate coefficient of variation
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = std_val / mean_val if mean_val > 0 else 0.0
    
    # Calculate skewness
    skew = 0.0
    if len(values) > 2:
        skew = float(stats.skew(values)) if hasattr(stats, 'skew') else 0.0
    
    # Decision logic
    if risk_preference == "risk_averse":
        # Use harmonic mean or OWA with higher weights on lower values
        if cv < 0.1:
            return AggregationMethod.HARMONIC, AdvancedAggregator.weighted_harmonic_mean(values, weights)
        else:
            # Use OWA with risk-averse weights
            n = len(values)
            owa_weights = [2.0 * (n - i) / (n * (n + 1)) for i in range(n)]  # Decreasing weights
            return AggregationMethod.OWA, AdvancedAggregator.ordered_weighted_average(values, owa_weights)
    
    elif risk_preference == "risk_seeking":
        # Use power mean with p > 1 or OWA with higher weights on higher values
        if cv < 0.1:
            return AggregationMethod.POWER, AdvancedAggregator.power_mean(values, 2.0, weights)
        else:
            # Use OWA with risk-seeking weights
            n = len(values)
            owa_weights = [2.0 * (i + 1) / (n * (n + 1)) for i in range(n)]  # Increasing weights
            return AggregationMethod.OWA, AdvancedAggregator.ordered_weighted_average(values, owa_weights)
    
    else:  # balanced
        # Use geometric mean (current default)
        if cv < 0.2 and abs(skew) < 0.5:
            return AggregationMethod.GEOMETRIC, AdvancedAggregator.weighted_geometric_mean(values, weights)
        else:
            # Use ensemble for robustness
            result = AdvancedAggregator.ensemble_aggregate(values, weights)
            return AggregationMethod.GEOMETRIC, result.value  # Return as geometric for compatibility
