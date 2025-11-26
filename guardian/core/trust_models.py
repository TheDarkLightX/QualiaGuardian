"""
Non-Linear Trust Models for Quality Metrics

Implements advanced trust coefficient calculations that better capture
the impact of flakiness and reliability issues on quality scores.
"""

import math
import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrustModel(Enum):
    """Enumeration of trust model types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"
    POWER_LAW = "power_law"
    PIECEWISE = "piecewise"
    ADAPTIVE = "adaptive"


@dataclass
class TrustModelConfig:
    """Configuration for trust model."""
    model_type: TrustModel
    parameters: Dict[str, float]
    calibration_data: Optional[Dict] = None


class TrustModelCalculator:
    """
    Advanced trust coefficient calculator with multiple non-linear models.
    
    Trust coefficient T represents confidence in test results:
    - T = 1.0: Perfect reliability (no flakiness)
    - T = 0.0: Complete unreliability (100% flakiness)
    
    Different models capture different assumptions about flakiness impact.
    """
    
    @staticmethod
    def linear_trust(flakiness_rate: float) -> float:
        """
        Linear trust model: T = 1 - flakiness_rate
        
        Current default model. Simple but may not capture true impact.
        
        Args:
            flakiness_rate: Flakiness rate in [0, 1]
            
        Returns:
            Trust coefficient in [0, 1]
        """
        return max(0.0, min(1.0, 1.0 - flakiness_rate))
    
    @staticmethod
    def exponential_trust(
        flakiness_rate: float,
        decay_rate: float = 2.0
    ) -> float:
        """
        Exponential decay trust model: T = exp(-λ * flakiness)
        
        More aggressive penalty for flakiness. Small flakiness rates
        have significant impact.
        
        Args:
            flakiness_rate: Flakiness rate in [0, 1]
            decay_rate: Decay rate parameter (λ)
            
        Returns:
            Trust coefficient in [0, 1]
        """
        return math.exp(-decay_rate * flakiness_rate)
    
    @staticmethod
    def sigmoid_trust(
        flakiness_rate: float,
        steepness: float = 10.0,
        threshold: float = 0.1
    ) -> float:
        """
        Sigmoid trust model: T = 1 / (1 + exp(k * (flakiness - threshold)))
        
        S-shaped curve with configurable threshold and steepness.
        Allows for tolerance of low flakiness before significant penalty.
        
        Args:
            flakiness_rate: Flakiness rate in [0, 1]
            steepness: Steepness parameter (k)
            threshold: Threshold where penalty becomes significant
            
        Returns:
            Trust coefficient in [0, 1]
        """
        exponent = -steepness * (flakiness_rate - threshold)
        # Clamp exponent to prevent overflow
        exponent = max(-700.0, min(700.0, exponent))
        return 1.0 / (1.0 + math.exp(exponent))
    
    @staticmethod
    def power_law_trust(
        flakiness_rate: float,
        exponent: float = 2.0
    ) -> float:
        """
        Power law trust model: T = (1 - flakiness)^α
        
        Configurable penalty strength via exponent.
        - α = 1: Linear (same as linear model)
        - α > 1: More aggressive penalty
        - α < 1: Less aggressive penalty
        
        Args:
            flakiness_rate: Flakiness rate in [0, 1]
            exponent: Power law exponent (α)
            
        Returns:
            Trust coefficient in [0, 1]
        """
        return (1.0 - flakiness_rate) ** exponent
    
    @staticmethod
    def piecewise_trust(
        flakiness_rate: float,
        thresholds: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Piecewise linear trust model with different slopes for different ranges.
        
        Allows different sensitivity to flakiness in different ranges:
        - Low flakiness (0-5%): Minimal penalty
        - Medium flakiness (5-20%): Moderate penalty
        - High flakiness (20%+): Severe penalty
        
        Args:
            flakiness_rate: Flakiness rate in [0, 1]
            thresholds: Dictionary with 'low', 'medium', 'high' thresholds
                        and 'low_slope', 'medium_slope', 'high_slope' values
            
        Returns:
            Trust coefficient in [0, 1]
        """
        if thresholds is None:
            thresholds = {
                'low': 0.05,
                'medium': 0.20,
                'low_slope': 0.5,      # 50% penalty per 1% flakiness
                'medium_slope': 2.0,    # 200% penalty per 1% flakiness
                'high_slope': 5.0       # 500% penalty per 1% flakiness
            }
        
        low_threshold = thresholds.get('low', 0.05)
        medium_threshold = thresholds.get('medium', 0.20)
        low_slope = thresholds.get('low_slope', 0.5)
        medium_slope = thresholds.get('medium_slope', 2.0)
        high_slope = thresholds.get('high_slope', 5.0)
        
        if flakiness_rate <= low_threshold:
            # Low flakiness: minimal penalty
            penalty = low_slope * flakiness_rate
        elif flakiness_rate <= medium_threshold:
            # Medium flakiness: moderate penalty
            low_penalty = low_slope * low_threshold
            additional_flakiness = flakiness_rate - low_threshold
            penalty = low_penalty + medium_slope * additional_flakiness
        else:
            # High flakiness: severe penalty
            low_penalty = low_slope * low_threshold
            medium_penalty = medium_slope * (medium_threshold - low_threshold)
            additional_flakiness = flakiness_rate - medium_threshold
            penalty = low_penalty + medium_penalty + high_slope * additional_flakiness
        
        return max(0.0, min(1.0, 1.0 - penalty))
    
    @staticmethod
    def adaptive_trust(
        flakiness_rate: float,
        historical_flakiness: Optional[list] = None,
        project_maturity: float = 0.5
    ) -> float:
        """
        Adaptive trust model that adjusts based on historical context.
        
        Considers:
        - Current flakiness rate
        - Historical flakiness trends
        - Project maturity (new projects more tolerant)
        
        Args:
            flakiness_rate: Current flakiness rate in [0, 1]
            historical_flakiness: List of historical flakiness rates
            project_maturity: Project maturity factor [0, 1] (1 = mature)
            
        Returns:
            Trust coefficient in [0, 1]
        """
        if historical_flakiness is None or len(historical_flakiness) == 0:
            # No history: use exponential with moderate decay
            return TrustModelCalculator.exponential_trust(flakiness_rate, decay_rate=2.0)
        
        # Calculate trend
        recent_flakiness = historical_flakiness[-10:] if len(historical_flakiness) > 10 else historical_flakiness
        avg_historical = np.mean(recent_flakiness)
        
        # If current is better than historical, be more lenient
        if flakiness_rate < avg_historical:
            improvement_factor = (avg_historical - flakiness_rate) / max(avg_historical, 0.01)
            # Use less aggressive model for improving projects
            decay_rate = 1.5 - 0.5 * improvement_factor
        else:
            # If current is worse, be more strict
            deterioration_factor = (flakiness_rate - avg_historical) / max(1.0 - avg_historical, 0.01)
            decay_rate = 2.0 + 2.0 * deterioration_factor
        
        # Adjust for project maturity (mature projects less tolerant)
        decay_rate *= (0.5 + 0.5 * project_maturity)
        
        return TrustModelCalculator.exponential_trust(flakiness_rate, decay_rate=decay_rate)
    
    @staticmethod
    def calculate_trust(
        flakiness_rate: float,
        model: TrustModel = TrustModel.LINEAR,
        config: Optional[TrustModelConfig] = None
    ) -> float:
        """
        Calculate trust coefficient using specified model.
        
        Args:
            flakiness_rate: Flakiness rate in [0, 1]
            model: Trust model type
            config: Optional configuration with parameters
            
        Returns:
            Trust coefficient in [0, 1]
        """
        # Clamp flakiness rate
        flakiness_rate = max(0.0, min(1.0, flakiness_rate))
        
        if model == TrustModel.LINEAR:
            return TrustModelCalculator.linear_trust(flakiness_rate)
        
        elif model == TrustModel.EXPONENTIAL:
            decay_rate = 2.0
            if config and 'decay_rate' in config.parameters:
                decay_rate = config.parameters['decay_rate']
            return TrustModelCalculator.exponential_trust(flakiness_rate, decay_rate)
        
        elif model == TrustModel.SIGMOID:
            steepness = 10.0
            threshold = 0.1
            if config:
                steepness = config.parameters.get('steepness', 10.0)
                threshold = config.parameters.get('threshold', 0.1)
            return TrustModelCalculator.sigmoid_trust(flakiness_rate, steepness, threshold)
        
        elif model == TrustModel.POWER_LAW:
            exponent = 2.0
            if config and 'exponent' in config.parameters:
                exponent = config.parameters['exponent']
            return TrustModelCalculator.power_law_trust(flakiness_rate, exponent)
        
        elif model == TrustModel.PIECEWISE:
            thresholds = None
            if config:
                thresholds = config.parameters
            return TrustModelCalculator.piecewise_trust(flakiness_rate, thresholds)
        
        elif model == TrustModel.ADAPTIVE:
            historical = None
            maturity = 0.5
            if config:
                historical = config.calibration_data.get('historical_flakiness') if config.calibration_data else None
                maturity = config.parameters.get('project_maturity', 0.5)
            return TrustModelCalculator.adaptive_trust(flakiness_rate, historical, maturity)
        
        else:
            logger.warning(f"Unknown trust model: {model}, using linear")
            return TrustModelCalculator.linear_trust(flakiness_rate)
    
    @staticmethod
    def calibrate_trust_model(
        flakiness_rates: list,
        quality_scores: list,
        model_type: TrustModel = TrustModel.EXPONENTIAL
    ) -> TrustModelConfig:
        """
        Calibrate trust model parameters using historical data.
        
        Finds optimal parameters that best predict quality scores
        from flakiness rates.
        
        Args:
            flakiness_rates: List of historical flakiness rates
            quality_scores: List of corresponding quality scores
            model_type: Trust model to calibrate
            
        Returns:
            TrustModelConfig with calibrated parameters
        """
        if len(flakiness_rates) != len(quality_scores) or len(flakiness_rates) < 3:
            # Insufficient data, return defaults
            return TrustModelConfig(
                model_type=model_type,
                parameters={}
            )
        
        # Simple grid search for parameter optimization
        best_params = {}
        best_error = float('inf')
        
        if model_type == TrustModel.EXPONENTIAL:
            # Optimize decay_rate
            for decay_rate in np.linspace(0.5, 5.0, 20):
                predicted_trust = [TrustModelCalculator.exponential_trust(f, decay_rate) for f in flakiness_rates]
                # Assume quality = base_quality * trust (simplified)
                # Error = sum of squared differences
                error = sum((q - t) ** 2 for q, t in zip(quality_scores, predicted_trust))
                if error < best_error:
                    best_error = error
                    best_params = {'decay_rate': decay_rate}
        
        elif model_type == TrustModel.POWER_LAW:
            # Optimize exponent
            for exponent in np.linspace(0.5, 5.0, 20):
                predicted_trust = [TrustModelCalculator.power_law_trust(f, exponent) for f in flakiness_rates]
                error = sum((q - t) ** 2 for q, t in zip(quality_scores, predicted_trust))
                if error < best_error:
                    best_error = error
                    best_params = {'exponent': exponent}
        
        # For other models, use defaults
        if not best_params:
            best_params = {}
        
        return TrustModelConfig(
            model_type=model_type,
            parameters=best_params,
            calibration_data={
                'calibration_error': best_error,
                'sample_size': len(flakiness_rates)
            }
        )


def compare_trust_models(
    flakiness_rate: float,
    models: Optional[list] = None
) -> Dict[str, float]:
    """
    Compare different trust models for a given flakiness rate.
    
    Args:
        flakiness_rate: Flakiness rate to evaluate
        models: List of models to compare (default: all)
        
    Returns:
        Dictionary mapping model names to trust coefficients
    """
    if models is None:
        models = [
            TrustModel.LINEAR,
            TrustModel.EXPONENTIAL,
            TrustModel.SIGMOID,
            TrustModel.POWER_LAW,
            TrustModel.PIECEWISE
        ]
    
    results = {}
    for model in models:
        trust = TrustModelCalculator.calculate_trust(flakiness_rate, model)
        results[model.value] = trust
    
    return results
