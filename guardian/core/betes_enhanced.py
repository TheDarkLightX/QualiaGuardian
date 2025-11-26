"""
Enhanced bE-TES Calculator with Advanced Features

Integrates:
- Uncertainty quantification
- Alternative aggregation methods
- Non-linear trust models
- Information-theoretic metrics
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from .betes import BETESCalculator, BETESComponents, BETESWeights, BETESSettingsV31
from .uncertainty_quantification import (
    BayesianQualityCalculator, BayesianMetric, UncertaintyInterval,
    estimate_mutation_score_uncertainty, estimate_flakiness_uncertainty
)
from .aggregation_methods import (
    AdvancedAggregator, AggregationMethod, AggregationResult,
    select_optimal_aggregation
)
from .trust_models import (
    TrustModelCalculator, TrustModel, TrustModelConfig
)
from .information_theoretic_metrics import (
    InformationTheoreticAnalyzer, InformationMetrics
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedBETESComponents(BETESComponents):
    """Extended BETES components with uncertainty and information metrics."""
    # Uncertainty quantification
    uncertainty_intervals: Dict[str, UncertaintyInterval] = field(default_factory=dict)
    bayesian_metrics: Dict[str, BayesianMetric] = field(default_factory=dict)
    
    # Aggregation details
    aggregation_method: AggregationMethod = AggregationMethod.GEOMETRIC
    aggregation_result: Optional[AggregationResult] = None
    
    # Trust model details
    trust_model: TrustModel = TrustModel.LINEAR
    trust_config: Optional[TrustModelConfig] = None
    
    # Information-theoretic metrics
    information_metrics: Optional[InformationMetrics] = None
    
    # Risk-adjusted score
    risk_adjusted_score: float = 0.0
    risk_aversion: float = 0.5


class EnhancedBETESCalculator:
    """
    Enhanced bE-TES calculator with advanced features.
    
    Features:
    1. Bayesian uncertainty quantification
    2. Multiple aggregation methods
    3. Non-linear trust models
    4. Information-theoretic analysis
    5. Risk-adjusted scoring
    """
    
    def __init__(
        self,
        weights: Optional[BETESWeights] = None,
        settings_v3_1: Optional[BETESSettingsV31] = None,
        aggregation_method: AggregationMethod = AggregationMethod.GEOMETRIC,
        trust_model: TrustModel = TrustModel.EXPONENTIAL,
        trust_config: Optional[TrustModelConfig] = None,
        risk_aversion: float = 0.5,
        enable_uncertainty: bool = True,
        enable_information_metrics: bool = False
    ):
        """
        Initialize enhanced calculator.
        
        Args:
            weights: Component weights
            settings_v3_1: bE-TES v3.1 settings
            aggregation_method: Aggregation method to use
            trust_model: Trust model for flakiness
            trust_config: Trust model configuration
            risk_aversion: Risk aversion parameter [0, 1]
            enable_uncertainty: Enable uncertainty quantification
            enable_information_metrics: Enable information-theoretic metrics
        """
        self.base_calculator = BETESCalculator(weights, settings_v3_1)
        self.bayesian_calculator = BayesianQualityCalculator() if enable_uncertainty else None
        self.aggregation_method = aggregation_method
        self.trust_model = trust_model
        self.trust_config = trust_config
        self.risk_aversion = risk_aversion
        self.enable_uncertainty = enable_uncertainty
        self.enable_information_metrics = enable_information_metrics
    
    def calculate(
        self,
        raw_mutation_score: float,
        raw_emt_gain: float,
        raw_assertion_iq: float,
        raw_behaviour_coverage: float,
        raw_median_test_time_ms: float,
        raw_flakiness_rate: float,
        # Additional data for uncertainty and information metrics
        mutation_data: Optional[Dict[str, Any]] = None,
        flakiness_data: Optional[Dict[str, Any]] = None,
        test_coverage: Optional[Dict[str, Set[str]]] = None,
        code_importance: Optional[Dict[str, float]] = None
    ) -> EnhancedBETESComponents:
        """
        Calculate enhanced bE-TES score with all advanced features.
        
        Args:
            raw_mutation_score: Raw mutation score [0, 1]
            raw_emt_gain: Raw EMT gain
            raw_assertion_iq: Raw assertion IQ [1, 5]
            raw_behaviour_coverage: Raw behavior coverage [0, 1]
            raw_median_test_time_ms: Median test execution time (ms)
            raw_flakiness_rate: Flakiness rate [0, 1]
            mutation_data: Optional dict with 'killed_mutants' and 'total_mutants'
            flakiness_data: Optional dict with 'flaky_runs' and 'total_runs'
            test_coverage: Optional test coverage mapping for information metrics
            code_importance: Optional code importance scores
            
        Returns:
            EnhancedBETESComponents with all metrics
        """
        start_time = time.monotonic()
        
        # Calculate base components
        base_components = self.base_calculator.calculate(
            raw_mutation_score,
            raw_emt_gain,
            raw_assertion_iq,
            raw_behaviour_coverage,
            raw_median_test_time_ms,
            raw_flakiness_rate
        )
        
        # Create enhanced components
        enhanced = EnhancedBETESComponents(
            # Copy base components
            raw_mutation_score=base_components.raw_mutation_score,
            raw_emt_gain=base_components.raw_emt_gain,
            raw_assertion_iq=base_components.raw_assertion_iq,
            raw_behaviour_coverage=base_components.raw_behaviour_coverage,
            raw_median_test_time_ms=base_components.raw_median_test_time_ms,
            raw_flakiness_rate=base_components.raw_flakiness_rate,
            norm_mutation_score=base_components.norm_mutation_score,
            norm_emt_gain=base_components.norm_emt_gain,
            norm_assertion_iq=base_components.norm_assertion_iq,
            norm_behaviour_coverage=base_components.norm_behaviour_coverage,
            norm_speed_factor=base_components.norm_speed_factor,
            geometric_mean_g=base_components.geometric_mean_g,
            trust_coefficient_t=base_components.trust_coefficient_t,
            betes_score=base_components.betes_score,
            calculation_time_s=base_components.calculation_time_s,
            applied_weights=base_components.applied_weights,
            insights=base_components.insights.copy(),
            aggregation_method=self.aggregation_method,
            trust_model=self.trust_model,
            trust_config=self.trust_config,
            risk_aversion=self.risk_aversion
        )
        
        # Uncertainty quantification
        if self.enable_uncertainty and self.bayesian_calculator:
            enhanced = self._add_uncertainty_quantification(
                enhanced, mutation_data, flakiness_data
            )
        
        # Alternative aggregation
        enhanced = self._apply_alternative_aggregation(enhanced)
        
        # Non-linear trust model
        enhanced = self._apply_trust_model(enhanced)
        
        # Information-theoretic metrics
        if self.enable_information_metrics and test_coverage:
            enhanced = self._add_information_metrics(
                enhanced, test_coverage, code_importance
            )
        
        # Risk-adjusted score
        enhanced = self._calculate_risk_adjusted_score(enhanced)
        
        enhanced.calculation_time_s = time.monotonic() - start_time
        
        return enhanced
    
    def _add_uncertainty_quantification(
        self,
        components: EnhancedBETESComponents,
        mutation_data: Optional[Dict[str, Any]],
        flakiness_data: Optional[Dict[str, Any]]
    ) -> EnhancedBETESComponents:
        """Add uncertainty quantification to components."""
        if mutation_data:
            killed = mutation_data.get('killed_mutants', 0)
            total = mutation_data.get('total_mutants', 1)
            if total > 0:
                mutation_metric = estimate_mutation_score_uncertainty(killed, total)
                components.bayesian_metrics['mutation_score'] = mutation_metric
                components.uncertainty_intervals['mutation_score'] = mutation_metric.uncertainty
        
        if flakiness_data:
            flaky = flakiness_data.get('flaky_runs', 0)
            total = flakiness_data.get('total_runs', 1)
            if total > 0:
                flakiness_metric = estimate_flakiness_uncertainty(flaky, total)
                components.bayesian_metrics['flakiness'] = flakiness_metric
                components.uncertainty_intervals['flakiness'] = flakiness_metric.uncertainty
        
        return components
    
    def _apply_alternative_aggregation(
        self,
        components: EnhancedBETESComponents
    ) -> EnhancedBETESComponents:
        """Apply alternative aggregation method."""
        factors = [
            components.norm_mutation_score,
            components.norm_emt_gain,
            components.norm_assertion_iq,
            components.norm_behaviour_coverage,
            components.norm_speed_factor
        ]
        
        weights = [
            components.applied_weights.w_m,
            components.applied_weights.w_e,
            components.applied_weights.w_a,
            components.applied_weights.w_b,
            components.applied_weights.w_s
        ]
        
        if self.aggregation_method == AggregationMethod.GEOMETRIC:
            # Use base calculation
            pass
        elif self.aggregation_method == AggregationMethod.HARMONIC:
            aggregated = AdvancedAggregator.weighted_harmonic_mean(factors, weights)
            components.geometric_mean_g = aggregated
        elif self.aggregation_method == AggregationMethod.ARITHMETIC:
            aggregated = AdvancedAggregator.weighted_arithmetic_mean(factors, weights)
            components.geometric_mean_g = aggregated
        elif self.aggregation_method == AggregationMethod.POWER:
            # Use power mean with p=0.5 (between geometric and arithmetic)
            aggregated = AdvancedAggregator.power_mean(factors, 0.5, weights)
            components.geometric_mean_g = aggregated
        else:
            # Use ensemble
            result = AdvancedAggregator.ensemble_aggregate(factors, weights)
            components.geometric_mean_g = result.value
            components.aggregation_result = result
        
        return components
    
    def _apply_trust_model(
        self,
        components: EnhancedBETESComponents
    ) -> EnhancedBETESComponents:
        """Apply non-linear trust model."""
        trust = TrustModelCalculator.calculate_trust(
            components.raw_flakiness_rate,
            self.trust_model,
            self.trust_config
        )
        
        # Update trust coefficient
        components.trust_coefficient_t = trust
        
        # Recalculate final score
        components.betes_score = max(0.0, min(1.0, components.geometric_mean_g * trust))
        
        return components
    
    def _add_information_metrics(
        self,
        components: EnhancedBETESComponents,
        test_coverage: Dict[str, Set[str]],
        code_importance: Optional[Dict[str, float]]
    ) -> EnhancedBETESComponents:
        """Add information-theoretic metrics."""
        info_metrics = InformationTheoreticAnalyzer.analyze_test_suite(
            test_coverage,
            code_importance
        )
        components.information_metrics = info_metrics
        
        # Add insights based on information metrics
        if info_metrics.redundancy_score > 0.7:
            components.insights.append(
                f"High test redundancy detected ({info_metrics.redundancy_score:.2f}). "
                "Consider removing redundant tests."
            )
        
        if info_metrics.diversity_score < 0.3:
            components.insights.append(
                f"Low test diversity ({info_metrics.diversity_score:.2f}). "
                "Tests may be covering similar code paths."
            )
        
        return components
    
    def _calculate_risk_adjusted_score(
        self,
        components: EnhancedBETESComponents
    ) -> EnhancedBETESComponents:
        """Calculate risk-adjusted score using uncertainty."""
        if self.enable_uncertainty and components.bayesian_metrics:
            # Use lower bound for risk-averse scoring
            base_score = components.betes_score
            
            # Adjust based on uncertainty
            if 'mutation_score' in components.bayesian_metrics:
                mutation_metric = components.bayesian_metrics['mutation_score']
                uncertainty_width = (
                    mutation_metric.uncertainty.upper_bound -
                    mutation_metric.uncertainty.lower_bound
                )
                # Penalize high uncertainty
                uncertainty_penalty = min(0.1, uncertainty_width * 0.5)
                base_score *= (1.0 - uncertainty_penalty)
            
            components.risk_adjusted_score = self.bayesian_calculator.calculate_uncertainty_aware_score(
                BayesianMetric(
                    value=base_score,
                    uncertainty=UncertaintyInterval(
                        mean=base_score,
                        lower_bound=base_score * 0.9,
                        upper_bound=base_score * 1.1,
                        confidence_level=0.95
                    ),
                    prior_parameters={},
                    posterior_parameters={},
                    sample_size=1,
                    effective_sample_size=1.0
                ),
                self.risk_aversion
            )
        else:
            components.risk_adjusted_score = components.betes_score
        
        return components
