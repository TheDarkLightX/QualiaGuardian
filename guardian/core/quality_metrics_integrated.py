"""
Integrated Quality Metrics Framework

Brings together all advanced features:
- Uncertainty quantification
- Alternative aggregation
- Non-linear trust models
- Information-theoretic metrics
- Adaptive normalization
- Sensitivity analysis
- Monte Carlo propagation
- Temporal analysis
- Robust statistics
- Model selection
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np
import logging

from .betes_enhanced import EnhancedBETESCalculator, EnhancedBETESComponents
from .uncertainty_quantification import BayesianQualityCalculator
from .aggregation_methods import AggregationMethod
from .trust_models import TrustModel
from .sensitivity_analysis import SensitivityAnalyzer, analyze_betes_sensitivity
from .monte_carlo_uncertainty import MonteCarloPropagator
from .temporal_analysis import TemporalAnalyzer
from .robust_statistics import RobustStatistician
from .model_selection import ModelSelector

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveQualityAnalysis:
    """Comprehensive quality analysis with all advanced features."""
    # Core metrics
    enhanced_betes: EnhancedBETESComponents
    
    # Uncertainty
    monte_carlo_result: Optional[Any] = None
    uncertainty_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Sensitivity
    sensitivity_results: List[Any] = field(default_factory=list)
    critical_factors: List[str] = field(default_factory=list)
    
    # Temporal
    temporal_decomposition: Optional[Any] = None
    trend_forecast: Optional[np.ndarray] = None
    
    # Robust statistics
    robust_stats: Optional[Any] = None
    
    # Model selection
    recommended_aggregation: Optional[str] = None
    recommended_trust_model: Optional[str] = None
    
    # Information metrics
    information_summary: Dict[str, float] = field(default_factory=dict)
    
    # Insights
    comprehensive_insights: List[str] = field(default_factory=list)


class ComprehensiveQualityAnalyzer:
    """
    Comprehensive quality analyzer that integrates all advanced features.
    """
    
    def __init__(
        self,
        enable_all_features: bool = True,
        risk_aversion: float = 0.5
    ):
        """
        Initialize comprehensive analyzer.
        
        Args:
            enable_all_features: Enable all advanced features
            risk_aversion: Risk aversion parameter
        """
        self.enable_all_features = enable_all_features
        self.risk_aversion = risk_aversion
        self.calculator = EnhancedBETESCalculator(
            aggregation_method=AggregationMethod.GEOMETRIC,
            trust_model=TrustModel.EXPONENTIAL,
            enable_uncertainty=True,
            enable_information_metrics=True,
            risk_aversion=risk_aversion
        )
    
    def analyze(
        self,
        raw_mutation_score: float,
        raw_emt_gain: float,
        raw_assertion_iq: float,
        raw_behaviour_coverage: float,
        raw_median_test_time_ms: float,
        raw_flakiness_rate: float,
        # Additional data
        mutation_data: Optional[Dict[str, Any]] = None,
        flakiness_data: Optional[Dict[str, Any]] = None,
        test_coverage: Optional[Dict[str, Set[str]]] = None,
        code_importance: Optional[Dict[str, float]] = None,
        historical_scores: Optional[np.ndarray] = None,
        historical_factors: Optional[Dict[str, np.ndarray]] = None
    ) -> ComprehensiveQualityAnalysis:
        """
        Perform comprehensive quality analysis.
        
        Args:
            raw_mutation_score: Raw mutation score
            raw_emt_gain: Raw EMT gain
            raw_assertion_iq: Raw assertion IQ
            raw_behaviour_coverage: Raw behavior coverage
            raw_median_test_time_ms: Median test time (ms)
            raw_flakiness_rate: Flakiness rate
            mutation_data: Optional mutation testing data
            flakiness_data: Optional flakiness data
            test_coverage: Optional test coverage mapping
            code_importance: Optional code importance scores
            historical_scores: Optional historical quality scores
            historical_factors: Optional historical factor values
            
        Returns:
            ComprehensiveQualityAnalysis with all results
        """
        # Core enhanced bE-TES calculation
        enhanced_betes = self.calculator.calculate(
            raw_mutation_score,
            raw_emt_gain,
            raw_assertion_iq,
            raw_behaviour_coverage,
            raw_median_test_time_ms,
            raw_flakiness_rate,
            mutation_data,
            flakiness_data,
            test_coverage,
            code_importance
        )
        
        analysis = ComprehensiveQualityAnalysis(enhanced_betes=enhanced_betes)
        
        if not self.enable_all_features:
            return analysis
        
        # Sensitivity analysis
        base_inputs = {
            'mutation_score': raw_mutation_score,
            'emt_gain': raw_emt_gain,
            'assertion_iq': raw_assertion_iq,
            'behaviour_coverage': raw_behaviour_coverage,
            'test_time_ms': raw_median_test_time_ms,
            'flakiness_rate': raw_flakiness_rate
        }
        
        try:
            sensitivity_results = analyze_betes_sensitivity(
                self.calculator.base_calculator,
                base_inputs
            )
            analysis.sensitivity_results = sensitivity_results
            analysis.critical_factors = [
                r.factor_name for r in sensitivity_results[:3]  # Top 3
            ]
        except Exception as e:
            logger.warning(f"Sensitivity analysis failed: {e}")
        
        # Monte Carlo uncertainty propagation
        if mutation_data and flakiness_data:
            try:
                killed = mutation_data.get('killed_mutants', 0)
                total = mutation_data.get('total_mutants', 1)
                flaky = flakiness_data.get('flaky_runs', 0)
                flaky_total = flakiness_data.get('total_runs', 1)
                
                if total > 0 and flaky_total > 0:
                    # Estimate distribution parameters
                    mutation_alpha = killed + 0.5
                    mutation_beta = (total - killed) + 0.5
                    flakiness_alpha = flaky + 0.5
                    flakiness_beta = (flaky_total - flaky) + 0.5
                    
                    mc_result = MonteCarloPropagator.propagate_betes_uncertainty(
                        self.calculator.base_calculator,
                        mutation_alpha,
                        mutation_beta,
                        flakiness_alpha,
                        flakiness_beta,
                        {
                            'emt_gain': raw_emt_gain,
                            'assertion_iq': raw_assertion_iq,
                            'behaviour_coverage': raw_behaviour_coverage,
                            'test_time_ms': raw_median_test_time_ms
                        }
                    )
                    analysis.monte_carlo_result = mc_result
                    analysis.uncertainty_summary = {
                        'mean': mc_result.mean,
                        'std': mc_result.std,
                        'ci_lower': mc_result.confidence_interval[0],
                        'ci_upper': mc_result.confidence_interval[1]
                    }
            except Exception as e:
                logger.warning(f"Monte Carlo propagation failed: {e}")
        
        # Temporal analysis
        if historical_scores is not None and len(historical_scores) > 5:
            try:
                temporal_decomp = TemporalAnalyzer.decompose_multi_scale(historical_scores)
                analysis.temporal_decomposition = temporal_decomp
                analysis.trend_forecast = TemporalAnalyzer.forecast_trend(
                    temporal_decomp, n_periods=10
                )
            except Exception as e:
                logger.warning(f"Temporal analysis failed: {e}")
        
        # Robust statistics
        if historical_scores is not None:
            try:
                robust_stats = RobustStatistician.calculate_robust_stats(
                    historical_scores
                )
                analysis.robust_stats = robust_stats
            except Exception as e:
                logger.warning(f"Robust statistics failed: {e}")
        
        # Information metrics summary
        if enhanced_betes.information_metrics:
            info = enhanced_betes.information_metrics
            analysis.information_summary = {
                'entropy': info.test_suite_entropy,
                'redundancy': info.redundancy_score,
                'diversity': info.diversity_score,
                'coverage_efficiency': info.coverage_efficiency
            }
        
        # Generate comprehensive insights
        analysis.comprehensive_insights = self._generate_insights(analysis)
        
        return analysis
    
    def _generate_insights(self, analysis: ComprehensiveQualityAnalysis) -> List[str]:
        """Generate comprehensive insights from all analyses."""
        insights = list(analysis.enhanced_betes.insights)
        
        # Sensitivity insights
        if analysis.sensitivity_results:
            top_factor = analysis.sensitivity_results[0]
            insights.append(
                f"Most sensitive factor: {top_factor.factor_name} "
                f"(elasticity: {top_factor.elasticity:.2f})"
            )
        
        # Uncertainty insights
        if analysis.uncertainty_summary:
            ci_width = (analysis.uncertainty_summary['ci_upper'] - 
                       analysis.uncertainty_summary['ci_lower'])
            if ci_width > 0.2:
                insights.append(
                    f"High uncertainty detected (CI width: {ci_width:.3f}). "
                    "Consider collecting more data."
                )
        
        # Temporal insights
        if analysis.temporal_decomposition:
            if analysis.temporal_decomposition.trend_slope > 0.01:
                insights.append("Positive quality trend detected.")
            elif analysis.temporal_decomposition.trend_slope < -0.01:
                insights.append("Negative quality trend detected - investigate.")
            
            if analysis.temporal_decomposition.change_points:
                insights.append(
                    f"Structural break detected at point {analysis.temporal_decomposition.change_points[0]}"
                )
        
        # Robust statistics insights
        if analysis.robust_stats and analysis.robust_stats.outlier_count > 0:
            insights.append(
                f"{analysis.robust_stats.outlier_count} outliers detected in historical data. "
                "Consider investigating these points."
            )
        
        return insights
