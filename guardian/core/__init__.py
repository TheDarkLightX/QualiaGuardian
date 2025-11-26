"""
Guardian Core Quality Metrics

Enhanced quality metrics with advanced features:
- Uncertainty quantification
- Alternative aggregation methods
- Non-linear trust models
- Information-theoretic metrics
- Adaptive normalization
- Sensitivity analysis
- Monte Carlo propagation
- Temporal analysis
- Robust statistics
- Model selection
"""

# Core calculators
from .betes import BETESCalculator, BETESComponents, BETESWeights, BETESSettingsV31
from .betes_enhanced import EnhancedBETESCalculator, EnhancedBETESComponents
from .osqi import OSQICalculator, OSQIResult, OSQIRawPillarsInput
from .etes import ETESCalculator, ETESComponents, QualityConfig
from .tes import calculate_quality_score

# Advanced features
from .uncertainty_quantification import (
    BayesianQualityCalculator,
    BayesianMetric,
    UncertaintyInterval,
    estimate_mutation_score_uncertainty,
    estimate_flakiness_uncertainty
)

from .aggregation_methods import (
    AdvancedAggregator,
    AggregationMethod,
    AggregationResult,
    select_optimal_aggregation
)

from .trust_models import (
    TrustModelCalculator,
    TrustModel,
    TrustModelConfig,
    compare_trust_models
)

from .information_theoretic_metrics import (
    InformationTheoreticAnalyzer,
    InformationMetrics
)

from .adaptive_normalization import (
    AdaptiveNormalizer,
    AdaptiveThresholds,
    NormalizationConfig
)

from .sensitivity_analysis import (
    SensitivityAnalyzer,
    SensitivityResult,
    analyze_betes_sensitivity
)

from .monte_carlo_uncertainty import (
    MonteCarloPropagator,
    MonteCarloResult,
    AdaptiveMonteCarlo
)

from .temporal_analysis import (
    TemporalAnalyzer,
    TemporalDecomposition
)

from .robust_statistics import (
    RobustStatistician,
    RobustStatistics,
    RobustAggregator
)

from .model_selection import (
    ModelSelector,
    ModelComparison,
    ModelSelectionCriterion
)

from .quality_metrics_integrated import (
    ComprehensiveQualityAnalyzer,
    ComprehensiveQualityAnalysis
)

__all__ = [
    # Core
    'BETESCalculator',
    'BETESComponents',
    'BETESWeights',
    'BETESSettingsV31',
    'EnhancedBETESCalculator',
    'EnhancedBETESComponents',
    'OSQICalculator',
    'OSQIResult',
    'OSQIRawPillarsInput',
    'ETESCalculator',
    'ETESComponents',
    'QualityConfig',
    'calculate_quality_score',
    
    # Uncertainty
    'BayesianQualityCalculator',
    'BayesianMetric',
    'UncertaintyInterval',
    'estimate_mutation_score_uncertainty',
    'estimate_flakiness_uncertainty',
    
    # Aggregation
    'AdvancedAggregator',
    'AggregationMethod',
    'AggregationResult',
    'select_optimal_aggregation',
    
    # Trust
    'TrustModelCalculator',
    'TrustModel',
    'TrustModelConfig',
    'compare_trust_models',
    
    # Information
    'InformationTheoreticAnalyzer',
    'InformationMetrics',
    
    # Adaptive
    'AdaptiveNormalizer',
    'AdaptiveThresholds',
    'NormalizationConfig',
    
    # Sensitivity
    'SensitivityAnalyzer',
    'SensitivityResult',
    'analyze_betes_sensitivity',
    
    # Monte Carlo
    'MonteCarloPropagator',
    'MonteCarloResult',
    'AdaptiveMonteCarlo',
    
    # Temporal
    'TemporalAnalyzer',
    'TemporalDecomposition',
    
    # Robust
    'RobustStatistician',
    'RobustStatistics',
    'RobustAggregator',
    
    # Model Selection
    'ModelSelector',
    'ModelComparison',
    'ModelSelectionCriterion',
    
    # Integrated
    'ComprehensiveQualityAnalyzer',
    'ComprehensiveQualityAnalysis',
]
