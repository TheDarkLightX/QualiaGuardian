"""
Adaptive Normalization Framework

Project-adaptive normalization that adjusts thresholds based on:
- Project history
- Industry benchmarks
- Dynamic performance baselines
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThresholds:
    """Adaptive thresholds for normalization."""
    percentile_10: float = 0.0
    percentile_50: float = 0.0  # Median
    percentile_90: float = 0.0
    min_observed: float = 0.0
    max_observed: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    sample_size: int = 0


@dataclass
class NormalizationConfig:
    """Configuration for adaptive normalization."""
    use_percentiles: bool = True
    use_industry_benchmarks: bool = False
    min_samples: int = 10
    window_size: int = 100  # Rolling window for history
    decay_factor: float = 0.95  # Exponential decay for old samples


class AdaptiveNormalizer:
    """
    Adaptive normalizer that learns thresholds from project history.
    
    Instead of fixed thresholds, uses:
    - Percentile-based thresholds (10th, 50th, 90th)
    - Industry benchmarks (if available)
    - Dynamic baselines that adapt over time
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize adaptive normalizer.
        
        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
        self.history: Dict[str, deque] = {}
        self.thresholds: Dict[str, AdaptiveThresholds] = {}
        self.industry_benchmarks: Dict[str, Dict[str, float]] = {}
    
    def add_observation(self, metric_name: str, value: float):
        """
        Add a new observation to history.
        
        Args:
            metric_name: Name of the metric
            value: Observed value
        """
        if metric_name not in self.history:
            self.history[metric_name] = deque(maxlen=self.config.window_size)
        
        self.history[metric_name].append(value)
        
        # Update thresholds if we have enough samples
        if len(self.history[metric_name]) >= self.config.min_samples:
            self._update_thresholds(metric_name)
    
    def _update_thresholds(self, metric_name: str):
        """Update thresholds for a metric based on history."""
        values = list(self.history[metric_name])
        
        thresholds = AdaptiveThresholds(
            percentile_10=np.percentile(values, 10),
            percentile_50=np.percentile(values, 50),
            percentile_90=np.percentile(values, 90),
            min_observed=min(values),
            max_observed=max(values),
            mean=np.mean(values),
            std=np.std(values),
            sample_size=len(values)
        )
        
        self.thresholds[metric_name] = thresholds
    
    def normalize(
        self,
        metric_name: str,
        value: float,
        higher_is_better: bool = True
    ) -> float:
        """
        Normalize a value using adaptive thresholds.
        
        Args:
            metric_name: Name of the metric
            value: Value to normalize
            higher_is_better: Whether higher values are better
            
        Returns:
            Normalized value in [0, 1]
        """
        if metric_name not in self.thresholds:
            # Not enough history, use default normalization
            logger.warning(f"Insufficient history for {metric_name}, using default normalization")
            return max(0.0, min(1.0, value))
        
        thresholds = self.thresholds[metric_name]
        
        if self.config.use_percentiles:
            # Use percentile-based normalization
            if higher_is_better:
                # Higher is better: 90th percentile = 1.0, 10th percentile = 0.0
                if value >= thresholds.percentile_90:
                    return 1.0
                elif value <= thresholds.percentile_10:
                    return 0.0
                else:
                    # Linear interpolation
                    range_size = thresholds.percentile_90 - thresholds.percentile_10
                    if range_size == 0:
                        return 0.5
                    normalized = (value - thresholds.percentile_10) / range_size
                    return max(0.0, min(1.0, normalized))
            else:
                # Lower is better: 10th percentile = 1.0, 90th percentile = 0.0
                if value <= thresholds.percentile_10:
                    return 1.0
                elif value >= thresholds.percentile_90:
                    return 0.0
                else:
                    range_size = thresholds.percentile_90 - thresholds.percentile_10
                    if range_size == 0:
                        return 0.5
                    normalized = (thresholds.percentile_90 - value) / range_size
                    return max(0.0, min(1.0, normalized))
        else:
            # Use min-max normalization
            range_size = thresholds.max_observed - thresholds.min_observed
            if range_size == 0:
                return 0.5
            
            if higher_is_better:
                normalized = (value - thresholds.min_observed) / range_size
            else:
                normalized = (thresholds.max_observed - value) / range_size
            
            return max(0.0, min(1.0, normalized))
    
    def get_thresholds(self, metric_name: str) -> Optional[AdaptiveThresholds]:
        """Get current thresholds for a metric."""
        return self.thresholds.get(metric_name)
    
    def set_industry_benchmark(
        self,
        metric_name: str,
        benchmark_data: Dict[str, float]
    ):
        """
        Set industry benchmark for a metric.
        
        Args:
            metric_name: Name of the metric
            benchmark_data: Dictionary with 'p10', 'p50', 'p90' percentiles
        """
        self.industry_benchmarks[metric_name] = benchmark_data
    
    def normalize_with_benchmark(
        self,
        metric_name: str,
        value: float,
        higher_is_better: bool = True,
        blend_factor: float = 0.5
    ) -> float:
        """
        Normalize using blend of project history and industry benchmarks.
        
        Args:
            metric_name: Name of the metric
            value: Value to normalize
            higher_is_better: Whether higher values are better
            blend_factor: Weight for project history (1.0 = only project, 0.0 = only benchmark)
            
        Returns:
            Normalized value in [0, 1]
        """
        project_normalized = self.normalize(metric_name, value, higher_is_better)
        
        if metric_name not in self.industry_benchmarks:
            return project_normalized
        
        benchmark = self.industry_benchmarks[metric_name]
        
        # Normalize using benchmark
        if higher_is_better:
            if value >= benchmark.get('p90', value):
                benchmark_normalized = 1.0
            elif value <= benchmark.get('p10', value):
                benchmark_normalized = 0.0
            else:
                p10 = benchmark.get('p10', value)
                p90 = benchmark.get('p90', value)
                range_size = p90 - p10
                if range_size == 0:
                    benchmark_normalized = 0.5
                else:
                    benchmark_normalized = (value - p10) / range_size
        else:
            if value <= benchmark.get('p10', value):
                benchmark_normalized = 1.0
            elif value >= benchmark.get('p90', value):
                benchmark_normalized = 0.0
            else:
                p10 = benchmark.get('p10', value)
                p90 = benchmark.get('p90', value)
                range_size = p90 - p10
                if range_size == 0:
                    benchmark_normalized = 0.5
                else:
                    benchmark_normalized = (p90 - value) / range_size
        
        # Blend project and benchmark
        blended = blend_factor * project_normalized + (1 - blend_factor) * benchmark_normalized
        return max(0.0, min(1.0, blended))
