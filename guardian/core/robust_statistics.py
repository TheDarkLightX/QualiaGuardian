"""
Robust Statistics for Quality Metrics

Uses robust statistical methods that are less sensitive to outliers:
- Median and IQR instead of mean and std
- Robust regression
- Outlier detection and handling
- Winsorization and trimming
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class RobustStatistics:
    """Robust statistical measures."""
    median: float
    iqr: float  # Interquartile Range
    mad: float  # Median Absolute Deviation
    trimmed_mean: float
    winsorized_mean: float
    outlier_count: int
    outlier_indices: List[int]


class RobustStatistician:
    """
    Calculates robust statistics that are resistant to outliers.
    """
    
    @staticmethod
    def calculate_robust_stats(
        data: np.ndarray,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5
    ) -> RobustStatistics:
        """
        Calculate robust statistics for a dataset.
        
        Args:
            data: Array of values
            outlier_method: Method for outlier detection ("iqr" or "zscore")
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            RobustStatistics object
        """
        if len(data) == 0:
            return RobustStatistics(
                median=0.0, iqr=0.0, mad=0.0,
                trimmed_mean=0.0, winsorized_mean=0.0,
                outlier_count=0, outlier_indices=[]
            )
        
        # Basic robust statistics
        median = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        
        # Median Absolute Deviation (MAD)
        mad = np.median(np.abs(data - median))
        
        # Detect outliers
        outliers = RobustStatistician.detect_outliers(
            data, outlier_method, outlier_threshold
        )
        outlier_indices = outliers
        outlier_count = len(outliers)
        
        # Trimmed mean (remove top and bottom 10%)
        trimmed_data = RobustStatistician.trim_outliers(data, outliers)
        trimmed_mean = np.mean(trimmed_data) if len(trimmed_data) > 0 else median
        
        # Winsorized mean (cap outliers at 5th and 95th percentiles)
        winsorized_data = RobustStatistician.winsorize(data, 0.05, 0.95)
        winsorized_mean = np.mean(winsorized_data)
        
        return RobustStatistics(
            median=median,
            iqr=iqr,
            mad=mad,
            trimmed_mean=trimmed_mean,
            winsorized_mean=winsorized_mean,
            outlier_count=outlier_count,
            outlier_indices=outlier_indices
        )
    
    @staticmethod
    def detect_outliers(
        data: np.ndarray,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> List[int]:
        """
        Detect outliers in data.
        
        Args:
            data: Array of values
            method: Detection method ("iqr" or "zscore" or "mad")
            threshold: Threshold multiplier
            
        Returns:
            List of indices where outliers occur
        """
        if len(data) < 3:
            return []
        
        outliers = []
        
        if method == "iqr":
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
        
        elif method == "zscore":
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                z_scores = np.abs((data - mean) / std)
                outliers = np.where(z_scores > threshold)[0].tolist()
        
        elif method == "mad":
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad > 0:
                modified_z_scores = 0.6745 * np.abs((data - median) / mad)
                outliers = np.where(modified_z_scores > threshold)[0].tolist()
        
        return outliers
    
    @staticmethod
    def trim_outliers(
        data: np.ndarray,
        outlier_indices: List[int]
    ) -> np.ndarray:
        """
        Remove outliers from data.
        
        Args:
            data: Original data
            outlier_indices: Indices of outliers to remove
            
        Returns:
            Data with outliers removed
        """
        mask = np.ones(len(data), dtype=bool)
        mask[outlier_indices] = False
        return data[mask]
    
    @staticmethod
    def winsorize(
        data: np.ndarray,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ) -> np.ndarray:
        """
        Winsorize data by capping values at percentiles.
        
        Args:
            data: Original data
            lower_percentile: Lower percentile to cap at
            upper_percentile: Upper percentile to cap at
            
        Returns:
            Winsorized data
        """
        lower_bound = np.percentile(data, lower_percentile * 100)
        upper_bound = np.percentile(data, upper_percentile * 100)
        
        winsorized = data.copy()
        winsorized[winsorized < lower_bound] = lower_bound
        winsorized[winsorized > upper_bound] = upper_bound
        
        return winsorized
    
    @staticmethod
    def robust_normalize(
        value: float,
        data: np.ndarray,
        use_median: bool = True
    ) -> float:
        """
        Normalize a value using robust statistics.
        
        Args:
            value: Value to normalize
            data: Reference data
            use_median: Use median/IQR (True) or trimmed mean/std (False)
            
        Returns:
            Normalized value (z-score like)
        """
        if len(data) == 0:
            return 0.0
        
        if use_median:
            center = np.median(data)
            scale = stats.iqr(data) / 1.349  # IQR to std approximation
        else:
            stats_obj = RobustStatistician.calculate_robust_stats(data)
            center = stats_obj.trimmed_mean
            scale = stats_obj.mad * 1.4826  # MAD to std approximation
        
        if scale == 0:
            return 0.0 if value == center else (1.0 if value > center else -1.0)
        
        return (value - center) / scale


class RobustAggregator:
    """
    Robust aggregation methods that handle outliers.
    """
    
    @staticmethod
    def robust_geometric_mean(
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
        trim_fraction: float = 0.1
    ) -> float:
        """
        Calculate robust geometric mean by trimming outliers.
        
        Args:
            values: Values to aggregate
            weights: Optional weights
            trim_fraction: Fraction to trim from each end
            
        Returns:
            Robust geometric mean
        """
        if len(values) == 0:
            return 0.0
        
        # Remove zeros and negatives
        positive_mask = values > 0
        if not np.any(positive_mask):
            return 0.0
        
        values = values[positive_mask]
        if weights is not None:
            weights = weights[positive_mask]
        
        # Trim outliers
        n_trim = int(len(values) * trim_fraction)
        if n_trim > 0:
            sorted_indices = np.argsort(values)
            keep_indices = sorted_indices[n_trim:-n_trim] if n_trim * 2 < len(values) else sorted_indices
            values = values[keep_indices]
            if weights is not None:
                weights = weights[keep_indices]
        
        # Calculate geometric mean
        if weights is None:
            return np.exp(np.mean(np.log(values)))
        else:
            weights = weights / np.sum(weights)
            return np.exp(np.sum(weights * np.log(values)))
    
    @staticmethod
    def median_of_means(
        values: np.ndarray,
        n_groups: int = 5
    ) -> float:
        """
        Calculate median of means (MoM) for robust aggregation.
        
        Divides data into groups, calculates mean of each, then takes median.
        
        Args:
            values: Values to aggregate
            n_groups: Number of groups to divide into
            
        Returns:
            Median of means
        """
        if len(values) == 0:
            return 0.0
        
        if len(values) < n_groups:
            return np.median(values)
        
        group_size = len(values) // n_groups
        group_means = []
        
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else len(values)
            group_means.append(np.mean(values[start:end]))
        
        return np.median(group_means)
