"""
Multi-Scale Temporal Analysis for Quality Metrics

Decomposes quality time series into different time scales:
- Short-term noise (days)
- Medium-term trends (weeks)
- Long-term patterns (months)
- Structural breaks (regime changes)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import signal
from scipy.stats import linregress
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalDecomposition:
    """Results of temporal decomposition."""
    original: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    change_points: List[int]
    trend_slope: float
    trend_p_value: float
    volatility: float


class TemporalAnalyzer:
    """
    Analyzes quality metrics over time using multi-scale decomposition.
    """
    
    @staticmethod
    def decompose_hodrick_prescott(
        time_series: np.ndarray,
        lambda_param: float = 1600.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose time series into trend and cyclical components using HP filter.
        
        Args:
            time_series: Time series data
            lambda_param: Smoothing parameter (higher = smoother trend)
            
        Returns:
            Tuple of (trend, cyclical)
        """
        n = len(time_series)
        if n < 3:
            return time_series.copy(), np.zeros(n)
        
        # Build HP filter matrix
        # Minimize: sum((y - trend)^2) + lambda * sum((trend_diff2)^2)
        # This is solved as: trend = (I + lambda * D'D)^(-1) * y
        
        # Second difference matrix
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            D[i, i] = 1
            D[i, i + 1] = -2
            D[i, i + 2] = 1
        
        # Solve for trend
        I = np.eye(n)
        H = I + lambda_param * D.T @ D
        trend = np.linalg.solve(H, time_series)
        cyclical = time_series - trend
        
        return trend, cyclical
    
    @staticmethod
    def detect_change_points(
        time_series: np.ndarray,
        method: str = "cusum"
    ) -> List[int]:
        """
        Detect structural break points in time series.
        
        Args:
            time_series: Time series data
            method: Detection method ("cusum" or "pelt")
            
        Returns:
            List of indices where changes occur
        """
        if len(time_series) < 4:
            return []
        
        change_points = []
        
        if method == "cusum":
            # CUSUM (Cumulative Sum) method
            mean = np.mean(time_series)
            cumsum = np.cumsum(time_series - mean)
            
            # Find points where cumulative sum deviates significantly
            threshold = 2 * np.std(time_series)
            
            for i in range(1, len(cumsum)):
                if abs(cumsum[i]) > threshold:
                    # Check if this is a new change point
                    if not change_points or i - change_points[-1] > len(time_series) // 10:
                        change_points.append(i)
        
        elif method == "pelt":
            # PELT (Pruned Exact Linear Time) - simplified version
            # Use binary segmentation
            def find_best_split(start, end):
                if end - start < 4:
                    return None, float('inf')
                
                best_split = None
                best_cost = float('inf')
                
                for split in range(start + 2, end - 2):
                    left_mean = np.mean(time_series[start:split])
                    right_mean = np.mean(time_series[split:end])
                    
                    left_cost = np.sum((time_series[start:split] - left_mean) ** 2)
                    right_cost = np.sum((time_series[split:end] - right_mean) ** 2)
                    total_cost = left_cost + right_cost
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_split = split
                
                return best_split, best_cost
            
            # Recursive binary segmentation
            def segment(start, end, depth=0):
                if end - start < 4 or depth > 5:
                    return []
                
                split, cost = find_best_split(start, end)
                if split is None:
                    return []
                
                # Check if split is significant
                no_split_cost = np.sum((time_series[start:end] - np.mean(time_series[start:end])) ** 2)
                if cost < no_split_cost * 0.8:  # 20% improvement threshold
                    return [split] + segment(start, split, depth + 1) + segment(split, end, depth + 1)
                return []
            
            change_points = sorted(segment(0, len(time_series)))
        
        return change_points
    
    @staticmethod
    def calculate_trend_statistics(
        time_series: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate trend statistics.
        
        Args:
            time_series: Time series data
            
        Returns:
            Tuple of (slope, p_value, r_squared)
        """
        if len(time_series) < 2:
            return 0.0, 1.0, 0.0
        
        x = np.arange(len(time_series))
        slope, intercept, r_value, p_value, std_err = linregress(x, time_series)
        r_squared = r_value ** 2
        
        return slope, p_value, r_squared
    
    @staticmethod
    def decompose_multi_scale(
        time_series: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> TemporalDecomposition:
        """
        Multi-scale decomposition of quality time series.
        
        Args:
            time_series: Time series data
            timestamps: Optional timestamps (if None, assumes uniform spacing)
            
        Returns:
            TemporalDecomposition with all components
        """
        if timestamps is None:
            timestamps = np.arange(len(time_series))
        
        # HP filter for trend
        trend, cyclical = TemporalAnalyzer.decompose_hodrick_prescott(time_series)
        
        # Detect change points
        change_points = TemporalAnalyzer.detect_change_points(time_series)
        
        # Calculate trend statistics
        trend_slope, trend_p_value, r_squared = TemporalAnalyzer.calculate_trend_statistics(trend)
        
        # Residual (noise)
        residual = time_series - trend
        
        # Volatility (rolling standard deviation of residuals)
        if len(residual) > 10:
            window = min(10, len(residual) // 3)
            volatility = np.std([np.std(residual[i:i+window]) 
                                for i in range(len(residual) - window + 1)])
        else:
            volatility = np.std(residual)
        
        return TemporalDecomposition(
            original=time_series,
            trend=trend,
            seasonal=cyclical,  # Using cyclical as seasonal proxy
            residual=residual,
            change_points=change_points,
            trend_slope=trend_slope,
            trend_p_value=trend_p_value,
            volatility=volatility
        )
    
    @staticmethod
    def forecast_trend(
        decomposition: TemporalDecomposition,
        n_periods: int = 10
    ) -> np.ndarray:
        """
        Forecast future values based on trend.
        
        Args:
            decomposition: Temporal decomposition
            n_periods: Number of periods to forecast
            
        Returns:
            Forecasted values
        """
        trend = decomposition.trend
        if len(trend) < 2:
            return np.full(n_periods, trend[0] if len(trend) > 0 else 0.0)
        
        # Linear extrapolation of trend
        x = np.arange(len(trend))
        slope, intercept, _, _, _ = linregress(x, trend)
        
        future_x = np.arange(len(trend), len(trend) + n_periods)
        forecast = slope * future_x + intercept
        
        return forecast
    
    @staticmethod
    def detect_anomalies(
        time_series: np.ndarray,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> List[int]:
        """
        Detect anomalies in time series.
        
        Args:
            time_series: Time series data
            method: Detection method ("zscore" or "isolation")
            threshold: Threshold for detection
            
        Returns:
            List of indices where anomalies occur
        """
        anomalies = []
        
        if method == "zscore":
            mean = np.mean(time_series)
            std = np.std(time_series)
            
            if std > 0:
                z_scores = np.abs((time_series - mean) / std)
                anomalies = np.where(z_scores > threshold)[0].tolist()
        
        elif method == "isolation":
            # Simplified isolation forest
            # Use median absolute deviation (MAD)
            median = np.median(time_series)
            mad = np.median(np.abs(time_series - median))
            
            if mad > 0:
                modified_z_scores = 0.6745 * (time_series - median) / mad
                anomalies = np.where(np.abs(modified_z_scores) > threshold)[0].tolist()
        
        return anomalies
