"""
Temporal Importance Tracking for Test Analysis

This module provides time series modeling of test importance with trend analysis,
seasonality detection, change point detection, and forecasting capabilities.

Author: DarkLightX/Dana Edwards
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
from enum import Enum
import warnings
from collections import defaultdict

# Statistical libraries for time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Some features will be limited.")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    warnings.warn("ruptures not available. Change point detection will be limited.")


class DecayType(Enum):
    """Types of decay functions for importance modeling."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"
    LOGARITHMIC = "logarithmic"
    POWER = "power"


class AlertType(Enum):
    """Types of importance alerts."""
    SUDDEN_INCREASE = "sudden_increase"
    SUDDEN_DECREASE = "sudden_decrease"
    TREND_REVERSAL = "trend_reversal"
    ANOMALY = "anomaly"
    DECAY_THRESHOLD = "decay_threshold"
    FORECAST_WARNING = "forecast_warning"


@dataclass
class TimeSeriesComponents:
    """Components of decomposed time series."""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    original: np.ndarray
    timestamps: List[datetime]


@dataclass
class ChangePoint:
    """Detected change point in time series."""
    index: int
    timestamp: datetime
    confidence: float
    magnitude: float
    direction: str  # "increase" or "decrease"


@dataclass
class ImportanceForecast:
    """Forecast of future importance values."""
    timestamps: List[datetime]
    values: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    model_type: str


@dataclass
class ImportanceAlert:
    """Alert for significant importance changes."""
    test_name: str
    alert_type: AlertType
    timestamp: datetime
    severity: float  # 0-1
    message: str
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalPattern:
    """Detected temporal pattern in importance."""
    pattern_type: str  # "weekly", "daily", "monthly", etc.
    strength: float  # 0-1
    period: float  # in days
    phase: float  # phase shift in days


class TemporalImportanceTracker:
    """
    Tracks and analyzes temporal patterns in test importance.
    
    Features:
    - Time series decomposition (trend, seasonality, residual)
    - Change point detection
    - Multiple decay function models
    - Importance forecasting
    - Anomaly detection and alerting
    - Historical data persistence
    """
    
    def __init__(self, db_path: Optional[str] = None, window_size: int = 30):
        """
        Initialize temporal importance tracker.
        
        Args:
            db_path: Path to SQLite database for persistence
            window_size: Default window size for rolling calculations
        """
        self.db_path = db_path or ".guardian/temporal_importance.db"
        self.window_size = window_size
        self._init_database()
        self._cache = {}
        
    def _init_database(self):
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical importance data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS importance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                importance REAL NOT NULL,
                metadata TEXT,
                UNIQUE(test_name, timestamp)
            )
        """)
        
        # Detected patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS temporal_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                strength REAL NOT NULL,
                period REAL NOT NULL,
                phase REAL NOT NULL,
                detected_at DATETIME NOT NULL,
                metadata TEXT
            )
        """)
        
        # Alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS importance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                severity REAL NOT NULL,
                message TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                metadata TEXT,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Forecasts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS importance_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                forecast_data TEXT NOT NULL,
                model_type TEXT NOT NULL,
                accuracy_metrics TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
    def record_importance(self, test_name: str, importance: float, 
                         timestamp: Optional[datetime] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Record test importance at a specific timestamp.
        
        Args:
            test_name: Name of the test
            importance: Importance value (0-1)
            timestamp: Timestamp (defaults to now)
            metadata: Additional metadata
        """
        timestamp = timestamp or datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO importance_history 
                (test_name, timestamp, importance, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                test_name,
                timestamp.isoformat(),
                importance,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
            
            # Check for alerts
            self._check_alerts(test_name, importance, timestamp)
            
        finally:
            conn.close()
            
        # Invalidate cache
        if test_name in self._cache:
            del self._cache[test_name]
            
    def decompose_time_series(self, test_name: str, 
                            period: Optional[int] = None) -> Optional[TimeSeriesComponents]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            test_name: Name of the test
            period: Period for seasonal decomposition (auto-detected if None)
            
        Returns:
            TimeSeriesComponents or None if insufficient data
        """
        if not STATSMODELS_AVAILABLE:
            warnings.warn("statsmodels required for time series decomposition")
            return None
            
        data = self._get_time_series(test_name)
        if len(data) < 2 * (period or 7):  # Need at least 2 periods
            return None
            
        timestamps, values = zip(*data)
        values = np.array(values)
        
        # Auto-detect period if not provided
        if period is None:
            period = self._detect_period(values)
            
        try:
            decomposition = seasonal_decompose(
                values, 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )
            
            return TimeSeriesComponents(
                trend=decomposition.trend,
                seasonal=decomposition.seasonal,
                residual=decomposition.resid,
                original=values,
                timestamps=list(timestamps)
            )
        except Exception as e:
            warnings.warn(f"Decomposition failed: {e}")
            return None
            
    def detect_change_points(self, test_name: str, 
                           method: str = "pelt",
                           min_size: int = 5) -> List[ChangePoint]:
        """
        Detect change points in importance time series.
        
        Args:
            test_name: Name of the test
            method: Detection method ("pelt", "binseg", "window")
            min_size: Minimum segment size
            
        Returns:
            List of detected change points
        """
        data = self._get_time_series(test_name)
        if len(data) < 2 * min_size:
            return []
            
        timestamps, values = zip(*data)
        values = np.array(values)
        
        if RUPTURES_AVAILABLE:
            # Use ruptures library for advanced detection
            algo = {
                "pelt": rpt.Pelt(model="rbf", min_size=min_size),
                "binseg": rpt.Binseg(model="rbf", min_size=min_size),
                "window": rpt.Window(width=min_size, model="rbf")
            }.get(method, rpt.Pelt(model="rbf", min_size=min_size))
            
            algo.fit(values)
            change_points = algo.predict(pen=0.1 * np.var(values))
            
        else:
            # Simple threshold-based detection
            change_points = self._simple_change_detection(values, min_size)
            
        # Convert to ChangePoint objects
        results = []
        for idx in change_points[:-1]:  # Exclude last point (end of series)
            if idx < len(values) - 1:
                magnitude = abs(np.mean(values[idx:idx+min_size]) - 
                              np.mean(values[max(0,idx-min_size):idx]))
                direction = "increase" if values[idx] > values[max(0, idx-1)] else "decrease"
                
                results.append(ChangePoint(
                    index=idx,
                    timestamp=timestamps[idx],
                    confidence=min(1.0, magnitude / np.std(values)),
                    magnitude=magnitude,
                    direction=direction
                ))
                
        return results
        
    def model_decay(self, test_name: str, 
                   decay_type: DecayType = DecayType.EXPONENTIAL,
                   lookback_days: int = 90) -> Dict[str, Any]:
        """
        Model importance decay over time.
        
        Args:
            test_name: Name of the test
            decay_type: Type of decay function
            lookback_days: Days to look back for modeling
            
        Returns:
            Dictionary with decay parameters and goodness of fit
        """
        data = self._get_time_series(test_name, lookback_days)
        if len(data) < 3:
            return {"error": "Insufficient data"}
            
        timestamps, values = zip(*data)
        
        # Convert to days since first observation
        first_timestamp = timestamps[0]
        days = np.array([(t - first_timestamp).total_seconds() / 86400 
                        for t in timestamps])
        values = np.array(values)
        
        # Fit decay model
        params = self._fit_decay_model(days, values, decay_type)
        
        # Calculate goodness of fit
        predicted = self._apply_decay_function(days, params, decay_type)
        r_squared = 1 - np.sum((values - predicted)**2) / np.sum((values - np.mean(values))**2)
        
        return {
            "decay_type": decay_type.value,
            "parameters": params,
            "r_squared": r_squared,
            "half_life": self._calculate_half_life(params, decay_type),
            "current_decay_rate": self._calculate_decay_rate(days[-1], params, decay_type)
        }
        
    def forecast_importance(self, test_name: str, 
                          horizon: int = 7,
                          method: str = "holt_winters") -> Optional[ImportanceForecast]:
        """
        Forecast future importance values.
        
        Args:
            test_name: Name of the test
            horizon: Forecast horizon in days
            method: Forecasting method
            
        Returns:
            ImportanceForecast or None if insufficient data
        """
        data = self._get_time_series(test_name)
        if len(data) < 14:  # Need at least 2 weeks of data
            return None
            
        timestamps, values = zip(*data)
        values = np.array(values)
        
        # Generate future timestamps
        last_timestamp = timestamps[-1]
        future_timestamps = [
            last_timestamp + timedelta(days=i+1) 
            for i in range(horizon)
        ]
        
        if method == "holt_winters" and STATSMODELS_AVAILABLE:
            return self._forecast_holt_winters(values, future_timestamps)
        else:
            # Simple moving average forecast
            return self._forecast_moving_average(values, future_timestamps)
            
    def get_temporal_patterns(self, test_name: str) -> List[TemporalPattern]:
        """
        Detect and return temporal patterns (weekly, daily cycles, etc.).
        
        Args:
            test_name: Name of the test
            
        Returns:
            List of detected temporal patterns
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pattern_type, strength, period, phase, metadata
            FROM temporal_patterns
            WHERE test_name = ?
            ORDER BY detected_at DESC
            LIMIT 10
        """, (test_name,))
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append(TemporalPattern(
                pattern_type=row[0],
                strength=row[1],
                period=row[2],
                phase=row[3]
            ))
            
        conn.close()
        
        # Also detect new patterns if data available
        new_patterns = self._detect_patterns(test_name)
        patterns.extend(new_patterns)
        
        return patterns
        
    def get_alerts(self, test_name: Optional[str] = None,
                  unacknowledged_only: bool = True,
                  limit: int = 100) -> List[ImportanceAlert]:
        """
        Get importance alerts.
        
        Args:
            test_name: Filter by test name (all if None)
            unacknowledged_only: Only return unacknowledged alerts
            limit: Maximum number of alerts to return
            
        Returns:
            List of importance alerts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT test_name, alert_type, timestamp, severity, 
                   message, recommendation, metadata
            FROM importance_alerts
            WHERE 1=1
        """
        params = []
        
        if test_name:
            query += " AND test_name = ?"
            params.append(test_name)
            
        if unacknowledged_only:
            query += " AND acknowledged = FALSE"
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append(ImportanceAlert(
                test_name=row[0],
                alert_type=AlertType(row[1]),
                timestamp=datetime.fromisoformat(row[2]),
                severity=row[3],
                message=row[4],
                recommendation=row[5],
                metadata=json.loads(row[6]) if row[6] else {}
            ))
            
        conn.close()
        return alerts
        
    def visualize_temporal_importance(self, test_name: str, 
                                    output_format: str = "dict") -> Dict[str, Any]:
        """
        Generate visualization data for temporal importance.
        
        Args:
            test_name: Name of the test
            output_format: Format for visualization data
            
        Returns:
            Dictionary with visualization data
        """
        data = self._get_time_series(test_name)
        if not data:
            return {"error": "No data available"}
            
        timestamps, values = zip(*data)
        
        # Decompose if possible
        components = self.decompose_time_series(test_name)
        
        # Detect change points
        change_points = self.detect_change_points(test_name)
        
        # Get forecast
        forecast = self.forecast_importance(test_name)
        
        viz_data = {
            "test_name": test_name,
            "time_series": {
                "timestamps": [t.isoformat() for t in timestamps],
                "values": list(values)
            },
            "statistics": {
                "mean": np.mean(values),
                "std": np.std(values),
                "trend": self._calculate_trend(values),
                "volatility": np.std(np.diff(values)) if len(values) > 1 else 0
            }
        }
        
        if components:
            viz_data["decomposition"] = {
                "trend": components.trend.tolist(),
                "seasonal": components.seasonal.tolist(),
                "residual": components.residual.tolist()
            }
            
        if change_points:
            viz_data["change_points"] = [
                {
                    "timestamp": cp.timestamp.isoformat(),
                    "magnitude": cp.magnitude,
                    "direction": cp.direction,
                    "confidence": cp.confidence
                }
                for cp in change_points
            ]
            
        if forecast:
            viz_data["forecast"] = {
                "timestamps": [t.isoformat() for t in forecast.timestamps],
                "values": forecast.values.tolist(),
                "confidence_lower": forecast.confidence_lower.tolist(),
                "confidence_upper": forecast.confidence_upper.tolist()
            }
            
        return viz_data
        
    # Private helper methods
    
    def _get_time_series(self, test_name: str, 
                        lookback_days: Optional[int] = None) -> List[Tuple[datetime, float]]:
        """Get time series data from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT timestamp, importance
            FROM importance_history
            WHERE test_name = ?
        """
        params = [test_name]
        
        if lookback_days:
            cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
            query += " AND timestamp >= ?"
            params.append(cutoff)
            
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        
        data = [
            (datetime.fromisoformat(row[0]), row[1])
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return data
        
    def _detect_period(self, values: np.ndarray) -> int:
        """Auto-detect period in time series."""
        if len(values) < 14:
            return 7  # Default to weekly
            
        # Try common periods
        for period in [7, 30, 365]:
            if len(values) >= 2 * period:
                # Simple autocorrelation check
                acf = np.correlate(values - np.mean(values), 
                                 values - np.mean(values), 
                                 mode='full')
                acf = acf[len(acf)//2:]
                acf = acf / acf[0]
                
                if period < len(acf) and acf[period] > 0.3:
                    return period
                    
        return 7  # Default to weekly
        
    def _simple_change_detection(self, values: np.ndarray, 
                               min_size: int) -> List[int]:
        """Simple threshold-based change point detection."""
        change_points = []
        
        # Use a more sensitive threshold for change detection
        global_std = np.std(values)
        if global_std < 1e-6:  # Handle constant series
            return [len(values)]
            
        # Use 1.5 standard deviations as threshold, more sensitive than 2
        threshold = 1.5 * global_std
        
        for i in range(min_size, len(values) - min_size):
            left_mean = np.mean(values[i-min_size:i])
            right_mean = np.mean(values[i:i+min_size])
            
            # Check for significant difference
            if abs(left_mean - right_mean) > threshold:
                # Avoid duplicate change points too close together
                if not change_points or i - change_points[-1] >= min_size:
                    change_points.append(i)
                
        # Add end point
        change_points.append(len(values))
        return change_points
        
    def _fit_decay_model(self, days: np.ndarray, values: np.ndarray, 
                        decay_type: DecayType) -> Dict[str, float]:
        """Fit decay model parameters."""
        from scipy.optimize import curve_fit
        
        def exponential_decay(t, a, b, c):
            return a * np.exp(-b * t) + c
            
        def linear_decay(t, a, b):
            return np.maximum(0, a - b * t)
            
        def logarithmic_decay(t, a, b, c):
            # Ensure argument to log is always positive
            arg = np.maximum(b * t + 1, 1e-10)
            return a * np.log(arg) + c
            
        def power_decay(t, a, b, c):
            return a * (t + 1) ** (-b) + c
            
        try:
            if decay_type == DecayType.EXPONENTIAL:
                popt, _ = curve_fit(exponential_decay, days, values, 
                                   p0=[1.0, 0.01, 0.1])
                return {"a": popt[0], "b": popt[1], "c": popt[2]}
                
            elif decay_type == DecayType.LINEAR:
                # Use bounds to ensure positive parameters for linear decay
                popt, _ = curve_fit(linear_decay, days, values,
                                   p0=[1.0, 0.01],
                                   bounds=([0, 0], [np.inf, np.inf]))
                return {"a": popt[0], "b": popt[1]}
                
            elif decay_type == DecayType.LOGARITHMIC:
                # Suppress warnings for log of invalid values during fitting
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    popt, _ = curve_fit(logarithmic_decay, days, values,
                                       p0=[0.5, 1.0, 0.1])
                return {"a": popt[0], "b": popt[1], "c": popt[2]}
                
            elif decay_type == DecayType.POWER:
                popt, _ = curve_fit(power_decay, days, values,
                                   p0=[1.0, 0.5, 0.1])
                return {"a": popt[0], "b": popt[1], "c": popt[2]}
                
            else:  # STEP
                # Simple step function - find best step point
                best_error = float('inf')
                best_params = {"step_day": 0, "before": 1.0, "after": 0.1}
                
                for step_day in range(1, len(days)-1):
                    before = np.mean(values[:step_day])
                    after = np.mean(values[step_day:])
                    error = np.sum((values[:step_day] - before)**2) + \
                           np.sum((values[step_day:] - after)**2)
                    
                    if error < best_error:
                        best_error = error
                        best_params = {
                            "step_day": days[step_day],
                            "before": before,
                            "after": after
                        }
                        
                return best_params
                
        except Exception:
            # Return default parameters on failure
            return {"a": 1.0, "b": 0.01, "c": 0.1}
            
    def _apply_decay_function(self, days: np.ndarray, params: Dict[str, float], 
                            decay_type: DecayType) -> np.ndarray:
        """Apply decay function with parameters."""
        if decay_type == DecayType.EXPONENTIAL:
            return params["a"] * np.exp(-params["b"] * days) + params["c"]
            
        elif decay_type == DecayType.LINEAR:
            return np.maximum(0, params["a"] - params["b"] * days)
            
        elif decay_type == DecayType.LOGARITHMIC:
            # Ensure argument to log is always positive
            arg = np.maximum(params["b"] * days + 1, 1e-10)
            return params["a"] * np.log(arg) + params["c"]
            
        elif decay_type == DecayType.POWER:
            return params["a"] * (days + 1) ** (-params["b"]) + params["c"]
            
        else:  # STEP
            return np.where(days < params["step_day"], 
                          params["before"], 
                          params["after"])
            
    def _calculate_half_life(self, params: Dict[str, float], 
                           decay_type: DecayType) -> Optional[float]:
        """Calculate half-life for decay model."""
        if decay_type == DecayType.EXPONENTIAL:
            # Solve: 0.5 = exp(-b * t)
            return np.log(2) / params["b"]
            
        elif decay_type == DecayType.LINEAR:
            # Solve: 0.5 * a = a - b * t
            return 0.5 * params["a"] / params["b"]
            
        elif decay_type == DecayType.POWER:
            # Solve: 0.5 = (t + 1)^(-b)
            return 2**(1/params["b"]) - 1
            
        return None
        
    def _calculate_decay_rate(self, day: float, params: Dict[str, float], 
                            decay_type: DecayType) -> float:
        """Calculate instantaneous decay rate."""
        epsilon = 0.001
        current = self._apply_decay_function(np.array([day]), params, decay_type)[0]
        next_val = self._apply_decay_function(np.array([day + epsilon]), params, decay_type)[0]
        
        return (current - next_val) / epsilon
        
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate overall trend (-1 to 1)."""
        if len(values) < 2:
            return 0.0
            
        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize by mean value
        return np.tanh(slope / (np.mean(values) + 1e-6))
        
    def _forecast_holt_winters(self, values: np.ndarray, 
                             future_timestamps: List[datetime]) -> ImportanceForecast:
        """Forecast using Holt-Winters exponential smoothing."""
        try:
            model = ExponentialSmoothing(
                values,
                seasonal_periods=7,
                trend='add',
                seasonal='add',
                use_boxcox=False
            )
            fit = model.fit()
            
            forecast = fit.forecast(len(future_timestamps))
            
            # Simple confidence intervals (±2 std of residuals)
            residual_std = np.std(fit.fittedvalues - values)
            confidence_margin = 2 * residual_std
            
            return ImportanceForecast(
                timestamps=future_timestamps,
                values=np.clip(forecast, 0, 1),
                confidence_lower=np.clip(forecast - confidence_margin, 0, 1),
                confidence_upper=np.clip(forecast + confidence_margin, 0, 1),
                model_type="holt_winters"
            )
        except Exception:
            # Fallback to moving average
            return self._forecast_moving_average(values, future_timestamps)
            
    def _forecast_moving_average(self, values: np.ndarray, 
                                future_timestamps: List[datetime]) -> ImportanceForecast:
        """Simple moving average forecast."""
        window = min(7, len(values) // 2)
        ma = np.mean(values[-window:])
        
        # Decay forecast slightly
        decay_rate = 0.99
        forecast = np.array([ma * (decay_rate ** i) 
                           for i in range(len(future_timestamps))])
        
        # Confidence based on recent volatility
        recent_std = np.std(values[-window:])
        confidence_margin = 2 * recent_std
        
        return ImportanceForecast(
            timestamps=future_timestamps,
            values=np.clip(forecast, 0, 1),
            confidence_lower=np.clip(forecast - confidence_margin, 0, 1),
            confidence_upper=np.clip(forecast + confidence_margin, 0, 1),
            model_type="moving_average"
        )
        
    def _detect_patterns(self, test_name: str) -> List[TemporalPattern]:
        """Detect new temporal patterns in data."""
        data = self._get_time_series(test_name)
        if len(data) < 28:  # Need at least 4 weeks
            return []
            
        timestamps, values = zip(*data)
        values = np.array(values)
        
        patterns = []
        
        # Check for weekly pattern
        if len(values) >= 14:
            weekly_strength = self._check_periodicity(values, 7)
            if weekly_strength > 0.3:
                patterns.append(TemporalPattern(
                    pattern_type="weekly",
                    strength=weekly_strength,
                    period=7.0,
                    phase=self._find_phase(values, 7)
                ))
                
        # Check for daily pattern (if enough granularity)
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600
                     for i in range(len(timestamps)-1)]
        avg_hours = np.mean(time_diffs)
        
        if avg_hours < 12:  # Sub-daily data
            daily_strength = self._check_periodicity(values, int(24 / avg_hours))
            if daily_strength > 0.3:
                patterns.append(TemporalPattern(
                    pattern_type="daily",
                    strength=daily_strength,
                    period=1.0,
                    phase=self._find_phase(values, int(24 / avg_hours))
                ))
                
        # Store detected patterns
        if patterns:
            self._store_patterns(test_name, patterns)
            
        return patterns
        
    def _check_periodicity(self, values: np.ndarray, period: int) -> float:
        """Check strength of periodicity."""
        if len(values) < 2 * period:
            return 0.0
            
        # Autocorrelation at lag = period
        acf = np.correlate(values - np.mean(values), 
                         values - np.mean(values), 
                         mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / (acf[0] + 1e-10)
        
        if period < len(acf):
            return abs(acf[period])
        return 0.0
        
    def _find_phase(self, values: np.ndarray, period: int) -> float:
        """Find phase shift of periodic pattern."""
        if len(values) < period:
            return 0.0
            
        # Find peak in first period
        first_period = values[:period]
        peak_idx = np.argmax(first_period)
        
        return peak_idx / period
        
    def _store_patterns(self, test_name: str, patterns: List[TemporalPattern]):
        """Store detected patterns in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pattern in patterns:
            cursor.execute("""
                INSERT INTO temporal_patterns 
                (test_name, pattern_type, strength, period, phase, detected_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                test_name,
                pattern.pattern_type,
                pattern.strength,
                pattern.period,
                pattern.phase,
                datetime.now().isoformat(),
                "{}"
            ))
            
        conn.commit()
        conn.close()
        
    def _check_alerts(self, test_name: str, importance: float, timestamp: datetime):
        """Check for alert conditions."""
        # Get recent history
        data = self._get_time_series(test_name, lookback_days=7)
        
        # Check for decay threshold (independent of history length)
        if importance < 0.1:
            self._create_alert(
                test_name=test_name,
                alert_type=AlertType.DECAY_THRESHOLD,
                timestamp=timestamp,
                severity=0.7,
                message="Test importance has decayed below threshold",
                recommendation="Consider removing or updating this test"
            )
        
        # Check for sudden changes (requires sufficient history)
        if len(data) < 3:
            return
            
        values = np.array([v for _, v in data])
        
        if len(values) > 1:
            recent_mean = np.mean(values[:-1])
            recent_std = np.std(values[:-1]) + 1e-6
            z_score = abs(importance - recent_mean) / recent_std
            
            if z_score > 3:
                alert_type = (AlertType.SUDDEN_INCREASE if importance > recent_mean 
                            else AlertType.SUDDEN_DECREASE)
                self._create_alert(
                    test_name=test_name,
                    alert_type=alert_type,
                    timestamp=timestamp,
                    severity=min(1.0, z_score / 5),
                    message=f"Test importance changed by {z_score:.1f} standard deviations",
                    recommendation="Investigate recent code changes affecting this test"
                )
            
    def _create_alert(self, **kwargs):
        """Create and store an alert."""
        alert = ImportanceAlert(**kwargs)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO importance_alerts 
            (test_name, alert_type, timestamp, severity, message, recommendation, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.test_name,
            alert.alert_type.value,
            alert.timestamp.isoformat(),
            alert.severity,
            alert.message,
            alert.recommendation,
            json.dumps(alert.metadata)
        ))
        
        conn.commit()
        conn.close()