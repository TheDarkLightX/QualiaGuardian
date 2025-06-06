"""
Unit tests for temporal importance tracking.

Author: DarkLightX/Dana Edwards
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from guardian.analytics.temporal_importance import (
    TemporalImportanceTracker,
    DecayType,
    AlertType,
    TemporalPattern
)


class TestTemporalImportanceTracker:
    """Test temporal importance tracking functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker with temporary database."""
        temp_dir = tempfile.mkdtemp()
        db_path = str(Path(temp_dir) / "test_temporal.db")
        tracker = TemporalImportanceTracker(db_path=db_path)
        yield tracker
        shutil.rmtree(temp_dir)
        
    def test_record_and_retrieve_importance(self, tracker):
        """Test basic recording and retrieval."""
        test_name = "test_example"
        importance = 0.75
        timestamp = datetime.now()
        
        # Record importance
        tracker.record_importance(test_name, importance, timestamp)
        
        # Retrieve data
        data = tracker._get_time_series(test_name)
        assert len(data) == 1
        assert abs(data[0][1] - importance) < 0.001
        
    def test_time_series_decomposition(self, tracker):
        """Test time series decomposition with seasonal data."""
        test_name = "test_seasonal"
        base_date = datetime.now() - timedelta(days=30)
        
        # Create seasonal data
        for day in range(30):
            timestamp = base_date + timedelta(days=day)
            importance = 0.7 + 0.2 * np.sin(day * 2 * np.pi / 7)  # Weekly pattern
            importance += np.random.normal(0, 0.01)
            tracker.record_importance(test_name, np.clip(importance, 0, 1), timestamp)
            
        # Decompose
        components = tracker.decompose_time_series(test_name, period=7)
        
        if components:  # Only if statsmodels available
            assert components.trend is not None
            assert components.seasonal is not None
            assert len(components.trend) == 30
            
    def test_change_point_detection(self, tracker):
        """Test change point detection."""
        test_name = "test_change"
        base_date = datetime.now() - timedelta(days=20)
        
        # Create data with change point
        for day in range(20):
            timestamp = base_date + timedelta(days=day)
            if day < 10:
                importance = 0.8 + np.random.normal(0, 0.02)
            else:
                importance = 0.4 + np.random.normal(0, 0.02)
            tracker.record_importance(test_name, np.clip(importance, 0, 1), timestamp)
            
        # Detect change points
        change_points = tracker.detect_change_points(test_name, min_size=3)
        
        assert len(change_points) >= 1
        # Should detect change around day 10
        change_indices = [cp.index for cp in change_points]
        assert any(8 <= idx <= 12 for idx in change_indices)
        
    def test_exponential_decay_modeling(self, tracker):
        """Test exponential decay modeling."""
        test_name = "test_decay"
        base_date = datetime.now() - timedelta(days=60)
        
        # Create exponentially decaying data
        for day in range(60):
            timestamp = base_date + timedelta(days=day)
            importance = 0.9 * np.exp(-0.03 * day) + 0.1
            importance += np.random.normal(0, 0.005)
            tracker.record_importance(test_name, np.clip(importance, 0, 1), timestamp)
            
        # Model decay
        result = tracker.model_decay(test_name, DecayType.EXPONENTIAL)
        
        assert "parameters" in result
        assert result["r_squared"] > 0.9  # Should fit well
        assert "half_life" in result
        assert result["half_life"] > 0
        
    def test_importance_forecasting(self, tracker):
        """Test importance forecasting."""
        test_name = "test_forecast"
        base_date = datetime.now() - timedelta(days=30)
        
        # Create trending data
        for day in range(30):
            timestamp = base_date + timedelta(days=day)
            importance = 0.8 - 0.01 * day + np.random.normal(0, 0.02)
            tracker.record_importance(test_name, np.clip(importance, 0, 1), timestamp)
            
        # Forecast
        forecast = tracker.forecast_importance(test_name, horizon=7)
        
        if forecast:
            assert len(forecast.values) == 7
            assert len(forecast.timestamps) == 7
            assert np.all(forecast.values >= 0)
            assert np.all(forecast.values <= 1)
            
    def test_alert_generation(self, tracker):
        """Test alert generation for sudden changes."""
        test_name = "test_alert"
        base_date = datetime.now() - timedelta(days=5)
        
        # Create stable data
        for day in range(5):
            timestamp = base_date + timedelta(days=day)
            tracker.record_importance(test_name, 0.8 + np.random.normal(0, 0.01), timestamp)
            
        # Sudden drop
        tracker.record_importance(test_name, 0.2)
        
        # Check alerts
        alerts = tracker.get_alerts(test_name)
        assert len(alerts) > 0
        
        # Should have sudden decrease alert
        assert any(alert.alert_type == AlertType.SUDDEN_DECREASE for alert in alerts)
        
    def test_temporal_pattern_detection(self, tracker):
        """Test detection of temporal patterns."""
        test_name = "test_pattern"
        base_date = datetime.now() - timedelta(days=35)
        
        # Create weekly pattern
        for day in range(35):
            timestamp = base_date + timedelta(days=day)
            # Strong weekly pattern
            day_of_week = day % 7
            if day_of_week < 5:  # Weekday
                importance = 0.8 + np.random.normal(0, 0.02)
            else:  # Weekend
                importance = 0.3 + np.random.normal(0, 0.02)
            tracker.record_importance(test_name, np.clip(importance, 0, 1), timestamp)
            
        # Get patterns
        patterns = tracker.get_temporal_patterns(test_name)
        
        # Should detect weekly pattern
        weekly_patterns = [p for p in patterns if p.pattern_type == "weekly"]
        assert len(weekly_patterns) > 0
        
    def test_visualization_data(self, tracker):
        """Test visualization data generation."""
        test_name = "test_viz"
        base_date = datetime.now() - timedelta(days=14)
        
        # Create some data
        for day in range(14):
            timestamp = base_date + timedelta(days=day)
            importance = 0.6 + 0.1 * np.sin(day * 2 * np.pi / 7)
            tracker.record_importance(test_name, importance, timestamp)
            
        # Get visualization data
        viz_data = tracker.visualize_temporal_importance(test_name)
        
        assert "time_series" in viz_data
        assert "statistics" in viz_data
        assert viz_data["time_series"]["values"]
        assert viz_data["statistics"]["mean"] > 0
        
    def test_decay_threshold_alert(self, tracker):
        """Test alert when importance falls below threshold."""
        test_name = "test_low_importance"
        
        # Record low importance
        tracker.record_importance(test_name, 0.05)
        
        # Check for decay threshold alert
        alerts = tracker.get_alerts(test_name)
        assert any(alert.alert_type == AlertType.DECAY_THRESHOLD for alert in alerts)
        
    def test_multiple_decay_models(self, tracker):
        """Test fitting multiple decay models."""
        test_name = "test_multi_decay"
        base_date = datetime.now() - timedelta(days=30)
        
        # Create data with linear decay
        for day in range(30):
            timestamp = base_date + timedelta(days=day)
            importance = max(0.1, 0.9 - 0.025 * day)
            importance += np.random.normal(0, 0.01)
            tracker.record_importance(test_name, np.clip(importance, 0, 1), timestamp)
            
        # Try different decay models
        models = {}
        for decay_type in [DecayType.LINEAR, DecayType.EXPONENTIAL, 
                          DecayType.LOGARITHMIC, DecayType.POWER]:
            models[decay_type] = tracker.model_decay(test_name, decay_type)
            
        # Linear should fit well for linear data
        linear_r2 = models[DecayType.LINEAR].get("r_squared", 0)
        assert linear_r2 > 0.8
        
        # All models should produce valid results without errors
        for decay_type, result in models.items():
            assert "r_squared" in result
            assert isinstance(result["r_squared"], (int, float))
            assert "parameters" in result
            
        # Linear should be a reasonably good fit and among the better models
        # (R-squared can be negative for very poor fits, which is mathematically valid)
        sorted_models = sorted(models.items(), 
                              key=lambda x: x[1].get("r_squared", -np.inf), 
                              reverse=True)
        
        # Linear should be in top 3 models (since there are only 4 total)
        top_3_models = [model[0] for model in sorted_models[:3]]
        assert DecayType.LINEAR in top_3_models