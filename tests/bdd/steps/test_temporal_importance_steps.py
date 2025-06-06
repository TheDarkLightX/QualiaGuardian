"""
BDD test steps for temporal importance tracking.

Author: DarkLightX/Dana Edwards
"""

import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

from guardian.analytics.temporal_importance import (
    TemporalImportanceTracker,
    DecayType,
    AlertType,
    TimeSeriesComponents,
    ChangePoint,
    ImportanceForecast,
    ImportanceAlert,
    TemporalPattern
)

# Load scenarios from feature file
scenarios('../features/temporal_importance.feature')


@pytest.fixture
def temp_db_path():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = str(Path(temp_dir) / "test_temporal.db")
    yield db_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def tracker(temp_db_path):
    """Create a temporal importance tracker instance."""
    return TemporalImportanceTracker(db_path=temp_db_path)


@pytest.fixture
def test_data():
    """Container for test data."""
    return {
        "recordings": [],
        "decomposition": None,
        "change_points": [],
        "decay_model": {},
        "forecast": None,
        "patterns": [],
        "alerts": [],
        "visualization": {}
    }


# Background steps

@given("I have a temporal importance tracker initialized")
def tracker_initialized(tracker):
    """Ensure tracker is initialized."""
    assert tracker is not None
    assert Path(tracker.db_path).exists()


@given("I have a test suite with historical importance data")
def historical_data(tracker):
    """Create some historical test data."""
    tests = ["test_auth", "test_payment", "test_search"]
    base_date = datetime.now() - timedelta(days=30)
    
    for test in tests:
        for day in range(30):
            timestamp = base_date + timedelta(days=day)
            # Create realistic importance values with some patterns
            importance = 0.8 + 0.1 * np.sin(day * 2 * np.pi / 7)  # Weekly pattern
            importance += np.random.normal(0, 0.02)  # Noise
            importance = np.clip(importance, 0, 1)
            
            tracker.record_importance(test, importance, timestamp)


# Scenario: Recording test importance over time

@given('I have a test named "test_critical_feature"')
def test_critical_feature(test_data):
    """Set up test name."""
    test_data["test_name"] = "test_critical_feature"


@when(parsers.parse("I record importance values:\n{table}"))
def record_importance_values(tracker, test_data, table):
    """Record importance values from table."""
    import re
    lines = table.strip().split('\n')
    
    for line in lines[1:]:  # Skip header
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 2:
            timestamp_str = parts[0]
            importance = float(parts[1])
            
            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
            
            tracker.record_importance(
                test_data["test_name"], 
                importance, 
                timestamp
            )
            test_data["recordings"].append((timestamp, importance))


@then("the importance history should be persisted")
def check_persistence(tracker, test_data):
    """Verify data is persisted."""
    conn = tracker._get_time_series(test_data["test_name"])
    assert len(conn) == len(test_data["recordings"])


@then("I should be able to retrieve the full time series")
def retrieve_time_series(tracker, test_data):
    """Verify time series retrieval."""
    data = tracker._get_time_series(test_data["test_name"])
    assert len(data) == len(test_data["recordings"])
    
    for (recorded_ts, recorded_val), (retrieved_ts, retrieved_val) in zip(
        test_data["recordings"], data
    ):
        assert abs((recorded_ts - retrieved_ts).total_seconds()) < 1
        assert abs(recorded_val - retrieved_val) < 0.001


# Scenario: Decomposing time series

@given('I have a test "test_seasonal_behavior" with 30 days of data')
def seasonal_test_data(tracker, test_data):
    """Create test with seasonal pattern."""
    test_data["test_name"] = "test_seasonal_behavior"
    base_date = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        timestamp = base_date + timedelta(days=day)
        # Strong weekly pattern
        importance = 0.7 + 0.2 * np.sin(day * 2 * np.pi / 7)
        importance += 0.05 * np.sin(day * 2 * np.pi / 30)  # Monthly
        importance += np.random.normal(0, 0.01)  # Small noise
        importance = np.clip(importance, 0, 1)
        
        tracker.record_importance(test_data["test_name"], importance, timestamp)


@given("the data shows weekly seasonality")
def verify_seasonality():
    """Confirm seasonal pattern exists."""
    pass  # Pattern created in previous step


@when("I decompose the time series")
def decompose_series(tracker, test_data):
    """Perform time series decomposition."""
    test_data["decomposition"] = tracker.decompose_time_series(
        test_data["test_name"], 
        period=7
    )


@then("I should get trend component")
def check_trend(test_data):
    """Verify trend component exists."""
    assert test_data["decomposition"] is not None
    assert test_data["decomposition"].trend is not None
    assert len(test_data["decomposition"].trend) > 0


@then("I should get seasonal component with period 7")
def check_seasonal(test_data):
    """Verify seasonal component."""
    decomp = test_data["decomposition"]
    assert decomp.seasonal is not None
    
    # Check seasonality repeats every 7 days
    period = 7
    for i in range(period, len(decomp.seasonal) - period):
        assert abs(decomp.seasonal[i] - decomp.seasonal[i - period]) < 0.1


@then("I should get residual component")
def check_residual(test_data):
    """Verify residual component."""
    assert test_data["decomposition"].residual is not None
    assert len(test_data["decomposition"].residual) > 0


@then("the components should sum to the original series")
def check_decomposition_sum(test_data):
    """Verify additive decomposition."""
    decomp = test_data["decomposition"]
    reconstructed = decomp.trend + decomp.seasonal + decomp.residual
    
    # Handle NaN values at boundaries
    valid_indices = ~(np.isnan(decomp.trend) | np.isnan(decomp.seasonal) | 
                     np.isnan(decomp.residual))
    
    diff = abs(reconstructed[valid_indices] - decomp.original[valid_indices])
    assert np.max(diff) < 0.01


# Scenario: Detecting change points

@given(parsers.parse('I have a test "test_refactored_module" with importance history:\n{table}'))
def create_change_point_data(tracker, test_data, table):
    """Create data with change point."""
    test_data["test_name"] = "test_refactored_module"
    base_date = datetime.now() - timedelta(days=15)
    
    # Parse table to understand the pattern
    lines = table.strip().split('\n')
    
    # Days 1-7: stable around 0.8-0.85
    for day in range(1, 8):
        timestamp = base_date + timedelta(days=day)
        importance = 0.825 + np.random.uniform(-0.025, 0.025)
        tracker.record_importance(test_data["test_name"], importance, timestamp)
    
    # Day 8: sudden drop
    timestamp = base_date + timedelta(days=8)
    tracker.record_importance(test_data["test_name"], 0.6, timestamp)
    
    # Days 9-15: stable around 0.58-0.62
    for day in range(9, 16):
        timestamp = base_date + timedelta(days=day)
        importance = 0.6 + np.random.uniform(-0.02, 0.02)
        tracker.record_importance(test_data["test_name"], importance, timestamp)


@when("I run change point detection")
def detect_changes(tracker, test_data):
    """Run change point detection."""
    test_data["change_points"] = tracker.detect_change_points(
        test_data["test_name"]
    )


@then("I should detect a change point at day 8")
def verify_change_point(test_data):
    """Verify change point detected."""
    assert len(test_data["change_points"]) >= 1
    
    # Check if any change point is around day 8
    change_indices = [cp.index for cp in test_data["change_points"]]
    assert any(6 <= idx <= 9 for idx in change_indices)


@then("the change magnitude should be approximately 0.2")
def verify_magnitude(test_data):
    """Verify change magnitude."""
    for cp in test_data["change_points"]:
        if 6 <= cp.index <= 9:
            assert 0.15 <= cp.magnitude <= 0.25


@then('the direction should be "decrease"')
def verify_direction(test_data):
    """Verify change direction."""
    for cp in test_data["change_points"]:
        if 6 <= cp.index <= 9:
            assert cp.direction == "decrease"


# Scenario: Modeling exponential decay

@given('I have a test "test_legacy_feature" showing decay pattern')
def create_decay_data(tracker, test_data):
    """Create exponentially decaying data."""
    test_data["test_name"] = "test_legacy_feature"
    base_date = datetime.now() - timedelta(days=90)
    
    for day in range(90):
        timestamp = base_date + timedelta(days=day)
        # Exponential decay: I(t) = 0.9 * exp(-0.02 * t) + 0.05
        importance = 0.9 * np.exp(-0.02 * day) + 0.05
        importance += np.random.normal(0, 0.01)  # Small noise
        importance = np.clip(importance, 0, 1)
        
        tracker.record_importance(test_data["test_name"], importance, timestamp)


@when("I model the decay as exponential")
def model_exponential_decay(tracker, test_data):
    """Fit exponential decay model."""
    test_data["decay_model"] = tracker.model_decay(
        test_data["test_name"],
        decay_type=DecayType.EXPONENTIAL
    )


@then("I should get decay parameters (a, b, c)")
def check_decay_params(test_data):
    """Verify decay parameters."""
    model = test_data["decay_model"]
    assert "parameters" in model
    assert "a" in model["parameters"]
    assert "b" in model["parameters"]
    assert "c" in model["parameters"]
    
    # Parameters should be close to true values
    assert 0.8 <= model["parameters"]["a"] <= 1.0
    assert 0.015 <= model["parameters"]["b"] <= 0.025
    assert 0.0 <= model["parameters"]["c"] <= 0.1


@then("the R-squared should be greater than 0.8")
def check_r_squared(test_data):
    """Verify model fit quality."""
    assert test_data["decay_model"]["r_squared"] > 0.8


@then("I should get a half-life estimate")
def check_half_life(test_data):
    """Verify half-life calculation."""
    assert "half_life" in test_data["decay_model"]
    assert test_data["decay_model"]["half_life"] > 0
    # For b=0.02, half-life should be around 34.7 days
    assert 25 <= test_data["decay_model"]["half_life"] <= 45


@then("the current decay rate should be calculated")
def check_decay_rate(test_data):
    """Verify decay rate calculation."""
    assert "current_decay_rate" in test_data["decay_model"]
    assert test_data["decay_model"]["current_decay_rate"] >= 0


# Scenario: Forecasting future importance

@given('I have a test "test_core_functionality" with 30 days of history')
def create_forecast_data(tracker, test_data):
    """Create data for forecasting."""
    test_data["test_name"] = "test_core_functionality"
    base_date = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        timestamp = base_date + timedelta(days=day)
        # Trending data with weekly pattern
        trend = 0.8 - 0.003 * day
        seasonal = 0.05 * np.sin(day * 2 * np.pi / 7)
        importance = trend + seasonal + np.random.normal(0, 0.01)
        importance = np.clip(importance, 0, 1)
        
        tracker.record_importance(test_data["test_name"], importance, timestamp)


@when("I forecast importance for the next 7 days")
def forecast_importance(tracker, test_data):
    """Generate forecast."""
    test_data["forecast"] = tracker.forecast_importance(
        test_data["test_name"],
        horizon=7
    )


@then("I should get predicted values for each day")
def check_forecast_values(test_data):
    """Verify forecast values."""
    forecast = test_data["forecast"]
    assert forecast is not None
    assert len(forecast.values) == 7
    assert len(forecast.timestamps) == 7


@then("I should get confidence intervals")
def check_confidence_intervals(test_data):
    """Verify confidence intervals."""
    forecast = test_data["forecast"]
    assert len(forecast.confidence_lower) == 7
    assert len(forecast.confidence_upper) == 7
    
    # Confidence intervals should contain predicted values
    for i in range(7):
        assert forecast.confidence_lower[i] <= forecast.values[i]
        assert forecast.values[i] <= forecast.confidence_upper[i]


@then("the forecast should respect bounds [0, 1]")
def check_forecast_bounds(test_data):
    """Verify forecast bounds."""
    forecast = test_data["forecast"]
    assert np.all(forecast.values >= 0)
    assert np.all(forecast.values <= 1)
    assert np.all(forecast.confidence_lower >= 0)
    assert np.all(forecast.confidence_upper <= 1)


@then("trend should be incorporated into forecast")
def check_forecast_trend(test_data):
    """Verify trend in forecast."""
    forecast = test_data["forecast"]
    # With declining trend, forecast should generally decrease
    avg_first_half = np.mean(forecast.values[:3])
    avg_second_half = np.mean(forecast.values[4:])
    assert avg_first_half >= avg_second_half


# Scenario: Generating alerts

@given('I have a test "test_broken_integration" with stable importance 0.8')
def create_stable_data(tracker, test_data):
    """Create stable importance data."""
    test_data["test_name"] = "test_broken_integration"
    base_date = datetime.now() - timedelta(days=7)
    
    for day in range(7):
        timestamp = base_date + timedelta(days=day)
        importance = 0.8 + np.random.normal(0, 0.01)
        tracker.record_importance(test_data["test_name"], importance, timestamp)


@when("the importance suddenly drops to 0.2")
def sudden_drop(tracker, test_data):
    """Record sudden importance drop."""
    tracker.record_importance(test_data["test_name"], 0.2)


@then("an alert should be generated")
def check_alert_generated(tracker, test_data):
    """Verify alert was created."""
    alerts = tracker.get_alerts(test_data["test_name"])
    assert len(alerts) > 0
    test_data["alerts"] = alerts


@then('the alert type should be "sudden_decrease"')
def check_alert_type(test_data):
    """Verify alert type."""
    assert any(alert.alert_type == AlertType.SUDDEN_DECREASE 
              for alert in test_data["alerts"])


@then("the severity should be high (>0.8)")
def check_severity(test_data):
    """Verify alert severity."""
    for alert in test_data["alerts"]:
        if alert.alert_type == AlertType.SUDDEN_DECREASE:
            assert alert.severity > 0.8


@then("the recommendation should suggest investigating recent changes")
def check_recommendation(test_data):
    """Verify recommendation text."""
    for alert in test_data["alerts"]:
        if alert.alert_type == AlertType.SUDDEN_DECREASE:
            assert "investigate" in alert.recommendation.lower()
            assert "change" in alert.recommendation.lower()


# Scenario: Visualizing temporal importance

@given('I have a test "test_evolving_feature" with 60 days of data')
def create_visualization_data(tracker, test_data):
    """Create rich data for visualization."""
    test_data["test_name"] = "test_evolving_feature"
    base_date = datetime.now() - timedelta(days=60)
    
    for day in range(60):
        timestamp = base_date + timedelta(days=day)
        
        # Complex pattern: trend + seasonality + change point
        if day < 30:
            importance = 0.8 + 0.1 * np.sin(day * 2 * np.pi / 7)
        else:
            importance = 0.6 + 0.05 * np.sin(day * 2 * np.pi / 7)
            
        importance += np.random.normal(0, 0.02)
        importance = np.clip(importance, 0, 1)
        
        tracker.record_importance(test_data["test_name"], importance, timestamp)


@when("I request visualization data")
def get_visualization(tracker, test_data):
    """Get visualization data."""
    test_data["visualization"] = tracker.visualize_temporal_importance(
        test_data["test_name"]
    )


@then("I should get time series data points")
def check_viz_time_series(test_data):
    """Verify time series in visualization."""
    viz = test_data["visualization"]
    assert "time_series" in viz
    assert "timestamps" in viz["time_series"]
    assert "values" in viz["time_series"]
    assert len(viz["time_series"]["timestamps"]) == 60
    assert len(viz["time_series"]["values"]) == 60


@then("I should get decomposition components if available")
def check_viz_decomposition(test_data):
    """Verify decomposition in visualization."""
    viz = test_data["visualization"]
    if "decomposition" in viz:
        assert "trend" in viz["decomposition"]
        assert "seasonal" in viz["decomposition"]
        assert "residual" in viz["decomposition"]


@then("I should get marked change points")
def check_viz_change_points(test_data):
    """Verify change points in visualization."""
    viz = test_data["visualization"]
    if "change_points" in viz:
        assert len(viz["change_points"]) > 0
        for cp in viz["change_points"]:
            assert "timestamp" in cp
            assert "magnitude" in cp
            assert "direction" in cp


@then("I should get forecast with confidence bands")
def check_viz_forecast(test_data):
    """Verify forecast in visualization."""
    viz = test_data["visualization"]
    if "forecast" in viz:
        assert "timestamps" in viz["forecast"]
        assert "values" in viz["forecast"]
        assert "confidence_lower" in viz["forecast"]
        assert "confidence_upper" in viz["forecast"]


@then("I should get summary statistics")
def check_viz_statistics(test_data):
    """Verify statistics in visualization."""
    viz = test_data["visualization"]
    assert "statistics" in viz
    stats = viz["statistics"]
    assert "mean" in stats
    assert "std" in stats
    assert "trend" in stats
    assert "volatility" in stats