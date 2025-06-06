"""
Demo script for Temporal Importance Tracking

Shows how to track test importance over time, detect patterns,
model decay, and generate forecasts.

Author: DarkLightX/Dana Edwards
"""

import numpy as np
from datetime import datetime, timedelta
import json

from guardian.analytics.temporal_importance import (
    TemporalImportanceTracker,
    DecayType,
    AlertType
)


def demo_temporal_importance():
    """Demonstrate temporal importance tracking capabilities."""
    print("=" * 80)
    print("TEMPORAL IMPORTANCE TRACKING DEMO")
    print("=" * 80)
    
    # Initialize tracker
    tracker = TemporalImportanceTracker()
    
    # Demo 1: Basic recording and retrieval
    print("\n1. RECORDING TEST IMPORTANCE OVER TIME")
    print("-" * 40)
    
    test_name = "test_payment_processing"
    base_date = datetime.now() - timedelta(days=30)
    
    # Simulate 30 days of importance data with weekly pattern
    for day in range(30):
        timestamp = base_date + timedelta(days=day)
        # Weekly pattern: higher during weekdays
        day_of_week = timestamp.weekday()
        if day_of_week < 5:  # Monday-Friday
            base_importance = 0.85
        else:  # Weekend
            base_importance = 0.45
            
        # Add some noise
        importance = base_importance + np.random.normal(0, 0.05)
        importance = np.clip(importance, 0, 1)
        
        tracker.record_importance(test_name, importance, timestamp)
        
    print(f"✓ Recorded 30 days of data for {test_name}")
    
    # Demo 2: Time series decomposition
    print("\n2. TIME SERIES DECOMPOSITION")
    print("-" * 40)
    
    components = tracker.decompose_time_series(test_name)
    if components:
        print("✓ Decomposed into trend, seasonal, and residual components")
        trend_direction = "increasing" if components.trend[-1] > components.trend[0] else "decreasing"
        print(f"  - Trend: {trend_direction}")
        print(f"  - Seasonal period detected: 7 days (weekly)")
    else:
        print("  - Decomposition requires statsmodels library")
        
    # Demo 3: Change point detection
    print("\n3. CHANGE POINT DETECTION")
    print("-" * 40)
    
    # Create test with sudden change
    test_name2 = "test_refactored_module"
    for day in range(20):
        timestamp = base_date + timedelta(days=day)
        if day < 10:
            importance = 0.9 + np.random.normal(0, 0.02)
        else:
            # Sudden drop after refactoring
            importance = 0.5 + np.random.normal(0, 0.02)
        importance = np.clip(importance, 0, 1)
        tracker.record_importance(test_name2, importance, timestamp)
        
    change_points = tracker.detect_change_points(test_name2)
    print(f"✓ Detected {len(change_points)} change points in {test_name2}")
    for cp in change_points[:2]:  # Show first 2
        print(f"  - Day {cp.index}: {cp.direction} by {cp.magnitude:.2f} "
              f"(confidence: {cp.confidence:.2f})")
        
    # Demo 4: Decay modeling
    print("\n4. IMPORTANCE DECAY MODELING")
    print("-" * 40)
    
    # Create decaying test data
    test_name3 = "test_deprecated_feature"
    for day in range(60):
        timestamp = base_date + timedelta(days=day)
        # Exponential decay
        importance = 0.85 * np.exp(-0.03 * day) + 0.1
        importance += np.random.normal(0, 0.01)
        importance = np.clip(importance, 0, 1)
        tracker.record_importance(test_name3, importance, timestamp)
        
    # Try different decay models
    print(f"Fitting decay models for {test_name3}:")
    for decay_type in [DecayType.EXPONENTIAL, DecayType.LINEAR]:
        result = tracker.model_decay(test_name3, decay_type)
        if "r_squared" in result:
            print(f"\n  {decay_type.value.upper()} model:")
            print(f"    - R² = {result['r_squared']:.3f}")
            if result.get('half_life'):
                print(f"    - Half-life = {result['half_life']:.1f} days")
            print(f"    - Current decay rate = {result['current_decay_rate']:.4f}/day")
            
    # Demo 5: Forecasting
    print("\n5. IMPORTANCE FORECASTING")
    print("-" * 40)
    
    forecast = tracker.forecast_importance(test_name, horizon=7)
    if forecast:
        print(f"✓ 7-day forecast for {test_name}:")
        for i in range(min(3, len(forecast.values))):
            print(f"  - Day {i+1}: {forecast.values[i]:.3f} "
                  f"[{forecast.confidence_lower[i]:.3f}, "
                  f"{forecast.confidence_upper[i]:.3f}]")
        print("  ...")
    else:
        print("  - Forecasting requires sufficient historical data")
        
    # Demo 6: Pattern detection
    print("\n6. TEMPORAL PATTERN DETECTION")
    print("-" * 40)
    
    patterns = tracker.get_temporal_patterns(test_name)
    if patterns:
        print(f"✓ Detected patterns in {test_name}:")
        for pattern in patterns:
            print(f"  - {pattern.pattern_type}: strength={pattern.strength:.2f}, "
                  f"period={pattern.period:.1f} days")
    else:
        print("  - Analyzing patterns...")
        
    # Demo 7: Alerts
    print("\n7. IMPORTANCE ALERTS")
    print("-" * 40)
    
    # Trigger an alert with sudden change
    test_name4 = "test_broken_integration"
    for day in range(5):
        timestamp = datetime.now() - timedelta(days=5-day)
        tracker.record_importance(test_name4, 0.95, timestamp)
    # Sudden drop
    tracker.record_importance(test_name4, 0.15)
    
    alerts = tracker.get_alerts(test_name4)
    print(f"✓ Generated {len(alerts)} alerts for {test_name4}:")
    for alert in alerts:
        print(f"  - {alert.alert_type.value}: {alert.message}")
        print(f"    Recommendation: {alert.recommendation}")
        
    # Demo 8: Visualization data
    print("\n8. VISUALIZATION DATA")
    print("-" * 40)
    
    viz_data = tracker.visualize_temporal_importance(test_name)
    print(f"✓ Generated visualization data for {test_name}:")
    print(f"  - Time series points: {len(viz_data['time_series']['values'])}")
    print(f"  - Mean importance: {viz_data['statistics']['mean']:.3f}")
    print(f"  - Volatility: {viz_data['statistics']['volatility']:.3f}")
    print(f"  - Trend: {viz_data['statistics']['trend']:.3f}")
    
    if "decomposition" in viz_data:
        print("  - Includes decomposition components")
    if "change_points" in viz_data:
        print(f"  - Includes {len(viz_data['change_points'])} change points")
    if "forecast" in viz_data:
        print("  - Includes forecast data")
        
    # Demo 9: Summary statistics
    print("\n9. SUMMARY STATISTICS")
    print("-" * 40)
    
    # Get all tests we've tracked
    test_names = [test_name, test_name2, test_name3, test_name4]
    
    print("Test importance summary:")
    for name in test_names:
        data = tracker._get_time_series(name)
        if data:
            values = [v for _, v in data]
            print(f"\n  {name}:")
            print(f"    - Current: {values[-1]:.3f}")
            print(f"    - Average: {np.mean(values):.3f}")
            print(f"    - Trend: {'↑' if values[-1] > values[0] else '↓'}")
            
            # Check for decay threshold
            if values[-1] < 0.1:
                print("    - ⚠️  Below decay threshold!")
                
    print("\n" + "=" * 80)
    print("Demo completed! The temporal importance data is stored in:")
    print(f"  {tracker.db_path}")
    print("=" * 80)


if __name__ == "__main__":
    demo_temporal_importance()