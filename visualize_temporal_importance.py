"""
Visualization example for Temporal Importance data

Shows how to use the visualization data to create charts.
This example generates ASCII charts for terminal display.

Author: DarkLightX/Dana Edwards
"""

import numpy as np
from datetime import datetime, timedelta
from guardian.analytics.temporal_importance import TemporalImportanceTracker


def create_ascii_chart(values, width=60, height=10, title=""):
    """Create a simple ASCII line chart."""
    if not values:
        return "No data"
        
    # Normalize values to chart height
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1
    
    # Create chart
    chart = []
    
    # Title
    if title:
        chart.append(title.center(width + 10))
        chart.append("")
        
    # Y-axis labels and chart
    for h in range(height, -1, -1):
        threshold = min_val + (h / height) * range_val
        label = f"{threshold:6.3f} |"
        
        line = ""
        for i, val in enumerate(values):
            x_pos = int(i * width / len(values))
            if h == 0:
                line += "-"
            elif abs(val - threshold) < range_val / (2 * height):
                line += "*"
            else:
                line += " "
                
        chart.append(label + line)
        
    # X-axis
    chart.append("        " + "-" * width)
    chart.append(f"        0" + " " * (width - 10) + f"{len(values)} days")
    
    return "\n".join(chart)


def visualize_test_importance(test_name: str, tracker: TemporalImportanceTracker):
    """Visualize temporal importance for a specific test."""
    print(f"\nTEMPORAL IMPORTANCE VISUALIZATION: {test_name}")
    print("=" * 70)
    
    # Get visualization data
    viz_data = tracker.visualize_temporal_importance(test_name)
    
    if "error" in viz_data:
        print(f"Error: {viz_data['error']}")
        return
        
    # 1. Time series chart
    values = viz_data["time_series"]["values"]
    print("\n1. Importance Over Time")
    print("-" * 70)
    print(create_ascii_chart(values, title="Test Importance"))
    
    # 2. Statistics
    stats = viz_data["statistics"]
    print("\n2. Summary Statistics")
    print("-" * 70)
    print(f"  Mean:       {stats['mean']:.3f}")
    print(f"  Std Dev:    {stats['std']:.3f}")
    print(f"  Trend:      {stats['trend']:.3f} ({'↑' if stats['trend'] > 0 else '↓' if stats['trend'] < 0 else '→'})")
    print(f"  Volatility: {stats['volatility']:.3f}")
    
    # 3. Change points
    if "change_points" in viz_data and viz_data["change_points"]:
        print("\n3. Detected Change Points")
        print("-" * 70)
        for i, cp in enumerate(viz_data["change_points"][:5]):
            print(f"  {i+1}. {cp['timestamp'][:10]}: {cp['direction']} by {cp['magnitude']:.3f} "
                  f"(confidence: {cp['confidence']:.2f})")
                  
    # 4. Decomposition (if available)
    if "decomposition" in viz_data:
        print("\n4. Time Series Components")
        print("-" * 70)
        
        # Trend component
        trend = viz_data["decomposition"]["trend"]
        print("\nTrend Component:")
        print(create_ascii_chart(trend, height=5))
        
        # Seasonal component (show one period)
        seasonal = viz_data["decomposition"]["seasonal"][:7]
        print("\nSeasonal Component (first week):")
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, (day, val) in enumerate(zip(weekdays, seasonal)):
            bar_length = int((val + 0.1) * 20) if val > -0.1 else 0
            print(f"  {day}: {'█' * bar_length} {val:.3f}")
            
    # 5. Forecast
    if "forecast" in viz_data:
        print("\n5. 7-Day Forecast")
        print("-" * 70)
        forecast = viz_data["forecast"]
        
        # Combine historical and forecast
        combined_values = values[-14:] + forecast["values"]
        print(create_ascii_chart(combined_values, 
                               title="Historical (last 14 days) + Forecast (7 days)"))
        
        # Forecast details
        print("\nForecast Values:")
        for i in range(min(7, len(forecast["values"]))):
            print(f"  Day +{i+1}: {forecast['values'][i]:.3f} "
                  f"[{forecast['confidence_lower'][i]:.3f}, "
                  f"{forecast['confidence_upper'][i]:.3f}]")
                  
    # 6. Patterns
    patterns = tracker.get_temporal_patterns(test_name)
    if patterns:
        print("\n6. Temporal Patterns")
        print("-" * 70)
        for pattern in patterns:
            print(f"  - {pattern.pattern_type.capitalize()} pattern: "
                  f"strength={pattern.strength:.2f}, period={pattern.period:.1f} days")
                  
    # 7. Recent alerts
    alerts = tracker.get_alerts(test_name, limit=5)
    if alerts:
        print("\n7. Recent Alerts")
        print("-" * 70)
        for alert in alerts:
            severity_bar = "█" * int(alert.severity * 5)
            print(f"  [{alert.timestamp.strftime('%Y-%m-%d')}] "
                  f"{alert.alert_type.value} {severity_bar} ({alert.severity:.2f})")
            print(f"    → {alert.recommendation}")
            
    print("\n" + "=" * 70)


def main():
    """Run visualization demo."""
    # Use existing tracker with data
    tracker = TemporalImportanceTracker()
    
    # Create some sample data if needed
    test_name = "test_example_visualization"
    base_date = datetime.now() - timedelta(days=45)
    
    print("Creating sample data...")
    for day in range(45):
        timestamp = base_date + timedelta(days=day)
        
        # Create interesting pattern
        trend = 0.8 - 0.005 * day  # Declining trend
        weekly = 0.15 * np.sin(day * 2 * np.pi / 7)  # Weekly cycle
        noise = np.random.normal(0, 0.02)
        
        # Add a change point at day 25
        if day >= 25:
            trend -= 0.2
            
        importance = np.clip(trend + weekly + noise, 0, 1)
        tracker.record_importance(test_name, importance, timestamp)
        
    # Visualize the data
    visualize_test_importance(test_name, tracker)
    
    # Also visualize any existing test data
    print("\n\nChecking for other tests with data...")
    test_names = ["test_payment_processing", "test_refactored_module", 
                  "test_deprecated_feature"]
    
    for name in test_names:
        data = tracker._get_time_series(name)
        if data and len(data) > 10:
            print(f"\nFound data for {name} ({len(data)} points)")
            visualize_test_importance(name, tracker)
            break


if __name__ == "__main__":
    main()