"""
Meta-Metrics: Metrics about the Metrics System

Tracks the quality of QualiaGuardian itself, creating a recursive quality assessment.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MetaMetricType(Enum):
    """Types of meta-metrics"""
    SELF_ANALYSIS_SCORE = "self_analysis_score"
    QUALITY_TOOL_QUALITY = "quality_tool_quality"
    SELF_IMPROVEMENT_CAPABILITY = "self_improvement_capability"
    METRIC_ACCURACY = "metric_accuracy"
    FEEDBACK_LOOP_EFFECTIVENESS = "feedback_loop_effectiveness"
    AUTOMATION_LEVEL = "automation_level"


@dataclass
class MetaMetric:
    """A meta-metric about the quality system itself"""
    metric_type: MetaMetricType
    value: float
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class MetaMetricsTracker:
    """
    Tracks meta-metrics about QualiaGuardian itself.
    
    This creates a recursive quality assessment where we measure:
    - How good is our quality measurement?
    - How effective is our self-improvement?
    - How accurate are our metrics?
    """
    
    def __init__(self):
        self.metrics: List[MetaMetric] = []
    
    def record_metric(
        self,
        metric_type: MetaMetricType,
        value: float,
        description: str,
        context: Dict[str, Any] = None,
    ) -> None:
        """Record a meta-metric."""
        self.metrics.append(MetaMetric(
            metric_type=metric_type,
            value=value,
            description=description,
            context=context or {},
        ))
    
    def calculate_meta_quality_score(self) -> float:
        """
        Calculate overall meta-quality score.
        
        This is a score about how good our quality measurement system is.
        """
        if not self.metrics:
            return 0.0
        
        # Weighted average of recent metrics
        recent_metrics = [m for m in self.metrics if m.metric_type in [
            MetaMetricType.SELF_ANALYSIS_SCORE,
            MetaMetricType.QUALITY_TOOL_QUALITY,
            MetaMetricType.SELF_IMPROVEMENT_CAPABILITY,
        ]]
        
        if not recent_metrics:
            return 0.0
        
        # Get most recent value for each type
        latest_values = {}
        for metric in reversed(recent_metrics):
            if metric.metric_type not in latest_values:
                latest_values[metric.metric_type] = metric.value
        
        # Average the latest values
        if latest_values:
            return sum(latest_values.values()) / len(latest_values)
        
        return 0.0
    
    def get_improvement_trend(self) -> Dict[str, Any]:
        """Get trend of meta-metrics over time."""
        if len(self.metrics) < 2:
            return {"trend": "insufficient_data"}
        
        # Group by type and calculate trends
        trends = {}
        for metric_type in MetaMetricType:
            type_metrics = [m for m in self.metrics if m.metric_type == metric_type]
            if len(type_metrics) >= 2:
                first = type_metrics[0].value
                last = type_metrics[-1].value
                change = last - first
                trends[metric_type.value] = {
                    "first": first,
                    "last": last,
                    "change": change,
                    "improving": change > 0,
                }
        
        return trends
