//! Trend analysis and forecasting utilities

use crate::metrics::{TrendAnalysis, TrendDirection, calculate_trend, exponential_smoothing};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quality trend analyzer
#[derive(Debug)]
pub struct QualityTrendAnalyzer {
    /// Historical quality scores
    history: Vec<QualitySnapshot>,
    /// Smoothing parameter for forecasting
    smoothing_alpha: f64,
}

/// A snapshot of quality metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySnapshot {
    /// Timestamp
    pub timestamp: i64,
    /// Overall quality score
    pub overall_score: f64,
    /// Component scores
    pub component_scores: HashMap<String, f64>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Quality forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityForecast {
    /// Predicted future quality scores
    pub predictions: Vec<f64>,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Component trends
    pub component_trends: HashMap<String, TrendAnalysis>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Risk assessment based on trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub level: RiskLevel,
    /// Risk factors identified
    pub factors: Vec<RiskFactor>,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Risk level
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Individual risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Component or metric name
    pub component: String,
    /// Type of risk
    pub risk_type: RiskType,
    /// Severity (0-1)
    pub severity: f64,
    /// Description
    pub description: String,
}

/// Type of risk
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RiskType {
    DecliningQuality,
    HighVolatility,
    BelowThreshold,
    NegativeTrend,
}

impl QualityTrendAnalyzer {
    /// Create a new trend analyzer
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            smoothing_alpha: 0.3,
        }
    }
    
    /// Set smoothing parameter
    pub fn with_smoothing(mut self, alpha: f64) -> Self {
        self.smoothing_alpha = alpha;
        self
    }
    
    /// Add a quality snapshot
    pub fn add_snapshot(&mut self, snapshot: QualitySnapshot) {
        self.history.push(snapshot);
        // Keep sorted by timestamp
        self.history.sort_by_key(|s| s.timestamp);
    }
    
    /// Analyze trends and forecast future quality
    pub fn analyze_and_forecast(&self, periods: usize) -> Result<QualityForecast> {
        if self.history.len() < 3 {
            anyhow::bail!("Need at least 3 snapshots for trend analysis");
        }
        
        // Extract overall scores
        let overall_scores: Vec<f64> = self.history.iter()
            .map(|s| s.overall_score)
            .collect();
        
        // Analyze overall trend
        let overall_trend = calculate_trend(&overall_scores)?;
        
        // Forecast future values
        let predictions = exponential_smoothing(
            &overall_scores,
            self.smoothing_alpha,
            periods,
        )?;
        
        // Calculate confidence intervals
        let confidence_intervals = self.calculate_confidence_intervals(&predictions, &overall_trend);
        
        // Analyze component trends
        let component_trends = self.analyze_component_trends()?;
        
        // Assess risks
        let risk_assessment = self.assess_risks(&overall_trend, &component_trends);
        
        Ok(QualityForecast {
            predictions,
            confidence_intervals,
            component_trends,
            risk_assessment,
        })
    }
    
    /// Calculate confidence intervals for predictions
    fn calculate_confidence_intervals(
        &self,
        predictions: &[f64],
        trend: &TrendAnalysis,
    ) -> Vec<(f64, f64)> {
        let historical_std = self.calculate_historical_volatility();
        let base_interval = 1.96 * historical_std; // 95% confidence
        
        predictions.iter().enumerate()
            .map(|(i, &pred)| {
                // Widen interval for further predictions
                let interval = base_interval * (1.0 + 0.1 * i as f64);
                (pred - interval, pred + interval)
            })
            .collect()
    }
    
    /// Calculate historical volatility
    fn calculate_historical_volatility(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        
        let scores: Vec<f64> = self.history.iter()
            .map(|s| s.overall_score)
            .collect();
        
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / scores.len() as f64;
        
        variance.sqrt()
    }
    
    /// Analyze trends for each component
    fn analyze_component_trends(&self) -> Result<HashMap<String, TrendAnalysis>> {
        let mut component_trends = HashMap::new();
        
        // Get all component names
        let component_names: Vec<String> = self.history.first()
            .map(|s| s.component_scores.keys().cloned().collect())
            .unwrap_or_default();
        
        for component in component_names {
            let scores: Vec<f64> = self.history.iter()
                .filter_map(|s| s.component_scores.get(&component).copied())
                .collect();
            
            if scores.len() >= 3 {
                if let Ok(trend) = calculate_trend(&scores) {
                    component_trends.insert(component, trend);
                }
            }
        }
        
        Ok(component_trends)
    }
    
    /// Assess risks based on trends
    fn assess_risks(
        &self,
        overall_trend: &TrendAnalysis,
        component_trends: &HashMap<String, TrendAnalysis>,
    ) -> RiskAssessment {
        let mut factors = Vec::new();
        let mut max_severity: f64 = 0.0;
        
        // Check overall trend
        if overall_trend.direction == TrendDirection::Decreasing && overall_trend.slope < -0.01 {
            let severity = (-overall_trend.slope * 10.0).min(1.0);
            factors.push(RiskFactor {
                component: "Overall Quality".to_string(),
                risk_type: RiskType::DecliningQuality,
                severity,
                description: format!("Quality declining at {:.2}% per period", -overall_trend.slope * 100.0),
            });
            max_severity = max_severity.max(severity);
        }
        
        // Check if below threshold
        if let Some(latest) = self.history.last() {
            if latest.overall_score < 0.6 {
                let severity = (0.6 - latest.overall_score) * 2.0;
                factors.push(RiskFactor {
                    component: "Overall Quality".to_string(),
                    risk_type: RiskType::BelowThreshold,
                    severity,
                    description: format!("Quality score {:.2} is below acceptable threshold", latest.overall_score),
                });
                max_severity = max_severity.max(severity);
            }
        }
        
        // Check component trends
        for (component, trend) in component_trends {
            if trend.direction == TrendDirection::Decreasing && trend.slope < -0.02 {
                let severity = (-trend.slope * 5.0).min(1.0);
                factors.push(RiskFactor {
                    component: component.clone(),
                    risk_type: RiskType::NegativeTrend,
                    severity,
                    description: format!("{} declining rapidly", component),
                });
                max_severity = max_severity.max(severity);
            }
        }
        
        // Check volatility
        let volatility = self.calculate_historical_volatility();
        if volatility > 0.15 {
            let severity = (volatility * 3.0).min(1.0);
            factors.push(RiskFactor {
                component: "Overall Quality".to_string(),
                risk_type: RiskType::HighVolatility,
                severity,
                description: format!("High quality volatility: {:.2}", volatility),
            });
            max_severity = max_severity.max(severity);
        }
        
        // Determine risk level
        let level = if max_severity > 0.8 {
            RiskLevel::Critical
        } else if max_severity > 0.6 {
            RiskLevel::High
        } else if max_severity > 0.3 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&factors);
        
        RiskAssessment {
            level,
            factors,
            recommendations,
        }
    }
    
    /// Generate recommendations based on risk factors
    fn generate_recommendations(&self, factors: &[RiskFactor]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        for factor in factors {
            match factor.risk_type {
                RiskType::DecliningQuality => {
                    recommendations.push(format!(
                        "Investigate root cause of declining {} quality",
                        factor.component
                    ));
                    recommendations.push("Run comprehensive test suite analysis".to_string());
                }
                RiskType::HighVolatility => {
                    recommendations.push("Stabilize test environment to reduce flakiness".to_string());
                    recommendations.push("Review recent changes for instability sources".to_string());
                }
                RiskType::BelowThreshold => {
                    recommendations.push(format!(
                        "Immediate action required: {} is below minimum threshold",
                        factor.component
                    ));
                    recommendations.push("Consider evolutionary test improvement".to_string());
                }
                RiskType::NegativeTrend => {
                    recommendations.push(format!(
                        "Address negative trend in {} before it impacts overall quality",
                        factor.component
                    ));
                }
            }
        }
        
        // Remove duplicates while preserving order
        let mut seen = std::collections::HashSet::new();
        recommendations.retain(|r| seen.insert(r.clone()));
        
        recommendations
    }
}

impl Default for QualityTrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quality_trend_analysis() {
        let mut analyzer = QualityTrendAnalyzer::new();
        
        // Add historical snapshots with declining quality trend
        for i in 0..5 {
            let mut component_scores = HashMap::new();
            component_scores.insert("coverage".to_string(), 0.8 - i as f64 * 0.08);
            component_scores.insert("mutation".to_string(), 0.7 - i as f64 * 0.05);
            
            analyzer.add_snapshot(QualitySnapshot {
                timestamp: i as i64,
                overall_score: 0.75 - i as f64 * 0.05,
                component_scores,
                metadata: HashMap::new(),
            });
        }
        
        let forecast = analyzer.analyze_and_forecast(3).unwrap();
        
        assert_eq!(forecast.predictions.len(), 3);
        assert_eq!(forecast.confidence_intervals.len(), 3);
        assert!(forecast.component_trends.contains_key("coverage"));
        assert!(forecast.component_trends.contains_key("mutation"));
        
        // Should detect declining quality
        assert!(!forecast.risk_assessment.factors.is_empty());
        assert!(matches!(
            forecast.risk_assessment.level,
            RiskLevel::Medium | RiskLevel::High
        ));
    }
}