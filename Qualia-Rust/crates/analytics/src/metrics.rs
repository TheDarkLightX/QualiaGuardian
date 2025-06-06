//! Advanced metrics and statistical analysis

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Normal, ContinuousCDF};
use statrs::statistics::{Statistics, OrderStatistics};
use std::collections::HashMap;

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction: positive, negative, or stable
    pub direction: TrendDirection,
    /// Slope of the trend line
    pub slope: f64,
    /// R-squared value (goodness of fit)
    pub r_squared: f64,
    /// Predicted next value
    pub next_prediction: f64,
    /// Confidence interval for prediction
    pub prediction_interval: (f64, f64),
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Statistical summary of a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Number of data points
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 25th percentile
    pub q1: f64,
    /// 75th percentile
    pub q3: f64,
    /// Coefficient of variation
    pub cv: f64,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Indices of detected anomalies
    pub anomaly_indices: Vec<usize>,
    /// Anomaly scores for each data point
    pub anomaly_scores: Vec<f64>,
    /// Threshold used for detection
    pub threshold: f64,
    /// Method used for detection
    pub method: AnomalyMethod,
}

/// Anomaly detection method
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnomalyMethod {
    ZScore,
    IQR,
    IsolationForest,
}

/// Calculate linear regression trend
pub fn calculate_trend(values: &[f64]) -> Result<TrendAnalysis> {
    if values.len() < 2 {
        anyhow::bail!("Need at least 2 data points for trend analysis");
    }
    
    let n = values.len() as f64;
    let x: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
    
    // Calculate means
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = values.iter().sum::<f64>() / n;
    
    // Calculate slope and intercept
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..values.len() {
        numerator += (x[i] - x_mean) * (values[i] - y_mean);
        denominator += (x[i] - x_mean).powi(2);
    }
    
    let slope = if denominator != 0.0 {
        numerator / denominator
    } else {
        0.0
    };
    
    let intercept = y_mean - slope * x_mean;
    
    // Calculate R-squared
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    
    for i in 0..values.len() {
        let y_pred = slope * x[i] + intercept;
        ss_tot += (values[i] - y_mean).powi(2);
        ss_res += (values[i] - y_pred).powi(2);
    }
    
    let r_squared = if ss_tot != 0.0 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };
    
    // Predict next value
    let next_x = values.len() as f64;
    let next_prediction = slope * next_x + intercept;
    
    // Calculate prediction interval
    let residual_std = (ss_res / (n - 2.0)).sqrt();
    let t_value = 1.96; // Approximate 95% confidence
    let prediction_error = t_value * residual_std * (1.0 + 1.0/n + (next_x - x_mean).powi(2) / denominator).sqrt();
    
    let prediction_interval = (
        next_prediction - prediction_error,
        next_prediction + prediction_error,
    );
    
    // Determine direction
    let direction = if slope.abs() < 0.001 {
        TrendDirection::Stable
    } else if slope > 0.0 {
        TrendDirection::Increasing
    } else {
        TrendDirection::Decreasing
    };
    
    Ok(TrendAnalysis {
        direction,
        slope,
        r_squared,
        next_prediction,
        prediction_interval,
    })
}

/// Calculate comprehensive metric summary
pub fn calculate_summary(values: &[f64]) -> Result<MetricSummary> {
    if values.is_empty() {
        anyhow::bail!("Cannot calculate summary for empty data");
    }
    
    let data = Array1::from_vec(values.to_vec());
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let count = values.len();
    let mean = data.clone().mean();
    let median = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    } else {
        sorted[count / 2]
    };
    
    let std_dev = data.std_dev();
    let min = *sorted.first().unwrap();
    let max = *sorted.last().unwrap();
    
    let q1 = sorted[count / 4];
    let q3 = sorted[3 * count / 4];
    
    let cv = if mean != 0.0 {
        std_dev / mean.abs()
    } else {
        0.0
    };
    
    Ok(MetricSummary {
        count,
        mean,
        median,
        std_dev,
        min,
        max,
        q1,
        q3,
        cv,
    })
}

/// Detect anomalies using Z-score method
pub fn detect_anomalies_zscore(values: &[f64], threshold: f64) -> Result<AnomalyResult> {
    if values.len() < 3 {
        anyhow::bail!("Need at least 3 data points for anomaly detection");
    }
    
    let data = Array1::from_vec(values.to_vec());
    let mean = data.clone().mean();
    let std_dev = data.std_dev();
    
    let mut anomaly_indices = Vec::new();
    let mut anomaly_scores = Vec::new();
    
    for (i, &value) in values.iter().enumerate() {
        let z_score = (value - mean).abs() / std_dev;
        anomaly_scores.push(z_score);
        
        if z_score > threshold {
            anomaly_indices.push(i);
        }
    }
    
    Ok(AnomalyResult {
        anomaly_indices,
        anomaly_scores,
        threshold,
        method: AnomalyMethod::ZScore,
    })
}

/// Detect anomalies using IQR method
pub fn detect_anomalies_iqr(values: &[f64], multiplier: f64) -> Result<AnomalyResult> {
    if values.len() < 4 {
        anyhow::bail!("Need at least 4 data points for IQR anomaly detection");
    }
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let q1 = sorted[values.len() / 4];
    let q3 = sorted[3 * values.len() / 4];
    let iqr = q3 - q1;
    
    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;
    
    let mut anomaly_indices = Vec::new();
    let mut anomaly_scores = Vec::new();
    
    for (i, &value) in values.iter().enumerate() {
        let score = if value < lower_bound {
            (lower_bound - value) / iqr
        } else if value > upper_bound {
            (value - upper_bound) / iqr
        } else {
            0.0
        };
        
        anomaly_scores.push(score);
        
        if score > 0.0 {
            anomaly_indices.push(i);
        }
    }
    
    Ok(AnomalyResult {
        anomaly_indices,
        anomaly_scores,
        threshold: multiplier,
        method: AnomalyMethod::IQR,
    })
}

/// Calculate correlation matrix for multiple metrics
pub fn calculate_correlation_matrix(metrics: &HashMap<String, Vec<f64>>) -> Result<Array2<f64>> {
    let metric_names: Vec<_> = metrics.keys().cloned().collect();
    let n_metrics = metric_names.len();
    
    if n_metrics == 0 {
        anyhow::bail!("No metrics provided");
    }
    
    let data_len = metrics.values().next().unwrap().len();
    for (name, values) in metrics {
        if values.len() != data_len {
            anyhow::bail!("Metric {} has different length", name);
        }
    }
    
    let mut correlation_matrix = Array2::zeros((n_metrics, n_metrics));
    
    for i in 0..n_metrics {
        for j in 0..n_metrics {
            if i == j {
                correlation_matrix[[i, j]] = 1.0;
            } else {
                let x = &metrics[&metric_names[i]];
                let y = &metrics[&metric_names[j]];
                let corr = calculate_correlation(x, y)?;
                correlation_matrix[[i, j]] = corr;
                correlation_matrix[[j, i]] = corr;
            }
        }
    }
    
    Ok(correlation_matrix)
}

/// Calculate Pearson correlation coefficient
pub fn calculate_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() || x.is_empty() {
        anyhow::bail!("Arrays must have same non-zero length");
    }
    
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;
    
    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        numerator += x_diff * y_diff;
        x_var += x_diff * x_diff;
        y_var += y_diff * y_diff;
    }
    
    if x_var == 0.0 || y_var == 0.0 {
        return Ok(0.0);
    }
    
    Ok(numerator / (x_var * y_var).sqrt())
}

/// Forecast future values using simple exponential smoothing
pub fn exponential_smoothing(values: &[f64], alpha: f64, periods: usize) -> Result<Vec<f64>> {
    if values.is_empty() {
        anyhow::bail!("Cannot forecast from empty data");
    }
    
    if alpha <= 0.0 || alpha >= 1.0 {
        anyhow::bail!("Alpha must be between 0 and 1");
    }
    
    let mut forecasts = Vec::new();
    let mut level = values[0];
    
    // Smooth historical data
    for &value in values.iter().skip(1) {
        level = alpha * value + (1.0 - alpha) * level;
    }
    
    // Forecast future periods
    for _ in 0..periods {
        forecasts.push(level);
    }
    
    Ok(forecasts)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trend_analysis() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = calculate_trend(&values).unwrap();
        
        assert_eq!(trend.direction, TrendDirection::Increasing);
        assert!((trend.slope - 1.0).abs() < 0.001);
        assert!((trend.r_squared - 1.0).abs() < 0.001);
        assert!((trend.next_prediction - 6.0).abs() < 0.001);
    }
    
    #[test]
    fn test_anomaly_detection() {
        let values = vec![1.0, 2.0, 3.0, 2.0, 100.0, 2.5, 3.0];
        
        let result = detect_anomalies_zscore(&values, 2.0).unwrap();
        assert!(result.anomaly_indices.contains(&4));
        
        let result_iqr = detect_anomalies_iqr(&values, 1.5).unwrap();
        assert!(result_iqr.anomaly_indices.contains(&4));
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = calculate_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 0.001); // Perfect positive correlation
    }
}