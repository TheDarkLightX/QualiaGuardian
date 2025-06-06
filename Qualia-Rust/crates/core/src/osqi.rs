//! OSQI: Overall Software Quality Index v1.0

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use std::fs;
use crate::{
    Result, QualityScore,
    config::OSQIWeights,
    traits::QualityMetric,
};

/// Raw inputs for OSQI calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OSQIInput {
    /// bE-TES score (already normalized 0-1)
    pub betes_score: f64,
    /// Raw code health sub-metrics
    pub raw_code_health_sub_metrics: HashMap<String, f64>,
    /// Weighted vulnerability density
    pub raw_weighted_vulnerability_density: f64,
    /// Algebraic connectivity (0-1, higher is better)
    pub raw_algebraic_connectivity: Option<f64>,
    /// Wasserstein distance for risk/robustness
    pub raw_wasserstein_distance: Option<f64>,
}

/// Normalized OSQI pillars
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OSQINormalizedPillars {
    pub betes_score: f64,
    pub code_health_score_c_hs: f64,
    pub security_score_sec_s: f64,
    pub architecture_score_arch_s: f64,
    pub risk_robustness_score: f64,
}

/// OSQI calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OSQIResult {
    pub raw_pillars_input: Option<OSQIInput>,
    pub normalized_pillars: Option<OSQINormalizedPillars>,
    pub applied_weights: Option<OSQIWeights>,
    pub osqi_score: f64,
    pub calculation_time_s: f64,
    pub insights: Vec<String>,
}

/// CHS threshold configuration
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct CHSThresholds {
    #[serde(flatten)]
    languages: HashMap<String, LanguageThresholds>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct LanguageThresholds {
    #[serde(flatten)]
    metrics: HashMap<String, MetricThreshold>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct MetricThreshold {
    ideal_max: Option<f64>,
    acceptable_max: Option<f64>,
    poor_min: Option<f64>,
    higher_is_better: Option<bool>,
}

/// OSQI calculator
#[derive(Debug)]
pub struct OSQICalculator {
    weights: OSQIWeights,
    chs_thresholds_path: String,
    chs_thresholds: HashMap<String, HashMap<String, MetricThreshold>>,
    wasserstein_90th_percentile: f64,
}

impl OSQICalculator {
    const DEFAULT_WASSERSTEIN_90TH_PERCENTILE: f64 = 1.0;
    
    /// Create a new OSQI calculator
    pub fn new(
        weights: OSQIWeights,
        chs_thresholds_path: &str,
        wasserstein_90th_percentile: Option<f64>,
    ) -> Self {
        let mut calculator = Self {
            weights,
            chs_thresholds_path: chs_thresholds_path.to_string(),
            chs_thresholds: HashMap::new(),
            wasserstein_90th_percentile: wasserstein_90th_percentile
                .unwrap_or(Self::DEFAULT_WASSERSTEIN_90TH_PERCENTILE),
        };
        
        // Load CHS thresholds
        calculator.load_chs_thresholds();
        
        calculator
    }
    
    /// Load CHS thresholds from YAML file
    fn load_chs_thresholds(&mut self) {
        match fs::read_to_string(&self.chs_thresholds_path) {
            Ok(content) => {
                match serde_yaml::from_str::<HashMap<String, HashMap<String, MetricThreshold>>>(&content) {
                    Ok(thresholds) => {
                        self.chs_thresholds = thresholds;
                    }
                    Err(e) => {
                        eprintln!("Error parsing CHS thresholds YAML: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading CHS thresholds file: {}", e);
            }
        }
    }
    
    /// Normalize a CHS sub-metric
    fn normalize_chs_sub_metric(
        &self,
        metric_name: &str,
        raw_value: f64,
        language: &str,
    ) -> f64 {
        let lang_thresholds = self.chs_thresholds.get(&language.to_lowercase());
        if let Some(lang_thresh) = lang_thresholds {
            if let Some(metric_config) = lang_thresh.get(metric_name) {
                let higher_is_better = metric_config.higher_is_better.unwrap_or(false);
                
                if higher_is_better {
                    // Higher values are better
                    if let (Some(poor_max), Some(ideal_min)) = 
                        (metric_config.poor_min, metric_config.ideal_max) {
                        if raw_value >= ideal_min {
                            return 1.0;
                        } else if raw_value <= poor_max {
                            return 0.0;
                        } else {
                            return (raw_value - poor_max) / (ideal_min - poor_max);
                        }
                    }
                } else {
                    // Lower values are better (default)
                    if let (Some(ideal_max), Some(poor_min)) = 
                        (metric_config.ideal_max, metric_config.poor_min) {
                        if raw_value <= ideal_max {
                            return 1.0;
                        } else if raw_value >= poor_min {
                            return 0.0;
                        } else {
                            return 1.0 - (raw_value - ideal_max) / (poor_min - ideal_max);
                        }
                    }
                }
            }
        }
        
        // Fallback: use reasonable defaults based on metric name
        match metric_name {
            "maintainability_index" => {
                // MI is 0-100, higher is better
                (raw_value / 100.0).clamp(0.0, 1.0)
            }
            "cyclomatic_complexity" => {
                // CC lower is better, use exponential decay
                (-raw_value / 10.0).exp().clamp(0.0, 1.0)
            }
            "shannon_entropy" => {
                // Entropy ~3-5 is good, normalize to that range
                ((raw_value - 2.0) / 3.0).clamp(0.0, 1.0)
            }
            _ => {
                // Generic fallback: assume lower is better with 10 as scale
                (1.0 - raw_value / 10.0).clamp(0.0, 1.0)
            }
        }
    }
    
    /// Calculate Code Health Score
    fn calculate_code_health_score(
        &self,
        sub_metrics: &HashMap<String, f64>,
        language: &str,
    ) -> f64 {
        if sub_metrics.is_empty() {
            return 0.0;
        }
        
        let mut total_score = 0.0;
        let mut count = 0;
        
        for (metric_name, &raw_value) in sub_metrics {
            let normalized = self.normalize_chs_sub_metric(metric_name, raw_value, language);
            total_score += normalized;
            count += 1;
        }
        
        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate Security Score
    fn calculate_security_score(&self, vuln_density: f64) -> f64 {
        // sec_s = e^(-3 * vuln_density)
        (-3.0 * vuln_density).exp().clamp(0.0, 1.0)
    }
    
    /// Calculate Architecture Score
    fn calculate_architecture_score(&self, connectivity: Option<f64>) -> f64 {
        connectivity.unwrap_or(0.5).clamp(0.0, 1.0)
    }
    
    /// Calculate Risk/Robustness Score
    fn calculate_risk_robustness_score(&self, wasserstein_distance: Option<f64>) -> f64 {
        if let Some(distance) = wasserstein_distance {
            // R_r = e^(-W/W_90%)
            (-distance / self.wasserstein_90th_percentile).exp().clamp(0.0, 1.0)
        } else {
            0.5 // Default neutral score
        }
    }
    
    /// Calculate the final OSQI score with language parameter
    pub fn calculate_with_language(&self, input: &OSQIInput, language: &str) -> Result<(QualityScore, OSQIResult)> {
        let start_time = Instant::now();
        
        // Calculate normalized pillars
        let code_health_score = self.calculate_code_health_score(
            &input.raw_code_health_sub_metrics,
            language,
        );
        let security_score = self.calculate_security_score(
            input.raw_weighted_vulnerability_density,
        );
        let architecture_score = self.calculate_architecture_score(
            input.raw_algebraic_connectivity,
        );
        let risk_robustness_score = self.calculate_risk_robustness_score(
            input.raw_wasserstein_distance,
        );
        
        let normalized_pillars = OSQINormalizedPillars {
            betes_score: input.betes_score,
            code_health_score_c_hs: code_health_score,
            security_score_sec_s: security_score,
            architecture_score_arch_s: architecture_score,
            risk_robustness_score,
        };
        
        
        // Calculate weighted harmonic mean
        let weights = vec![
            self.weights.w_test,
            self.weights.w_code,
            self.weights.w_sec,
            self.weights.w_arch,
        ];
        
        let scores = vec![
            normalized_pillars.betes_score,
            normalized_pillars.code_health_score_c_hs,
            normalized_pillars.security_score_sec_s,
            normalized_pillars.architecture_score_arch_s,
        ];
        
        let mut weighted_sum_reciprocals = 0.0;
        let sum_weights: f64 = weights.iter().sum();
        
        for (i, &score) in scores.iter().enumerate() {
            if score > 0.0 {
                weighted_sum_reciprocals += weights[i] / score;
            } else {
                // If any pillar is 0, the harmonic mean is 0
                weighted_sum_reciprocals = f64::INFINITY;
                break;
            }
        }
        
        let osqi_score = if weighted_sum_reciprocals.is_infinite() {
            0.0
        } else {
            (sum_weights / weighted_sum_reciprocals).clamp(0.0, 1.0)
        };
        
        let insights = generate_osqi_insights(&normalized_pillars, osqi_score);
        
        let result = OSQIResult {
            raw_pillars_input: Some(input.clone()),
            normalized_pillars: Some(normalized_pillars),
            applied_weights: Some(self.weights),
            osqi_score,
            calculation_time_s: start_time.elapsed().as_secs_f64(),
            insights,
        };
        
        Ok((QualityScore::new(osqi_score)?, result))
    }
}

impl QualityMetric for OSQICalculator {
    type Input = OSQIInput;
    type Components = OSQIResult;
    
    fn calculate(&self, input: &Self::Input) -> Result<(QualityScore, Self::Components)> {
        // Default to Python language if not specified
        self.calculate_with_language(input, "python")
    }
    
    fn name(&self) -> &'static str {
        "OSQI"
    }
    
    fn version(&self) -> &'static str {
        "1.0"
    }
}

/// Generate insights for OSQI score
fn generate_osqi_insights(pillars: &OSQINormalizedPillars, osqi_score: f64) -> Vec<String> {
    let mut insights = Vec::new();
    
    // Overall score insight
    if osqi_score >= 0.9 {
        insights.push("Exceptional overall software quality".to_string());
    } else if osqi_score >= 0.8 {
        insights.push("Strong overall software quality".to_string());
    } else if osqi_score >= 0.7 {
        insights.push("Good overall software quality".to_string());
    } else if osqi_score >= 0.6 {
        insights.push("Acceptable overall software quality".to_string());
    } else {
        insights.push("Software quality needs significant improvement".to_string());
    }
    
    // Pillar-specific insights
    if pillars.betes_score < 0.7 {
        insights.push("Test effectiveness is below recommended levels".to_string());
    }
    
    if pillars.code_health_score_c_hs < 0.7 {
        insights.push("Code health metrics indicate maintainability issues".to_string());
    }
    
    if pillars.security_score_sec_s < 0.8 {
        insights.push("Security vulnerabilities detected - remediation recommended".to_string());
    }
    
    if pillars.architecture_score_arch_s < 0.7 {
        insights.push("Architectural cohesion could be improved".to_string());
    }
    
    // Identify weakest pillar
    let mut min_score = pillars.betes_score;
    let mut weakest = "Test Effectiveness";
    
    if pillars.code_health_score_c_hs < min_score {
        min_score = pillars.code_health_score_c_hs;
        weakest = "Code Health";
    }
    if pillars.security_score_sec_s < min_score {
        min_score = pillars.security_score_sec_s;
        weakest = "Security";
    }
    if pillars.architecture_score_arch_s < min_score {
        weakest = "Architecture";
    }
    
    insights.push(format!("Focus area: {} is the weakest pillar", weakest));
    
    insights
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_osqi_calculation() {
        let weights = OSQIWeights::default();
        let calculator = OSQICalculator::new(weights, "config/chs_thresholds.yml", None);
        
        let input = OSQIInput {
            betes_score: 0.85,
            raw_code_health_sub_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("cyclomatic_complexity".to_string(), 5.0);
                metrics.insert("maintainability_index".to_string(), 85.0);
                metrics
            },
            raw_weighted_vulnerability_density: 0.1,
            raw_algebraic_connectivity: Some(0.9),
            raw_wasserstein_distance: None,
        };
        
        let (score, result) = calculator.calculate_with_language(&input, "python").unwrap();
        assert!(score.value() > 0.0);
        assert!(!result.insights.is_empty());
    }
}