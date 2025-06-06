//! TES: Test Effectiveness Score - Main quality score dispatcher

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{
    Result, QualityScore, QualityError,
    config::{QualityConfig, QualityMode},
    betes::{BETESCalculator, BETESInput, BETESComponents},
    osqi::{OSQICalculator, OSQIInput, OSQIResult},
    traits::QualityMetric,
};

/// Input data for quality score calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityInput {
    /// Raw metrics for bE-TES modes
    pub betes_metrics: Option<BETESMetrics>,
    /// Test suite data for E-TES v2 mode
    pub test_suite_data: Option<HashMap<String, serde_json::Value>>,
    /// Codebase data for E-TES v2 mode
    pub codebase_data: Option<HashMap<String, serde_json::Value>>,
    /// Previous score for evolution tracking
    pub previous_score: Option<f64>,
    /// Project path for OSQI sensors
    pub project_path: Option<String>,
    /// Project language for CHS normalization
    pub project_language: Option<String>,
}

/// Raw metrics for bE-TES calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BETESMetrics {
    pub raw_mutation_score: f64,
    pub raw_emt_gain: f64,
    pub raw_assertion_iq: f64,
    pub raw_behaviour_coverage: f64,
    pub raw_median_test_time_ms: f64,
    pub raw_flakiness_rate: f64,
}

/// Output from quality score calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum QualityOutput {
    BETES(BETESComponents),
    OSQI(OSQIResult),
    ETES(ETESComponents),
}

/// E-TES components (placeholder for now)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ETESComponents {
    pub mutation_score: f64,
    pub evolution_gain: f64,
    pub assertion_iq: f64,
    pub behavior_coverage: f64,
    pub speed_factor: f64,
    pub quality_factor: f64,
    pub etes_score: f64,
    pub insights: Vec<String>,
}

/// Main quality score calculator
pub struct QualityCalculator {
    config: QualityConfig,
}

impl QualityCalculator {
    /// Create a new quality calculator
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }
    
    /// Calculate quality score based on configured mode
    pub fn calculate(&self, input: &QualityInput) -> Result<(QualityScore, QualityOutput)> {
        match self.config.mode {
            QualityMode::BETESv3 | QualityMode::BETESv31 => {
                self.calculate_betes(input)
            }
            QualityMode::OSQIv1 => {
                self.calculate_osqi(input)
            }
            QualityMode::ETESv2 => {
                self.calculate_etes_v2(input)
            }
        }
    }
    
    /// Calculate bE-TES score
    fn calculate_betes(&self, input: &QualityInput) -> Result<(QualityScore, QualityOutput)> {
        let metrics = input.betes_metrics.as_ref()
            .ok_or_else(|| QualityError::MissingInput("bE-TES metrics".to_string()))?;
        
        let settings_v3_1 = match self.config.mode {
            QualityMode::BETESv31 => self.config.betes_v3_1_settings.clone(),
            _ => None,
        };
        
        let calculator = BETESCalculator::new(
            self.config.betes_weights,
            settings_v3_1,
        );
        
        let betes_input = BETESInput {
            raw_mutation_score: metrics.raw_mutation_score,
            raw_emt_gain: metrics.raw_emt_gain,
            raw_assertion_iq: metrics.raw_assertion_iq,
            raw_behaviour_coverage: metrics.raw_behaviour_coverage,
            raw_median_test_time_ms: metrics.raw_median_test_time_ms,
            raw_flakiness_rate: metrics.raw_flakiness_rate,
        };
        
        let (score, components) = calculator.calculate(&betes_input)?;
        Ok((score, QualityOutput::BETES(components)))
    }
    
    /// Calculate OSQI score
    fn calculate_osqi(&self, input: &QualityInput) -> Result<(QualityScore, QualityOutput)> {
        let _project_path = input.project_path.as_ref()
            .ok_or_else(|| QualityError::MissingInput("project path for OSQI".to_string()))?;
        
        let project_language = input.project_language.as_deref()
            .unwrap_or("python");
        
        // First calculate bE-TES as a pillar
        let (betes_score, _) = self.calculate_betes(input)?;
        
        // TODO: Collect other sensor data (CHS, Security, Architecture)
        // For now, using placeholder values
        let osqi_input = OSQIInput {
            betes_score: betes_score.value(),
            raw_code_health_sub_metrics: HashMap::new(),
            raw_weighted_vulnerability_density: 0.0,
            raw_algebraic_connectivity: Some(1.0),
            raw_wasserstein_distance: None,
        };
        
        let chs_path = self.config.chs_thresholds_path.as_deref()
            .unwrap_or("config/chs_thresholds.yml");
        
        let calculator = OSQICalculator::new(
            self.config.osqi_weights,
            chs_path,
            None,
        );
        
        let (score, result) = calculator.calculate_with_language(&osqi_input, project_language)?;
        Ok((score, QualityOutput::OSQI(result)))
    }
    
    /// Calculate E-TES v2 score (placeholder)
    fn calculate_etes_v2(&self, _input: &QualityInput) -> Result<(QualityScore, QualityOutput)> {
        // TODO: Implement E-TES v2 calculation
        let components = ETESComponents {
            mutation_score: 0.0,
            evolution_gain: 0.0,
            assertion_iq: 0.0,
            behavior_coverage: 0.0,
            speed_factor: 0.0,
            quality_factor: 0.0,
            etes_score: 0.0,
            insights: vec!["E-TES v2 calculation not yet implemented".to_string()],
        };
        
        Ok((QualityScore::new(0.0)?, QualityOutput::ETES(components)))
    }
}

/// Get letter grade for a quality score
pub fn get_quality_grade(score: f64) -> &'static str {
    if score >= 0.9 {
        "A+"
    } else if score >= 0.8 {
        "A"
    } else if score >= 0.7 {
        "B"
    } else if score >= 0.6 {
        "C"
    } else {
        "F"
    }
}

/// Classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub score: f64,
    pub risk_class: Option<String>,
    pub threshold: Option<f64>,
    pub verdict: String,
    pub message: String,
}

/// Classify a score against risk definitions
pub fn classify_score(
    score: f64,
    risk_class_name: Option<&str>,
    risk_definitions: &HashMap<String, HashMap<String, serde_json::Value>>,
    metric_name: &str,
) -> ClassificationResult {
    let mut result = ClassificationResult {
        score: (score * 1000.0).round() / 1000.0,
        risk_class: risk_class_name.map(String::from),
        threshold: None,
        verdict: "UNKNOWN".to_string(),
        message: String::new(),
    };
    
    if let Some(class_name) = risk_class_name {
        if let Some(class_def) = risk_definitions.get(class_name) {
            if let Some(min_score_val) = class_def.get("min_score") {
                if let Some(min_score) = min_score_val.as_f64() {
                    result.threshold = Some((min_score * 1000.0).round() / 1000.0);
                    
                    if score >= min_score {
                        result.verdict = "PASS".to_string();
                        result.message = format!(
                            "{} score meets the threshold for {}",
                            metric_name, class_name
                        );
                    } else {
                        result.verdict = "FAIL".to_string();
                        result.message = format!(
                            "{} score does not meet the threshold for {}",
                            metric_name, class_name
                        );
                    }
                }
            }
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BETESWeights;
    
    #[test]
    fn test_quality_calculator() {
        let config = QualityConfig::builder()
            .mode(QualityMode::BETESv31)
            .build()
            .unwrap();
        
        let calculator = QualityCalculator::new(config);
        
        let input = QualityInput {
            betes_metrics: Some(BETESMetrics {
                raw_mutation_score: 0.85,
                raw_emt_gain: 0.15,
                raw_assertion_iq: 4.0,
                raw_behaviour_coverage: 0.9,
                raw_median_test_time_ms: 50.0,
                raw_flakiness_rate: 0.05,
            }),
            test_suite_data: None,
            codebase_data: None,
            previous_score: None,
            project_path: None,
            project_language: None,
        };
        
        let (score, output) = calculator.calculate(&input).unwrap();
        assert!(score.value() > 0.0);
        
        if let QualityOutput::BETES(components) = output {
            assert_eq!(components.raw_mutation_score, 0.85);
        } else {
            panic!("Expected BETES output");
        }
    }
}