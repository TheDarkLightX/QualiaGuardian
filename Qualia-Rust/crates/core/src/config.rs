//! Configuration structures for quality metrics

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::types::RiskClass;

pub mod builder;
pub use builder::QualityConfigBuilder;

/// Quality calculation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityMode {
    /// E-TES v2.0: Evolutionary Test Effectiveness Score
    #[serde(rename = "etes_v2")]
    ETESv2,
    /// bE-TES v3.0: Bounded Evolutionary TES
    #[serde(rename = "betes_v3")]
    BETESv3,
    /// bE-TES v3.1: With sigmoid smoothing options
    #[serde(rename = "betes_v3.1")]
    BETESv31,
    /// OSQI v1.0: Overall Software Quality Index
    #[serde(rename = "osqi_v1")]
    OSQIv1,
}

/// Main quality configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Quality scoring mode
    pub mode: QualityMode,
    /// Risk class for threshold evaluation
    pub risk_class: Option<RiskClass>,
    /// bE-TES component weights
    pub betes_weights: BETESWeights,
    /// bE-TES v3.1 specific settings
    pub betes_v3_1_settings: Option<BETESSettingsV31>,
    /// OSQI pillar weights
    pub osqi_weights: OSQIWeights,
    /// Path to CHS thresholds file
    pub chs_thresholds_path: Option<String>,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            mode: QualityMode::BETESv31,
            risk_class: None,
            betes_weights: BETESWeights::default(),
            betes_v3_1_settings: Some(BETESSettingsV31::default()),
            osqi_weights: OSQIWeights::default(),
            chs_thresholds_path: Some("config/chs_thresholds.yml".to_string()),
        }
    }
}

/// bE-TES component weights
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BETESWeights {
    /// Weight for mutation score (M)
    pub w_m: f64,
    /// Weight for EMT gain (E)
    pub w_e: f64,
    /// Weight for assertion IQ (A)
    pub w_a: f64,
    /// Weight for behavior coverage (B)
    pub w_b: f64,
    /// Weight for speed factor (S)
    pub w_s: f64,
}

impl Default for BETESWeights {
    fn default() -> Self {
        Self {
            w_m: 1.0,
            w_e: 1.0,
            w_a: 1.0,
            w_b: 1.0,
            w_s: 1.0,
        }
    }
}

/// bE-TES v3.1 sigmoid smoothing settings
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BETESSettingsV31 {
    /// Apply sigmoid smoothing to mutation score
    pub smooth_m: bool,
    /// Sigmoid steepness parameter for M
    pub k_m: f64,
    /// Apply sigmoid smoothing to EMT gain
    pub smooth_e: bool,
    /// Sigmoid steepness parameter for E
    pub k_e: f64,
}

impl Default for BETESSettingsV31 {
    fn default() -> Self {
        Self {
            smooth_m: true,
            k_m: 10.0,
            smooth_e: true,
            k_e: 10.0,
        }
    }
}

/// OSQI pillar weights
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OSQIWeights {
    /// Weight for bE-TES score
    pub w_test: f64,
    /// Weight for Code Health Score
    pub w_code: f64,
    /// Weight for Security Score
    pub w_sec: f64,
    /// Weight for Architecture Score
    pub w_arch: f64,
}

impl Default for OSQIWeights {
    fn default() -> Self {
        Self {
            w_test: 2.0,
            w_code: 1.0,
            w_sec: 1.5,
            w_arch: 1.0,
        }
    }
}

/// Risk definitions loaded from configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDefinitions {
    /// Map of risk class names to their definitions
    pub classes: HashMap<String, RiskClassDefinition>,
}

/// Individual risk class definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskClassDefinition {
    /// Minimum required score
    pub min_score: f64,
    /// Human-readable description
    pub description: Option<String>,
    /// Example use cases
    pub examples: Option<Vec<String>>,
}