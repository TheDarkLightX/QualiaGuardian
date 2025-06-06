//! Builder pattern implementations for configuration types

use crate::{Result, QualityError};
use super::{QualityConfig, QualityMode, BETESWeights, BETESSettingsV31, OSQIWeights};
use crate::types::RiskClass;

/// Builder for `QualityConfig`
#[derive(Debug, Default)]
pub struct QualityConfigBuilder {
    mode: Option<QualityMode>,
    risk_class: Option<RiskClass>,
    betes_weights: Option<BETESWeights>,
    betes_v3_1_settings: Option<BETESSettingsV31>,
    osqi_weights: Option<OSQIWeights>,
    chs_thresholds_path: Option<String>,
}

impl QualityConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the quality mode
    #[must_use]
    pub fn mode(mut self, mode: QualityMode) -> Self {
        self.mode = Some(mode);
        self
    }
    
    /// Set the risk class
    #[must_use]
    pub fn risk_class(mut self, risk_class: RiskClass) -> Self {
        self.risk_class = Some(risk_class);
        self
    }
    
    /// Set bE-TES weights
    #[must_use]
    pub fn betes_weights(mut self, weights: BETESWeights) -> Self {
        self.betes_weights = Some(weights);
        self
    }
    
    /// Set bE-TES v3.1 settings
    #[must_use]
    pub fn betes_v3_1_settings(mut self, settings: BETESSettingsV31) -> Self {
        self.betes_v3_1_settings = Some(settings);
        self
    }
    
    /// Set OSQI weights
    #[must_use]
    pub fn osqi_weights(mut self, weights: OSQIWeights) -> Self {
        self.osqi_weights = Some(weights);
        self
    }
    
    /// Set CHS thresholds path
    #[must_use]
    pub fn chs_thresholds_path(mut self, path: impl Into<String>) -> Self {
        self.chs_thresholds_path = Some(path.into());
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> Result<QualityConfig> {
        let mode = self.mode
            .ok_or_else(|| QualityError::Config("Quality mode is required".to_string()))?;
        
        // Set appropriate defaults based on mode
        let (betes_weights, betes_v3_1_settings) = match mode {
            QualityMode::BETESv31 => {
                (
                    self.betes_weights.unwrap_or_default(),
                    Some(self.betes_v3_1_settings.unwrap_or_default())
                )
            }
            QualityMode::BETESv3 => {
                (self.betes_weights.unwrap_or_default(), None)
            }
            _ => (BETESWeights::default(), None)
        };
        
        Ok(QualityConfig {
            mode,
            risk_class: self.risk_class,
            betes_weights,
            betes_v3_1_settings,
            osqi_weights: self.osqi_weights.unwrap_or_default(),
            chs_thresholds_path: self.chs_thresholds_path
                .or_else(|| Some("config/chs_thresholds.yml".to_string())),
        })
    }
}

impl QualityConfig {
    /// Create a builder for `QualityConfig`
    pub fn builder() -> QualityConfigBuilder {
        QualityConfigBuilder::new()
    }
}