//! Core traits for quality metrics

use crate::{Result, QualityScore};
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;

/// Base trait for all quality metrics
pub trait QualityMetric: Debug + Send + Sync {
    /// The input type for this metric
    type Input: Debug + Send + Sync;
    
    /// The detailed output/components type
    type Components: Debug + Clone + Serialize + DeserializeOwned + Send + Sync;
    
    /// Calculate the quality score
    fn calculate(&self, input: &Self::Input) -> Result<(QualityScore, Self::Components)>;
    
    /// Get the metric name
    fn name(&self) -> &'static str;
    
    /// Get metric version
    fn version(&self) -> &'static str;
}

/// Trait for normalizing raw values
pub trait Normalizer {
    /// Normalize a raw value to [0.0, 1.0] range
    fn normalize(&self, raw_value: f64) -> f64;
}

/// Min-max normalizer
pub struct MinMaxNormalizer {
    pub min: f64,
    pub max: f64,
}

impl Normalizer for MinMaxNormalizer {
    fn normalize(&self, raw_value: f64) -> f64 {
        if self.max <= self.min {
            return if raw_value < self.min { 0.0 } else { 1.0 };
        }
        ((raw_value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }
}

/// Sigmoid normalizer
pub struct SigmoidNormalizer {
    pub center: f64,
    pub steepness: f64,
}

impl Normalizer for SigmoidNormalizer {
    fn normalize(&self, raw_value: f64) -> f64 {
        let val = -self.steepness * (raw_value - self.center);
        match val.exp() {
            v if v.is_infinite() => if val > 0.0 { 0.0 } else { 1.0 },
            v => 1.0 / (1.0 + v),
        }
    }
}