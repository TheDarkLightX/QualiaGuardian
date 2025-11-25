//! Core traits for quality metrics
//!
//! This module defines the core traits used throughout the quality metric system,
//! providing a consistent interface for all metric implementations.

use crate::{Result, QualityScore};
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;

/// Base trait for all quality metrics
///
/// This trait provides a uniform interface for calculating quality scores.
/// All metrics must implement this trait to be used in the quality system.
///
/// # Type Parameters
/// - `Input`: The input data type for this metric
/// - `Components`: The detailed output type containing component breakdowns
///
/// # Examples
///
/// ```rust
/// use qualia_core::traits::QualityMetric;
/// use qualia_core::{Result, QualityScore};
///
/// struct MyMetric;
///
/// impl QualityMetric for MyMetric {
///     type Input = MyInput;
///     type Components = MyComponents;
///
///     fn calculate(&self, input: &Self::Input) -> Result<(QualityScore, Self::Components)> {
///         // Implementation
///     }
///
///     fn name(&self) -> &'static str { "MyMetric" }
///     fn version(&self) -> &'static str { "1.0" }
/// }
/// ```
pub trait QualityMetric: Debug + Send + Sync {
    /// The input type for this metric
    type Input: Debug + Send + Sync;
    
    /// The detailed output/components type
    type Components: Debug + Clone + Serialize + DeserializeOwned + Send + Sync;
    
    /// Calculate the quality score
    ///
    /// # Errors
    /// Returns an error if the calculation fails or inputs are invalid.
    fn calculate(&self, input: &Self::Input) -> Result<(QualityScore, Self::Components)>;
    
    /// Get the metric name
    fn name(&self) -> &'static str;
    
    /// Get metric version
    fn version(&self) -> &'static str;
}

/// Trait for normalizing raw values to [0.0, 1.0] range
///
/// Normalizers are used to convert raw metric values into a standardized
/// [0.0, 1.0] range for consistent scoring.
///
/// # Properties
/// - Output is always in [0.0, 1.0]
/// - Should be monotonic (increasing input increases output)
/// - Should be continuous (no discontinuities)
pub trait Normalizer {
    /// Normalize a raw value to [0.0, 1.0] range
    ///
    /// # Arguments
    /// * `raw_value` - The raw value to normalize
    ///
    /// # Returns
    /// A value in [0.0, 1.0]
    fn normalize(&self, raw_value: f64) -> f64;
}

/// Min-max normalizer
///
/// Linearly maps values from [min, max] to [0.0, 1.0].
/// Values outside the range are clamped.
///
/// # Examples
///
/// ```rust
/// use qualia_core::traits::{Normalizer, MinMaxNormalizer};
///
/// let normalizer = MinMaxNormalizer { min: 0.0, max: 100.0 };
/// assert_eq!(normalizer.normalize(50.0), 0.5);
/// assert_eq!(normalizer.normalize(150.0), 1.0); // Clamped
/// ```
#[derive(Debug, Clone, Copy)]
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
///
/// Uses a sigmoid (logistic) function to normalize values.
/// Provides smooth transitions and handles outliers gracefully.
///
/// The sigmoid function is: 1 / (1 + exp(-steepness * (value - center)))
///
/// # Examples
///
/// ```rust
/// use qualia_core::traits::{Normalizer, SigmoidNormalizer};
///
/// let normalizer = SigmoidNormalizer {
///     center: 0.5,
///     steepness: 10.0,
/// };
/// let normalized = normalizer.normalize(0.5);
/// assert!((0.4..=0.6).contains(&normalized));
/// ```
#[derive(Debug, Clone, Copy)]
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_min_max_normalizer() {
        let normalizer = MinMaxNormalizer { min: 0.0, max: 100.0 };
        
        assert_eq!(normalizer.normalize(0.0), 0.0);
        assert_eq!(normalizer.normalize(50.0), 0.5);
        assert_eq!(normalizer.normalize(100.0), 1.0);
        assert_eq!(normalizer.normalize(150.0), 1.0); // Clamped
        assert_eq!(normalizer.normalize(-50.0), 0.0); // Clamped
    }
    
    #[test]
    fn test_min_max_normalizer_edge_cases() {
        let normalizer = MinMaxNormalizer { min: 10.0, max: 10.0 };
        assert_eq!(normalizer.normalize(10.0), 1.0);
        assert_eq!(normalizer.normalize(5.0), 0.0);
        
        let normalizer = MinMaxNormalizer { min: 100.0, max: 0.0 };
        assert_eq!(normalizer.normalize(50.0), 1.0);
    }
    
    #[test]
    fn test_sigmoid_normalizer() {
        let normalizer = SigmoidNormalizer {
            center: 0.5,
            steepness: 10.0,
        };
        
        let result = normalizer.normalize(0.5);
        assert!((0.0..=1.0).contains(&result));
        
        // Should be close to 0.5 at center
        assert!((0.4..=0.6).contains(&result));
    }
    
    #[test]
    fn test_sigmoid_normalizer_extremes() {
        let normalizer = SigmoidNormalizer {
            center: 0.5,
            steepness: 10.0,
        };
        
        let low = normalizer.normalize(-100.0);
        let high = normalizer.normalize(100.0);
        
        assert!((0.0..=1.0).contains(&low));
        assert!((0.0..=1.0).contains(&high));
        assert!(low < 0.1);
        assert!(high > 0.9);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn prop_min_max_bounded(
            min in -100.0f64..=100.0f64,
            max in -100.0f64..=100.0f64,
            value in -200.0f64..=200.0f64,
        ) {
            let normalizer = MinMaxNormalizer { min, max };
            let result = normalizer.normalize(value);
            prop_assert!((0.0..=1.0).contains(&result));
        }
        
        #[test]
        fn prop_sigmoid_bounded(
            center in -10.0f64..=10.0f64,
            steepness in 0.1f64..=100.0f64,
            value in -100.0f64..=100.0f64,
        ) {
            let normalizer = SigmoidNormalizer { center, steepness };
            let result = normalizer.normalize(value);
            prop_assert!((0.0..=1.0).contains(&result));
        }
        
        #[test]
        fn prop_min_max_monotonic(
            min in 0.0f64..=50.0f64,
            max in 50.0f64..=100.0f64,
            value1 in 0.0f64..=100.0f64,
            delta in 0.1f64..=10.0f64,
        ) {
            prop_assume!(max > min);
            let normalizer = MinMaxNormalizer { min, max };
            let value2 = (value1 + delta).min(100.0);
            
            let result1 = normalizer.normalize(value1);
            let result2 = normalizer.normalize(value2);
            
            prop_assert!(result2 >= result1);
        }
    }
}
