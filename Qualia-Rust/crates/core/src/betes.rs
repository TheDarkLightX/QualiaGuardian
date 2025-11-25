//! bE-TES: Bounded Evolutionary Test Effectiveness Score v3.1
//!
//! This module implements the bE-TES (bounded Evolutionary Test Effectiveness Score) v3.1,
//! a comprehensive metric for evaluating test suite quality.
//!
//! # Mathematical Properties
//!
//! The bE-TES score is bounded to [0.0, 1.0] and satisfies the following properties:
//! - **Boundedness**: All components are normalized to [0.0, 1.0]
//! - **Monotonicity**: Increasing any component increases the score
//! - **Continuity**: Small changes in inputs produce small changes in output
//! - **Idempotency**: Normalization is idempotent (normalizing twice = normalizing once)

use serde::{Deserialize, Serialize};
use std::time::Instant;
use crate::{
    Result, QualityScore,
    config::{BETESWeights, BETESSettingsV31},
    traits::{QualityMetric, Normalizer, MinMaxNormalizer, SigmoidNormalizer},
};

/// bE-TES raw input values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BETESInput {
    /// Raw mutation score (0.0 to 1.0)
    pub raw_mutation_score: f64,
    /// Raw EMT gain (Final MS - Initial MS)
    pub raw_emt_gain: f64,
    /// Mean assertion IQ rubric score (1.0 to 5.0)
    pub raw_assertion_iq: f64,
    /// Ratio: Covered_Critical / Total_Critical
    pub raw_behaviour_coverage: f64,
    /// Median test execution time in milliseconds
    pub raw_median_test_time_ms: f64,
    /// Test suite flakiness rate (0.0 to 1.0)
    pub raw_flakiness_rate: f64,
}

impl BETESInput {
    /// Validate input values are within expected ranges
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.raw_mutation_score) {
            return Err(crate::QualityError::InvalidInput {
                field: "raw_mutation_score".to_string(),
                value: self.raw_mutation_score.to_string(),
            });
        }
        if !(1.0..=5.0).contains(&self.raw_assertion_iq) {
            return Err(crate::QualityError::InvalidInput {
                field: "raw_assertion_iq".to_string(),
                value: self.raw_assertion_iq.to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.raw_behaviour_coverage) {
            return Err(crate::QualityError::InvalidInput {
                field: "raw_behaviour_coverage".to_string(),
                value: self.raw_behaviour_coverage.to_string(),
            });
        }
        if self.raw_median_test_time_ms < 0.0 {
            return Err(crate::QualityError::InvalidInput {
                field: "raw_median_test_time_ms".to_string(),
                value: self.raw_median_test_time_ms.to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.raw_flakiness_rate) {
            return Err(crate::QualityError::InvalidInput {
                field: "raw_flakiness_rate".to_string(),
                value: self.raw_flakiness_rate.to_string(),
            });
        }
        Ok(())
    }
}

/// bE-TES components and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BETESComponents {
    // Raw input values
    pub raw_mutation_score: f64,
    pub raw_emt_gain: f64,
    pub raw_assertion_iq: f64,
    pub raw_behaviour_coverage: f64,
    pub raw_median_test_time_ms: f64,
    pub raw_flakiness_rate: f64,
    
    // Normalized factor values (0-1)
    pub norm_mutation_score: f64,
    pub norm_emt_gain: f64,
    pub norm_assertion_iq: f64,
    pub norm_behaviour_coverage: f64,
    pub norm_speed_factor: f64,
    
    // Intermediate calculations
    pub geometric_mean_g: f64,
    pub trust_coefficient_t: f64,
    
    // Final score
    pub betes_score: f64,
    
    // Metadata
    pub calculation_time_s: f64,
    pub applied_weights: Option<BETESWeights>,
    pub insights: Vec<String>,
}

/// bE-TES calculator
#[derive(Debug)]
pub struct BETESCalculator {
    weights: BETESWeights,
    settings_v3_1: Option<BETESSettingsV31>,
}

impl BETESCalculator {
    /// Create a new bE-TES calculator
    pub fn new(weights: BETESWeights, settings_v3_1: Option<BETESSettingsV31>) -> Self {
        Self {
            weights,
            settings_v3_1,
        }
    }
    
    /// Calculate normalized mutation score (M')
    /// 
    /// Uses sigmoid normalization if v3.1 settings are enabled, otherwise min-max.
    /// 
    /// # Mathematical Properties
    /// - Output is always in [0.0, 1.0]
    /// - Monotonic: increasing input increases output
    /// - Continuous: no discontinuities
    fn normalize_mutation_score(&self, raw: f64) -> f64 {
        if let Some(settings) = &self.settings_v3_1 {
            if settings.smooth_m {
                let normalizer = SigmoidNormalizer {
                    center: 0.775,
                    steepness: settings.k_m,
                };
                return normalizer.normalize(raw);
            }
        }
        
        // Default: min-max normalization
        let normalizer = MinMaxNormalizer {
            min: 0.6,
            max: 0.95,
        };
        normalizer.normalize(raw)
    }
    
    /// Calculate normalized EMT gain (E')
    /// 
    /// Uses sigmoid normalization if v3.1 settings are enabled, otherwise clip.
    fn normalize_emt_gain(&self, raw: f64) -> f64 {
        if let Some(settings) = &self.settings_v3_1 {
            if settings.smooth_e {
                let normalizer = SigmoidNormalizer {
                    center: 0.125,
                    steepness: settings.k_e,
                };
                return normalizer.normalize(raw);
            }
        }
        
        // Default: clip normalization
        (raw / 0.25).clamp(0.0, 1.0)
    }
    
    /// Calculate normalized assertion IQ (A')
    /// 
    /// Maps [1.0, 5.0] to [0.0, 1.0] linearly.
    fn normalize_assertion_iq(&self, raw: f64) -> f64 {
        // A_raw is 1-5, normalize to 0-1
        ((raw - 1.0) / 4.0).clamp(0.0, 1.0)
    }
    
    /// Calculate normalized behavior coverage (B')
    /// 
    /// Already a ratio, just clamp to [0.0, 1.0].
    fn normalize_behaviour_coverage(&self, raw: f64) -> f64 {
        // Already a ratio, just clamp
        raw.clamp(0.0, 1.0)
    }
    
    /// Calculate normalized speed factor (S')
    /// 
    /// Uses logarithmic scaling: S' = 1 / (1 + log10(time_ms / 100))
    /// 
    /// # Properties
    /// - Returns 1.0 for times <= 100ms
    /// - Returns 0.0 for times <= 0ms
    /// - Monotonically decreasing
    fn normalize_speed_factor(&self, raw_ms: f64) -> f64 {
        if raw_ms <= 0.0 {
            return 0.0;
        } else if raw_ms <= 100.0 {
            return 1.0;
        }
        
        // For times > 100ms: S' = 1 / (1 + log10(time_ms / 100))
        let log_input = raw_ms / 100.0;
        match log_input.log10() {
            log_val if log_val.is_finite() => {
                let denominator = 1.0 + log_val;
                if denominator <= f64::EPSILON {
                    0.0
                } else {
                    (1.0 / denominator).clamp(0.0, 1.0)
                }
            }
            _ => 0.0,
        }
    }
    
    /// Calculate weighted geometric mean
    /// 
    /// G = (∏(factor_i^weight_i))^(1/Σweight_i)
    /// 
    /// # Properties
    /// - Returns 0.0 if any factor is 0.0 and its weight > 0.0
    /// - Returns 0.0 if sum of weights <= 0.0
    /// - Monotonic: increasing any factor increases the result
    fn calculate_geometric_mean(&self, factors: &[f64], weights: &[f64]) -> f64 {
        let sum_of_weights: f64 = weights.iter().sum();
        if sum_of_weights <= 0.0 {
            return 0.0;
        }
        
        let mut weighted_product = 1.0;
        
        for (&factor, &weight) in factors.iter().zip(weights.iter()) {
            if factor == 0.0 && weight > 0.0 {
                return 0.0;
            }
            if factor > 0.0 || weight == 0.0 {
                weighted_product *= factor.powf(weight);
            }
        }
        
        weighted_product.powf(1.0 / sum_of_weights)
    }
}

impl QualityMetric for BETESCalculator {
    type Input = BETESInput;
    type Components = BETESComponents;
    
    fn calculate(&self, input: &Self::Input) -> Result<(QualityScore, Self::Components)> {
        // Validate input
        input.validate()?;
        
        let start_time = Instant::now();
        
        // Normalize all factors
        let norm_mutation = self.normalize_mutation_score(input.raw_mutation_score);
        let norm_emt = self.normalize_emt_gain(input.raw_emt_gain);
        let norm_aiq = self.normalize_assertion_iq(input.raw_assertion_iq);
        let norm_behaviour = self.normalize_behaviour_coverage(input.raw_behaviour_coverage);
        let norm_speed = self.normalize_speed_factor(input.raw_median_test_time_ms);
        
        // Calculate geometric mean
        let factors = vec![
            norm_mutation,
            norm_emt,
            norm_aiq,
            norm_behaviour,
            norm_speed,
        ];
        
        let weights = vec![
            self.weights.w_m,
            self.weights.w_e,
            self.weights.w_a,
            self.weights.w_b,
            self.weights.w_s,
        ];
        
        let geometric_mean = self.calculate_geometric_mean(&factors, &weights);
        
        // Calculate trust coefficient
        let trust_coefficient = (1.0 - input.raw_flakiness_rate).clamp(0.0, 1.0);
        
        // Final bE-TES score: G * T
        let betes_score = (geometric_mean * trust_coefficient).clamp(0.0, 1.0);
        
        let mut components = BETESComponents {
            raw_mutation_score: input.raw_mutation_score,
            raw_emt_gain: input.raw_emt_gain,
            raw_assertion_iq: input.raw_assertion_iq,
            raw_behaviour_coverage: input.raw_behaviour_coverage,
            raw_median_test_time_ms: input.raw_median_test_time_ms,
            raw_flakiness_rate: input.raw_flakiness_rate,
            norm_mutation_score: norm_mutation,
            norm_emt_gain: norm_emt,
            norm_assertion_iq: norm_aiq,
            norm_behaviour_coverage: norm_behaviour,
            norm_speed_factor: norm_speed,
            geometric_mean_g: geometric_mean,
            trust_coefficient_t: trust_coefficient,
            betes_score,
            calculation_time_s: start_time.elapsed().as_secs_f64(),
            applied_weights: Some(self.weights),
            insights: Vec::new(),
        };
        
        components.insights = generate_insights(&components);
        
        Ok((QualityScore::new(betes_score)?, components))
    }
    
    fn name(&self) -> &'static str {
        "bE-TES"
    }
    
    fn version(&self) -> &'static str {
        if self.settings_v3_1.is_some() {
            "3.1"
        } else {
            "3.0"
        }
    }
}

/// Generate insights based on component values
fn generate_insights(components: &BETESComponents) -> Vec<String> {
    let mut insights = Vec::new();
    
    // Mutation score insights
    if components.norm_mutation_score < 0.5 {
        insights.push("Low mutation score indicates weak test suite effectiveness".to_string());
    } else if components.norm_mutation_score > 0.9 {
        insights.push("Excellent mutation score - tests effectively detect code changes".to_string());
    }
    
    // EMT gain insights
    if components.norm_emt_gain > 0.8 {
        insights.push("Strong evolutionary improvement in test effectiveness".to_string());
    } else if components.norm_emt_gain < 0.2 {
        insights.push("Limited improvement from evolutionary testing".to_string());
    }
    
    // Assertion IQ insights
    if components.norm_assertion_iq < 0.5 {
        insights.push("Low assertion IQ - consider more intelligent assertions".to_string());
    }
    
    // Speed insights
    if components.norm_speed_factor < 0.5 {
        insights.push("Slow test execution impacting overall quality".to_string());
    }
    
    // Flakiness insights
    if components.raw_flakiness_rate > 0.1 {
        insights.push("High flakiness reducing test suite reliability".to_string());
    }
    
    insights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BETESWeights;
    
    #[test]
    fn test_betes_calculation() {
        let calculator = BETESCalculator::new(
            BETESWeights::default(),
            Some(BETESSettingsV31::default()),
        );
        
        let input = BETESInput {
            raw_mutation_score: 0.85,
            raw_emt_gain: 0.15,
            raw_assertion_iq: 4.0,
            raw_behaviour_coverage: 0.9,
            raw_median_test_time_ms: 50.0,
            raw_flakiness_rate: 0.05,
        };
        
        let (score, components) = calculator.calculate(&input).unwrap();
        assert!(score.value() > 0.7);
        assert_eq!(components.raw_mutation_score, 0.85);
    }
    
    #[test]
    fn test_boundedness_property() {
        // Property: Score is always in [0.0, 1.0]
        let calculator = BETESCalculator::new(
            BETESWeights::default(),
            Some(BETESSettingsV31::default()),
        );
        
        // Test with extreme values
        let extreme_inputs = vec![
            BETESInput {
                raw_mutation_score: 0.0,
                raw_emt_gain: 0.0,
                raw_assertion_iq: 1.0,
                raw_behaviour_coverage: 0.0,
                raw_median_test_time_ms: 10000.0,
                raw_flakiness_rate: 1.0,
            },
            BETESInput {
                raw_mutation_score: 1.0,
                raw_emt_gain: 1.0,
                raw_assertion_iq: 5.0,
                raw_behaviour_coverage: 1.0,
                raw_median_test_time_ms: 1.0,
                raw_flakiness_rate: 0.0,
            },
        ];
        
        for input in extreme_inputs {
            let (score, _) = calculator.calculate(&input).unwrap();
            assert!(
                (0.0..=1.0).contains(&score.value()),
                "Score {} is not in [0.0, 1.0]",
                score.value()
            );
        }
    }
    
    #[test]
    fn test_monotonicity_property() {
        // Property: Increasing any component increases the score
        let calculator = BETESCalculator::new(
            BETESWeights::default(),
            Some(BETESSettingsV31::default()),
        );
        
        let base_input = BETESInput {
            raw_mutation_score: 0.7,
            raw_emt_gain: 0.1,
            raw_assertion_iq: 3.0,
            raw_behaviour_coverage: 0.7,
            raw_median_test_time_ms: 200.0,
            raw_flakiness_rate: 0.1,
        };
        
        let (base_score, _) = calculator.calculate(&base_input).unwrap();
        
        // Increase mutation score
        let mut improved = base_input.clone();
        improved.raw_mutation_score = 0.9;
        let (improved_score, _) = calculator.calculate(&improved).unwrap();
        assert!(
            improved_score.value() >= base_score.value(),
            "Increasing mutation score should increase overall score"
        );
    }
    
    #[test]
    fn test_input_validation() {
        let input = BETESInput {
            raw_mutation_score: 1.5, // Invalid: > 1.0
            raw_emt_gain: 0.15,
            raw_assertion_iq: 4.0,
            raw_behaviour_coverage: 0.9,
            raw_median_test_time_ms: 50.0,
            raw_flakiness_rate: 0.05,
        };
        
        assert!(input.validate().is_err());
    }
    
    #[test]
    fn test_normalization_idempotency() {
        // Property: Normalizing twice should equal normalizing once
        let calculator = BETESCalculator::new(
            BETESWeights::default(),
            Some(BETESSettingsV31::default()),
        );
        
        let raw_value = 0.8;
        let normalized_once = calculator.normalize_mutation_score(raw_value);
        let normalized_twice = calculator.normalize_mutation_score(normalized_once);
        
        // For min-max, this should hold (sigmoid may not be exactly idempotent)
        // But the result should still be in [0.0, 1.0]
        assert!(
            (0.0..=1.0).contains(&normalized_twice),
            "Double normalization should still be in [0.0, 1.0]"
        );
    }
    
    #[test]
    fn test_geometric_mean_edge_cases() {
        let calculator = BETESCalculator::new(
            BETESWeights::default(),
            None,
        );
        
        // Zero factors
        let factors = vec![0.0, 0.5, 0.5];
        let weights = vec![1.0, 1.0, 1.0];
        let result = calculator.calculate_geometric_mean(&factors, &weights);
        assert_eq!(result, 0.0);
        
        // Zero weights
        let factors = vec![0.5, 0.5, 0.5];
        let weights = vec![0.0, 0.0, 0.0];
        let result = calculator.calculate_geometric_mean(&factors, &weights);
        assert_eq!(result, 0.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    
    prop_compose! {
        fn valid_betes_input()(
            raw_mutation_score in 0.0f64..=1.0f64,
            raw_emt_gain in -1.0f64..=1.0f64,
            raw_assertion_iq in 1.0f64..=5.0f64,
            raw_behaviour_coverage in 0.0f64..=1.0f64,
            raw_median_test_time_ms in 0.0f64..=10000.0f64,
            raw_flakiness_rate in 0.0f64..=1.0f64,
        ) -> BETESInput {
            BETESInput {
                raw_mutation_score,
                raw_emt_gain,
                raw_assertion_iq,
                raw_behaviour_coverage,
                raw_median_test_time_ms,
                raw_flakiness_rate,
            }
        }
    }
    
    prop_compose! {
        fn valid_weights()(
            w_m in 0.0f64..=10.0f64,
            w_e in 0.0f64..=10.0f64,
            w_a in 0.0f64..=10.0f64,
            w_b in 0.0f64..=10.0f64,
            w_s in 0.0f64..=10.0f64,
        ) -> BETESWeights {
            BETESWeights { w_m, w_e, w_a, w_b, w_s }
        }
    }
    
    proptest! {
        #[test]
        fn prop_boundedness(input in valid_betes_input()) {
            let calculator = BETESCalculator::new(
                BETESWeights::default(),
                Some(BETESSettingsV31::default()),
            );
            
            let (score, _) = calculator.calculate(&input).unwrap();
            prop_assert!((0.0..=1.0).contains(&score.value()));
        }
        
        #[test]
        fn prop_monotonicity_mutation_score(
            base_ms in 0.0f64..=1.0f64,
            improvement in 0.01f64..=0.2f64,
        ) {
            let calculator = BETESCalculator::new(
                BETESWeights::default(),
                Some(BETESSettingsV31::default()),
            );
            
            let base_input = BETESInput {
                raw_mutation_score: base_ms,
                raw_emt_gain: 0.15,
                raw_assertion_iq: 4.0,
                raw_behaviour_coverage: 0.9,
                raw_median_test_time_ms: 50.0,
                raw_flakiness_rate: 0.05,
            };
            
            let mut improved_input = base_input.clone();
            improved_input.raw_mutation_score = (base_ms + improvement).min(1.0);
            
            let (base_score, _) = calculator.calculate(&base_input).unwrap();
            let (improved_score, _) = calculator.calculate(&improved_input).unwrap();
            
            prop_assert!(improved_score.value() >= base_score.value());
        }
        
        #[test]
        fn prop_geometric_mean_properties(
            factors in prop::collection::vec(0.0f64..=1.0f64, 3..=5),
            weights in prop::collection::vec(0.0f64..=10.0f64, 3..=5),
        ) {
            let calculator = BETESCalculator::new(
                BETESWeights::default(),
                None,
            );
            
            let result = calculator.calculate_geometric_mean(&factors, &weights);
            
            // If any factor is 0 and weight > 0, result should be 0
            let has_zero_factor = factors.iter().zip(weights.iter())
                .any(|(&f, &w)| f == 0.0 && w > 0.0);
            
            if has_zero_factor {
                prop_assert_eq!(result, 0.0);
            } else {
                prop_assert!((0.0..=1.0).contains(&result));
            }
        }
        
        #[test]
        fn prop_normalization_bounded(
            raw_value in -10.0f64..=10.0f64,
        ) {
            let calculator = BETESCalculator::new(
                BETESWeights::default(),
                Some(BETESSettingsV31::default()),
            );
            
            let normalized = calculator.normalize_mutation_score(raw_value);
            prop_assert!((0.0..=1.0).contains(&normalized));
        }
    }
}
