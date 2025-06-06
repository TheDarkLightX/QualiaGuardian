//! bE-TES: Bounded Evolutionary Test Effectiveness Score v3.1

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
    fn normalize_assertion_iq(&self, raw: f64) -> f64 {
        // A_raw is 1-5, normalize to 0-1
        ((raw - 1.0) / 4.0).clamp(0.0, 1.0)
    }
    
    /// Calculate normalized behavior coverage (B')
    fn normalize_behaviour_coverage(&self, raw: f64) -> f64 {
        // Already a ratio, just clamp
        raw.clamp(0.0, 1.0)
    }
    
    /// Calculate normalized speed factor (S')
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
        
        // Final bE-TES score
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
}