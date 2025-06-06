//! Shapley value calculation for test importance analysis

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Shapley value calculator for test importance
#[derive(Debug)]
pub struct ShapleyCalculator {
    /// Number of Monte Carlo samples for approximation
    n_samples: usize,
    /// Whether to use parallel computation
    parallel: bool,
}

/// Result of Shapley value calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapleyResult {
    /// Shapley values for each test
    pub values: HashMap<String, f64>,
    /// Normalized importance scores (0-1)
    pub importance_scores: HashMap<String, f64>,
    /// Ranking of tests by importance
    pub ranking: Vec<(String, f64)>,
    /// Confidence intervals if calculated
    pub confidence_intervals: Option<HashMap<String, (f64, f64)>>,
}

/// Coalition value function trait
pub trait ValueFunction: Send + Sync {
    /// Calculate the value of a coalition (subset of tests)
    fn value(&self, coalition: &[String]) -> f64;
    
    /// Get all player IDs
    fn players(&self) -> Vec<String>;
}

/// Test suite value function based on coverage
#[derive(Debug)]
pub struct CoverageValueFunction {
    /// Coverage data: test -> set of covered lines
    coverage_data: HashMap<String, HashSet<u32>>,
    /// Total lines in codebase
    total_lines: usize,
}

impl CoverageValueFunction {
    pub fn new(coverage_data: HashMap<String, HashSet<u32>>, total_lines: usize) -> Self {
        Self {
            coverage_data,
            total_lines,
        }
    }
}

impl ValueFunction for CoverageValueFunction {
    fn value(&self, coalition: &[String]) -> f64 {
        let mut covered_lines: HashSet<u32> = HashSet::new();
        
        for test in coalition {
            if let Some(lines) = self.coverage_data.get(test) {
                covered_lines.extend(lines);
            }
        }
        
        if self.total_lines > 0 {
            covered_lines.len() as f64 / self.total_lines as f64
        } else {
            0.0
        }
    }
    
    fn players(&self) -> Vec<String> {
        self.coverage_data.keys().cloned().collect()
    }
}

/// Quality-based value function
pub struct QualityValueFunction {
    /// Function to calculate quality score for a test subset
    quality_fn: Box<dyn Fn(&[String]) -> f64 + Send + Sync>,
    /// All test IDs
    test_ids: Vec<String>,
}

impl QualityValueFunction {
    pub fn new<F>(test_ids: Vec<String>, quality_fn: F) -> Self
    where
        F: Fn(&[String]) -> f64 + Send + Sync + 'static,
    {
        Self {
            quality_fn: Box::new(quality_fn),
            test_ids,
        }
    }
}

impl ValueFunction for QualityValueFunction {
    fn value(&self, coalition: &[String]) -> f64 {
        (self.quality_fn)(coalition)
    }
    
    fn players(&self) -> Vec<String> {
        self.test_ids.clone()
    }
}

/// Multi-criteria value function combining multiple objectives
pub struct MultiCriteriaValueFunction {
    /// Individual value functions with weights
    functions: Vec<(Box<dyn ValueFunction>, f64)>,
    /// Cached player list
    players: Vec<String>,
}

impl MultiCriteriaValueFunction {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            players: Vec::new(),
        }
    }
    
    pub fn add_criterion(mut self, function: Box<dyn ValueFunction>, weight: f64) -> Self {
        if self.players.is_empty() {
            self.players = function.players();
        }
        self.functions.push((function, weight));
        self
    }
}

impl ValueFunction for MultiCriteriaValueFunction {
    fn value(&self, coalition: &[String]) -> f64 {
        let mut total_value = 0.0;
        let mut total_weight = 0.0;
        
        for (function, weight) in &self.functions {
            total_value += function.value(coalition) * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            total_value / total_weight
        } else {
            0.0
        }
    }
    
    fn players(&self) -> Vec<String> {
        self.players.clone()
    }
}

impl Default for ShapleyCalculator {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            parallel: true,
        }
    }
}

impl ShapleyCalculator {
    /// Create a new Shapley calculator
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            parallel: true,
        }
    }
    
    /// Set whether to use parallel computation
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
    
    /// Calculate exact Shapley values (for small sets)
    pub fn calculate_exact(&self, value_fn: &dyn ValueFunction) -> Result<ShapleyResult> {
        let players = value_fn.players();
        let n = players.len();
        
        if n > 20 {
            anyhow::bail!("Exact calculation is only feasible for <= 20 players. Use calculate_monte_carlo instead.");
        }
        
        let mut shapley_values = HashMap::new();
        
        for (i, player) in players.iter().enumerate() {
            let mut value = 0.0;
            
            // Iterate over all possible coalitions
            for mask in 0..(1 << n) {
                if mask & (1 << i) != 0 {
                    continue; // Player already in coalition
                }
                
                let coalition = Self::mask_to_coalition(&players, mask);
                let mut coalition_with_player = coalition.clone();
                coalition_with_player.push(player.clone());
                
                let marginal = value_fn.value(&coalition_with_player) - value_fn.value(&coalition);
                let coalition_size = coalition.len();
                let weight = Self::shapley_weight(coalition_size, n);
                
                value += weight * marginal;
            }
            
            shapley_values.insert(player.clone(), value);
        }
        
        self.create_result(shapley_values)
    }
    
    /// Calculate Shapley values using Monte Carlo approximation
    pub fn calculate_monte_carlo(&self, value_fn: &dyn ValueFunction) -> Result<ShapleyResult> {
        let players = value_fn.players();
        let n = players.len();
        
        if n == 0 {
            return Ok(ShapleyResult {
                values: HashMap::new(),
                importance_scores: HashMap::new(),
                ranking: Vec::new(),
                confidence_intervals: None,
            });
        }
        
        // Run Monte Carlo samples
        let samples = if self.parallel {
            (0..self.n_samples)
                .into_par_iter()
                .map(|_| self.monte_carlo_sample(&players, value_fn))
                .collect::<Vec<_>>()
        } else {
            (0..self.n_samples)
                .map(|_| self.monte_carlo_sample(&players, value_fn))
                .collect::<Vec<_>>()
        };
        
        // Aggregate results
        let mut shapley_values = HashMap::new();
        let mut value_samples: HashMap<String, Vec<f64>> = HashMap::new();
        
        for player in &players {
            shapley_values.insert(player.clone(), 0.0);
            value_samples.insert(player.clone(), Vec::new());
        }
        
        for sample in samples {
            for (player, value) in sample {
                *shapley_values.get_mut(&player).unwrap() += value;
                value_samples.get_mut(&player).unwrap().push(value);
            }
        }
        
        // Average values
        for value in shapley_values.values_mut() {
            *value /= self.n_samples as f64;
        }
        
        // Calculate confidence intervals
        let confidence_intervals = self.calculate_confidence_intervals(&value_samples);
        
        let mut result = self.create_result(shapley_values)?;
        result.confidence_intervals = Some(confidence_intervals);
        
        Ok(result)
    }
    
    /// Perform one Monte Carlo sample
    fn monte_carlo_sample(
        &self,
        players: &[String],
        value_fn: &dyn ValueFunction,
    ) -> HashMap<String, f64> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        // Random permutation
        let mut perm = players.to_vec();
        perm.shuffle(&mut rng);
        
        let mut marginal_contributions = HashMap::new();
        let mut coalition = Vec::new();
        
        for player in perm {
            // Calculate marginal contribution
            let value_without = value_fn.value(&coalition);
            coalition.push(player.clone());
            let value_with = value_fn.value(&coalition);
            
            marginal_contributions.insert(player, value_with - value_without);
        }
        
        marginal_contributions
    }
    
    /// Calculate Shapley weight for a coalition size
    fn shapley_weight(coalition_size: usize, total_players: usize) -> f64 {
        let s = coalition_size as f64;
        let n = total_players as f64;
        
        // Weight = |S|!(n-|S|-1)!/n!
        let factorial = |x: f64| -> f64 {
            if x <= 1.0 { 1.0 } else { (1..=x as u64).product::<u64>() as f64 }
        };
        
        factorial(s) * factorial(n - s - 1.0) / factorial(n)
    }
    
    /// Convert bit mask to coalition
    fn mask_to_coalition(players: &[String], mask: usize) -> Vec<String> {
        let mut coalition = Vec::new();
        
        for (i, player) in players.iter().enumerate() {
            if mask & (1 << i) != 0 {
                coalition.push(player.clone());
            }
        }
        
        coalition
    }
    
    /// Create result from Shapley values
    fn create_result(&self, shapley_values: HashMap<String, f64>) -> Result<ShapleyResult> {
        // Normalize to importance scores
        let max_value = shapley_values.values()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(1.0)
            .max(f64::EPSILON);
        
        let importance_scores: HashMap<String, f64> = shapley_values.iter()
            .map(|(k, v)| (k.clone(), v / max_value))
            .collect();
        
        // Create ranking
        let mut ranking: Vec<(String, f64)> = shapley_values.into_iter().collect();
        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(ShapleyResult {
            values: ranking.iter().cloned().collect(),
            importance_scores,
            ranking,
            confidence_intervals: None,
        })
    }
    
    /// Calculate confidence intervals for Monte Carlo estimates
    fn calculate_confidence_intervals(
        &self,
        samples: &HashMap<String, Vec<f64>>,
    ) -> HashMap<String, (f64, f64)> {
        use statrs::statistics::{Statistics, OrderStatistics};
        
        let mut intervals = HashMap::new();
        
        for (player, values) in samples {
            if values.is_empty() {
                intervals.insert(player.clone(), (0.0, 0.0));
                continue;
            }
            
            let data = Array1::from_vec(values.clone());
            let mean = data.clone().mean();
            let std_dev = data.std_dev();
            
            // 95% confidence interval
            let z = 1.96; // z-score for 95% CI
            let margin = z * std_dev / (values.len() as f64).sqrt();
            
            intervals.insert(player.clone(), (mean - margin, mean + margin));
        }
        
        intervals
    }
}

/// Helper to create a coverage-based value function from test data
pub fn create_coverage_value_function(
    test_coverage: HashMap<String, Vec<u32>>,
) -> CoverageValueFunction {
    let mut all_lines: HashSet<u32> = HashSet::new();
    let mut coverage_sets = HashMap::new();
    
    for (test, lines) in test_coverage {
        let line_set: HashSet<u32> = lines.into_iter().collect();
        all_lines.extend(&line_set);
        coverage_sets.insert(test, line_set);
    }
    
    CoverageValueFunction::new(coverage_sets, all_lines.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shapley_exact_calculation() {
        let mut coverage_data = HashMap::new();
        coverage_data.insert("test1".to_string(), vec![1, 2, 3].into_iter().collect());
        coverage_data.insert("test2".to_string(), vec![2, 3, 4].into_iter().collect());
        coverage_data.insert("test3".to_string(), vec![4, 5].into_iter().collect());
        
        let value_fn = CoverageValueFunction::new(coverage_data, 5);
        let calculator = ShapleyCalculator::new(100);
        
        let result = calculator.calculate_exact(&value_fn).unwrap();
        
        assert_eq!(result.values.len(), 3);
        assert!(result.values.contains_key("test1"));
        assert!(result.values.contains_key("test2"));
        assert!(result.values.contains_key("test3"));
        
        // Check that values sum to total value
        let total_value = value_fn.value(&["test1".to_string(), "test2".to_string(), "test3".to_string()]);
        let sum_shapley: f64 = result.values.values().sum();
        assert!((sum_shapley - total_value).abs() < 0.001);
    }
    
    #[test]
    fn test_monte_carlo_approximation() {
        let test_coverage = HashMap::from([
            ("test1".to_string(), vec![1, 2, 3]),
            ("test2".to_string(), vec![3, 4, 5]),
            ("test3".to_string(), vec![5, 6, 7]),
            ("test4".to_string(), vec![7, 8, 9]),
        ]);
        
        let value_fn = create_coverage_value_function(test_coverage);
        let calculator = ShapleyCalculator::new(500);
        
        let result = calculator.calculate_monte_carlo(&value_fn).unwrap();
        
        assert_eq!(result.values.len(), 4);
        assert!(result.confidence_intervals.is_some());
        assert_eq!(result.ranking.len(), 4);
        
        // Check ranking is sorted
        for i in 1..result.ranking.len() {
            assert!(result.ranking[i - 1].1 >= result.ranking[i].1);
        }
    }
}