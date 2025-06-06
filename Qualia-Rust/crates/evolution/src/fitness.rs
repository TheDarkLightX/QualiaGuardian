//! Fitness functions for test suite optimization

use crate::types::FitnessFunction;
use qualia_core::QualityScore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Test suite representation for evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    /// Unique identifier for this suite
    pub id: usize,
    /// Test cases in this suite
    pub test_cases: Vec<TestCase>,
    /// Fitness values for multiple objectives
    pub fitness_values: Vec<f64>,
    /// Domination rank (for NSGA-II)
    pub rank: Option<usize>,
    /// Crowding distance (for NSGA-II)
    pub crowding_distance: f64,
}

/// Individual test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Test name/identifier
    pub name: String,
    /// Test content/code
    pub content: String,
    /// Number of assertions
    pub assertions: usize,
}

impl crate::types::Evolvable for TestSuite {
    fn random() -> Self {
        // For now, return an empty suite
        Self {
            id: 0,
            test_cases: Vec::new(),
            fitness_values: Vec::new(),
            rank: None,
            crowding_distance: 0.0,
        }
    }
    
    fn len(&self) -> usize {
        self.test_cases.len()
    }
}

/// Metadata for individual tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetadata {
    /// Test execution time in milliseconds
    pub execution_time_ms: f64,
    /// Lines covered by this test
    pub covered_lines: Vec<u32>,
    /// Mutation score for this test
    pub mutation_score: f64,
    /// Historical flakiness rate
    pub flakiness_rate: f64,
}

/// Multi-objective fitness function for test suites
#[derive(Debug)]
pub struct TestSuiteFitness {
    /// Test metadata for fitness calculation
    test_data: HashMap<String, TestMetadata>,
    /// Total lines in the codebase
    total_lines: usize,
}

impl TestSuiteFitness {
    pub fn new(test_data: HashMap<String, TestMetadata>, total_lines: usize) -> Self {
        Self { test_data, total_lines }
    }
    
    /// Calculate coverage ratio for a test suite
    fn calculate_coverage(&self, suite: &TestSuite) -> f64 {
        let mut covered_lines: std::collections::HashSet<u32> = std::collections::HashSet::new();
        
        for test_case in &suite.test_cases {
            if let Some(metadata) = self.test_data.get(&test_case.name) {
                covered_lines.extend(&metadata.covered_lines);
            }
        }
        
        if self.total_lines > 0 {
            covered_lines.len() as f64 / self.total_lines as f64
        } else {
            0.0
        }
    }
    
    /// Calculate total execution time
    fn calculate_execution_time(&self, suite: &TestSuite) -> f64 {
        suite.test_cases.iter()
            .filter_map(|tc| self.test_data.get(&tc.name))
            .map(|metadata| metadata.execution_time_ms)
            .sum()
    }
    
    /// Calculate average mutation score
    fn calculate_mutation_score(&self, suite: &TestSuite) -> f64 {
        let scores: Vec<f64> = suite.test_cases.iter()
            .filter_map(|tc| self.test_data.get(&tc.name))
            .map(|metadata| metadata.mutation_score)
            .collect();
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        }
    }
    
    /// Calculate suite stability (inverse of flakiness)
    fn calculate_stability(&self, suite: &TestSuite) -> f64 {
        let flakiness_rates: Vec<f64> = suite.test_cases.iter()
            .filter_map(|tc| self.test_data.get(&tc.name))
            .map(|metadata| metadata.flakiness_rate)
            .collect();
        
        if flakiness_rates.is_empty() {
            1.0
        } else {
            let avg_flakiness = flakiness_rates.iter().sum::<f64>() / flakiness_rates.len() as f64;
            1.0 - avg_flakiness
        }
    }
}

impl FitnessFunction<TestSuite> for TestSuiteFitness {
    fn evaluate(&self, suite: &TestSuite) -> Vec<f64> {
        // Multi-objective optimization:
        // 1. Maximize coverage (positive)
        // 2. Minimize execution time (convert to maximization)
        // 3. Maximize mutation score (positive)
        // 4. Maximize stability (positive)
        
        let coverage = self.calculate_coverage(suite);
        let exec_time = self.calculate_execution_time(suite);
        let mutation_score = self.calculate_mutation_score(suite);
        let stability = self.calculate_stability(suite);
        
        // Convert execution time to maximization objective
        // Using inverse with small epsilon to avoid division by zero
        let speed_fitness = 1.0 / (exec_time + 1.0);
        
        vec![coverage, speed_fitness, mutation_score, stability]
    }
    
    fn num_objectives(&self) -> usize {
        4
    }
    
    fn objective_names(&self) -> Vec<&'static str> {
        vec!["Coverage", "Speed", "Mutation Score", "Stability"]
    }
}

/// Quality-based fitness function using Guardian metrics
pub struct QualityFitness {
    /// Function to calculate quality score
    quality_fn: Box<dyn Fn(&TestSuite) -> QualityScore + Send + Sync>,
}

impl std::fmt::Debug for QualityFitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QualityFitness")
            .field("quality_fn", &"<quality function>")
            .finish()
    }
}

impl QualityFitness {
    pub fn new<F>(quality_fn: F) -> Self
    where
        F: Fn(&TestSuite) -> QualityScore + Send + Sync + 'static,
    {
        Self {
            quality_fn: Box::new(quality_fn),
        }
    }
}

impl FitnessFunction<TestSuite> for QualityFitness {
    fn evaluate(&self, suite: &TestSuite) -> Vec<f64> {
        let quality_score = (self.quality_fn)(suite);
        vec![quality_score.value()]
    }
    
    fn num_objectives(&self) -> usize {
        1
    }
    
    fn objective_names(&self) -> Vec<&'static str> {
        vec!["Quality Score"]
    }
}

/// Combined fitness function that uses multiple objectives
pub struct CombinedFitness {
    functions: Vec<Box<dyn FitnessFunction<TestSuite>>>,
}

impl CombinedFitness {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }
    
    pub fn add_function(mut self, function: Box<dyn FitnessFunction<TestSuite>>) -> Self {
        self.functions.push(function);
        self
    }
}

impl FitnessFunction<TestSuite> for CombinedFitness {
    fn evaluate(&self, suite: &TestSuite) -> Vec<f64> {
        let mut all_scores = Vec::new();
        
        for function in &self.functions {
            all_scores.extend(function.evaluate(suite));
        }
        
        all_scores
    }
    
    fn num_objectives(&self) -> usize {
        self.functions.iter()
            .map(|f| f.num_objectives())
            .sum()
    }
    
    fn objective_names(&self) -> Vec<&'static str> {
        self.functions.iter()
            .flat_map(|f| f.objective_names())
            .collect()
    }
}

/// Mutation score fitness function using actual sensor
#[derive(Debug)]
pub struct MutationScoreFitness<S> {
    sensor: S,
}

impl<S> MutationScoreFitness<S> {
    pub fn new(sensor: S) -> Self {
        Self { sensor }
    }
}

impl<S> FitnessFunction<TestSuite> for MutationScoreFitness<S>
where
    S: qualia_sensors::Sensor + Send + Sync,
{
    fn evaluate(&self, _suite: &TestSuite) -> Vec<f64> {
        // In real implementation, this would run mutation testing on the suite
        // For now, return a placeholder based on test count
        vec![0.5 + (_suite.test_cases.len() as f64 * 0.01).min(0.4)]
    }
    
    fn num_objectives(&self) -> usize {
        1
    }
    
    fn objective_names(&self) -> Vec<&'static str> {
        vec!["Mutation Score"]
    }
}

/// Speed fitness function using actual sensor
#[derive(Debug)]
pub struct SpeedFitness<S> {
    sensor: S,
}

impl<S> SpeedFitness<S> {
    pub fn new(sensor: S) -> Self {
        Self { sensor }
    }
}

impl<S> FitnessFunction<TestSuite> for SpeedFitness<S>
where
    S: qualia_sensors::Sensor + Send + Sync,
{
    fn evaluate(&self, suite: &TestSuite) -> Vec<f64> {
        // In real implementation, this would measure actual execution time
        // For now, return inverse of test count as proxy for speed
        vec![1.0 / (suite.test_cases.len() as f64 + 1.0)]
    }
    
    fn num_objectives(&self) -> usize {
        1
    }
    
    fn objective_names(&self) -> Vec<&'static str> {
        vec!["Execution Speed"]
    }
}

/// Composite fitness function
pub struct CompositeFitness {
    functions: Vec<Box<dyn FitnessFunction<TestSuite>>>,
}

impl CompositeFitness {
    pub fn new(functions: Vec<Box<dyn FitnessFunction<TestSuite>>>) -> Self {
        Self { functions }
    }
}

impl FitnessFunction<TestSuite> for CompositeFitness {
    fn evaluate(&self, suite: &TestSuite) -> Vec<f64> {
        let mut all_scores = Vec::new();
        
        for function in &self.functions {
            all_scores.extend(function.evaluate(suite));
        }
        
        all_scores
    }
    
    fn num_objectives(&self) -> usize {
        self.functions.iter()
            .map(|f| f.num_objectives())
            .sum()
    }
    
    fn objective_names(&self) -> Vec<&'static str> {
        self.functions.iter()
            .flat_map(|f| f.objective_names())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_test_suite_fitness() {
        let mut test_data = HashMap::new();
        test_data.insert("test1".to_string(), TestMetadata {
            execution_time_ms: 100.0,
            covered_lines: vec![1, 2, 3],
            mutation_score: 0.8,
            flakiness_rate: 0.1,
        });
        test_data.insert("test2".to_string(), TestMetadata {
            execution_time_ms: 200.0,
            covered_lines: vec![3, 4, 5],
            mutation_score: 0.9,
            flakiness_rate: 0.0,
        });
        
        let fitness = TestSuiteFitness::new(test_data, 10);
        let suite = TestSuite {
            id: 0,
            test_cases: vec![
                TestCase { name: "test1".to_string(), content: "".to_string(), assertions: 1 },
                TestCase { name: "test2".to_string(), content: "".to_string(), assertions: 2 },
            ],
            fitness_values: vec![],
            rank: None,
            crowding_distance: 0.0,
        };
        
        let scores = fitness.evaluate(&suite);
        assert_eq!(scores.len(), 4);
        
        // Coverage should be 5/10 = 0.5
        assert!((scores[0] - 0.5).abs() < 0.001);
        
        // Execution time is 300ms, speed fitness = 1/(300+1) ≈ 0.00332
        assert!(scores[1] > 0.0 && scores[1] < 0.01);
        
        // Mutation score average = (0.8 + 0.9) / 2 = 0.85
        assert!((scores[2] - 0.85).abs() < 0.001);
        
        // Stability = 1 - avg_flakiness = 1 - 0.05 = 0.95
        assert!((scores[3] - 0.95).abs() < 0.001);
    }
}