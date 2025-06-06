//! Adaptive Evolutionary Mutation Testing (EMT) Engine

use crate::types::*;
use crate::fitness::{TestSuite, TestSuiteFitness, CombinedFitness, TestMetadata, TestCase};
use crate::operators::{TestSuiteMutator, TestSuiteCrossover};
use crate::nsga2::NSGA2;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Adaptive EMT engine for test suite optimization
pub struct AdaptiveEMT {
    /// Evolution configuration
    config: EvolutionConfig,
    /// Test metadata for fitness calculation
    test_data: HashMap<String, TestMetadata>,
    /// Project path
    project_path: String,
}

/// EMT result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMTResult {
    /// Optimized test suites (Pareto front)
    pub optimized_suites: Vec<OptimizedTestSuite>,
    /// Evolution statistics
    pub evolution_stats: EvolutionStats,
    /// Improvement metrics
    pub improvements: ImprovementMetrics,
}

/// An optimized test suite with its metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedTestSuite {
    /// Test IDs in the suite
    pub test_ids: Vec<String>,
    /// Fitness scores
    pub fitness_scores: HashMap<String, f64>,
    /// Quality metrics
    pub metrics: TestSuiteMetrics,
}

/// Test suite metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteMetrics {
    /// Code coverage percentage
    pub coverage: f64,
    /// Total execution time in ms
    pub execution_time_ms: f64,
    /// Average mutation score
    pub mutation_score: f64,
    /// Suite stability (1 - flakiness)
    pub stability: f64,
    /// Number of tests
    pub test_count: usize,
}

/// Evolution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionStats {
    /// Number of generations completed
    pub generations_completed: usize,
    /// Final Pareto front size
    pub pareto_front_size: usize,
    /// Best fitness values achieved
    pub best_fitness: HashMap<String, f64>,
    /// Generation history
    pub history: Vec<GenerationStats>,
}

/// Improvement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    /// Coverage improvement percentage
    pub coverage_improvement: f64,
    /// Speed improvement percentage
    pub speed_improvement: f64,
    /// Mutation score improvement
    pub mutation_score_improvement: f64,
    /// Test count reduction percentage
    pub test_reduction: f64,
}


impl AdaptiveEMT {
    /// Create a new adaptive EMT engine
    pub fn new(project_path: String) -> Self {
        Self {
            config: EvolutionConfig::default(),
            test_data: HashMap::new(),
            project_path,
        }
    }
    
    /// Configure evolution parameters
    pub fn with_config(mut self, config: EvolutionConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Load test metadata
    pub fn with_test_data(mut self, test_data: HashMap<String, TestMetadata>) -> Self {
        self.test_data = test_data;
        self
    }
    
    /// Collect test metadata from the project
    pub async fn collect_test_metadata(&mut self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Run tests with coverage
        // 2. Measure execution times
        // 3. Run mutation testing
        // 4. Analyze historical CI data
        
        // For now, generate sample data
        self.test_data = self.generate_sample_metadata();
        Ok(())
    }
    
    /// Generate sample test metadata for demonstration
    fn generate_sample_metadata(&self) -> HashMap<String, TestMetadata> {
        let mut metadata = HashMap::new();
        
        // Sample test data
        for i in 1..=20 {
            metadata.insert(
                format!("test_{}", i),
                TestMetadata {
                    execution_time_ms: (50.0 + (i as f64) * 10.0) % 200.0,
                    covered_lines: ((i * 10)..((i + 1) * 10)).collect(),
                    mutation_score: 0.5 + (i as f64 % 5.0) / 10.0,
                    flakiness_rate: if i % 7 == 0 { 0.1 } else { 0.0 },
                },
            );
        }
        
        metadata
    }
    
    /// Run the evolutionary optimization
    pub async fn optimize(&self) -> Result<EMTResult> {
        // Get all available tests
        let available_tests: Vec<String> = self.test_data.keys().cloned().collect();
        let total_lines = self.estimate_total_lines();
        
        // Create fitness function
        let fitness = Box::new(TestSuiteFitness::new(
            self.test_data.clone(),
            total_lines,
        ));
        
        // Create genetic operators
        let mutator = Box::new(TestSuiteMutator::new(available_tests.clone()));
        let crossover = Box::new(TestSuiteCrossover);
        
        // Initialize NSGA-II
        let nsga2 = NSGA2::new(
            self.config.clone(),
            fitness,
            mutator,
            crossover,
        );
        
        // Run evolution
        let evolution_result = nsga2.evolve();
        
        // Process results
        let optimized_suites = self.process_pareto_front(&evolution_result.pareto_front);
        let evolution_stats = self.calculate_evolution_stats(&evolution_result);
        let improvements = self.calculate_improvements(&optimized_suites);
        
        Ok(EMTResult {
            optimized_suites,
            evolution_stats,
            improvements,
        })
    }
    
    /// Estimate total lines in codebase
    fn estimate_total_lines(&self) -> usize {
        // Sum of all covered lines as estimate
        self.test_data.values()
            .flat_map(|meta| &meta.covered_lines)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
    
    /// Process Pareto front into optimized test suites
    fn process_pareto_front(&self, pareto_front: &Population<TestSuite>) -> Vec<OptimizedTestSuite> {
        pareto_front.iter()
            .map(|individual| {
                let suite = &individual.genome;
                let metrics = self.calculate_suite_metrics(suite);
                
                let mut fitness_scores = HashMap::new();
                fitness_scores.insert("Coverage".to_string(), individual.fitness[0]);
                fitness_scores.insert("Speed".to_string(), individual.fitness[1]);
                fitness_scores.insert("Mutation Score".to_string(), individual.fitness[2]);
                fitness_scores.insert("Stability".to_string(), individual.fitness[3]);
                
                OptimizedTestSuite {
                    test_ids: suite.test_cases.iter().map(|tc| tc.name.clone()).collect(),
                    fitness_scores,
                    metrics,
                }
            })
            .collect()
    }
    
    /// Calculate metrics for a test suite
    fn calculate_suite_metrics(&self, suite: &TestSuite) -> TestSuiteMetrics {
        let mut covered_lines: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut total_time = 0.0;
        let mut mutation_scores = Vec::new();
        let mut flakiness_rates = Vec::new();
        
        for test_case in &suite.test_cases {
            if let Some(metadata) = self.test_data.get(&test_case.name) {
                covered_lines.extend(&metadata.covered_lines);
                total_time += metadata.execution_time_ms;
                mutation_scores.push(metadata.mutation_score);
                flakiness_rates.push(metadata.flakiness_rate);
            }
        }
        
        let coverage = covered_lines.len() as f64 / self.estimate_total_lines() as f64;
        let avg_mutation = if mutation_scores.is_empty() {
            0.0
        } else {
            mutation_scores.iter().sum::<f64>() / mutation_scores.len() as f64
        };
        
        let avg_flakiness = if flakiness_rates.is_empty() {
            0.0
        } else {
            flakiness_rates.iter().sum::<f64>() / flakiness_rates.len() as f64
        };
        
        TestSuiteMetrics {
            coverage,
            execution_time_ms: total_time,
            mutation_score: avg_mutation,
            stability: 1.0 - avg_flakiness,
            test_count: suite.test_cases.len(),
        }
    }
    
    /// Calculate evolution statistics
    fn calculate_evolution_stats(&self, result: &EvolutionResult<TestSuite>) -> EvolutionStats {
        let mut best_fitness = HashMap::new();
        best_fitness.insert("Coverage".to_string(), result.best_per_objective[0].fitness[0]);
        best_fitness.insert("Speed".to_string(), result.best_per_objective[1].fitness[1]);
        best_fitness.insert("Mutation Score".to_string(), result.best_per_objective[2].fitness[2]);
        best_fitness.insert("Stability".to_string(), result.best_per_objective[3].fitness[3]);
        
        EvolutionStats {
            generations_completed: self.config.generations,
            pareto_front_size: result.pareto_front.len(),
            best_fitness,
            history: result.history.clone(),
        }
    }
    
    /// Calculate improvements over baseline
    fn calculate_improvements(&self, optimized_suites: &[OptimizedTestSuite]) -> ImprovementMetrics {
        // Baseline: all tests
        let all_tests = TestSuite {
            id: 0,
            test_cases: self.test_data.keys().map(|name| TestCase {
                name: name.clone(),
                content: "".to_string(),
                assertions: 1,
            }).collect(),
            fitness_values: vec![],
            rank: None,
            crowding_distance: 0.0,
        };
        let baseline_metrics = self.calculate_suite_metrics(&all_tests);
        
        // Find best suite for each metric
        let best_coverage = optimized_suites.iter()
            .map(|s| s.metrics.coverage)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        let best_speed = optimized_suites.iter()
            .map(|s| s.metrics.execution_time_ms)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f64::MAX);
        
        let best_mutation = optimized_suites.iter()
            .map(|s| s.metrics.mutation_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        let avg_test_count = optimized_suites.iter()
            .map(|s| s.metrics.test_count)
            .sum::<usize>() as f64 / optimized_suites.len() as f64;
        
        ImprovementMetrics {
            coverage_improvement: ((best_coverage - baseline_metrics.coverage) / baseline_metrics.coverage) * 100.0,
            speed_improvement: ((baseline_metrics.execution_time_ms - best_speed) / baseline_metrics.execution_time_ms) * 100.0,
            mutation_score_improvement: ((best_mutation - baseline_metrics.mutation_score) / baseline_metrics.mutation_score) * 100.0,
            test_reduction: ((baseline_metrics.test_count as f64 - avg_test_count) / baseline_metrics.test_count as f64) * 100.0,
        }
    }
}

/// Builder for AdaptiveEMT
pub struct EMTBuilder {
    project_path: String,
    config: Option<EvolutionConfig>,
    test_data: Option<HashMap<String, TestMetadata>>,
}

impl EMTBuilder {
    pub fn new(project_path: impl Into<String>) -> Self {
        Self {
            project_path: project_path.into(),
            config: None,
            test_data: None,
        }
    }
    
    pub fn config(mut self, config: EvolutionConfig) -> Self {
        self.config = Some(config);
        self
    }
    
    pub fn test_data(mut self, data: HashMap<String, TestMetadata>) -> Self {
        self.test_data = Some(data);
        self
    }
    
    pub fn build(self) -> AdaptiveEMT {
        let mut emt = AdaptiveEMT::new(self.project_path);
        
        if let Some(config) = self.config {
            emt = emt.with_config(config);
        }
        
        if let Some(data) = self.test_data {
            emt = emt.with_test_data(data);
        }
        
        emt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_adaptive_emt() {
        let mut emt = AdaptiveEMT::new("/tmp/test_project".to_string());
        emt.collect_test_metadata().await.unwrap();
        
        let config = EvolutionConfig {
            population_size: 20,
            generations: 5,
            ..Default::default()
        };
        
        let emt = emt.with_config(config);
        let result = emt.optimize().await.unwrap();
        
        assert!(!result.optimized_suites.is_empty());
        assert!(result.evolution_stats.generations_completed == 5);
        assert!(result.optimized_suites.len() <= 20); // At most population size
    }
}