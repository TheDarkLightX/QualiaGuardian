//! Core types for evolutionary algorithms

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// An individual in the evolutionary population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual<T> {
    /// The genetic representation (e.g., test suite)
    pub genome: T,
    /// Fitness values for multiple objectives
    pub fitness: Vec<f64>,
    /// Crowding distance for NSGA-II
    pub crowding_distance: f64,
    /// Domination rank
    pub rank: usize,
}

impl<T> Individual<T> {
    /// Create a new individual
    pub fn new(genome: T) -> Self {
        Self {
            genome,
            fitness: Vec::new(),
            crowding_distance: 0.0,
            rank: 0,
        }
    }
    
    /// Check if this individual dominates another
    pub fn dominates(&self, other: &Self) -> bool {
        if self.fitness.len() != other.fitness.len() {
            return false;
        }
        
        let mut at_least_one_better = false;
        
        for (a, b) in self.fitness.iter().zip(&other.fitness) {
            if a < b {
                return false; // Worse in at least one objective
            }
            if a > b {
                at_least_one_better = true;
            }
        }
        
        at_least_one_better
    }
}

/// A population of individuals
pub type Population<T> = Vec<Individual<T>>;

/// Selection strategy for evolutionary algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Tournament selection with given size
    Tournament(usize),
    /// Roulette wheel selection
    RouletteWheel,
    /// Rank-based selection
    RankBased,
}

/// Crossover strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CrossoverStrategy {
    /// Single-point crossover
    SinglePoint,
    /// Two-point crossover
    TwoPoint,
    /// Uniform crossover with given probability
    Uniform(f64),
}

/// Evolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Mutation probability
    pub mutation_rate: f64,
    /// Selection strategy
    pub selection: SelectionStrategy,
    /// Crossover strategy
    pub crossover: CrossoverStrategy,
    /// Number of objectives
    pub num_objectives: usize,
    /// Elitism size (preserve best individuals)
    pub elitism: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Elitism size (number of best individuals to preserve)
    pub elitism_size: usize,
    /// Whether to use parallel evaluation
    pub parallel: bool,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 50,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            selection: SelectionStrategy::Tournament(2),
            crossover: CrossoverStrategy::SinglePoint,
            num_objectives: 2,
            elitism: 10,
            tournament_size: 3,
            elitism_size: 2,
            parallel: true,
        }
    }
}

/// Result of an evolution run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult<T> {
    /// Final population
    pub final_population: Population<T>,
    /// Pareto front (non-dominated solutions)
    pub pareto_front: Population<T>,
    /// Best individual for each objective
    pub best_per_objective: Vec<Individual<T>>,
    /// Evolution history (fitness over generations)
    pub history: Vec<GenerationStats>,
}

/// Statistics for a single generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    /// Generation number
    pub generation: usize,
    /// Average fitness for each objective
    pub avg_fitness: Vec<f64>,
    /// Best fitness for each objective
    pub best_fitness: Vec<f64>,
    /// Size of Pareto front
    pub pareto_size: usize,
}

/// Trait for types that can be evolved
pub trait Evolvable: Clone + Send + Sync {
    /// Create a random individual
    fn random() -> Self;
    
    /// Get the size/length of the genome
    fn len(&self) -> usize;
    
    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for fitness evaluation
pub trait FitnessFunction<T>: Send + Sync {
    /// Evaluate fitness for all objectives
    fn evaluate(&self, individual: &T) -> Vec<f64>;
    
    /// Get number of objectives
    fn num_objectives(&self) -> usize;
    
    /// Get objective names for reporting
    fn objective_names(&self) -> Vec<&'static str>;
}

/// Trait for mutation operators
pub trait MutationOperator<T>: Send + Sync {
    /// Mutate an individual
    fn mutate(&self, individual: &mut T, rate: f64);
}

/// Trait for crossover operators
pub trait CrossoverOperator<T>: Send + Sync {
    /// Perform crossover between two parents
    fn crossover(&self, parent1: &T, parent2: &T) -> (T, T);
}