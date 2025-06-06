//! NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation

use crate::types::*;
use ordered_float::OrderedFloat;
use rand::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;

/// NSGA-II algorithm implementation
pub struct NSGA2<T> {
    config: EvolutionConfig,
    fitness_fn: Box<dyn FitnessFunction<T>>,
    mutation_op: Box<dyn MutationOperator<T>>,
    crossover_op: Box<dyn CrossoverOperator<T>>,
}

impl<T> NSGA2<T>
where
    T: Evolvable + Send + Sync,
{
    /// Create a new NSGA-II instance
    pub fn new(
        config: EvolutionConfig,
        fitness_fn: Box<dyn FitnessFunction<T>>,
        mutation_op: Box<dyn MutationOperator<T>>,
        crossover_op: Box<dyn CrossoverOperator<T>>,
    ) -> Self {
        Self {
            config,
            fitness_fn,
            mutation_op,
            crossover_op,
        }
    }
    
    /// Run the evolutionary algorithm
    pub fn evolve(&self) -> EvolutionResult<T> {
        let mut population = self.initialize_population();
        let mut history = Vec::new();
        
        for generation in 0..self.config.generations {
            // Evaluate fitness
            self.evaluate_population(&mut population);
            
            // Non-dominated sorting
            let fronts = self.fast_non_dominated_sort(&population);
            
            // Assign crowding distance
            for front in &fronts {
                self.assign_crowding_distance(&mut population, front);
            }
            
            // Record statistics
            let stats = self.calculate_generation_stats(generation, &population, &fronts);
            history.push(stats);
            
            // Create offspring
            let offspring = self.create_offspring(&population);
            
            // Combine populations
            let mut combined = population.clone();
            combined.extend(offspring);
            
            // Select next generation
            population = self.environmental_selection(combined, &fronts);
        }
        
        // Final evaluation
        self.evaluate_population(&mut population);
        let fronts = self.fast_non_dominated_sort(&population);
        
        // Extract Pareto front
        let pareto_front = if !fronts.is_empty() {
            fronts[0].iter()
                .map(|&idx| population[idx].clone())
                .collect()
        } else {
            Vec::new()
        };
        
        // Find best per objective
        let best_per_objective = self.find_best_per_objective(&population);
        
        EvolutionResult {
            final_population: population,
            pareto_front,
            best_per_objective,
            history,
        }
    }
    
    /// Run the evolutionary algorithm with initial population
    pub async fn evolve_with_initial(&mut self, initial_population: Vec<T>) -> anyhow::Result<Vec<T>> {
        let mut population: Population<T> = initial_population.into_iter()
            .map(Individual::new)
            .collect();
        
        for generation in 0..self.config.generations {
            // Evaluate fitness
            self.evaluate_population(&mut population);
            
            // Non-dominated sorting
            let fronts = self.fast_non_dominated_sort(&population);
            
            // Assign crowding distance
            for front in &fronts {
                self.assign_crowding_distance(&mut population, front);
            }
            
            // Create offspring
            let offspring = self.create_offspring(&population);
            
            // Combine populations
            let mut combined = population.clone();
            combined.extend(offspring);
            
            // Select next generation
            population = self.environmental_selection(combined, &fronts);
        }
        
        // Return final population genomes
        Ok(population.into_iter().map(|ind| ind.genome).collect())
    }
    
    /// Run the evolutionary algorithm with a callback
    pub async fn evolve_with_callback<F>(&mut self, initial_population: Vec<T>, mut callback: F) -> anyhow::Result<Vec<T>>
    where
        F: FnMut(usize, &[T]),
    {
        let mut population: Population<T> = initial_population.into_iter()
            .map(Individual::new)
            .collect();
        
        for generation in 0..self.config.generations {
            // Evaluate fitness
            self.evaluate_population(&mut population);
            
            // Non-dominated sorting
            let fronts = self.fast_non_dominated_sort(&population);
            
            // Assign crowding distance
            for front in &fronts {
                self.assign_crowding_distance(&mut population, front);
            }
            
            // Call the callback with current generation info
            let genomes: Vec<T> = population.iter().map(|ind| ind.genome.clone()).collect();
            callback(generation, &genomes);
            
            // Create offspring
            let offspring = self.create_offspring(&population);
            
            // Combine populations
            let mut combined = population.clone();
            combined.extend(offspring);
            
            // Select next generation
            population = self.environmental_selection(combined, &fronts);
        }
        
        // Final evaluation
        self.evaluate_population(&mut population);
        
        // Return final population genomes
        Ok(population.into_iter().map(|ind| ind.genome).collect())
    }
    
    /// Initialize random population
    fn initialize_population(&self) -> Population<T> {
        (0..self.config.population_size)
            .map(|_| Individual::new(T::random()))
            .collect()
    }
    
    /// Evaluate fitness for entire population
    fn evaluate_population(&self, population: &mut Population<T>) {
        // Parallel fitness evaluation
        population.par_iter_mut().for_each(|individual| {
            if individual.fitness.is_empty() {
                individual.fitness = self.fitness_fn.evaluate(&individual.genome);
            }
        });
    }
    
    /// Fast non-dominated sorting algorithm
    fn fast_non_dominated_sort(&self, population: &Population<T>) -> Vec<Vec<usize>> {
        let n = population.len();
        let mut fronts = vec![Vec::new()];
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];
        
        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if population[i].dominates(&population[j]) {
                        dominated_solutions[i].push(j);
                    } else if population[j].dominates(&population[i]) {
                        domination_count[i] += 1;
                    }
                }
            }
            
            // First front (non-dominated solutions)
            if domination_count[i] == 0 {
                fronts[0].push(i);
            }
        }
        
        // Find remaining fronts
        let mut current_front = 0;
        while !fronts[current_front].is_empty() {
            let mut next_front = Vec::new();
            
            for &i in &fronts[current_front] {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            
            if !next_front.is_empty() {
                fronts.push(next_front);
            }
            current_front += 1;
        }
        
        fronts
    }
    
    /// Assign crowding distance to individuals in a front
    fn assign_crowding_distance(&self, population: &mut Population<T>, front: &[usize]) {
        let n = front.len();
        if n == 0 {
            return;
        }
        
        // Reset distances
        for &idx in front {
            population[idx].crowding_distance = 0.0;
        }
        
        let num_objectives = self.fitness_fn.num_objectives();
        
        // For each objective
        for obj in 0..num_objectives {
            // Sort by objective
            let mut sorted_indices: Vec<_> = front.to_vec();
            sorted_indices.sort_by(|&a, &b| {
                OrderedFloat(population[a].fitness[obj])
                    .cmp(&OrderedFloat(population[b].fitness[obj]))
            });
            
            // Boundary points get infinite distance
            population[sorted_indices[0]].crowding_distance = f64::INFINITY;
            population[sorted_indices[n - 1]].crowding_distance = f64::INFINITY;
            
            // Calculate range
            let min_val = population[sorted_indices[0]].fitness[obj];
            let max_val = population[sorted_indices[n - 1]].fitness[obj];
            let range = max_val - min_val;
            
            if range > 0.0 {
                // Assign distances
                for i in 1..n - 1 {
                    let prev_val = population[sorted_indices[i - 1]].fitness[obj];
                    let next_val = population[sorted_indices[i + 1]].fitness[obj];
                    let distance = (next_val - prev_val) / range;
                    population[sorted_indices[i]].crowding_distance += distance;
                }
            }
        }
    }
    
    /// Create offspring through selection, crossover, and mutation
    fn create_offspring(&self, population: &Population<T>) -> Population<T> {
        let mut offspring = Vec::new();
        let mut rng = thread_rng();
        
        while offspring.len() < self.config.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(population, &mut rng);
            let parent2 = self.tournament_selection(population, &mut rng);
            
            // Crossover
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.config.crossover_rate {
                self.crossover_op.crossover(
                    &population[parent1].genome,
                    &population[parent2].genome,
                )
            } else {
                (
                    population[parent1].genome.clone(),
                    population[parent2].genome.clone(),
                )
            };
            
            // Mutation
            self.mutation_op.mutate(&mut child1, self.config.mutation_rate);
            self.mutation_op.mutate(&mut child2, self.config.mutation_rate);
            
            offspring.push(Individual::new(child1));
            if offspring.len() < self.config.population_size {
                offspring.push(Individual::new(child2));
            }
        }
        
        offspring.truncate(self.config.population_size);
        offspring
    }
    
    /// Tournament selection
    fn tournament_selection(&self, population: &Population<T>, rng: &mut ThreadRng) -> usize {
        let tournament_size = match self.config.selection {
            SelectionStrategy::Tournament(size) => size,
            _ => 2,
        };
        
        let mut best_idx = rng.gen_range(0..population.len());
        
        for _ in 1..tournament_size {
            let candidate_idx = rng.gen_range(0..population.len());
            
            if self.crowded_comparison(&population[candidate_idx], &population[best_idx]) == Ordering::Less {
                best_idx = candidate_idx;
            }
        }
        
        best_idx
    }
    
    /// Crowded comparison operator
    fn crowded_comparison(&self, a: &Individual<T>, b: &Individual<T>) -> Ordering {
        if a.rank < b.rank {
            Ordering::Less
        } else if a.rank > b.rank {
            Ordering::Greater
        } else {
            // Same rank, compare crowding distance
            OrderedFloat(b.crowding_distance).cmp(&OrderedFloat(a.crowding_distance))
        }
    }
    
    /// Environmental selection for next generation
    fn environmental_selection(
        &self,
        mut combined: Population<T>,
        _fronts: &[Vec<usize>],
    ) -> Population<T> {
        // Re-evaluate and sort
        self.evaluate_population(&mut combined);
        let fronts = self.fast_non_dominated_sort(&combined);
        
        // Assign ranks
        for (rank, front) in fronts.iter().enumerate() {
            for &idx in front {
                combined[idx].rank = rank;
            }
        }
        
        let mut selected = Vec::new();
        
        // Add fronts until we exceed population size
        for front in &fronts {
            self.assign_crowding_distance(&mut combined, front);
            
            if selected.len() + front.len() <= self.config.population_size {
                // Add entire front
                for &idx in front {
                    selected.push(combined[idx].clone());
                }
            } else {
                // Add partial front sorted by crowding distance
                let mut sorted_front: Vec<_> = front.iter().copied().collect();
                sorted_front.sort_by(|&a, &b| {
                    OrderedFloat(combined[b].crowding_distance)
                        .cmp(&OrderedFloat(combined[a].crowding_distance))
                });
                
                let remaining = self.config.population_size - selected.len();
                for &idx in sorted_front.iter().take(remaining) {
                    selected.push(combined[idx].clone());
                }
                break;
            }
        }
        
        selected
    }
    
    /// Calculate statistics for current generation
    fn calculate_generation_stats(
        &self,
        generation: usize,
        population: &Population<T>,
        fronts: &[Vec<usize>],
    ) -> GenerationStats {
        let num_objectives = self.fitness_fn.num_objectives();
        let mut avg_fitness = vec![0.0; num_objectives];
        let mut best_fitness = vec![f64::NEG_INFINITY; num_objectives];
        
        for individual in population {
            for (i, &fitness) in individual.fitness.iter().enumerate() {
                avg_fitness[i] += fitness;
                best_fitness[i] = best_fitness[i].max(fitness);
            }
        }
        
        for avg in &mut avg_fitness {
            *avg /= population.len() as f64;
        }
        
        GenerationStats {
            generation,
            avg_fitness,
            best_fitness,
            pareto_size: fronts.get(0).map(|f| f.len()).unwrap_or(0),
        }
    }
    
    /// Find best individual for each objective
    fn find_best_per_objective(&self, population: &Population<T>) -> Vec<Individual<T>> {
        let num_objectives = self.fitness_fn.num_objectives();
        let mut best_individuals = Vec::new();
        
        for obj in 0..num_objectives {
            if let Some(best) = population.iter()
                .max_by(|a, b| {
                    OrderedFloat(a.fitness.get(obj).copied().unwrap_or(f64::NEG_INFINITY))
                        .cmp(&OrderedFloat(b.fitness.get(obj).copied().unwrap_or(f64::NEG_INFINITY)))
                })
            {
                best_individuals.push(best.clone());
            }
        }
        
        best_individuals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::{TestSuite, TestSuiteFitness};
    use crate::operators::{TestSuiteMutator, TestSuiteCrossover};
    
    #[test]
    fn test_domination() {
        let mut ind1 = Individual::new(());
        ind1.fitness = vec![1.0, 2.0];
        
        let mut ind2 = Individual::new(());
        ind2.fitness = vec![0.5, 1.5];
        
        assert!(ind1.dominates(&ind2));
        assert!(!ind2.dominates(&ind1));
    }
    
    #[test]
    fn test_non_dominated_sorting() {
        // Create a simple fitness function
        struct SimpleFitness;
        impl FitnessFunction<()> for SimpleFitness {
            fn evaluate(&self, _: &()) -> Vec<f64> {
                vec![1.0, 1.0]
            }
            fn num_objectives(&self) -> usize { 2 }
            fn objective_names(&self) -> Vec<&'static str> {
                vec!["Obj1", "Obj2"]
            }
        }
        
        let config = EvolutionConfig {
            population_size: 4,
            generations: 1,
            ..Default::default()
        };
        
        struct DummyMutator;
        impl MutationOperator<()> for DummyMutator {
            fn mutate(&self, _: &mut (), _: f64) {}
        }
        
        struct DummyCrossover;
        impl CrossoverOperator<()> for DummyCrossover {
            fn crossover(&self, _: &(), _: &()) -> ((), ()) {
                ((), ())
            }
        }
        
        impl Evolvable for () {
            fn random() -> Self { () }
            fn len(&self) -> usize { 0 }
        }
        
        let nsga2 = NSGA2::new(
            config,
            Box::new(SimpleFitness),
            Box::new(DummyMutator),
            Box::new(DummyCrossover),
        );
        
        let mut population = vec![
            Individual { genome: (), fitness: vec![1.0, 4.0], crowding_distance: 0.0, rank: 0 },
            Individual { genome: (), fitness: vec![2.0, 3.0], crowding_distance: 0.0, rank: 0 },
            Individual { genome: (), fitness: vec![3.0, 2.0], crowding_distance: 0.0, rank: 0 },
            Individual { genome: (), fitness: vec![4.0, 1.0], crowding_distance: 0.0, rank: 0 },
        ];
        
        let fronts = nsga2.fast_non_dominated_sort(&population);
        
        // All individuals are non-dominated (Pareto front)
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0].len(), 4);
    }
}