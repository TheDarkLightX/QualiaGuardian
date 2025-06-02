"""
Tests for Guardian Evolution Engine

Testing the evolutionary algorithms, mutation testing, and optimization components.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guardian'))

from guardian.evolution.adaptive_emt import AdaptiveEMT, TestIndividual, EvolutionHistory
from guardian.evolution.smart_mutator import SmartMutator, Mutant, MutantType
from guardian.evolution.operators import CrossoverOperator, MutationOperator
from guardian.evolution.fitness import FitnessEvaluator, MultiObjectiveFitness, FitnessVector


class TestAdaptiveEMT:
    """Test Adaptive Evolutionary Mutation Testing"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test codebase
        self.test_file = self.temp_path / "test_code.py"
        self.test_file.write_text('''
def add(a, b):
    if a > 0 and b > 0:
        return a + b
    else:
        return 0

def divide(a, b):
    if b != 0:
        return a / b
    else:
        raise ValueError("Division by zero")
''')
        
        self.emt = AdaptiveEMT(
            codebase_path=str(self.temp_path),
            test_suite_path=str(self.temp_path),
            population_size=10,
            max_generations=3
        )
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_emt_initialization(self):
        """Test EMT initialization"""
        assert self.emt.codebase_path == str(self.temp_path)
        assert self.emt.population_size == 10
        assert self.emt.max_generations == 3
        assert self.emt.generation == 0
        assert len(self.emt.population) == 0
    
    def test_test_individual_creation(self):
        """Test TestIndividual creation and properties"""
        test_code = '''
def test_example():
    assert add(1, 2) == 3
    assert add(0, 1) == 0
'''
        
        individual = TestIndividual(
            test_code=test_code,
            assertions=[
                {'type': 'equality', 'code': 'assert add(1, 2) == 3', 'target_criticality': 1.0},
                {'type': 'boundary', 'code': 'assert add(0, 1) == 0', 'target_criticality': 1.5}
            ]
        )
        
        assert individual.test_code == test_code
        assert len(individual.assertions) == 2
        assert individual.generation == 0
        assert individual.id is not None
        
        # Test fitness vector
        fitness_vector = individual.get_fitness_vector()
        assert len(fitness_vector) == 4  # kill_rate, severity_score, speed_score, determinism_score
    
    def test_population_initialization(self):
        """Test population initialization"""
        existing_tests = [
            "def test_add(): assert add(1, 1) == 2",
            "def test_divide(): assert divide(4, 2) == 2"
        ]
        
        population = self.emt.initialize_population(existing_tests)
        
        assert len(population) == self.emt.population_size
        assert all(isinstance(individual, TestIndividual) for individual in population)
        
        # Should include the existing tests
        test_codes = [ind.test_code for ind in population]
        assert any("test_add" in code for code in test_codes)
        assert any("test_divide" in code for code in test_codes)
    
    def test_evolution_cycle(self):
        """Test basic evolution cycle"""
        existing_tests = ["def test_basic(): assert True"]
        
        # Run a short evolution cycle
        best_individuals = self.emt.evolve(early_stop=True)
        
        # Should return some individuals
        assert isinstance(best_individuals, list)
        assert len(best_individuals) > 0
        assert all(isinstance(ind, TestIndividual) for ind in best_individuals)
        
        # Should have evolution history
        assert len(self.emt.history.generations) > 0
        assert len(self.emt.history.best_fitness) > 0
    
    def test_evolution_history_tracking(self):
        """Test evolution history tracking"""
        history = EvolutionHistory()
        
        # Record some generations
        history.record(0, [0.5, 0.6, 0.7], 0.1, 0.8)
        history.record(1, [0.6, 0.7, 0.8], 0.12, 0.75)
        history.record(2, [0.7, 0.8, 0.9], 0.15, 0.7)
        
        assert len(history.generations) == 3
        assert len(history.best_fitness) == 3
        assert history.best_fitness[-1] == 0.9  # Best from last generation
        assert history.avg_fitness[-1] == 0.8   # Average from last generation
    
    def test_adaptive_parameters(self):
        """Test adaptive parameter adjustment"""
        # Start with default mutation rate
        initial_rate = self.emt.current_mutation_rate
        
        # Simulate low diversity scenario
        low_diversity_scores = [0.5, 0.51, 0.52]  # Very similar scores
        self.emt._update_adaptive_parameters(low_diversity_scores)
        
        # Mutation rate should increase to promote diversity
        assert self.emt.current_mutation_rate >= initial_rate


class TestSmartMutator:
    """Test Smart Mutation Generation"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.mutator = SmartMutator(str(self.temp_path), mutation_budget=20)
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_mutator_initialization(self):
        """Test mutator initialization"""
        assert self.mutator.codebase_path == str(self.temp_path)
        assert self.mutator.mutation_budget == 20
        assert len(self.mutator.fault_patterns) > 0
    
    def test_mutant_creation(self):
        """Test Mutant object creation"""
        mutant = Mutant(
            original_code="x + y",
            mutated_code="x - y",
            mutation_type=MutantType.ARITHMETIC,
            line_number=5,
            impact_score=1.5,
            likelihood=0.8,
            description="Change + to -"
        )
        
        assert mutant.original_code == "x + y"
        assert mutant.mutated_code == "x - y"
        assert mutant.mutation_type == MutantType.ARITHMETIC
        assert mutant.line_number == 5
        assert mutant.impact_score == 1.5
        assert mutant.likelihood == 0.8
        assert mutant.id is not None
    
    def test_fault_pattern_initialization(self):
        """Test fault pattern initialization"""
        patterns = self.mutator.fault_patterns
        
        # Should have common fault patterns
        pattern_names = [p.name for p in patterns]
        assert "Off-by-one errors" in pattern_names
        assert "Boolean logic errors" in pattern_names
        assert "Arithmetic operator errors" in pattern_names
        
        # Each pattern should have mutation rules
        for pattern in patterns:
            assert len(pattern.mutation_rules) > 0
            assert pattern.impact_weight > 0
    
    def test_smart_mutant_generation(self):
        """Test smart mutant generation"""
        # Create test file
        test_file = self.temp_path / "test.py"
        test_file.write_text('''
def calculate(x, y):
    if x > 0 and y > 0:
        return x + y
    elif x < 0 or y < 0:
        return x - y
    else:
        return 0

def process_list(items):
    for i in range(len(items)):
        if items[i] is not None:
            items[i] = items[i] * 2
    return items
''')
        
        mutants = self.mutator.generate_smart_mutants(str(test_file))
        
        # Should generate mutants
        assert len(mutants) > 0
        assert len(mutants) <= self.mutator.mutation_budget
        
        # All should be Mutant objects
        assert all(isinstance(m, Mutant) for m in mutants)
        
        # Should have different mutation types
        mutation_types = set(m.mutation_type for m in mutants)
        assert len(mutation_types) > 1
        
        # Should be prioritized (first mutants should have higher priority)
        if len(mutants) > 1:
            assert mutants[0].impact_score * mutants[0].likelihood >= \
                   mutants[-1].impact_score * mutants[-1].likelihood


class TestEvolutionOperators:
    """Test Crossover and Mutation Operators"""
    
    def test_crossover_operator(self):
        """Test crossover operator"""
        crossover = CrossoverOperator()
        
        # Create parent individuals
        parent1 = TestIndividual(
            test_code="def test1(): assert func(1) == 1",
            assertions=[{'type': 'equality', 'code': 'assert func(1) == 1', 'target_criticality': 1.0}]
        )
        
        parent2 = TestIndividual(
            test_code="def test2(): assert func(2) == 4",
            assertions=[{'type': 'equality', 'code': 'assert func(2) == 4', 'target_criticality': 1.0}]
        )
        
        # Perform crossover
        child1, child2 = crossover.crossover(parent1, parent2)
        
        # Should produce valid children
        assert isinstance(child1, TestIndividual)
        assert isinstance(child2, TestIndividual)
        assert child1.test_code is not None
        assert child2.test_code is not None
        assert len(child1.assertions) > 0
        assert len(child2.assertions) > 0
        
        # Children should have parent IDs
        assert parent1.id in child1.parent_ids
        assert parent2.id in child1.parent_ids
    
    def test_mutation_operator(self):
        """Test mutation operator"""
        mutation_op = MutationOperator()
        
        # Create individual to mutate
        individual = TestIndividual(
            test_code="def test(): assert func(1) == 1",
            assertions=[{'type': 'equality', 'code': 'assert func(1) == 1', 'target_criticality': 1.0}]
        )
        
        # Perform mutation
        mutated = mutation_op.mutate(individual, mutation_rate=1.0)  # Force mutation
        
        # Should produce valid mutated individual
        assert isinstance(mutated, TestIndividual)
        assert mutated.test_code is not None
        assert len(mutated.assertions) > 0
        
        # Should have mutation history
        assert len(mutated.mutation_history) > 0
        assert individual.id in mutated.parent_ids


class TestFitnessEvaluation:
    """Test Fitness Evaluation Components"""
    
    def test_fitness_vector_creation(self):
        """Test FitnessVector creation and operations"""
        fitness = FitnessVector(
            mutation_killing=0.8,
            execution_speed=0.9,
            assertion_quality=0.7,
            coverage_breadth=0.85,
            maintainability=0.75,
            fault_detection=0.8
        )
        
        # Test array conversion
        array = fitness.to_array()
        assert len(array) == 6
        assert array[0] == 0.8  # mutation_killing
        assert array[1] == 0.9  # execution_speed
        
        # Test weighted sum
        weights = {
            'mutation_killing': 0.3,
            'execution_speed': 0.2,
            'assertion_quality': 0.2,
            'coverage_breadth': 0.15,
            'maintainability': 0.1,
            'fault_detection': 0.05
        }
        
        weighted_score = fitness.weighted_sum(weights)
        assert 0.0 <= weighted_score <= 1.0
        
        # Should be reasonable given the input values
        assert weighted_score > 0.7  # All components are reasonably high
    
    def test_fitness_evaluator(self):
        """Test FitnessEvaluator"""
        evaluator = FitnessEvaluator("dummy_path")
        
        # Create test individual
        individual = TestIndividual(
            test_code="def test(): assert func(1) == 1",
            assertions=[
                {'type': 'equality', 'code': 'assert func(1) == 1', 'target_criticality': 1.0},
                {'type': 'boundary', 'code': 'assert func(0) == 0', 'target_criticality': 1.5}
            ]
        )
        
        # Evaluate fitness
        fitness = evaluator.evaluate_individual(individual)
        
        # Should return valid fitness vector
        assert isinstance(fitness, FitnessVector)
        assert 0.0 <= fitness.mutation_killing <= 1.0
        assert 0.0 <= fitness.execution_speed <= 1.0
        assert 0.0 <= fitness.assertion_quality <= 1.0
        assert 0.0 <= fitness.coverage_breadth <= 1.0
        assert 0.0 <= fitness.maintainability <= 1.0
        assert 0.0 <= fitness.fault_detection <= 1.0
    
    def test_multi_objective_fitness(self):
        """Test multi-objective fitness evaluation"""
        mo_fitness = MultiObjectiveFitness()
        
        # Create fitness vectors for population
        fitness_vectors = [
            FitnessVector(0.8, 0.9, 0.7, 0.8, 0.75, 0.8),
            FitnessVector(0.9, 0.7, 0.8, 0.7, 0.8, 0.85),
            FitnessVector(0.7, 0.8, 0.9, 0.9, 0.7, 0.75),
            FitnessVector(0.6, 0.95, 0.6, 0.85, 0.9, 0.7)
        ]
        
        # Evaluate population
        ranks, distances = mo_fitness.evaluate_population(fitness_vectors)
        
        # Should return valid ranks and distances
        assert len(ranks) == len(fitness_vectors)
        assert len(distances) == len(fitness_vectors)
        assert all(isinstance(rank, int) for rank in ranks)
        assert all(isinstance(dist, float) for dist in distances)
        
        # Ranks should start from 0
        assert min(ranks) == 0
        
        # Select best individuals
        selected = mo_fitness.select_best(fitness_vectors, 2)
        assert len(selected) == 2
        assert all(0 <= idx < len(fitness_vectors) for idx in selected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
