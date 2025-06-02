"""
Multi-Objective Fitness Evaluation for Test Suite Evolution

Advanced fitness evaluation with Pareto optimization and 
multi-dimensional quality assessment.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class FitnessObjective(Enum):
    """Fitness objectives for multi-objective optimization"""
    MUTATION_KILLING = "mutation_killing"
    EXECUTION_SPEED = "execution_speed"
    ASSERTION_QUALITY = "assertion_quality"
    COVERAGE_BREADTH = "coverage_breadth"
    TEST_MAINTAINABILITY = "test_maintainability"
    FAULT_DETECTION = "fault_detection"


@dataclass
class FitnessVector:
    """Multi-dimensional fitness representation"""
    mutation_killing: float = 0.0
    execution_speed: float = 0.0
    assertion_quality: float = 0.0
    coverage_breadth: float = 0.0
    maintainability: float = 0.0
    fault_detection: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for calculations"""
        return np.array([
            self.mutation_killing,
            self.execution_speed,
            self.assertion_quality,
            self.coverage_breadth,
            self.maintainability,
            self.fault_detection
        ])

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return {
            "mutation_killing": float(self.mutation_killing),
            "execution_speed": float(self.execution_speed),
            "assertion_quality": float(self.assertion_quality),
            "coverage_breadth": float(self.coverage_breadth),
            "maintainability": float(self.maintainability),
            "fault_detection": float(self.fault_detection)
        }
    
    def weighted_sum(self, weights: Dict[str, float]) -> float:
        """Calculate weighted sum of fitness components"""
        return (
            self.mutation_killing * weights.get('mutation_killing', 0.3) +
            self.execution_speed * weights.get('execution_speed', 0.15) +
            self.assertion_quality * weights.get('assertion_quality', 0.2) +
            self.coverage_breadth * weights.get('coverage_breadth', 0.15) +
            self.maintainability * weights.get('maintainability', 0.1) +
            self.fault_detection * weights.get('fault_detection', 0.1)
        )


class MultiObjectiveFitness:
    """
    Multi-objective fitness evaluation using NSGA-II principles
    """
    
    def __init__(self):
        self.pareto_fronts = []
        self.crowding_distances = {}
    
    def evaluate_population(self, fitness_vectors: List[FitnessVector]) -> Tuple[List[int], List[float]]:
        """
        Evaluate population using multi-objective optimization
        
        Args:
            fitness_vectors: List of fitness vectors for population
            
        Returns:
            Tuple of (pareto_ranks, crowding_distances)
        """
        if not fitness_vectors:
            return [], []
        
        # Convert to numpy array for efficient computation
        fitness_matrix = np.array([fv.to_array() for fv in fitness_vectors])
        
        # Calculate Pareto fronts
        pareto_ranks = self._calculate_pareto_ranks(fitness_matrix)
        
        # Calculate crowding distances
        crowding_distances = self._calculate_crowding_distances(fitness_matrix, pareto_ranks)
        
        return pareto_ranks, crowding_distances
    
    def _calculate_pareto_ranks(self, fitness_matrix: np.ndarray) -> List[int]:
        """Calculate Pareto rank for each individual"""
        n_individuals = fitness_matrix.shape[0]
        ranks = np.zeros(n_individuals, dtype=int)
        
        # For each individual, count how many dominate it
        domination_counts = np.zeros(n_individuals, dtype=int)
        dominated_solutions = [[] for _ in range(n_individuals)]
        
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i != j:
                    if self._dominates(fitness_matrix[i], fitness_matrix[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(fitness_matrix[j], fitness_matrix[i]):
                        domination_counts[i] += 1
        
        # Find first Pareto front (rank 0)
        current_front = []
        for i in range(n_individuals):
            if domination_counts[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        # Find subsequent fronts
        front_number = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        ranks[j] = front_number + 1
                        next_front.append(j)
            
            front_number += 1
            current_front = next_front
        
        return ranks.tolist()
    
    def _dominates(self, solution1: np.ndarray, solution2: np.ndarray) -> bool:
        """Check if solution1 dominates solution2 (all objectives >= and at least one >)"""
        return np.all(solution1 >= solution2) and np.any(solution1 > solution2)
    
    def _calculate_crowding_distances(self, fitness_matrix: np.ndarray, ranks: List[int]) -> List[float]:
        """Calculate crowding distance for diversity preservation"""
        n_individuals = fitness_matrix.shape[0]
        n_objectives = fitness_matrix.shape[1]
        distances = np.zeros(n_individuals)
        
        # Group by Pareto front
        fronts = {}
        for i, rank in enumerate(ranks):
            if rank not in fronts:
                fronts[rank] = []
            fronts[rank].append(i)
        
        # Calculate crowding distance for each front
        for front_indices in fronts.values():
            if len(front_indices) <= 2:
                # Boundary solutions get infinite distance
                for idx in front_indices:
                    distances[idx] = float('inf')
                continue
            
            front_fitness = fitness_matrix[front_indices]
            
            # For each objective
            for obj in range(n_objectives):
                # Sort by objective value
                sorted_indices = np.argsort(front_fitness[:, obj])
                
                # Boundary solutions get infinite distance
                distances[front_indices[sorted_indices[0]]] = float('inf')
                distances[front_indices[sorted_indices[-1]]] = float('inf')
                
                # Calculate distance for intermediate solutions
                obj_range = front_fitness[sorted_indices[-1], obj] - front_fitness[sorted_indices[0], obj]
                
                if obj_range > 0:
                    for i in range(1, len(sorted_indices) - 1):
                        idx = front_indices[sorted_indices[i]]
                        if distances[idx] != float('inf'):
                            distance_contribution = (
                                front_fitness[sorted_indices[i + 1], obj] - 
                                front_fitness[sorted_indices[i - 1], obj]
                            ) / obj_range
                            distances[idx] += distance_contribution
        
        return distances.tolist()
    
    def select_best(self, fitness_vectors: List[FitnessVector], 
                   selection_size: int) -> List[int]:
        """
        Select best individuals using NSGA-II selection
        
        Args:
            fitness_vectors: Population fitness vectors
            selection_size: Number of individuals to select
            
        Returns:
            Indices of selected individuals
        """
        if len(fitness_vectors) <= selection_size:
            return list(range(len(fitness_vectors)))
        
        ranks, distances = self.evaluate_population(fitness_vectors)
        
        # Sort by rank first, then by crowding distance (descending)
        combined_scores = list(zip(range(len(fitness_vectors)), ranks, distances))
        combined_scores.sort(key=lambda x: (x[1], -x[2] if x[2] != float('inf') else -1e10))
        
        selected_indices = [idx for idx, _, _ in combined_scores[:selection_size]]
        return selected_indices


class FitnessEvaluator:
    """
    Comprehensive fitness evaluator for test individuals
    """
    
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        self.multi_objective = MultiObjectiveFitness()
        
        # Fitness weights (can be adjusted)
        self.default_weights = {
            'mutation_killing': 0.3,
            'execution_speed': 0.15,
            'assertion_quality': 0.2,
            'coverage_breadth': 0.15,
            'maintainability': 0.1,
            'fault_detection': 0.1
        }
    
    def evaluate_individual(self, individual, mutants: List[Dict[str, Any]] = None) -> FitnessVector:
        """
        Evaluate comprehensive fitness of a test individual
        
        Args:
            individual: Test individual to evaluate
            mutants: Optional list of mutants for mutation testing
            
        Returns:
            Multi-dimensional fitness vector
        """
        try:
            fitness = FitnessVector()
            
            # Evaluate mutation killing effectiveness
            fitness.mutation_killing = self._evaluate_mutation_killing(individual, mutants)
            
            # Evaluate execution speed
            fitness.execution_speed = self._evaluate_execution_speed(individual)
            
            # Evaluate assertion quality
            fitness.assertion_quality = self._evaluate_assertion_quality(individual)
            
            # Evaluate coverage breadth
            fitness.coverage_breadth = self._evaluate_coverage_breadth(individual)
            
            # Evaluate maintainability
            fitness.maintainability = self._evaluate_maintainability(individual)
            
            # Evaluate fault detection capability
            fitness.fault_detection = self._evaluate_fault_detection(individual)
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Error evaluating individual fitness: {e}")
            return FitnessVector()  # Return zero fitness on error
    
    def calculate_composite_fitness(self, fitness_vector: np.ndarray, 
                                  weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate composite fitness score from vector
        
        Args:
            fitness_vector: Array of fitness components
            weights: Optional custom weights
            
        Returns:
            Composite fitness score
        """
        if weights is None:
            weights = self.default_weights
        
        # Map array to fitness components
        fitness = FitnessVector(
            mutation_killing=fitness_vector[0] if len(fitness_vector) > 0 else 0.0,
            execution_speed=fitness_vector[1] if len(fitness_vector) > 1 else 0.0,
            assertion_quality=fitness_vector[2] if len(fitness_vector) > 2 else 0.0,
            coverage_breadth=fitness_vector[3] if len(fitness_vector) > 3 else 0.0,
            maintainability=fitness_vector[4] if len(fitness_vector) > 4 else 0.0,
            fault_detection=fitness_vector[5] if len(fitness_vector) > 5 else 0.0
        )
        
        return fitness.weighted_sum(weights)
    
    def _evaluate_mutation_killing(self, individual, mutants: List[Dict[str, Any]] = None) -> float:
        """Evaluate mutation killing effectiveness"""
        if not mutants:
            # Simulate mutation testing (placeholder)
            return np.random.uniform(0.6, 0.95)
        
        killed_mutants = 0
        total_severity = 0
        
        for mutant in mutants:
            # Simulate test execution against mutant
            if self._test_kills_mutant(individual, mutant):
                killed_mutants += 1
                total_severity += mutant.get('severity_weight', 1.0)
        
        if not mutants:
            return 0.0
        
        # Weight by severity
        kill_rate = killed_mutants / len(mutants)
        severity_bonus = total_severity / (len(mutants) * 2.0)  # Normalize severity
        
        return min(kill_rate + severity_bonus, 1.0)
    
    def _evaluate_execution_speed(self, individual) -> float:
        """Evaluate test execution speed"""
        # Estimate execution time based on test complexity
        code_lines = len(individual.test_code.split('\n'))
        assertion_count = len(individual.assertions)
        
        # Simple heuristic: more code/assertions = slower
        estimated_ms = (code_lines * 10) + (assertion_count * 5)
        
        # Convert to speed factor (higher is better)
        speed_factor = 1.0 / (1.0 + math.log(1.0 + estimated_ms / 100.0))
        return min(speed_factor, 1.0)
    
    def _evaluate_assertion_quality(self, individual) -> float:
        """Evaluate quality and intelligence of assertions"""
        if not individual.assertions:
            return 0.0
        
        quality_score = 0.0
        assertion_weights = {
            'equality': 1.0,
            'inequality': 1.0,
            'type_check': 1.2,
            'exception': 1.5,
            'invariant': 2.0,
            'property': 1.8,
            'boundary': 1.6,
            'performance': 1.4
        }
        
        for assertion in individual.assertions:
            assertion_type = assertion.get('type', 'equality')
            weight = assertion_weights.get(assertion_type, 1.0)
            
            # Bonus for invariant checking
            if assertion.get('checks_invariant', False):
                weight *= 1.3
            
            # Penalty for redundancy
            if assertion.get('is_redundant', False):
                weight *= 0.5
            
            # Weight by target criticality
            criticality = assertion.get('target_criticality', 1.0)
            quality_score += weight * criticality
        
        # Normalize by number of assertions
        avg_quality = quality_score / len(individual.assertions)
        return min(avg_quality / 2.0, 1.0)  # Normalize to 0-1 range
    
    def _evaluate_coverage_breadth(self, individual) -> float:
        """Evaluate breadth of test coverage"""
        coverage_indicators = {
            'loop_coverage': 'for ' in individual.test_code,
            'branch_coverage': 'if ' in individual.test_code,
            'exception_coverage': 'try:' in individual.test_code or any(
                a.get('type') == 'exception' for a in individual.assertions
            ),
            'boundary_coverage': any(
                a.get('type') == 'boundary' for a in individual.assertions
            ),
            'type_coverage': any(
                a.get('type') == 'type_check' for a in individual.assertions
            ),
            'property_coverage': any(
                a.get('type') == 'property' for a in individual.assertions
            )
        }
        
        coverage_count = sum(coverage_indicators.values())
        max_coverage = len(coverage_indicators)
        
        return coverage_count / max_coverage if max_coverage > 0 else 0.0
    
    def _evaluate_maintainability(self, individual) -> float:
        """Evaluate test maintainability"""
        maintainability_score = 1.0
        
        # Penalize overly long tests
        code_lines = len([line for line in individual.test_code.split('\n') if line.strip()])
        if code_lines > 50:
            maintainability_score *= 0.8
        elif code_lines > 100:
            maintainability_score *= 0.5
        
        # Penalize too many assertions (cognitive overload)
        if len(individual.assertions) > 10:
            maintainability_score *= 0.9
        elif len(individual.assertions) > 20:
            maintainability_score *= 0.7
        
        # Bonus for good structure (setup/action/assert pattern)
        if self._has_good_structure(individual):
            maintainability_score *= 1.1
        
        # Penalize complex setup
        setup_complexity = len(individual.setup_code.split('\n')) if individual.setup_code else 0
        if setup_complexity > 10:
            maintainability_score *= 0.9
        
        return min(maintainability_score, 1.0)
    
    def _evaluate_fault_detection(self, individual) -> float:
        """Evaluate fault detection capability"""
        fault_detection_score = 0.0
        
        # Check for common fault detection patterns
        fault_patterns = {
            'null_check': any('None' in a.get('code', '') for a in individual.assertions),
            'boundary_check': any(a.get('type') == 'boundary' for a in individual.assertions),
            'type_validation': any(a.get('type') == 'type_check' for a in individual.assertions),
            'exception_handling': any(a.get('type') == 'exception' for a in individual.assertions),
            'invariant_check': any(a.get('checks_invariant', False) for a in individual.assertions),
            'range_validation': any('range' in individual.test_code.lower()),
            'state_validation': any('state' in a.get('code', '').lower() for a in individual.assertions)
        }
        
        # Weight different fault detection capabilities
        pattern_weights = {
            'null_check': 0.2,
            'boundary_check': 0.25,
            'type_validation': 0.15,
            'exception_handling': 0.2,
            'invariant_check': 0.3,
            'range_validation': 0.15,
            'state_validation': 0.1
        }
        
        for pattern, present in fault_patterns.items():
            if present:
                fault_detection_score += pattern_weights.get(pattern, 0.1)
        
        return min(fault_detection_score, 1.0)
    
    def _test_kills_mutant(self, individual, mutant: Dict[str, Any]) -> bool:
        """Simulate test execution against mutant"""
        # Placeholder simulation
        # In practice, would execute test against mutated code
        
        # Higher chance to kill if test has relevant assertions
        kill_probability = 0.3  # Base probability
        
        # Increase probability based on assertion types
        for assertion in individual.assertions:
            assertion_type = assertion.get('type', 'equality')
            mutant_type = mutant.get('mutation_type', 'unknown')
            
            # Type-specific kill probabilities
            if assertion_type == 'boundary' and 'boundary' in mutant_type:
                kill_probability += 0.4
            elif assertion_type == 'exception' and 'exception' in mutant_type:
                kill_probability += 0.5
            elif assertion_type == 'type_check' and 'type' in mutant_type:
                kill_probability += 0.3
            else:
                kill_probability += 0.1
        
        return np.random.random() < min(kill_probability, 0.95)
    
    def _has_good_structure(self, individual) -> bool:
        """Check if test follows good structural patterns"""
        code_lines = individual.test_code.split('\n')
        
        # Look for Arrange-Act-Assert pattern
        has_setup = any('setup' in line.lower() or '=' in line for line in code_lines[:3])
        has_action = any(line.strip() and not line.strip().startswith('assert') 
                        for line in code_lines[1:-2])
        has_assertions = any('assert' in line for line in code_lines)
        
        return has_setup and has_action and has_assertions
