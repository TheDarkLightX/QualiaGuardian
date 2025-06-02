"""
Evolution Types and Data Structures

Shared types and classes for the evolution package to avoid circular imports.
"""

import numpy as np
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field
from guardian.evolution.fitness import FitnessVector


@dataclass
class EvolutionHistory:
    """Track evolution progress and statistics"""
    generations: List[int] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    avg_fitness: List[float] = field(default_factory=list)
    diversity: List[float] = field(default_factory=list)
    mutation_rates: List[float] = field(default_factory=list)
    
    def record(self, generation: int, fitness_scores: np.ndarray, 
               mutation_rate: float, population_diversity: float):
        """Record generation statistics"""
        self.generations.append(generation)
        self.best_fitness.append(float(np.max(fitness_scores)))
        self.avg_fitness.append(float(np.mean(fitness_scores)))
        self.diversity.append(population_diversity)
        self.mutation_rates.append(mutation_rate)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "generations": self.generations,
            "best_fitness": [float(f) for f in self.best_fitness],
            "avg_fitness": [float(f) for f in self.avg_fitness],
            "diversity": [float(d) for d in self.diversity],
            "mutation_rates": [float(mr) for mr in self.mutation_rates]
        }


@dataclass
class TestIndividual:
    """Represents a test case in the evolutionary population"""
    __test__ = False  # Tell pytest this is not a test class
    test_code: str
    assertions: List[Dict[str, Any]]
    setup_code: str = ""
    teardown_code: str = ""
    
    # Fitness components
    fitness_values: FitnessVector = field(default_factory=FitnessVector)
    pareto_rank: int = -1  # Initialize with a value indicating not yet calculated
    crowding_distance: float = 0.0  # Initialize to 0.0
    
    # Metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = f"test_{hash(self.test_code + str(time.time()))}"
    
    def get_fitness_vector(self) -> np.ndarray:
        """Return multi-dimensional fitness vector"""
        return self.fitness_values.to_array()

    def code(self) -> str:
        """
        Returns a string representation of the executable test code.

        For M1, this assumes `self.test_code` is a complete, well-formed
        Python test function string. Future enhancements might involve
        constructing the code from parts or handling class-based tests
        with setup/teardown.
        """
        # Ensure there's a newline at the end for cleaner file concatenation
        code_str = self.test_code.strip()
        if self.setup_code:
            # Simple prepend for now. Assumes setup_code is standalone lines.
            code_str = self.setup_code.strip() + "\n" + code_str
        if self.teardown_code:
            # Simple append for now. Assumes teardown_code is standalone lines.
            code_str += "\n" + self.teardown_code.strip()
        
        return code_str + "\n"
