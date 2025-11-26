"""
Unit Tests for Evolutionary CQS

Tests genetic algorithms, population management, and evolution.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from guardian.core.cqs_evolutionary import (
    EvolutionaryCQS, CodeVariant, RefactoringType, RefactoringSuggestion
)
from guardian.core.cqs import CQSCalculator


class TestEvolutionaryCQS(unittest.TestCase):
    """Unit tests for Evolutionary CQS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evolutionary = EvolutionaryCQS(
            population_size=10,
            max_generations=3,
            mutation_rate=0.3,
            crossover_rate=0.7
        )
        self.cqs_calc = CQSCalculator()
    
    def test_initialization(self):
        """Test evolutionary system initialization."""
        self.assertEqual(self.evolutionary.population_size, 10)
        self.assertEqual(self.evolutionary.max_generations, 3)
        self.assertEqual(self.evolutionary.mutation_rate, 0.3)
        self.assertEqual(self.evolutionary.crossover_rate, 0.7)
        self.assertEqual(len(self.evolutionary.population), 0)
        self.assertIsNone(self.evolutionary.best_ever)
    
    def test_fitness_calculation(self):
        """Test fitness calculation."""
        variant = CodeVariant(
            code="def test(): return 1",
            cqs_score=0.8,
            readability=0.8,
            simplicity=0.9,
            maintainability=0.7,
            clarity=0.8,
            fitness=0.0,
            generation=0
        )
        
        fitness = self.evolutionary._calculate_fitness(variant, 0.9)
        
        self.assertGreater(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
    
    def test_refactoring_suggestion_generation(self):
        """Test refactoring suggestion generation."""
        code = """
def process(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x * y * z
            else:
                return 0
        else:
            return 0
    else:
        return 0
"""
        suggestions = self.evolutionary._generate_refactoring_suggestions(
            code,
            lambda c: self.cqs_calc.calculate_from_code(c)
        )
        
        self.assertIsInstance(suggestions, list)
        # Should find opportunities in complex code
        if len(suggestions) > 0:
            self.assertIsInstance(suggestions[0], RefactoringSuggestion)
    
    def test_code_variant_creation(self):
        """Test code variant creation."""
        variant = CodeVariant(
            code="def test(): return 1",
            cqs_score=0.8,
            readability=0.8,
            simplicity=0.9,
            maintainability=0.7,
            clarity=0.8,
            fitness=0.75,
            generation=1,
            parent_ids=[1, 2],
            refactorings_applied=[RefactoringType.SIMPLIFY_CONDITIONAL]
        )
        
        self.assertEqual(variant.cqs_score, 0.8)
        self.assertEqual(variant.generation, 1)
        self.assertEqual(len(variant.parent_ids), 2)
        self.assertEqual(len(variant.refactorings_applied), 1)
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        # Create mock population
        self.evolutionary.population = [
            CodeVariant(code=f"code{i}", cqs_score=0.5 + i*0.1, 
                       readability=0.5, simplicity=0.5,
                       maintainability=0.5, clarity=0.5,
                       fitness=0.5 + i*0.1, generation=0)
            for i in range(5)
        ]
        
        # Tournament selection should return a variant
        selected = self.evolutionary._tournament_selection()
        self.assertIsInstance(selected, CodeVariant)
        self.assertIn(selected, self.evolutionary.population)
    
    def test_crossover(self):
        """Test crossover operation."""
        code1 = "def func1(): return 1"
        code2 = "def func2(): return 2"
        
        child = self.evolutionary._crossover(code1, code2)
        
        self.assertIsInstance(child, str)
        # Child should be one of the parents (simplified crossover)
        self.assertIn(child, [code1, code2])
    
    def test_mutation(self):
        """Test mutation operation."""
        code = "def calc(x): return x * 2"
        
        mutated = self.evolutionary._random_mutation(code)
        
        self.assertIsInstance(mutated, (str, type(None)))
        if mutated:
            self.assertIsInstance(mutated, str)
    
    def test_estimate_improvement(self):
        """Test improvement estimation."""
        suggestion = RefactoringSuggestion(
            refactoring_type=RefactoringType.SIMPLIFY_CONDITIONAL,
            description="Test",
            location=(1, 5),
            current_code="if x: if y: return 1",
            suggested_code="if not x: return 0\nif not y: return 0\nreturn 1",
            expected_improvement=0.1,
            confidence=0.8
        )
        
        improvement = self.evolutionary._estimate_improvement(
            suggestion,
            lambda c: self.cqs_calc.calculate_from_code(c)
        )
        
        self.assertIsInstance(improvement, float)
    
    def test_confidence_calculation(self):
        """Test confidence calculation."""
        suggestion = RefactoringSuggestion(
            refactoring_type=RefactoringType.SIMPLIFY_CONDITIONAL,
            description="Test",
            location=(1, 5),
            current_code="code",
            suggested_code="better code",
            expected_improvement=0.1,
            confidence=0.8
        )
        
        confidence = self.evolutionary._calculate_confidence(suggestion)
        
        self.assertEqual(confidence, 0.8)


class TestCodeVariant(unittest.TestCase):
    """Unit tests for CodeVariant."""
    
    def test_variant_creation(self):
        """Test variant creation."""
        variant = CodeVariant(
            code="def test(): pass",
            cqs_score=0.8,
            readability=0.8,
            simplicity=0.8,
            maintainability=0.8,
            clarity=0.8,
            fitness=0.8,
            generation=0
        )
        
        self.assertEqual(variant.code, "def test(): pass")
        self.assertEqual(variant.cqs_score, 0.8)
        self.assertEqual(variant.generation, 0)


if __name__ == '__main__':
    unittest.main()
