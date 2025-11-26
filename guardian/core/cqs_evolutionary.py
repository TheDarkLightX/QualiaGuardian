"""
Evolutionary Code Quality Improvement System

Uses evolutionary algorithms to evolve code towards higher quality:
- Genetic programming for code generation
- Multi-objective optimization (readability, performance, maintainability)
- Pattern-based learning from high-quality code
- Automated refactoring with quality validation
"""

import ast
import random
import math
from typing import List, Dict, Tuple, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RefactoringType(Enum):
    """Types of refactoring operations."""
    EXTRACT_METHOD = "extract_method"
    RENAME_VARIABLE = "rename_variable"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    REDUCE_NESTING = "reduce_nesting"
    ADD_TYPE_HINTS = "add_type_hints"
    ADD_DOCSTRING = "add_docstring"
    REMOVE_DUPLICATION = "remove_duplication"
    IMPROVE_NAMING = "improve_naming"


@dataclass
class CodeVariant:
    """A variant of code with quality metrics."""
    code: str
    cqs_score: float
    readability: float
    simplicity: float
    maintainability: float
    clarity: float
    fitness: float  # Combined fitness for evolution
    generation: int
    parent_ids: List[int] = field(default_factory=list)
    refactorings_applied: List[RefactoringType] = field(default_factory=list)


@dataclass
class RefactoringSuggestion:
    """A specific refactoring suggestion."""
    refactoring_type: RefactoringType
    description: str
    location: Tuple[int, int]  # (start_line, end_line)
    current_code: str
    suggested_code: str
    expected_improvement: float
    confidence: float


class EvolutionaryCQS:
    """
    Evolutionary Code Quality Improvement System.
    
    Uses genetic algorithms to evolve code towards higher quality:
    1. Generate code variants through refactoring
    2. Evaluate quality using CQS
    3. Select best variants
    4. Evolve through crossover and mutation
    5. Converge to highest quality code
    """
    
    def __init__(
        self,
        population_size: int = 20,
        max_generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elitism_rate: float = 0.2
    ):
        """
        Initialize evolutionary CQS.
        
        Args:
            population_size: Number of code variants per generation
            max_generations: Maximum generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_rate: Fraction of best variants to keep
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.generation = 0
        self.population: List[CodeVariant] = []
        self.best_ever: Optional[CodeVariant] = None
    
    def evolve_code(
        self,
        initial_code: str,
        cqs_calculator: Callable,
        target_cqs: float = 0.9
    ) -> Tuple[CodeVariant, List[CodeVariant]]:
        """
        Evolve code towards higher quality.
        
        Args:
            initial_code: Starting code
            cqs_calculator: Function that calculates CQS from code
            target_cqs: Target CQS score (default: 0.9)
            
        Returns:
            Tuple of (best_variant, evolution_history)
        """
        # Initialize population with variants of initial code
        self.population = self._initialize_population(initial_code, cqs_calculator)
        self.best_ever = max(self.population, key=lambda v: v.fitness)
        evolution_history = [self.best_ever]
        
        # Evolve
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Evaluate fitness
            for variant in self.population:
                if variant.fitness == 0:  # Not yet evaluated
                    components = cqs_calculator(variant.code)
                    variant.cqs_score = components.cqs_score
                    variant.readability = components.readability_score
                    variant.simplicity = components.simplicity_score
                    variant.maintainability = components.maintainability_score
                    variant.clarity = components.clarity_score
                    variant.fitness = self._calculate_fitness(variant, target_cqs)
            
            # Sort by fitness
            self.population.sort(key=lambda v: v.fitness, reverse=True)
            
            # Update best ever
            current_best = self.population[0]
            if current_best.fitness > self.best_ever.fitness:
                self.best_ever = current_best
                evolution_history.append(current_best)
            
            # Check convergence
            if self.best_ever.cqs_score >= target_cqs:
                logger.info(f"Target CQS reached at generation {generation}")
                break
            
            # Create next generation
            self.population = self._create_next_generation(cqs_calculator)
        
        return self.best_ever, evolution_history
    
    def _initialize_population(
        self,
        initial_code: str,
        cqs_calculator: Callable
    ) -> List[CodeVariant]:
        """Initialize population with refactored variants."""
        population = []
        
        # Original code
        components = cqs_calculator(initial_code)
        population.append(CodeVariant(
            code=initial_code,
            cqs_score=components.cqs_score,
            readability=components.readability_score,
            simplicity=components.simplicity_score,
            maintainability=components.maintainability_score,
            clarity=components.clarity_score,
            fitness=self._calculate_fitness_from_components(components, 0.9),
            generation=0
        ))
        
        # Generate variants through refactoring
        refactorings = self._generate_refactoring_suggestions(initial_code, cqs_calculator)
        
        for i, refactoring in enumerate(refactorings[:self.population_size - 1]):
            variant_code = self._apply_refactoring(initial_code, refactoring)
            if variant_code:
                components = cqs_calculator(variant_code)
                population.append(CodeVariant(
                    code=variant_code,
                    cqs_score=components.cqs_score,
                    readability=components.readability_score,
                    simplicity=components.simplicity_score,
                    maintainability=components.maintainability_score,
                    clarity=components.clarity_score,
                    fitness=self._calculate_fitness_from_components(components, 0.9),
                    generation=0,
                    refactorings_applied=[refactoring.refactoring_type]
                ))
        
        # Fill remaining with random mutations
        while len(population) < self.population_size:
            variant_code = self._random_mutation(initial_code)
            if variant_code:
                components = cqs_calculator(variant_code)
                population.append(CodeVariant(
                    code=variant_code,
                    cqs_score=components.cqs_score,
                    readability=components.readability_score,
                    simplicity=components.simplicity_score,
                    maintainability=components.maintainability_score,
                    clarity=components.clarity_score,
                    fitness=self._calculate_fitness_from_components(components, 0.9),
                    generation=0
                ))
        
        return population
    
    def _generate_refactoring_suggestions(
        self,
        code: str,
        cqs_calculator: Callable
    ) -> List[RefactoringSuggestion]:
        """Generate intelligent refactoring suggestions."""
        suggestions = []
        
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            
            # Analyze code structure
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            for func in functions:
                func_code = '\n'.join(lines[func.lineno - 1:func.end_lineno if hasattr(func, 'end_lineno') else func.lineno + 10])
                
                # Check for extract method opportunities
                if self._should_extract_method(func, tree):
                    suggestion = self._suggest_extract_method(func, func_code, lines)
                    if suggestion:
                        suggestions.append(suggestion)
                
                # Check for simplify conditional
                if self._has_complex_conditional(func):
                    suggestion = self._suggest_simplify_conditional(func, func_code, lines)
                    if suggestion:
                        suggestions.append(suggestion)
                
                # Check for reduce nesting
                if self._has_deep_nesting(func):
                    suggestion = self._suggest_reduce_nesting(func, func_code, lines)
                    if suggestion:
                        suggestions.append(suggestion)
            
            # Check for naming improvements
            naming_suggestions = self._suggest_naming_improvements(tree, code)
            suggestions.extend(naming_suggestions)
            
            # Check for type hints
            if not self._has_type_hints(code):
                suggestion = self._suggest_add_type_hints(tree, code)
                if suggestion:
                    suggestions.append(suggestion)
            
            # Check for docstrings
            docstring_suggestions = self._suggest_add_docstrings(functions, code)
            suggestions.extend(docstring_suggestions)
            
        except SyntaxError:
            pass
        
        # Score suggestions by expected improvement
        for suggestion in suggestions:
            suggestion.expected_improvement = self._estimate_improvement(suggestion, cqs_calculator)
            suggestion.confidence = self._calculate_confidence(suggestion)
        
        # Sort by expected improvement
        suggestions.sort(key=lambda s: s.expected_improvement * s.confidence, reverse=True)
        
        return suggestions
    
    def _should_extract_method(self, func: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function should be split."""
        func_lines = func.end_lineno - func.lineno if hasattr(func, 'end_lineno') else 20
        return func_lines > 30  # Functions longer than 30 lines
    
    def _has_complex_conditional(self, func: ast.FunctionDef) -> bool:
        """Check for complex conditionals."""
        for node in ast.walk(func):
            if isinstance(node, ast.If):
                # Check if condition is complex
                if isinstance(node.test, (ast.BoolOp, ast.Compare)):
                    return True
        return False
    
    def _has_deep_nesting(self, func: ast.FunctionDef) -> bool:
        """Check for deep nesting."""
        max_depth = 0
        def visit(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                    visit(child, depth + 1)
                else:
                    visit(child, depth)
        visit(func, 0)
        return max_depth > 3
    
    def _has_type_hints(self, code: str) -> bool:
        """Check if code has type hints."""
        return '->' in code or ': ' in code and any(word in code for word in ['int', 'str', 'float', 'bool', 'List', 'Dict'])
    
    def _suggest_extract_method(self, func: ast.FunctionDef, func_code: str, lines: List[str]) -> Optional[RefactoringSuggestion]:
        """Suggest extracting a method."""
        # Simplified - would need more sophisticated analysis
        return None  # Placeholder
    
    def _suggest_simplify_conditional(self, func: ast.FunctionDef, func_code: str, lines: List[str]) -> Optional[RefactoringSuggestion]:
        """Suggest simplifying conditionals."""
        # Find complex conditionals and suggest early returns
        for node in ast.walk(func):
            if isinstance(node, ast.If):
                # Suggest early return pattern
                if node.orelse:
                    return RefactoringSuggestion(
                        refactoring_type=RefactoringType.SIMPLIFY_CONDITIONAL,
                        description="Simplify conditional with early return",
                        location=(node.lineno, node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 5),
                        current_code=func_code,
                        suggested_code=self._generate_early_return_variant(node, lines),
                        expected_improvement=0.1,
                        confidence=0.7
                    )
        return None
    
    def _suggest_reduce_nesting(self, func: ast.FunctionDef, func_code: str, lines: List[str]) -> Optional[RefactoringSuggestion]:
        """Suggest reducing nesting."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.REDUCE_NESTING,
            description="Reduce nesting depth with guard clauses",
            location=(func.lineno, func.end_lineno if hasattr(func, 'end_lineno') else func.lineno + 10),
            current_code=func_code,
            suggested_code=self._generate_guard_clause_variant(func, lines),
            expected_improvement=0.15,
            confidence=0.8
        )
    
    def _suggest_naming_improvements(self, tree: ast.AST, code: str) -> List[RefactoringSuggestion]:
        """Suggest naming improvements."""
        suggestions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.name) < 5 or not self._is_descriptive_name(node.name):
                    suggestions.append(RefactoringSuggestion(
                        refactoring_type=RefactoringType.IMPROVE_NAMING,
                        description=f"Improve function name: {node.name}",
                        location=(node.lineno, node.lineno),
                        current_code=node.name,
                        suggested_code=self._generate_better_name(node.name),
                        expected_improvement=0.05,
                        confidence=0.9
                    ))
        return suggestions
    
    def _suggest_add_type_hints(self, tree: ast.AST, code: str) -> Optional[RefactoringSuggestion]:
        """Suggest adding type hints."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if functions:
            return RefactoringSuggestion(
                refactoring_type=RefactoringType.ADD_TYPE_HINTS,
                description="Add type hints to functions",
                location=(functions[0].lineno, functions[-1].end_lineno if hasattr(functions[-1], 'end_lineno') else functions[-1].lineno + 5),
                current_code=code,
                suggested_code=self._add_type_hints_to_code(code, functions),
                expected_improvement=0.08,
                confidence=0.85
            )
        return None
    
    def _suggest_add_docstrings(self, functions: List[ast.FunctionDef], code: str) -> List[RefactoringSuggestion]:
        """Suggest adding docstrings."""
        suggestions = []
        for func in functions:
            if not ast.get_docstring(func):
                suggestions.append(RefactoringSuggestion(
                    refactoring_type=RefactoringType.ADD_DOCSTRING,
                    description=f"Add docstring to {func.name}",
                    location=(func.lineno, func.lineno + 1),
                    current_code=code,
                    suggested_code=self._add_docstring_to_function(code, func),
                    expected_improvement=0.06,
                    confidence=0.9
                ))
        return suggestions
    
    def _apply_refactoring(self, code: str, suggestion: RefactoringSuggestion) -> Optional[str]:
        """Apply a refactoring suggestion to code."""
        # Simplified implementation - would need full AST manipulation
        return suggestion.suggested_code
    
    def _random_mutation(self, code: str) -> Optional[str]:
        """Apply random mutation to code."""
        # Randomly apply one of several mutations
        mutations = [
            self._mutate_add_docstring,
            self._mutate_improve_naming,
            self._mutate_simplify_conditional
        ]
        mutation = random.choice(mutations)
        try:
            return mutation(code)
        except:
            return None
    
    def _mutate_add_docstring(self, code: str) -> str:
        """Mutation: Add docstring."""
        try:
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if functions and not ast.get_docstring(functions[0]):
                # Add simple docstring
                lines = code.split('\n')
                func_line = functions[0].lineno - 1
                docstring = f'    """{functions[0].name.replace("_", " ").title()}."""'
                lines.insert(func_line + 1, docstring)
                return '\n'.join(lines)
        except:
            pass
        return code
    
    def _mutate_improve_naming(self, code: str) -> str:
        """Mutation: Improve variable names."""
        # Simplified - would need more sophisticated renaming
        return code
    
    def _mutate_simplify_conditional(self, code: str) -> str:
        """Mutation: Simplify conditionals."""
        # Simplified - would need AST manipulation
        return code
    
    def _calculate_fitness(self, variant: CodeVariant, target_cqs: float) -> float:
        """Calculate fitness for evolution."""
        # Multi-objective fitness
        cqs_weight = 0.4
        readability_weight = 0.2
        simplicity_weight = 0.2
        maintainability_weight = 0.1
        clarity_weight = 0.1
        
        fitness = (
            cqs_weight * variant.cqs_score +
            readability_weight * variant.readability +
            simplicity_weight * variant.simplicity +
            maintainability_weight * variant.maintainability +
            clarity_weight * variant.clarity
        )
        
        # Bonus for reaching target
        if variant.cqs_score >= target_cqs:
            fitness += 0.1
        
        return fitness
    
    def _calculate_fitness_from_components(self, components, target_cqs: float) -> float:
        """Calculate fitness from CQS components."""
        variant = CodeVariant(
            code="",
            cqs_score=components.cqs_score,
            readability=components.readability_score,
            simplicity=components.simplicity_score,
            maintainability=components.maintainability_score,
            clarity=components.clarity_score,
            fitness=0.0,
            generation=0
        )
        return self._calculate_fitness(variant, target_cqs)
    
    def _create_next_generation(self, cqs_calculator: Callable) -> List[CodeVariant]:
        """Create next generation through selection, crossover, and mutation."""
        next_gen = []
        
        # Elitism: Keep best variants
        n_elite = int(self.population_size * self.elitism_rate)
        next_gen.extend(self.population[:n_elite])
        
        # Generate rest through crossover and mutation
        while len(next_gen) < self.population_size:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child_code = self._crossover(parent1.code, parent2.code)
                if child_code:
                    components = cqs_calculator(child_code)
                    next_gen.append(CodeVariant(
                        code=child_code,
                        cqs_score=components.cqs_score,
                        readability=components.readability_score,
                        simplicity=components.simplicity_score,
                        maintainability=components.maintainability_score,
                        clarity=components.clarity_score,
                        fitness=self._calculate_fitness_from_components(components, 0.9),
                        generation=self.generation + 1,
                        parent_ids=[id(parent1), id(parent2)]
                    ))
            else:
                # Mutation
                parent = self._tournament_selection()
                child_code = self._random_mutation(parent.code)
                if child_code:
                    components = cqs_calculator(child_code)
                    next_gen.append(CodeVariant(
                        code=child_code,
                        cqs_score=components.cqs_score,
                        readability=components.readability_score,
                        simplicity=components.simplicity_score,
                        maintainability=components.maintainability_score,
                        clarity=components.clarity_score,
                        fitness=self._calculate_fitness_from_components(components, 0.9),
                        generation=self.generation + 1,
                        parent_ids=[id(parent)],
                        refactorings_applied=parent.refactorings_applied.copy()
                    ))
        
        return next_gen
    
    def _tournament_selection(self, tournament_size: int = 3) -> CodeVariant:
        """Tournament selection for parents."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda v: v.fitness)
    
    def _crossover(self, code1: str, code2: str) -> Optional[str]:
        """Crossover two code variants."""
        # Simplified - would need sophisticated AST merging
        # For now, randomly choose one or combine functions
        return code1 if random.random() < 0.5 else code2
    
    def _estimate_improvement(self, suggestion: RefactoringSuggestion, cqs_calculator: Callable) -> float:
        """Estimate CQS improvement from refactoring."""
        try:
            current_components = cqs_calculator(suggestion.current_code)
            suggested_components = cqs_calculator(suggestion.suggested_code)
            return suggested_components.cqs_score - current_components.cqs_score
        except:
            return suggestion.expected_improvement
    
    def _calculate_confidence(self, suggestion: RefactoringSuggestion) -> float:
        """Calculate confidence in refactoring suggestion."""
        return suggestion.confidence
    
    def _generate_early_return_variant(self, node: ast.If, lines: List[str]) -> str:
        """Generate early return variant of conditional."""
        # Simplified - would need full AST manipulation
        return '\n'.join(lines)
    
    def _generate_guard_clause_variant(self, func: ast.FunctionDef, lines: List[str]) -> str:
        """Generate guard clause variant to reduce nesting."""
        # Simplified - would need full AST manipulation
        return '\n'.join(lines)
    
    def _is_descriptive_name(self, name: str) -> bool:
        """Check if name is descriptive."""
        return len(name) >= 5 and '_' in name
    
    def _generate_better_name(self, name: str) -> str:
        """Generate a better name."""
        # Simple improvement - add descriptive suffix
        if name.startswith('calc'):
            return name.replace('calc', 'calculate')
        if name.startswith('proc'):
            return name.replace('proc', 'process')
        if len(name) < 5:
            return name + '_value'
        return name
    
    def _add_type_hints_to_code(self, code: str, functions: List[ast.FunctionDef]) -> str:
        """Add type hints to code."""
        # Simplified - would need full AST manipulation
        return code
    
    def _add_docstring_to_function(self, code: str, func: ast.FunctionDef) -> str:
        """Add docstring to function."""
        lines = code.split('\n')
        func_line = func.lineno - 1
        docstring = f'    """{func.name.replace("_", " ").title()}."""'
        if func_line + 1 < len(lines):
            lines.insert(func_line + 1, docstring)
        return '\n'.join(lines)
