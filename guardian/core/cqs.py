"""
Code Quality Score (CQS)

A metric that actually measures and improves code quality:
- Readability (naming, structure, comments)
- Simplicity (small functions, low complexity)
- Maintainability (low duplication, low coupling)
- Clarity (clear intent, self-documenting)

CQS = (Readability × Simplicity × Maintainability × Clarity)^(1/4)

Where 1.0 = highest quality code.
"""

import math
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import re
import ast
import logging

logger = logging.getLogger(__name__)


@dataclass
class CQSComponents:
    """Components of CQS calculation."""
    # Raw metrics
    readability_score: float = 0.0  # [0, 1]
    simplicity_score: float = 0.0  # [0, 1]
    maintainability_score: float = 0.0  # [0, 1]
    clarity_score: float = 0.0  # [0, 1]
    
    # Sub-components
    naming_quality: float = 0.0
    structure_quality: float = 0.0
    comment_quality: float = 0.0
    function_size_score: float = 0.0
    complexity_score: float = 0.0
    duplication_score: float = 0.0
    coupling_score: float = 0.0
    intent_clarity: float = 0.0
    
    # Final score
    cqs_score: float = 0.0  # [0, 1] where 1 = highest quality
    
    # Metadata
    insights: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


class CQSCalculator:
    """
    Calculates Code Quality Score (CQS).
    
    CQS measures actual code quality (readability, simplicity, maintainability, clarity)
    rather than just predicting bugs.
    """
    
    def __init__(
        self,
        max_function_lines: int = 50,
        max_complexity: int = 10,
        max_parameters: int = 5,
        max_nesting: int = 4
    ):
        """
        Initialize CQS calculator.
        
        Args:
            max_function_lines: Maximum lines per function (for scoring)
            max_complexity: Maximum cyclomatic complexity (for scoring)
            max_parameters: Maximum parameters per function (for scoring)
            max_nesting: Maximum nesting depth (for scoring)
        """
        self.max_function_lines = max_function_lines
        self.max_complexity = max_complexity
        self.max_parameters = max_parameters
        self.max_nesting = max_nesting
    
    def calculate_from_code(
        self,
        code: str,
        file_path: Optional[str] = None
    ) -> CQSComponents:
        """
        Calculate CQS from source code.
        
        Args:
            code: Source code string
            file_path: Optional file path for context
            
        Returns:
            CQSComponents with all scores
        """
        components = CQSComponents()
        
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Calculate readability
            components.readability_score, readability_details = self._calculate_readability(code, tree)
            components.naming_quality = readability_details.get('naming', 0.0)
            components.structure_quality = readability_details.get('structure', 0.0)
            components.comment_quality = readability_details.get('comments', 0.0)
            
            # Calculate simplicity
            components.simplicity_score, simplicity_details = self._calculate_simplicity(code, tree)
            components.function_size_score = simplicity_details.get('function_size', 0.0)
            components.complexity_score = simplicity_details.get('complexity', 0.0)
            
            # Calculate maintainability
            components.maintainability_score, maintainability_details = self._calculate_maintainability(code, tree)
            components.duplication_score = maintainability_details.get('duplication', 0.0)
            components.coupling_score = maintainability_details.get('coupling', 0.0)
            
            # Calculate clarity
            components.clarity_score, clarity_details = self._calculate_clarity(code, tree)
            components.intent_clarity = clarity_details.get('intent', 0.0)
            
            # Calculate final CQS
            factors = [
                components.readability_score,
                components.simplicity_score,
                components.maintainability_score,
                components.clarity_score
            ]
            
            if all(f > 0 for f in factors):
                product = 1.0
                for f in factors:
                    product *= f
                components.cqs_score = product ** (1.0 / len(factors))
            else:
                components.cqs_score = 0.0
            
            # Generate insights and suggestions
            components.insights = self._generate_insights(components)
            components.improvement_suggestions = self._generate_suggestions(components)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in code: {e}")
            components.insights.append(f"Syntax error: {e}")
        except Exception as e:
            logger.error(f"Error calculating CQS: {e}")
            components.insights.append(f"Error: {e}")
        
        return components
    
    def _calculate_readability(self, code: str, tree: ast.AST) -> tuple:
        """Calculate readability score."""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if not non_empty_lines:
            return 0.0, {}
        
        scores = {}
        
        # Naming quality
        naming_score = self._assess_naming_quality(code, tree)
        scores['naming'] = naming_score
        
        # Structure quality
        structure_score = self._assess_structure_quality(code, tree)
        scores['structure'] = structure_score
        
        # Comment quality
        comment_score = self._assess_comment_quality(code)
        scores['comments'] = comment_score
        
        # Line length (penalize very long lines)
        long_lines = sum(1 for l in non_empty_lines if len(l) > 120)
        line_length_score = 1.0 - min(0.3, long_lines / len(non_empty_lines))
        
        # Overall readability (geometric mean)
        factors = [naming_score, structure_score, comment_score, line_length_score]
        readability = (math.prod(factors)) ** (1.0 / len(factors)) if all(f > 0 for f in factors) else 0.0
        
        return readability, scores
    
    def _assess_naming_quality(self, code: str, tree: ast.AST) -> float:
        """Assess naming quality."""
        score = 1.0
        
        # Check function names
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name
                # Good: descriptive, snake_case
                if len(name) < 3:
                    score *= 0.7  # Too short
                if not re.match(r'^[a-z_][a-z0-9_]*$', name):
                    score *= 0.8  # Not snake_case
                if name.startswith('_') and not name.startswith('__'):
                    score *= 0.9  # Private method (ok but slightly less readable)
        
        # Check variable names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                name = node.id
                if len(name) <= 2 and name not in ['i', 'j', 'k', 'x', 'y', 'z']:
                    score *= 0.9  # Short variable names
        
        return min(1.0, score)
    
    def _assess_structure_quality(self, code: str, tree: ast.AST) -> float:
        """Assess code structure quality."""
        score = 1.0
        
        # Check for proper function organization
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if len(functions) > 0:
            # Check function length
            for func in functions:
                func_lines = func.end_lineno - func.lineno if hasattr(func, 'end_lineno') else 10
                if func_lines > self.max_function_lines:
                    score *= 0.8  # Functions too long
        
        # Check for excessive nesting
        max_depth = self._calculate_max_nesting(tree)
        if max_depth > self.max_nesting:
            score *= 0.7  # Too much nesting
        
        return min(1.0, score)
    
    def _assess_comment_quality(self, code: str) -> float:
        """Assess comment quality."""
        lines = code.split('\n')
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        comment_lines = [l for l in lines if l.strip().startswith('#')]
        
        if not code_lines:
            return 1.0
        
        comment_ratio = len(comment_lines) / len(code_lines)
        
        # Optimal comment ratio is 10-20%
        if 0.1 <= comment_ratio <= 0.2:
            return 1.0
        elif comment_ratio < 0.05:
            return 0.7  # Too few comments
        elif comment_ratio > 0.5:
            return 0.6  # Too many comments (might indicate unclear code)
        else:
            return 0.9
    
    def _calculate_simplicity(self, code: str, tree: ast.AST) -> tuple:
        """Calculate simplicity score."""
        scores = {}
        
        # Function size
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if functions:
            avg_lines = sum(
                (f.end_lineno - f.lineno if hasattr(f, 'end_lineno') else 10)
                for f in functions
            ) / len(functions)
            function_size_score = 1.0 - min(0.5, (avg_lines - 10) / self.max_function_lines)
        else:
            function_size_score = 1.0
        scores['function_size'] = function_size_score
        
        # Complexity
        complexity = self._calculate_cyclomatic_complexity(tree)
        complexity_score = 1.0 - min(0.7, (complexity - 1) / self.max_complexity)
        scores['complexity'] = complexity_score
        
        # Parameter count
        if functions:
            avg_params = sum(len(f.args.args) for f in functions) / len(functions)
            param_score = 1.0 - min(0.5, (avg_params - 2) / self.max_parameters)
        else:
            param_score = 1.0
        
        # Overall simplicity
        factors = [function_size_score, complexity_score, param_score]
        simplicity = (math.prod(factors)) ** (1.0 / len(factors)) if all(f > 0 for f in factors) else 0.0
        
        return simplicity, scores
    
    def _calculate_maintainability(self, code: str, tree: ast.AST) -> tuple:
        """Calculate maintainability score."""
        scores = {}
        
        # Duplication (simple check - look for repeated code patterns)
        duplication_score = self._estimate_duplication(code)
        scores['duplication'] = duplication_score
        
        # Coupling (simplified - count imports and external dependencies)
        coupling_score = self._estimate_coupling(tree)
        scores['coupling'] = coupling_score
        
        # Overall maintainability
        factors = [duplication_score, coupling_score]
        maintainability = (math.prod(factors)) ** (1.0 / len(factors)) if all(f > 0 for f in factors) else 0.0
        
        return maintainability, scores
    
    def _estimate_duplication(self, code: str) -> float:
        """Estimate code duplication (simplified)."""
        lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        
        if len(lines) < 4:
            return 1.0
        
        # Look for repeated line sequences
        sequences = {}
        for i in range(len(lines) - 2):
            seq = tuple(lines[i:i+3])
            sequences[seq] = sequences.get(seq, 0) + 1
        
        # Count duplicates
        duplicates = sum(1 for count in sequences.values() if count > 1)
        duplication_ratio = duplicates / len(sequences) if sequences else 0.0
        
        return 1.0 - min(0.5, duplication_ratio)
    
    def _estimate_coupling(self, tree: ast.AST) -> float:
        """Estimate coupling (simplified)."""
        # Count imports
        imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        
        # Count external function calls
        external_calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    external_calls += 1
        
        # Normalize (more imports/calls = higher coupling = lower score)
        import_score = 1.0 - min(0.5, imports / 20.0)
        call_score = 1.0 - min(0.3, external_calls / 50.0)
        
        return (import_score * call_score) ** 0.5
    
    def _calculate_clarity(self, code: str, tree: ast.AST) -> tuple:
        """Calculate clarity score."""
        scores = {}
        
        # Intent clarity
        intent_score = self._assess_intent_clarity(code, tree)
        scores['intent'] = intent_score
        
        # Magic numbers
        magic_numbers = len(re.findall(r'\b\d{3,}\b', code))  # Numbers >= 3 digits
        magic_score = 1.0 - min(0.3, magic_numbers / 10.0)
        
        # Overall clarity
        factors = [intent_score, magic_score]
        clarity = (math.prod(factors)) ** (1.0 / len(factors)) if all(f > 0 for f in factors) else 0.0
        
        return clarity, scores
    
    def _assess_intent_clarity(self, code: str, tree: ast.AST) -> float:
        """Assess how clear the code intent is."""
        score = 1.0
        
        # Check for descriptive function names
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        for func in functions:
            name = func.name
            # Good function names are descriptive
            if len(name) < 5:
                score *= 0.9
            if not any(word in name.lower() for word in ['get', 'set', 'is', 'has', 'can', 'should', 'calculate', 'process', 'handle']):
                # Might be unclear what function does
                if len(name) < 8:
                    score *= 0.95
        
        return min(1.0, score)
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        
        def visit_node(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.Try)):
                    visit_node(child, depth + 1)
                else:
                    visit_node(child, depth)
        
        visit_node(tree, 0)
        return max_depth
    
    def _generate_insights(self, components: CQSComponents) -> List[str]:
        """Generate insights about code quality."""
        insights = []
        
        if components.cqs_score < 0.5:
            insights.append("Low code quality detected - significant improvements needed")
        elif components.cqs_score < 0.7:
            insights.append("Moderate code quality - some improvements recommended")
        else:
            insights.append("Good code quality")
        
        if components.readability_score < 0.7:
            insights.append(f"Readability needs improvement ({components.readability_score:.2f})")
        
        if components.simplicity_score < 0.7:
            insights.append(f"Code is too complex - simplify ({components.simplicity_score:.2f})")
        
        if components.maintainability_score < 0.7:
            insights.append(f"Maintainability needs improvement ({components.maintainability_score:.2f})")
        
        if components.clarity_score < 0.7:
            insights.append(f"Code intent is unclear - improve clarity ({components.clarity_score:.2f})")
        
        return insights
    
    def _generate_suggestions(self, components: CQSComponents) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []
        
        if components.naming_quality < 0.8:
            suggestions.append("Improve naming: Use descriptive, consistent names")
        
        if components.function_size_score < 0.8:
            suggestions.append("Reduce function size: Break large functions into smaller ones")
        
        if components.complexity_score < 0.8:
            suggestions.append("Reduce complexity: Simplify control flow, extract methods")
        
        if components.duplication_score < 0.8:
            suggestions.append("Remove duplication: Extract common code into functions")
        
        if components.coupling_score < 0.8:
            suggestions.append("Reduce coupling: Minimize dependencies between modules")
        
        if components.intent_clarity < 0.8:
            suggestions.append("Improve clarity: Make code intent more obvious, add comments if needed")
        
        return suggestions
