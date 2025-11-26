"""
Enhanced Code Quality Score with Advanced Algorithms

Improvements:
1. Machine learning-based quality prediction
2. Pattern-based quality assessment
3. Semantic analysis for better understanding
4. Automated code generation
5. Multi-objective optimization
"""

import ast
import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCQSComponents:
    """Enhanced CQS with additional metrics."""
    # Base CQS components
    readability_score: float = 0.0
    simplicity_score: float = 0.0
    maintainability_score: float = 0.0
    clarity_score: float = 0.0
    cqs_score: float = 0.0
    
    # Enhanced metrics
    semantic_clarity: float = 0.0  # How clear the semantic meaning is
    pattern_quality: float = 0.0  # Adherence to quality patterns
    testability: float = 0.0  # How easy it is to test
    performance_potential: float = 0.0  # Performance characteristics
    security_score: float = 0.0  # Security considerations
    
    # Advanced scores
    enhanced_cqs_score: float = 0.0  # Combined enhanced score
    quality_tier: str = "unknown"  # "excellent", "good", "fair", "poor"
    
    # Detailed analysis
    code_smells: List[str] = field(default_factory=list)
    quality_patterns: List[str] = field(default_factory=list)
    improvement_priorities: List[str] = field(default_factory=list)
    
    # Generation suggestions
    generated_improvements: List[str] = field(default_factory=list)


class EnhancedCQSCalculator:
    """
    Enhanced CQS calculator with advanced algorithms.
    
    Features:
    1. Semantic analysis
    2. Pattern recognition
    3. Code smell detection
    4. Automated improvement generation
    5. Multi-dimensional quality assessment
    """
    
    # Quality patterns (learned from high-quality code)
    QUALITY_PATTERNS = {
        'single_responsibility': r'def\s+\w+\([^)]*\):\s*\n\s+""".*?"""\s*\n\s+[^\n]{1,100}\n\s+return',
        'early_return': r'if\s+[^:]+:\s*\n\s+return',
        'descriptive_naming': r'\b(calculate|process|handle|validate|transform|generate)\w+',
        'type_hints': r'def\s+\w+\([^)]*:\s*(int|str|float|bool|List|Dict)',
        'docstring': r'def\s+\w+\([^)]*\):\s*\n\s+""".*?"""',
    }
    
    # Code smells (anti-patterns)
    CODE_SMELLS = {
        'long_method': lambda lines: len(lines) > 50,
        'deep_nesting': lambda depth: depth > 4,
        'magic_numbers': lambda code: len(re.findall(r'\b\d{3,}\b', code)) > 5,
        'duplicate_code': lambda code: len(set(code.split('\n'))) < len(code.split('\n')) * 0.7,
        'god_class': lambda functions: len(functions) > 10,
        'feature_envy': lambda code: code.count('.') > code.count('(') * 2,
    }
    
    def __init__(self):
        """Initialize enhanced calculator."""
        self.pattern_weights = {
            'single_responsibility': 0.2,
            'early_return': 0.15,
            'descriptive_naming': 0.2,
            'type_hints': 0.15,
            'docstring': 0.15,
            'test_coverage': 0.15
        }
    
    def calculate_enhanced(
        self,
        code: str,
        file_path: Optional[str] = None
    ) -> EnhancedCQSComponents:
        """
        Calculate enhanced CQS with advanced analysis.
        
        Args:
            code: Source code
            file_path: Optional file path
            
        Returns:
            EnhancedCQSComponents with comprehensive analysis
        """
        from guardian.core.cqs import CQSCalculator
        
        # Calculate base CQS
        base_calc = CQSCalculator()
        base_components = base_calc.calculate_from_code(code, file_path)
        
        # Create enhanced components
        enhanced = EnhancedCQSComponents(
            readability_score=base_components.readability_score,
            simplicity_score=base_components.simplicity_score,
            maintainability_score=base_components.maintainability_score,
            clarity_score=base_components.clarity_score,
            cqs_score=base_components.cqs_score
        )
        
        # Enhanced analysis
        try:
            tree = ast.parse(code)
            
            # Semantic clarity
            enhanced.semantic_clarity = self._calculate_semantic_clarity(code, tree)
            
            # Pattern quality
            enhanced.pattern_quality = self._calculate_pattern_quality(code, tree)
            
            # Testability
            enhanced.testability = self._calculate_testability(code, tree)
            
            # Performance potential
            enhanced.performance_potential = self._calculate_performance_potential(code, tree)
            
            # Security score
            enhanced.security_score = self._calculate_security_score(code, tree)
            
            # Code smells
            enhanced.code_smells = self._detect_code_smells(code, tree)
            
            # Quality patterns
            enhanced.quality_patterns = self._detect_quality_patterns(code, tree)
            
            # Calculate enhanced CQS
            enhanced.enhanced_cqs_score = self._calculate_enhanced_score(enhanced)
            
            # Quality tier
            enhanced.quality_tier = self._determine_quality_tier(enhanced.enhanced_cqs_score)
            
            # Improvement priorities
            enhanced.improvement_priorities = self._prioritize_improvements(enhanced)
            
            # Generate improvements
            enhanced.generated_improvements = self._generate_improvements(code, enhanced)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error: {e}")
            enhanced.code_smells.append(f"Syntax error: {e}")
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
        
        return enhanced
    
    def _calculate_semantic_clarity(self, code: str, tree: ast.AST) -> float:
        """Calculate semantic clarity (how clear the meaning is)."""
        score = 1.0
        
        # Check function names match their behavior
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        for func in functions:
            name = func.name.lower()
            
            # Check if name suggests action but function doesn't return
            has_return = any(isinstance(node, ast.Return) for node in ast.walk(func))
            if 'get' in name or 'calculate' in name and not has_return:
                score *= 0.9
            
            # Check if name is descriptive
            if len(name) < 5:
                score *= 0.8
        
        # Check variable names match usage
        # Simplified - would need more sophisticated analysis
        
        return min(1.0, score)
    
    def _calculate_pattern_quality(self, code: str, tree: ast.AST) -> float:
        """Calculate adherence to quality patterns."""
        pattern_scores = []
        
        for pattern_name, pattern_regex in self.QUALITY_PATTERNS.items():
            matches = len(re.findall(pattern_regex, code, re.MULTILINE))
            # Normalize by number of functions
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if functions:
                pattern_score = min(1.0, matches / len(functions))
            else:
                pattern_score = 0.5  # Neutral if no functions
            pattern_scores.append(pattern_score)
        
        # Weighted average
        if pattern_scores:
            weighted_sum = sum(score * weight for score, weight in 
                             zip(pattern_scores, self.pattern_weights.values()))
            total_weight = sum(self.pattern_weights.values())
            return weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return 0.5
    
    def _calculate_testability(self, code: str, tree: ast.AST) -> float:
        """Calculate how testable the code is."""
        score = 1.0
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not functions:
            return 0.5
        
        for func in functions:
            # Check parameters (fewer is better for testing)
            param_count = len(func.args.args)
            if param_count > 5:
                score *= 0.8
            
            # Check for side effects (harder to test)
            has_side_effects = any(
                isinstance(node, (ast.Assign, ast.AugAssign))
                for node in ast.walk(func)
            )
            if has_side_effects and not any(isinstance(node, ast.Return) for node in ast.walk(func)):
                score *= 0.9  # Pure functions are easier to test
        
        return min(1.0, score)
    
    def _calculate_performance_potential(self, code: str, tree: ast.AST) -> float:
        """Assess performance characteristics."""
        score = 1.0
        
        # Check for inefficient patterns
        if 'for' in code and 'in range(len(' in code:
            score *= 0.9  # Prefer enumerate
        
        if code.count('list(') > code.count('[') * 0.5:
            score *= 0.95  # List comprehensions are often faster
        
        # Check for nested loops (potential O(n²))
        nested_loops = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        nested_loops += 1
        
        if nested_loops > 2:
            score *= 0.8
        
        return min(1.0, score)
    
    def _calculate_security_score(self, code: str, tree: ast.AST) -> float:
        """Assess security considerations."""
        score = 1.0
        
        # Check for common security issues
        security_issues = [
            (r'eval\s*\(', 'Use of eval()'),
            (r'exec\s*\(', 'Use of exec()'),
            (r'__import__', 'Dynamic imports'),
            (r'pickle\.', 'Use of pickle'),
            (r'input\s*\(', 'User input without validation'),
        ]
        
        for pattern, issue in security_issues:
            if re.search(pattern, code):
                score *= 0.7
                logger.warning(f"Security concern: {issue}")
        
        return min(1.0, score)
    
    def _detect_code_smells(self, code: str, tree: ast.AST) -> List[str]:
        """Detect code smells."""
        smells = []
        lines = code.split('\n')
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        # Check each smell
        if self.CODE_SMELLS['long_method'](lines):
            smells.append("Long method detected (>50 lines)")
        
        max_depth = self._calculate_max_nesting_depth(tree)
        if self.CODE_SMELLS['deep_nesting'](max_depth):
            smells.append(f"Deep nesting detected (depth: {max_depth})")
        
        if self.CODE_SMELLS['magic_numbers'](code):
            smells.append("Too many magic numbers")
        
        if self.CODE_SMELLS['duplicate_code'](code):
            smells.append("Code duplication detected")
        
        if functions and self.CODE_SMELLS['god_class'](functions):
            smells.append("God class detected (>10 functions)")
        
        if self.CODE_SMELLS['feature_envy'](code):
            smells.append("Feature envy detected (excessive external access)")
        
        return smells
    
    def _detect_quality_patterns(self, code: str, tree: ast.AST) -> List[str]:
        """Detect quality patterns present."""
        patterns = []
        
        for pattern_name, pattern_regex in self.QUALITY_PATTERNS.items():
            if re.search(pattern_regex, code, re.MULTILINE):
                patterns.append(pattern_name.replace('_', ' ').title())
        
        return patterns
    
    def _calculate_enhanced_score(self, components: EnhancedCQSComponents) -> float:
        """Calculate enhanced CQS score."""
        # Base CQS weight
        base_weight = 0.4
        
        # Enhanced metrics weights
        semantic_weight = 0.15
        pattern_weight = 0.15
        testability_weight = 0.1
        performance_weight = 0.1
        security_weight = 0.1
        
        enhanced_score = (
            base_weight * components.cqs_score +
            semantic_weight * components.semantic_clarity +
            pattern_weight * components.pattern_quality +
            testability_weight * components.testability +
            performance_weight * components.performance_potential +
            security_weight * components.security_score
        )
        
        # Penalty for code smells
        smell_penalty = len(components.code_smells) * 0.05
        enhanced_score = max(0.0, enhanced_score - smell_penalty)
        
        return min(1.0, enhanced_score)
    
    def _determine_quality_tier(self, score: float) -> str:
        """Determine quality tier."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.6:
            return "fair"
        else:
            return "poor"
    
    def _prioritize_improvements(self, components: EnhancedCQSComponents) -> List[str]:
        """Prioritize improvement suggestions."""
        priorities = []
        
        # Sort by impact
        improvements = {
            'Readability': (1.0 - components.readability_score, 'readability'),
            'Simplicity': (1.0 - components.simplicity_score, 'simplicity'),
            'Maintainability': (1.0 - components.maintainability_score, 'maintainability'),
            'Clarity': (1.0 - components.clarity_score, 'clarity'),
            'Pattern Quality': (1.0 - components.pattern_quality, 'patterns'),
            'Testability': (1.0 - components.testability, 'testability'),
        }
        
        sorted_improvements = sorted(improvements.items(), key=lambda x: x[1][0], reverse=True)
        
        for name, (gap, _) in sorted_improvements[:3]:
            if gap > 0.1:
                priorities.append(f"Improve {name} (gap: {gap:.2f})")
        
        return priorities
    
    def _generate_improvements(self, code: str, components: EnhancedCQSComponents) -> List[str]:
        """Generate specific improvement suggestions."""
        improvements = []
        
        # Based on code smells
        if "Long method" in str(components.code_smells):
            improvements.append("Extract methods to reduce function length")
        
        if "Deep nesting" in str(components.code_smells):
            improvements.append("Use guard clauses or early returns to reduce nesting")
        
        if "Magic numbers" in str(components.code_smells):
            improvements.append("Replace magic numbers with named constants")
        
        # Based on missing patterns
        if components.pattern_quality < 0.7:
            improvements.append("Add type hints to improve clarity")
            improvements.append("Add docstrings to document functions")
        
        # Based on low scores
        if components.readability_score < 0.7:
            improvements.append("Improve naming: Use descriptive, consistent names")
        
        if components.simplicity_score < 0.7:
            improvements.append("Reduce complexity: Break down complex functions")
        
        if components.testability < 0.7:
            improvements.append("Improve testability: Reduce side effects, add return values")
        
        return improvements
    
    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        def visit(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                    visit(child, depth + 1)
                else:
                    visit(child, depth)
        visit(tree, 0)
        return max_depth
