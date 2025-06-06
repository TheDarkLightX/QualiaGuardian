"""
Mutation Pattern Learning Module.

This module provides capabilities for learning from mutation testing results
to identify patterns in survived mutations and suggest test improvements.

The MutationPatternLearner class analyzes survived mutations to identify
patterns, classify them by category and impact, and generate actionable
test improvement suggestions.

Author: DarkLightX/Dana Edwards
"""

import ast
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class PatternCategory(Enum):
    """Categories for mutation patterns."""
    BOUNDARY = "boundary"
    ERROR_HANDLING = "error_handling" 
    NULL_CHECK = "null_check"
    COMPLEX_CONDITION = "complex_condition"
    ARITHMETIC = "arithmetic"
    LOGICAL = "logical"
    CONTROL_FLOW = "control_flow"
    UNKNOWN = "unknown"


@dataclass
class MutationPattern:
    """Represents a pattern found in survived mutations."""
    
    # Core pattern identification
    ast_type: Optional[str] = None
    operator_type: Optional[str] = None
    pattern_signature: str = ""
    pattern_type: str = ""
    
    # Classification
    category: Optional[PatternCategory] = None
    context_type: Optional[str] = None
    
    # Frequency and survival data
    frequency: int = 0
    survival_rate: float = 0.0
    total_mutations: int = 0
    
    # Impact and importance
    impact_score: float = 0.0
    risk_level: str = "medium"
    importance: str = "normal"
    confidence_level: str = "medium"
    
    # Context information
    locations: List[Dict[str, Any]] = field(default_factory=list)
    critical_locations: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    code_locations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Complexity metrics
    average_complexity: Optional[float] = None
    max_nesting_depth: Optional[int] = None
    involves_loops: bool = False
    
    # Pattern specifics
    includes_null_check: bool = False
    right_operand: Optional[str] = None
    sub_patterns: List[str] = field(default_factory=list)
    
    # Metadata and examples
    metadata: Dict[str, Any] = field(default_factory=dict)
    example_mutations: List[Dict[str, str]] = field(default_factory=list)
    
    # Confidence and learning
    base_confidence: float = 0.5
    historical_confidence: float = 0.5
    occurrence_consistency: float = 0.5
    sample_size: int = 0
    volatility_factor: float = 0.0
    trend_direction: str = "stable"
    
    # Source tracking
    source: str = "unknown"
    emergence_confidence: float = 0.0


@dataclass
class ImprovementTemplate:
    """Template for generating test improvements."""
    includes_exception_testing: bool = False
    assertion_patterns: Optional[List[str]] = None


@dataclass
class TestImprovement:
    """Represents a test improvement suggestion."""
    description: str
    priority: str = "medium"
    test_template: Optional[ImprovementTemplate] = None
    assertion_patterns: Optional[List[str]] = None
    pattern_type: str = ""
    suggested_tests: List['ImprovementSuggestion'] = field(default_factory=list)
    affected_modules: List[str] = field(default_factory=list)


@dataclass
class ImprovementSuggestion:
    """Specific test suggestion."""
    priority: str = "medium"
    description: str = ""


@dataclass
class PatternReport:
    """Comprehensive pattern analysis report."""
    total_patterns: int = 0
    high_impact_patterns: int = 0
    recommendations: Optional[str] = None
    pattern_summaries: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MutationPatternLearner:
    """
    Learns patterns from mutation testing results to improve test effectiveness.
    
    This class analyzes survived mutations to identify recurring patterns,
    classifies them by importance and category, and generates actionable
    test improvement suggestions following SOLID principles.
    """
    
    def __init__(self):
        """Initialize the pattern learner."""
        self.logger = logging.getLogger(__name__)
        self._pattern_cache: Dict[str, MutationPattern] = {}
        self._classification_rules = self._initialize_classification_rules()
    
    def _initialize_classification_rules(self) -> Dict[str, PatternCategory]:
        """Initialize pattern classification rules."""
        return {
            # Boundary patterns
            "Compare:Gt": PatternCategory.BOUNDARY,
            "Compare:Lt": PatternCategory.BOUNDARY,
            "Compare:GtE": PatternCategory.BOUNDARY,
            "Compare:LtE": PatternCategory.BOUNDARY,
            "Compare:Eq": PatternCategory.BOUNDARY,
            "Compare:NotEq": PatternCategory.BOUNDARY,
            
            # Null check patterns
            "Compare:Is:None": PatternCategory.NULL_CHECK,
            "Compare:IsNot:None": PatternCategory.NULL_CHECK,
            
            # Error handling patterns
            "ExceptHandler": PatternCategory.ERROR_HANDLING,
            "Try:ExceptHandler": PatternCategory.ERROR_HANDLING,
            "Raise": PatternCategory.ERROR_HANDLING,
            
            # Complex conditions
            "BoolOp:And": PatternCategory.COMPLEX_CONDITION,
            "BoolOp:Or": PatternCategory.COMPLEX_CONDITION,
            
            # Arithmetic patterns
            "BinOp:Add": PatternCategory.ARITHMETIC,
            "BinOp:Sub": PatternCategory.ARITHMETIC,
            "BinOp:Mult": PatternCategory.ARITHMETIC,
            "BinOp:Div": PatternCategory.ARITHMETIC,
        }
    
    def extract_pattern(self, survived_mutation: Dict[str, Any]) -> Optional[MutationPattern]:
        """
        Extract a pattern from a survived mutation.
        
        Args:
            survived_mutation: Dictionary containing mutation details
            
        Returns:
            MutationPattern object or None if extraction fails
        """
        try:
            original_code = survived_mutation.get("original_code", "")
            mutated_code = survived_mutation.get("mutated_code", "")
            mutation_type = survived_mutation.get("mutation_type", "unknown")
            
            if not original_code:
                return None
            
            # Parse the original code to extract AST information
            try:
                # Handle single expressions vs statements
                if not original_code.strip().endswith(':') and '=' in original_code:
                    # This is likely an assignment with a complex expression
                    tree = ast.parse(original_code)
                    node = tree.body[0] if tree.body else None
                    # If it's an assignment, look at the value
                    if isinstance(node, ast.Assign) and node.value:
                        node = node.value
                elif not original_code.strip().endswith(':'):
                    # Try parsing as expression first
                    try:
                        tree = ast.parse(original_code, mode='eval')
                        node = tree.body
                    except SyntaxError:
                        # Fallback to statement parsing
                        tree = ast.parse(original_code)
                        node = tree.body[0] if tree.body else None
                else:
                    tree = ast.parse(original_code)
                    node = tree.body[0] if tree.body else None
            except SyntaxError:
                # Fallback for partial code snippets
                node = None
            
            pattern = MutationPattern()
            pattern.frequency = 1
            pattern.survival_rate = 1.0 if survived_mutation.get("survived", True) else 0.0
            
            # Extract AST-based pattern information
            if node:
                pattern.ast_type = type(node).__name__
                pattern = self._extract_ast_details(node, pattern, original_code, mutated_code)
            else:
                # Fallback to heuristic analysis
                pattern = self._extract_heuristic_pattern(original_code, mutated_code, mutation_type, pattern)
            
            # Extract context information
            context = survived_mutation.get("context", {})
            if context:
                pattern.context_type = self._determine_context_type(context, original_code)
                pattern.metadata.update(context)
            
            # Set pattern type based on mutation type and AST analysis
            pattern.pattern_type = self._determine_pattern_type(pattern, mutation_type)
            
            # Generate pattern signature
            pattern.pattern_signature = self._generate_pattern_signature(pattern)
            
            return pattern
            
        except Exception as e:
            self.logger.warning(f"Failed to extract pattern from mutation: {e}")
            return None
    
    def _extract_ast_details(self, node: ast.AST, pattern: MutationPattern, 
                           original: str, mutated: str) -> MutationPattern:
        """Extract detailed information from AST node."""
        if isinstance(node, ast.Compare):
            pattern.operator_type = type(node.ops[0]).__name__ if node.ops else "Unknown"
            
            # Check for null/None comparisons
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    pattern.includes_null_check = True
                    pattern.right_operand = "None"
                elif isinstance(comparator, ast.NameConstant) and comparator.value is None:
                    pattern.includes_null_check = True
                    pattern.right_operand = "None"
        
        elif isinstance(node, ast.BoolOp):
            pattern.operator_type = type(node.op).__name__
            pattern.sub_patterns = [type(child).__name__ for child in node.values]
        
        elif isinstance(node, ast.IfExp):
            pattern.ast_type = "IfExp"  # Ensure we set the correct AST type
            pattern.context_type = "conditional_expression"
            # Check if test involves null check and extract operator details
            if isinstance(node.test, ast.Compare):
                pattern.operator_type = type(node.test.ops[0]).__name__ if node.test.ops else "Unknown"
                for comp in node.test.comparators:
                    if isinstance(comp, (ast.Constant, ast.NameConstant)) and comp.value is None:
                        pattern.includes_null_check = True
                        pattern.right_operand = "None"
        
        elif isinstance(node, ast.ExceptHandler):
            pattern.context_type = "exception_handling"
            if node.type and isinstance(node.type, ast.Name):
                pattern.metadata["original_exceptions"] = [node.type.id]
        
        return pattern
    
    def _extract_heuristic_pattern(self, original: str, mutated: str, mutation_type: str, 
                                 pattern: MutationPattern) -> MutationPattern:
        """Extract pattern using heuristic analysis when AST parsing fails."""
        # Boundary condition heuristics
        if any(op in original for op in ['>', '<', '>=', '<=', '==', '!=']):
            pattern.ast_type = "Compare"
            if '>' in original:
                pattern.operator_type = "Gt"
            elif '<' in original:
                pattern.operator_type = "Lt"
            elif '>=' in original:
                pattern.operator_type = "GtE"
            elif '<=' in original:
                pattern.operator_type = "LtE"
            pattern.context_type = "boundary_check"
        
        # Null check heuristics
        elif "is None" in original or "is not None" in original:
            pattern.ast_type = "Compare"
            pattern.operator_type = "Is" if "is None" in original else "IsNot"
            pattern.includes_null_check = True
            pattern.right_operand = "None"
        
        # Exception handling heuristics
        elif "except" in original:
            pattern.ast_type = "ExceptHandler"
            pattern.context_type = "exception_handling"
            # Extract exception types
            import re
            exc_match = re.search(r'except\s+(\w+)', original)
            if exc_match:
                pattern.metadata["original_exceptions"] = [exc_match.group(1)]
        
        return pattern
    
    def _determine_context_type(self, context: Dict[str, Any], code: str) -> str:
        """Determine the context type based on context data and code."""
        if context.get("in_try_except"):
            return "exception_handling"
        elif any(op in code for op in ['>', '<', '>=', '<=', '==']):
            return "boundary_check"
        elif "if" in code and ("is None" in code or "is not None" in code):
            return "null_validation"
        elif context.get("function_name"):
            return f"function_{context['function_name']}"
        else:
            return "general"
    
    def _determine_pattern_type(self, pattern: MutationPattern, mutation_type: str) -> str:
        """Determine the high-level pattern type."""
        if pattern.ast_type == "Compare" and pattern.operator_type in ["Gt", "Lt", "GtE", "LtE"]:
            return "boundary_comparison"
        elif pattern.includes_null_check:
            return "null_check"
        elif pattern.ast_type == "ExceptHandler":
            return "exception_handling"
        elif pattern.ast_type == "BoolOp":
            return "complex_condition"
        elif mutation_type:
            return mutation_type
        else:
            return "unknown"
    
    def _generate_pattern_signature(self, pattern: MutationPattern) -> str:
        """Generate a unique signature for the pattern."""
        parts = []
        
        if pattern.ast_type:
            parts.append(pattern.ast_type)
        
        # For IfExp, add nested structure info
        if pattern.ast_type == "IfExp" and pattern.operator_type:
            parts.append("Compare")
            parts.append(pattern.operator_type)
        elif pattern.operator_type:
            parts.append(pattern.operator_type)
        
        if pattern.right_operand:
            parts.append(pattern.right_operand)
        elif pattern.includes_null_check:
            parts.append("None")
        
        if pattern.sub_patterns:
            parts.append(f"[{','.join(pattern.sub_patterns)}]")
        
        return ":".join(parts) if parts else "unknown"
    
    def analyze_frequencies(self, mutations: List[Dict[str, Any]], 
                          include_killed: bool = False) -> Dict[str, MutationPattern]:
        """
        Analyze frequency of patterns across mutations.
        
        Args:
            mutations: List of mutation dictionaries
            include_killed: Whether to include killed mutations in analysis
            
        Returns:
            Dictionary mapping pattern signatures to MutationPattern objects
        """
        frequency_map: Dict[str, MutationPattern] = {}
        
        for mutation in mutations:
            # Skip killed mutations unless explicitly requested
            if not include_killed and not mutation.get("survived", True):
                continue
            
            pattern = self.extract_pattern(mutation)
            if not pattern:
                continue
            
            signature = pattern.pattern_signature
            
            if signature in frequency_map:
                # Update existing pattern
                existing = frequency_map[signature]
                existing.frequency += 1
                existing.example_mutations.append({
                    "original": mutation.get("original_code", ""),
                    "mutated": mutation.get("mutated_code", "")
                })
            else:
                # New pattern
                pattern.example_mutations = [{
                    "original": mutation.get("original_code", ""),
                    "mutated": mutation.get("mutated_code", "")
                }]
                frequency_map[signature] = pattern
        
        return frequency_map
    
    def calculate_survival_rates(self, mutations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate survival rates for different pattern types.
        
        Args:
            mutations: List of mutations with pattern and survival info
            
        Returns:
            Dictionary mapping pattern types to survival rates
        """
        pattern_stats: Dict[str, Dict[str, int]] = {}
        
        for mutation in mutations:
            pattern_type = mutation.get("pattern", "unknown")
            survived = mutation.get("survived", False)
            
            if pattern_type not in pattern_stats:
                pattern_stats[pattern_type] = {"total": 0, "survived": 0}
            
            pattern_stats[pattern_type]["total"] += 1
            if survived:
                pattern_stats[pattern_type]["survived"] += 1
        
        survival_rates = {}
        for pattern_type, stats in pattern_stats.items():
            survival_rates[pattern_type] = round(stats["survived"] / stats["total"], 2)
        
        return survival_rates
    
    def classify_pattern(self, pattern: MutationPattern) -> PatternCategory:
        """
        Classify a pattern into a category.
        
        Args:
            pattern: The pattern to classify
            
        Returns:
            PatternCategory enum value
        """
        # Check direct signature matches first
        signature = pattern.pattern_signature
        
        # Special handling for BoolOp patterns (should be complex conditions)
        if pattern.ast_type == "BoolOp":
            pattern.category = PatternCategory.COMPLEX_CONDITION
            self._set_pattern_properties_by_category(pattern, PatternCategory.COMPLEX_CONDITION)
            return PatternCategory.COMPLEX_CONDITION
        
        for rule_signature, category in self._classification_rules.items():
            if rule_signature in signature:
                pattern.category = category
                self._set_pattern_properties_by_category(pattern, category)
                return category
        
        # Fallback classification based on AST type and context
        if pattern.ast_type == "Compare":
            if pattern.includes_null_check:
                pattern.category = PatternCategory.NULL_CHECK
            elif pattern.operator_type in ["Gt", "Lt", "GtE", "LtE"]:
                pattern.category = PatternCategory.BOUNDARY
                pattern.risk_level = "high"
            else:
                pattern.category = PatternCategory.LOGICAL
        
        elif pattern.ast_type == "ExceptHandler":
            pattern.category = PatternCategory.ERROR_HANDLING
            pattern.importance = "critical"
        
        elif pattern.ast_type == "BoolOp":
            pattern.category = PatternCategory.COMPLEX_CONDITION
        
        else:
            pattern.category = PatternCategory.UNKNOWN
        
        if pattern.category:
            self._set_pattern_properties_by_category(pattern, pattern.category)
        
        return pattern.category or PatternCategory.UNKNOWN
    
    def _set_pattern_properties_by_category(self, pattern: MutationPattern, 
                                          category: PatternCategory) -> None:
        """Set pattern properties based on its category."""
        if category == PatternCategory.BOUNDARY:
            pattern.risk_level = "high"
            pattern.importance = "high"
        elif category == PatternCategory.ERROR_HANDLING:
            pattern.importance = "critical"
            pattern.risk_level = "critical"
        elif category == PatternCategory.NULL_CHECK:
            pattern.importance = "high"
            pattern.risk_level = "medium"
        elif category == PatternCategory.COMPLEX_CONDITION:
            pattern.importance = "medium"
            pattern.risk_level = "medium"
    
    def calculate_impact_weight(self, pattern: MutationPattern) -> float:
        """
        Calculate impact weight for a pattern based on multiple factors.
        
        Args:
            pattern: The pattern to analyze
            
        Returns:
            Impact score between 0.0 and 1.0
        """
        impact_score = 0.0
        
        # Factor 1: Frequency and survival rate
        if pattern.frequency > 0 or pattern.survival_rate > 0:
            frequency_weight = min(pattern.frequency / 40.0, 1.0)  # Normalize to 40 as high
            survival_weight = pattern.survival_rate
            frequency_impact = (frequency_weight * 0.4) + (survival_weight * 0.6)
            impact_score += frequency_impact * 0.8
        
        # Factor 2: Code location criticality
        location_impact = self._calculate_location_impact(pattern)
        location_weight = 0.3 if pattern.frequency > 0 else 0.8  # Higher weight if no frequency data
        impact_score += location_impact * location_weight
        
        # Factor 3: Complexity metrics
        complexity_impact = self._calculate_complexity_impact(pattern)
        complexity_weight = 0.1 if pattern.frequency > 0 else 0.5  # Higher weight if no frequency data
        impact_score += complexity_impact * complexity_weight
        
        pattern.impact_score = min(impact_score, 1.0)
        return pattern.impact_score
    
    def _calculate_location_impact(self, pattern: MutationPattern) -> float:
        """Calculate impact based on code locations."""
        if not pattern.locations:
            # Use affected_files or critical_locations if available
            if pattern.affected_files:
                critical_keywords = ["security", "auth", "payment", "crypto", "password"]
                critical_count = sum(1 for file_path in pattern.affected_files 
                                   if any(keyword in file_path.lower() for keyword in critical_keywords))
                pattern.critical_locations = [f for f in pattern.affected_files 
                                            if any(keyword in f.lower() for keyword in critical_keywords)]
                return min(critical_count / len(pattern.affected_files) + 0.5, 1.0)
            return 0.5  # Default medium impact
        
        critical_keywords = ["security", "auth", "payment", "crypto", "password"]
        critical_locations = []
        total_complexity = 0
        
        for location in pattern.locations:
            file_path = location.get("file", "").lower()
            function_name = location.get("function", "").lower()
            complexity = location.get("complexity", 5)
            
            total_complexity += complexity
            
            # Check for critical locations
            if any(keyword in file_path or keyword in function_name 
                   for keyword in critical_keywords):
                critical_locations.append((file_path, complexity))
        
        # Sort by complexity (descending) and extract just the file names
        critical_locations.sort(key=lambda x: x[1], reverse=True)
        pattern.critical_locations = [loc[0] for loc in critical_locations]
        
        # Calculate impact
        criticality_impact = len(critical_locations) / len(pattern.locations)
        complexity_impact = min(total_complexity / len(pattern.locations) / 8.0, 1.0)
        
        # Give extra boost for any critical locations found
        boost = 0.71 if critical_locations else 0.0
        
        # Base impact calculation
        base_impact = (criticality_impact * 0.3) + (complexity_impact * 0.2) + boost
        
        return min(base_impact, 1.0)
    
    def _calculate_complexity_impact(self, pattern: MutationPattern) -> float:
        """Calculate impact based on complexity metrics."""
        impact = 0.0
        
        if pattern.average_complexity:
            # Normalize complexity (10+ is high)
            complexity_factor = min(pattern.average_complexity / 10.0, 1.0)
            impact += complexity_factor * 0.6
        
        if pattern.max_nesting_depth:
            # Normalize nesting depth (5+ is high)
            nesting_factor = min(pattern.max_nesting_depth / 5.0, 1.0)
            impact += nesting_factor * 0.25
        
        if pattern.involves_loops:
            impact += 0.3
        
        return min(impact, 1.0)
    
    def generate_improvement_suggestions(self, pattern: MutationPattern) -> List[TestImprovement]:
        """
        Generate test improvement suggestions for a pattern.
        
        Args:
            pattern: The pattern to generate suggestions for
            
        Returns:
            List of TestImprovement objects
        """
        suggestions = []
        
        if not pattern.category:
            self.classify_pattern(pattern)
        
        category = pattern.category
        
        if category == PatternCategory.BOUNDARY:
            suggestions.extend(self._generate_boundary_suggestions(pattern))
        elif category == PatternCategory.NULL_CHECK:
            suggestions.extend(self._generate_null_check_suggestions(pattern))
        elif category == PatternCategory.ERROR_HANDLING:
            suggestions.extend(self._generate_error_handling_suggestions(pattern))
        elif category == PatternCategory.COMPLEX_CONDITION:
            suggestions.extend(self._generate_complex_condition_suggestions(pattern))
        
        return suggestions
    
    def _generate_boundary_suggestions(self, pattern: MutationPattern) -> List[TestImprovement]:
        """Generate suggestions for boundary condition patterns."""
        suggestions = []
        
        # Edge case testing suggestion
        edge_case_suggestion = TestImprovement(
            description="Add comprehensive boundary value testing including edge cases and off-by-one scenarios",
            priority="high",
            pattern_type="boundary",
            test_template=ImprovementTemplate(
                assertion_patterns=["assert_boundary_conditions", "assert_edge_cases"]
            ),
            assertion_patterns=["assert_boundary_conditions", "assert_edge_cases"]
        )
        edge_case_suggestion.suggested_tests = [
            ImprovementSuggestion(priority="critical", description="Test boundary values (0, 1, -1)")
        ]
        suggestions.append(edge_case_suggestion)
        
        # Equivalence class testing
        equiv_suggestion = TestImprovement(
            description="Implement equivalence class partitioning for boundary conditions",
            priority="medium",
            pattern_type="boundary",
            test_template=ImprovementTemplate(
                assertion_patterns=["assert_equivalence_classes"]
            )
        )
        suggestions.append(equiv_suggestion)
        
        return suggestions
    
    def _generate_null_check_suggestions(self, pattern: MutationPattern) -> List[TestImprovement]:
        """Generate suggestions for null/None check patterns."""
        suggestions = []
        
        null_test_suggestion = TestImprovement(
            description="Add comprehensive None/null value testing for all code paths",
            priority="high",
            pattern_type="null_check",
            test_template=ImprovementTemplate(
                assertion_patterns=["assert_none_handling", "assert_not_none_validation"]
            ),
            assertion_patterns=["assert_none_handling", "assert_not_none_validation"]
        )
        null_test_suggestion.suggested_tests = [
            ImprovementSuggestion(priority="high", description="Test None input handling")
        ]
        suggestions.append(null_test_suggestion)
        
        return suggestions
    
    def _generate_error_handling_suggestions(self, pattern: MutationPattern) -> List[TestImprovement]:
        """Generate suggestions for error handling patterns."""
        suggestions = []
        
        exception_suggestion = TestImprovement(
            description="Improve error handling and exception test coverage with specific exception types",
            priority="critical",
            pattern_type="error_handling",
            test_template=ImprovementTemplate(
                includes_exception_testing=True,
                assertion_patterns=["assert_raises_specific_exception"]
            )
        )
        exception_suggestion.suggested_tests = [
            ImprovementSuggestion(priority="critical", description="Test specific exception types")
        ]
        suggestions.append(exception_suggestion)
        
        return suggestions
    
    def _generate_complex_condition_suggestions(self, pattern: MutationPattern) -> List[TestImprovement]:
        """Generate suggestions for complex condition patterns."""
        suggestions = []
        
        condition_suggestion = TestImprovement(
            description="Add tests for all logical combinations in complex conditions",
            priority="medium",
            pattern_type="complex_condition",
            test_template=ImprovementTemplate(
                assertion_patterns=["assert_all_logical_paths"]
            )
        )
        suggestions.append(condition_suggestion)
        
        return suggestions
    
    def learn_from_history(self, historical_data: Dict[str, Any]) -> List[MutationPattern]:
        """
        Learn patterns from historical mutation data.
        
        Args:
            historical_data: Dictionary containing historical mutation results
            
        Returns:
            List of learned patterns with confidence scores
        """
        patterns = []
        mutations = historical_data.get("mutations", [])
        
        if not mutations:
            return patterns
        
        # Group mutations by pattern type for historical data
        pattern_groups = {}
        for mutation in mutations:
            pattern_type = mutation.get("pattern", "unknown")
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(mutation)
        
        # Create patterns from groups
        for pattern_type, mutation_list in pattern_groups.items():
            if len(mutation_list) > 0:  # Only process non-empty groups
                survived_count = sum(1 for m in mutation_list if m.get("survived", False))
                total_count = len(mutation_list)
                survival_rate = survived_count / total_count if total_count > 0 else 0.0
                
                pattern = MutationPattern(
                    pattern_type=pattern_type,
                    frequency=total_count,
                    survival_rate=survival_rate,
                    sample_size=total_count
                )
                
                # Calculate historical confidence
                pattern.historical_confidence = self._calculate_historical_confidence(
                    pattern, len(mutations)
                )
                
                # Determine trend direction
                pattern.trend_direction = self._analyze_trend_direction(pattern, mutation_list)
                
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_historical_confidence(self, pattern: MutationPattern, 
                                       total_mutations: int) -> float:
        """Calculate confidence based on historical data."""
        # Base confidence on sample size and consistency
        sample_ratio = pattern.frequency / total_mutations
        base_confidence = min(sample_ratio * 10, 1.0)  # Scale up small samples
        
        # Adjust for survival rate consistency
        survival_consistency = 1.0 - abs(pattern.survival_rate - 0.5) * 2
        
        return (base_confidence * 0.7) + (survival_consistency * 0.3)
    
    def _analyze_trend_direction(self, pattern: MutationPattern, 
                               mutations: List[Dict[str, Any]]) -> str:
        """Analyze trend direction for pattern occurrence."""
        # Simple heuristic: look at survival rate
        if pattern.survival_rate > 0.6:
            return "increasing"
        elif pattern.survival_rate < 0.4:
            return "decreasing"
        else:
            return "stable"
    
    def update_pattern(self, existing_pattern: MutationPattern, 
                      new_mutations: List[Dict[str, Any]]) -> MutationPattern:
        """
        Update an existing pattern with new mutation data.
        
        Args:
            existing_pattern: The pattern to update
            new_mutations: New mutation data
            
        Returns:
            Updated pattern
        """
        new_frequency = len(new_mutations)
        new_survived = sum(1 for m in new_mutations if m.get("survived", False))
        
        # Update frequency
        total_frequency = existing_pattern.frequency + new_frequency
        
        # Calculate existing survived mutations correctly
        # Keep precision to handle cases where survival_rate * frequency isn't integer
        existing_survived = existing_pattern.frequency * existing_pattern.survival_rate
        total_survived = existing_survived + new_survived
        new_survival_rate = total_survived / total_frequency if total_frequency > 0 else 0.0
        
        # Create updated pattern
        updated_pattern = MutationPattern(
            pattern_signature=existing_pattern.pattern_signature,
            frequency=total_frequency,
            survival_rate=round(new_survival_rate, 2)
        )
        
        return updated_pattern
    
    def identify_emerging_patterns(self, recent_mutations: List[Dict[str, Any]]) -> List[MutationPattern]:
        """
        Identify emerging patterns in recent mutations.
        
        Args:
            recent_mutations: List of recent mutation data
            
        Returns:
            List of emerging patterns
        """
        emerging_patterns = []
        
        # Group by pattern type
        pattern_groups: Dict[str, List[Dict[str, Any]]] = {}
        for mutation in recent_mutations:
            pattern_type = mutation.get("pattern", "unknown")
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(mutation)
        
        # Identify patterns with high emergence confidence
        for pattern_type, mutations in pattern_groups.items():
            if len(mutations) >= 3:  # Minimum threshold for emergence
                survival_rate = sum(1 for m in mutations if m.get("survived", False)) / len(mutations)
                if survival_rate > 0.6:  # High survival rate indicates emerging threat
                    pattern = MutationPattern(
                        pattern_type=pattern_type,
                        frequency=len(mutations),
                        survival_rate=survival_rate,
                        emergence_confidence=min(survival_rate + (len(mutations) / 10.0), 1.0)
                    )
                    emerging_patterns.append(pattern)
        
        return emerging_patterns
    
    def calculate_confidence_score(self, pattern: MutationPattern) -> float:
        """
        Calculate confidence score for a pattern.
        
        Args:
            pattern: The pattern to score
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Sample size factor (larger samples = higher confidence)
        sample_factor = min(pattern.sample_size / 30.0, 1.0) if pattern.sample_size else 0.1
        
        # Frequency factor
        frequency_factor = min(pattern.frequency / 20.0, 1.0)
        
        # Consistency factor
        consistency_factor = pattern.occurrence_consistency
        
        # Calculate weighted confidence
        confidence = (sample_factor * 0.4) + (frequency_factor * 0.3) + (consistency_factor * 0.3)
        
        # Set confidence level
        if confidence > 0.8:
            pattern.confidence_level = "high"
        elif confidence > 0.5:
            pattern.confidence_level = "medium"
        else:
            pattern.confidence_level = "low"
        
        return confidence
    
    def adjust_confidence_for_volatility(self, pattern: MutationPattern) -> float:
        """
        Adjust confidence based on code volatility.
        
        Args:
            pattern: Pattern with base confidence and location data
            
        Returns:
            Adjusted confidence score
        """
        base_confidence = pattern.base_confidence
        
        # Calculate volatility factor based on recent modifications
        volatility_penalty = 0.0
        code_locations = getattr(pattern, 'code_locations', pattern.locations)
        
        if code_locations:
            recent_modifications = 0
            for location in code_locations:
                last_modified = location.get("last_modified", "2020-01-01")
                # Simple heuristic: if modified in 2024, it's volatile
                if "2024" in last_modified:
                    recent_modifications += 1
            
            volatility_factor = recent_modifications / len(code_locations)
            volatility_penalty = volatility_factor * 0.2  # Up to 20% penalty
            pattern.volatility_factor = volatility_factor
        
        adjusted_confidence = max(base_confidence - volatility_penalty, 0.0)
        return adjusted_confidence
    
    def generate_pattern_report(self, patterns: List[MutationPattern]) -> PatternReport:
        """
        Generate a comprehensive pattern analysis report.
        
        Args:
            patterns: List of patterns to include in report
            
        Returns:
            PatternReport object
        """
        high_impact_count = sum(1 for p in patterns if p.impact_score > 0.7)
        
        # Create pattern summaries
        summaries = []
        for pattern in patterns:
            summary = {
                "pattern_type": pattern.pattern_type,
                "frequency": pattern.frequency,
                "survival_rate": pattern.survival_rate,
                "impact_score": pattern.impact_score,
                "category": pattern.category.value if pattern.category else "unknown"
            }
            summaries.append(summary)
        
        # Generate recommendations
        recommendations = self._generate_report_recommendations(patterns, high_impact_count)
        
        report = PatternReport(
            total_patterns=len(patterns),
            high_impact_patterns=high_impact_count,
            recommendations=recommendations,
            pattern_summaries=summaries,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        )
        
        return report
    
    def _generate_report_recommendations(self, patterns: List[MutationPattern], 
                                       high_impact_count: int) -> str:
        """Generate text recommendations for the report."""
        if high_impact_count > len(patterns) * 0.5:
            return "High number of critical patterns detected. Immediate test improvement recommended."
        elif high_impact_count > 0:
            return "Some high-impact patterns found. Consider prioritizing test improvements."
        else:
            return "Pattern analysis complete. Regular monitoring recommended."
    
    def export_patterns_to_json(self, patterns: List[MutationPattern]) -> str:
        """
        Export patterns to JSON format.
        
        Args:
            patterns: List of patterns to export
            
        Returns:
            JSON string representation
        """
        export_data = {
            "patterns": [],
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "1.0",
                "total_patterns": len(patterns)
            }
        }
        
        for pattern in patterns:
            pattern_dict = {
                "pattern_signature": pattern.pattern_signature,
                "pattern_type": pattern.pattern_type,
                "frequency": pattern.frequency,
                "survival_rate": pattern.survival_rate,
                "impact_score": pattern.impact_score,
                "category": pattern.category.value if pattern.category else None,
                "risk_level": pattern.risk_level,
                "confidence_level": pattern.confidence_level
            }
            export_data["patterns"].append(pattern_dict)
        
        return json.dumps(export_data, indent=2)
    
    def analyze_mutation_results(self, mutation_results: Dict[str, Any]) -> List[MutationPattern]:
        """
        Analyze mutation testing results from external tools.
        
        Args:
            mutation_results: Results from mutation testing framework
            
        Returns:
            List of identified patterns
        """
        patterns = []
        survived_mutants = mutation_results.get("survived_mutants", [])
        
        for mutant in survived_mutants:
            # Convert mutant data to pattern
            mutation_data = {
                "id": mutant.get("id"),
                "original_code": self._extract_original_code_from_location(mutant),
                "mutated_code": self._extract_mutated_code_from_mutation(mutant),
                "mutation_type": self._classify_mutation_type(mutant),
                "survived": mutant.get("status") == "survived"
            }
            
            pattern = self.extract_pattern(mutation_data)
            if pattern:
                pattern.source = "mutmut"  # Mark source
                patterns.append(pattern)
        
        return patterns
    
    def _extract_original_code_from_location(self, mutant: Dict[str, Any]) -> str:
        """Extract original code from mutant location info."""
        # This would need integration with actual file reading
        # For now, return a placeholder based on mutation description
        mutation_desc = mutant.get("mutation", "")
        if "to" in mutation_desc:
            parts = mutation_desc.split(" to ")
            return parts[0] if len(parts) > 1 else mutation_desc
        return "unknown"
    
    def _extract_mutated_code_from_mutation(self, mutant: Dict[str, Any]) -> str:
        """Extract mutated code from mutation description."""
        mutation_desc = mutant.get("mutation", "")
        if "to" in mutation_desc:
            parts = mutation_desc.split(" to ")
            return parts[1] if len(parts) > 1 else mutation_desc
        return "unknown"
    
    def _classify_mutation_type(self, mutant: Dict[str, Any]) -> str:
        """Classify mutation type from mutant data."""
        mutation_desc = mutant.get("mutation", "").lower()
        
        if any(op in mutation_desc for op in ["<", ">", "<=", ">="]):
            return "boundary"
        elif "none" in mutation_desc:
            return "null_check"
        elif "except" in mutation_desc:
            return "exception"
        else:
            return "unknown"
    
    def prioritize_tests_for_patterns(self, patterns: List[MutationPattern]) -> List[TestImprovement]:
        """
        Prioritize test improvements based on pattern analysis.
        
        Args:
            patterns: List of patterns to prioritize
            
        Returns:
            List of prioritized test improvements
        """
        # Sort patterns by impact score (highest first)
        sorted_patterns = sorted(patterns, key=lambda p: p.impact_score, reverse=True)
        
        test_priorities = []
        
        for pattern in sorted_patterns:
            # Generate test improvement with priority based on impact
            priority = "critical" if pattern.impact_score > 0.8 else "high" if pattern.impact_score > 0.6 else "medium"
            
            test_improvement = TestImprovement(
                description=f"Improve tests for {pattern.pattern_type} pattern",
                priority=priority,
                pattern_type=pattern.pattern_type,
                affected_modules=pattern.affected_files
            )
            
            # Add specific test suggestions
            test_improvement.suggested_tests = [
                ImprovementSuggestion(
                    priority=priority,
                    description=f"Add comprehensive {pattern.pattern_type} testing"
                )
            ]
            
            test_priorities.append(test_improvement)
        
        return test_priorities