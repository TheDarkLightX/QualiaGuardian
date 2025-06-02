"""
Quality Factor Calculator

Comprehensive test quality assessment including determinism,
stability, clarity, and independence metrics.
"""

import numpy as np
import logging
import re
import ast
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import time

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for quality factor components"""
    determinism: float = 1.0
    stability: float = 1.0
    clarity: float = 1.0
    independence: float = 1.0
    
    # Detailed sub-metrics
    flakiness_score: float = 0.0
    readability_score: float = 1.0
    coupling_score: float = 0.0
    complexity_score: float = 0.0
    
    # Metadata
    measurement_confidence: float = 1.0
    sample_size: int = 1
    
    def calculate_quality_factor(self) -> float:
        """Calculate geometric mean quality factor"""
        return (self.determinism * self.stability * self.clarity * self.independence) ** 0.25


class QualityFactorCalculator:
    """
    Advanced quality factor calculator for test reliability assessment
    """
    
    def __init__(self, sample_runs: int = 10):
        self.sample_runs = sample_runs
        self.execution_cache = {}
        
    def calculate_quality_factor(self, test_data: Dict[str, Any], 
                               test_code: str = None) -> QualityMetrics:
        """
        Calculate comprehensive quality factor for a test
        
        Args:
            test_data: Test execution and metadata
            test_code: Optional test source code for static analysis
            
        Returns:
            QualityMetrics with detailed quality assessment
        """
        try:
            metrics = QualityMetrics()
            
            # Calculate determinism (consistency across runs)
            metrics.determinism = self._calculate_determinism(test_data)
            
            # Calculate stability (resistance to environmental changes)
            metrics.stability = self._calculate_stability(test_data)
            
            # Calculate clarity (readability and maintainability)
            if test_code:
                metrics.clarity = self._calculate_clarity(test_code)
            
            # Calculate independence (coupling with other tests)
            metrics.independence = self._calculate_independence(test_data)
            
            # Calculate detailed sub-metrics
            metrics.flakiness_score = self._calculate_flakiness(test_data)
            metrics.readability_score = self._calculate_readability(test_code) if test_code else 1.0
            metrics.coupling_score = self._calculate_coupling(test_data)
            metrics.complexity_score = self._calculate_complexity(test_code) if test_code else 0.0
            
            # Set measurement confidence
            metrics.measurement_confidence = self._calculate_confidence(test_data)
            metrics.sample_size = test_data.get('execution_count', 1)
            
            logger.debug(f"Quality factor calculated: {metrics.calculate_quality_factor():.3f}")
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating quality factor: {e}")
            return QualityMetrics()  # Return default metrics on error
    
    def _calculate_determinism(self, test_data: Dict[str, Any]) -> float:
        """Calculate test determinism (consistency across multiple runs)"""
        try:
            execution_results = test_data.get('execution_results', [])
            
            if len(execution_results) < 2:
                # Insufficient data, assume perfect determinism
                return 1.0
            
            # Check result consistency
            results = [result.get('passed', False) for result in execution_results]
            unique_results = set(results)
            
            if len(unique_results) == 1:
                # All results are the same
                determinism = 1.0
            else:
                # Calculate consistency ratio
                most_common_result = Counter(results).most_common(1)[0][1]
                determinism = most_common_result / len(results)
            
            # Check execution time consistency
            execution_times = [result.get('execution_time_ms', 0) for result in execution_results]
            if execution_times:
                time_cv = np.std(execution_times) / np.mean(execution_times) if np.mean(execution_times) > 0 else 0
                time_consistency = max(0, 1.0 - time_cv)
                determinism = (determinism + time_consistency) / 2
            
            return max(0.0, min(determinism, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating determinism: {e}")
            return 1.0
    
    def _calculate_stability(self, test_data: Dict[str, Any]) -> float:
        """Calculate test stability (resistance to environmental changes)"""
        try:
            stability_factors = []
            
            # Check for environment-dependent failures
            env_failures = test_data.get('environment_failures', 0)
            total_runs = test_data.get('total_runs', 1)
            env_stability = 1.0 - (env_failures / total_runs) if total_runs > 0 else 1.0
            stability_factors.append(env_stability)
            
            # Check for timing-dependent issues
            timing_issues = test_data.get('timing_issues', 0)
            timing_stability = 1.0 - (timing_issues / total_runs) if total_runs > 0 else 1.0
            stability_factors.append(timing_stability)
            
            # Check for resource-dependent issues
            resource_issues = test_data.get('resource_issues', 0)
            resource_stability = 1.0 - (resource_issues / total_runs) if total_runs > 0 else 1.0
            stability_factors.append(resource_stability)
            
            # Check modification frequency (lower is more stable)
            modification_frequency = test_data.get('modification_frequency', 0.0)
            modification_stability = max(0.0, 1.0 - modification_frequency)
            stability_factors.append(modification_stability)
            
            # Calculate overall stability as geometric mean
            if stability_factors:
                stability = np.prod(stability_factors) ** (1.0 / len(stability_factors))
            else:
                stability = 1.0
            
            return max(0.0, min(stability, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating stability: {e}")
            return 1.0
    
    def _calculate_clarity(self, test_code: str) -> float:
        """Calculate test clarity (readability and maintainability)"""
        try:
            clarity_score = 1.0
            
            # Calculate readability score
            readability = self._calculate_readability(test_code)
            clarity_score *= readability
            
            # Check for good naming conventions
            naming_score = self._assess_naming_quality(test_code)
            clarity_score *= naming_score
            
            # Check for appropriate comments
            comment_score = self._assess_comment_quality(test_code)
            clarity_score *= comment_score
            
            # Check for test structure (Arrange-Act-Assert)
            structure_score = self._assess_test_structure(test_code)
            clarity_score *= structure_score
            
            return max(0.0, min(clarity_score, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating clarity: {e}")
            return 1.0
    
    def _calculate_independence(self, test_data: Dict[str, Any]) -> float:
        """Calculate test independence (low coupling with other tests)"""
        try:
            # Check for shared state dependencies
            shared_state_deps = test_data.get('shared_state_dependencies', 0)
            max_deps = test_data.get('max_possible_dependencies', 10)
            state_independence = 1.0 - (shared_state_deps / max_deps) if max_deps > 0 else 1.0
            
            # Check for execution order dependencies
            order_deps = test_data.get('execution_order_dependencies', 0)
            order_independence = 1.0 - (order_deps / 5.0)  # Assume max 5 order deps
            
            # Check for external resource dependencies
            external_deps = test_data.get('external_dependencies', 0)
            external_independence = 1.0 - (external_deps / 3.0)  # Assume max 3 external deps
            
            # Calculate overall independence
            independence_factors = [state_independence, order_independence, external_independence]
            independence = np.mean([max(0.0, min(factor, 1.0)) for factor in independence_factors])
            
            return max(0.0, min(independence, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating independence: {e}")
            return 1.0
    
    def _calculate_flakiness(self, test_data: Dict[str, Any]) -> float:
        """Calculate test flakiness score (0 = not flaky, 1 = very flaky)"""
        try:
            execution_results = test_data.get('execution_results', [])
            
            if len(execution_results) < 3:
                return 0.0  # Insufficient data to determine flakiness
            
            # Check for inconsistent results
            results = [result.get('passed', False) for result in execution_results]
            result_changes = sum(1 for i in range(1, len(results)) if results[i] != results[i-1])
            
            # Flakiness based on result inconsistency
            flakiness = result_changes / (len(results) - 1) if len(results) > 1 else 0.0
            
            # Check for timeout issues
            timeouts = sum(1 for result in execution_results if result.get('timeout', False))
            timeout_flakiness = timeouts / len(execution_results)
            
            # Combine flakiness factors
            total_flakiness = max(flakiness, timeout_flakiness)
            
            return max(0.0, min(total_flakiness, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating flakiness: {e}")
            return 0.0
    
    def _calculate_readability(self, test_code: str) -> float:
        """Calculate code readability score"""
        try:
            if not test_code:
                return 1.0
            
            readability_score = 1.0
            lines = test_code.split('\n')
            
            # Check line length (penalize very long lines)
            long_lines = sum(1 for line in lines if len(line) > 120)
            if long_lines > 0:
                readability_score *= max(0.5, 1.0 - (long_lines / len(lines)))
            
            # Check for excessive nesting
            max_indent = 0
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    max_indent = max(max_indent, indent)
            
            if max_indent > 16:  # More than 4 levels of nesting
                readability_score *= 0.8
            
            # Check for magic numbers
            magic_numbers = len(re.findall(r'\b\d{2,}\b', test_code))
            if magic_numbers > 3:
                readability_score *= 0.9
            
            # Check for descriptive variable names
            variables = re.findall(r'\b[a-z_][a-z0-9_]*\b', test_code.lower())
            short_vars = sum(1 for var in variables if len(var) <= 2 and var not in ['i', 'j', 'k'])
            if short_vars > len(variables) * 0.3:
                readability_score *= 0.8
            
            return max(0.0, min(readability_score, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating readability: {e}")
            return 1.0
    
    def _calculate_coupling(self, test_data: Dict[str, Any]) -> float:
        """Calculate test coupling score (0 = no coupling, 1 = high coupling)"""
        try:
            coupling_factors = []
            
            # Data coupling (shared data dependencies)
            data_deps = test_data.get('data_dependencies', 0)
            coupling_factors.append(min(data_deps / 5.0, 1.0))
            
            # Control coupling (execution order dependencies)
            control_deps = test_data.get('control_dependencies', 0)
            coupling_factors.append(min(control_deps / 3.0, 1.0))
            
            # Common coupling (global state usage)
            global_state_usage = test_data.get('global_state_usage', 0)
            coupling_factors.append(min(global_state_usage / 2.0, 1.0))
            
            # Content coupling (direct access to other test internals)
            content_coupling = test_data.get('content_coupling', 0)
            coupling_factors.append(min(content_coupling / 1.0, 1.0))
            
            # Calculate overall coupling as maximum (worst case)
            coupling = max(coupling_factors) if coupling_factors else 0.0
            
            return max(0.0, min(coupling, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating coupling: {e}")
            return 0.0
    
    def _calculate_complexity(self, test_code: str) -> float:
        """Calculate test complexity score (0 = simple, 1 = very complex)"""
        try:
            if not test_code:
                return 0.0
            
            complexity_factors = []
            
            # Cyclomatic complexity
            try:
                tree = ast.parse(test_code)
                cyclomatic = self._calculate_cyclomatic_complexity(tree)
                complexity_factors.append(min(cyclomatic / 10.0, 1.0))
            except SyntaxError:
                complexity_factors.append(0.5)  # Assume moderate complexity if can't parse
            
            # Line count complexity
            lines = len([line for line in test_code.split('\n') if line.strip()])
            complexity_factors.append(min(lines / 50.0, 1.0))
            
            # Assertion count complexity
            assertion_count = test_code.count('assert')
            complexity_factors.append(min(assertion_count / 10.0, 1.0))
            
            # Nesting depth complexity
            max_nesting = self._calculate_max_nesting_depth(test_code)
            complexity_factors.append(min(max_nesting / 5.0, 1.0))
            
            # Calculate overall complexity as weighted average
            weights = [0.4, 0.2, 0.2, 0.2]
            complexity = sum(factor * weight for factor, weight in zip(complexity_factors, weights))
            
            return max(0.0, min(complexity, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating complexity: {e}")
            return 0.0
    
    def _assess_naming_quality(self, test_code: str) -> float:
        """Assess quality of naming conventions"""
        try:
            # Check for descriptive test function names
            function_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', test_code)
            
            naming_score = 1.0
            
            for name in function_names:
                if name.startswith('test_'):
                    # Good test naming convention
                    if len(name) < 10:
                        naming_score *= 0.9  # Prefer longer, descriptive names
                    if '_' not in name[5:]:  # No underscores after 'test_'
                        naming_score *= 0.8  # Prefer snake_case
                else:
                    naming_score *= 0.7  # Should start with 'test_'
            
            # Check variable naming
            variables = re.findall(r'\b([a-z_][a-z0-9_]*)\s*=', test_code)
            descriptive_vars = sum(1 for var in variables if len(var) > 3)
            if variables:
                var_quality = descriptive_vars / len(variables)
                naming_score *= (0.5 + 0.5 * var_quality)
            
            return max(0.0, min(naming_score, 1.0))
            
        except Exception as e:
            logger.warning(f"Error assessing naming quality: {e}")
            return 1.0
    
    def _assess_comment_quality(self, test_code: str) -> float:
        """Assess quality and appropriateness of comments"""
        try:
            lines = test_code.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            
            if not code_lines:
                return 1.0
            
            comment_ratio = len(comment_lines) / len(code_lines)
            
            # Optimal comment ratio is around 10-20%
            if 0.1 <= comment_ratio <= 0.2:
                comment_score = 1.0
            elif comment_ratio < 0.1:
                comment_score = 0.8  # Too few comments
            elif comment_ratio > 0.5:
                comment_score = 0.6  # Too many comments (might indicate unclear code)
            else:
                comment_score = 0.9
            
            # Check for meaningful comments (not just "# test" or "# TODO")
            meaningful_comments = sum(1 for comment in comment_lines 
                                    if len(comment.strip()) > 10 and 
                                    not any(word in comment.lower() for word in ['todo', 'fixme', 'hack']))
            
            if comment_lines:
                meaningful_ratio = meaningful_comments / len(comment_lines)
                comment_score *= (0.5 + 0.5 * meaningful_ratio)
            
            return max(0.0, min(comment_score, 1.0))
            
        except Exception as e:
            logger.warning(f"Error assessing comment quality: {e}")
            return 1.0
    
    def _assess_test_structure(self, test_code: str) -> float:
        """Assess test structure (Arrange-Act-Assert pattern)"""
        try:
            lines = [line.strip() for line in test_code.split('\n') if line.strip()]
            
            if len(lines) < 3:
                return 0.5  # Too short to have good structure
            
            structure_score = 1.0
            
            # Look for Arrange-Act-Assert pattern
            has_setup = any('=' in line and 'assert' not in line for line in lines[:len(lines)//2])
            has_action = any(line and not line.startswith('assert') and '=' not in line 
                           for line in lines[1:-1])
            has_assertions = any('assert' in line for line in lines)
            
            if not has_setup:
                structure_score *= 0.8
            if not has_action:
                structure_score *= 0.8
            if not has_assertions:
                structure_score *= 0.5
            
            # Check for logical grouping (blank lines between sections)
            blank_lines = sum(1 for i, line in enumerate(test_code.split('\n')) 
                            if not line.strip() and i > 0 and i < len(lines) - 1)
            
            if blank_lines >= 2:
                structure_score *= 1.1  # Bonus for good sectioning
            
            return max(0.0, min(structure_score, 1.0))
            
        except Exception as e:
            logger.warning(f"Error assessing test structure: {e}")
            return 1.0
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of AST"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _calculate_max_nesting_depth(self, test_code: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for line in test_code.split('\n'):
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                depth = indent // 4  # Assuming 4-space indentation
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_confidence(self, test_data: Dict[str, Any]) -> float:
        """Calculate confidence in quality measurements"""
        execution_count = test_data.get('execution_count', 1)
        
        # Confidence increases with more execution samples
        if execution_count >= 10:
            return 1.0
        elif execution_count >= 5:
            return 0.8
        elif execution_count >= 3:
            return 0.6
        else:
            return 0.4
