"""
Information-Theoretic Quality Metrics

Measures information content, redundancy, diversity, and dependencies
in test suites using information theory principles.
"""

import math
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class InformationMetrics:
    """Container for information-theoretic metrics."""
    test_suite_entropy: float = 0.0
    mutual_information: float = 0.0
    kl_divergence: float = 0.0
    information_gain: float = 0.0
    redundancy_score: float = 0.0
    diversity_score: float = 0.0
    coverage_efficiency: float = 0.0


class InformationTheoreticAnalyzer:
    """
    Analyzes test suites using information theory.
    
    Measures:
    - Entropy: Diversity and information content
    - Mutual Information: Dependencies between tests and code
    - KL Divergence: Distance from optimal distribution
    - Information Gain: Improvement in information content
    """
    
    @staticmethod
    def calculate_entropy(probabilities: List[float], base: float = 2.0) -> float:
        """
        Calculate Shannon entropy: H(X) = -Σ p(x) * log(p(x))
        
        Args:
            probabilities: List of probabilities (must sum to 1)
            base: Logarithm base (2 for bits, e for nats)
            
        Returns:
            Entropy value
        """
        if not probabilities:
            return 0.0
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            return 0.0
        
        normalized = [p / total for p in probabilities]
        
        entropy = 0.0
        for p in normalized:
            if p > 0:
                entropy -= p * math.log(p, base)
        
        return entropy
    
    @staticmethod
    def calculate_test_suite_entropy(
        test_coverage: Dict[str, Set[str]],
        normalize: bool = True
    ) -> float:
        """
        Calculate entropy of test suite based on coverage diversity.
        
        High entropy = diverse coverage (tests cover different code)
        Low entropy = redundant coverage (tests cover similar code)
        
        Args:
            test_coverage: Dictionary mapping test names to sets of covered code elements
            normalize: Whether to normalize entropy by maximum possible entropy
            
        Returns:
            Test suite entropy
        """
        if not test_coverage:
            return 0.0
        
        # Count coverage for each code element
        code_element_counts = Counter()
        for test, covered in test_coverage.items():
            for element in covered:
                code_element_counts[element] += 1
        
        total_coverage_events = sum(code_element_counts.values())
        if total_coverage_events == 0:
            return 0.0
        
        # Calculate probability distribution
        probabilities = [count / total_coverage_events for count in code_element_counts.values()]
        entropy = InformationTheoreticAnalyzer.calculate_entropy(probabilities)
        
        if normalize:
            # Maximum entropy occurs when all code elements are covered equally
            max_entropy = math.log(len(code_element_counts), 2) if code_element_counts else 0.0
            if max_entropy > 0:
                entropy = entropy / max_entropy
        
        return entropy
    
    @staticmethod
    def calculate_mutual_information(
        test_coverage: Dict[str, Set[str]],
        code_importance: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate mutual information I(T; C) between tests and code.
        
        Measures dependency between tests and code coverage.
        High MI = strong coupling, low MI = independence.
        
        Args:
            test_coverage: Dictionary mapping test names to sets of covered code elements
            code_importance: Optional dictionary mapping code elements to importance scores
            
        Returns:
            Mutual information value
        """
        if not test_coverage:
            return 0.0
        
        # Build joint distribution
        test_names = list(test_coverage.keys())
        all_code_elements = set()
        for covered in test_coverage.values():
            all_code_elements.update(covered)
        
        if not all_code_elements:
            return 0.0
        
        # Calculate marginal distributions
        test_probs = {}
        code_probs = {}
        joint_probs = {}
        
        total_events = 0
        for test, covered in test_coverage.items():
            for code_element in covered:
                key = (test, code_element)
                joint_probs[key] = joint_probs.get(key, 0) + 1
                test_probs[test] = test_probs.get(test, 0) + 1
                code_probs[code_element] = code_probs.get(code_element, 0) + 1
                total_events += 1
        
        if total_events == 0:
            return 0.0
        
        # Normalize to probabilities
        for key in joint_probs:
            joint_probs[key] /= total_events
        for test in test_probs:
            test_probs[test] /= total_events
        for code in code_probs:
            code_probs[code] /= total_events
        
        # Calculate mutual information: I(T;C) = Σ Σ p(t,c) * log(p(t,c) / (p(t) * p(c)))
        mi = 0.0
        for (test, code), joint_prob in joint_probs.items():
            test_prob = test_probs.get(test, 0)
            code_prob = code_probs.get(code, 0)
            
            if joint_prob > 0 and test_prob > 0 and code_prob > 0:
                mi += joint_prob * math.log2(joint_prob / (test_prob * code_prob))
        
        return max(0.0, mi)
    
    @staticmethod
    def calculate_kl_divergence(
        actual_distribution: Dict[str, float],
        target_distribution: Dict[str, float]
    ) -> float:
        """
        Calculate KL divergence: D_KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
        
        Measures how far actual distribution is from target (optimal) distribution.
        Lower KL divergence = closer to optimal.
        
        Args:
            actual_distribution: Actual probability distribution
            target_distribution: Target (optimal) probability distribution
            
        Returns:
            KL divergence value
        """
        # Normalize distributions
        actual_sum = sum(actual_distribution.values())
        target_sum = sum(target_distribution.values())
        
        if actual_sum == 0 or target_sum == 0:
            return float('inf')
        
        actual_norm = {k: v / actual_sum for k, v in actual_distribution.items()}
        target_norm = {k: v / target_sum for k, v in target_distribution.items()}
        
        # Calculate KL divergence
        kl = 0.0
        all_keys = set(actual_norm.keys()) | set(target_norm.keys())
        
        for key in all_keys:
            p = actual_norm.get(key, 0.0)
            q = target_norm.get(key, 0.0)
            
            if p > 0:
                if q > 0:
                    kl += p * math.log2(p / q)
                else:
                    # Infinite divergence if target has zero probability
                    return float('inf')
        
        return max(0.0, kl)
    
    @staticmethod
    def calculate_information_gain(
        entropy_before: float,
        entropy_after: float
    ) -> float:
        """
        Calculate information gain: IG = H(before) - H(after)
        
        Measures improvement in information content.
        Positive IG = improvement, negative IG = degradation.
        
        Args:
            entropy_before: Entropy before change
            entropy_after: Entropy after change
            
        Returns:
            Information gain value
        """
        return entropy_before - entropy_after
    
    @staticmethod
    def calculate_redundancy_score(
        test_coverage: Dict[str, Set[str]]
    ) -> float:
        """
        Calculate redundancy score based on coverage overlap.
        
        High redundancy = many tests cover same code
        Low redundancy = tests cover different code
        
        Args:
            test_coverage: Dictionary mapping test names to sets of covered code elements
            
        Returns:
            Redundancy score [0, 1] where 1 = maximum redundancy
        """
        if not test_coverage or len(test_coverage) < 2:
            return 0.0
        
        # Count how many tests cover each code element
        code_element_coverage_count = Counter()
        for covered in test_coverage.values():
            for element in covered:
                code_element_coverage_count[element] += 1
        
        if not code_element_coverage_count:
            return 0.0
        
        # Redundancy = average number of tests covering each element
        # Normalize by number of tests
        avg_coverage = np.mean(list(code_element_coverage_count.values()))
        max_possible = len(test_coverage)
        
        # Normalize to [0, 1]
        redundancy = min(1.0, avg_coverage / max_possible) if max_possible > 0 else 0.0
        
        return redundancy
    
    @staticmethod
    def calculate_diversity_score(
        test_coverage: Dict[str, Set[str]]
    ) -> float:
        """
        Calculate diversity score (inverse of redundancy).
        
        High diversity = tests cover different code
        Low diversity = tests cover similar code
        
        Args:
            test_coverage: Dictionary mapping test names to sets of covered code elements
            
        Returns:
            Diversity score [0, 1] where 1 = maximum diversity
        """
        redundancy = InformationTheoreticAnalyzer.calculate_redundancy_score(test_coverage)
        return 1.0 - redundancy
    
    @staticmethod
    def calculate_coverage_efficiency(
        test_coverage: Dict[str, Set[str]],
        code_importance: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate coverage efficiency using information theory.
        
        Measures how efficiently tests cover important code.
        High efficiency = tests focus on important code
        Low efficiency = tests waste effort on unimportant code
        
        Args:
            test_coverage: Dictionary mapping test names to sets of covered code elements
            code_importance: Optional dictionary mapping code elements to importance scores
            
        Returns:
            Coverage efficiency score [0, 1]
        """
        if not test_coverage:
            return 0.0
        
        # If no importance scores, assume uniform importance
        if code_importance is None:
            all_elements = set()
            for covered in test_coverage.values():
                all_elements.update(covered)
            code_importance = {element: 1.0 for element in all_elements}
        
        # Calculate weighted coverage
        total_importance_covered = 0.0
        total_importance = sum(code_importance.values())
        
        if total_importance == 0:
            return 0.0
        
        covered_elements = set()
        for covered in test_coverage.values():
            covered_elements.update(covered)
        
        for element in covered_elements:
            total_importance_covered += code_importance.get(element, 0.0)
        
        efficiency = total_importance_covered / total_importance if total_importance > 0 else 0.0
        
        return max(0.0, min(1.0, efficiency))
    
    @staticmethod
    def analyze_test_suite(
        test_coverage: Dict[str, Set[str]],
        code_importance: Optional[Dict[str, float]] = None,
        target_distribution: Optional[Dict[str, float]] = None,
        previous_entropy: Optional[float] = None
    ) -> InformationMetrics:
        """
        Comprehensive information-theoretic analysis of test suite.
        
        Args:
            test_coverage: Dictionary mapping test names to sets of covered code elements
            code_importance: Optional dictionary mapping code elements to importance scores
            target_distribution: Optional target distribution for KL divergence calculation
            previous_entropy: Optional previous entropy for information gain calculation
            
        Returns:
            InformationMetrics with all calculated metrics
        """
        metrics = InformationMetrics()
        
        # Test suite entropy
        metrics.test_suite_entropy = InformationTheoreticAnalyzer.calculate_test_suite_entropy(
            test_coverage
        )
        
        # Mutual information
        metrics.mutual_information = InformationTheoreticAnalyzer.calculate_mutual_information(
            test_coverage, code_importance
        )
        
        # KL divergence (if target distribution provided)
        if target_distribution:
            # Build actual distribution from coverage
            actual_dist = {}
            all_elements = set()
            for covered in test_coverage.values():
                all_elements.update(covered)
            
            total_coverage = len(all_elements)
            if total_coverage > 0:
                for element in all_elements:
                    actual_dist[element] = 1.0 / total_coverage  # Uniform for simplicity
                
                metrics.kl_divergence = InformationTheoreticAnalyzer.calculate_kl_divergence(
                    actual_dist, target_distribution
                )
        
        # Information gain
        if previous_entropy is not None:
            metrics.information_gain = InformationTheoreticAnalyzer.calculate_information_gain(
                previous_entropy, metrics.test_suite_entropy
            )
        
        # Redundancy and diversity
        metrics.redundancy_score = InformationTheoreticAnalyzer.calculate_redundancy_score(
            test_coverage
        )
        metrics.diversity_score = InformationTheoreticAnalyzer.calculate_diversity_score(
            test_coverage
        )
        
        # Coverage efficiency
        metrics.coverage_efficiency = InformationTheoreticAnalyzer.calculate_coverage_efficiency(
            test_coverage, code_importance
        )
        
        return metrics
