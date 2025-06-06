"""
Optimized Shapley Value Calculator with Convergence Detection and Variance Reduction

This module implements an optimized version of Shapley value computation with:
- Convergence detection for early stopping
- Antithetic variates for variance reduction
- Confidence interval estimation
- Memory-efficient computation

Author: DarkLightX/Dana Edwards
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for test identifiers
TestId = Union[Path, str]


@dataclass
class ConvergenceMetrics:
    """Metrics for convergence detection."""
    iterations: int
    mean_variance: float
    max_variance: float
    confidence_level: float


class ConvergenceStrategy(Enum):
    """Available convergence detection strategies."""
    VARIANCE_BASED = "variance"
    CONFIDENCE_INTERVAL = "confidence"
    RELATIVE_CHANGE = "relative"


# Interface Segregation Principle (ISP)
class IConvergenceDetector(ABC):
    """Interface for convergence detection strategies."""
    
    @abstractmethod
    def has_converged(self, threshold: float) -> bool:
        """Check if values have converged."""
        pass
    
    @abstractmethod
    def update(self, values: Dict[TestId, float]) -> None:
        """Update with new iteration values."""
        pass
    
    @abstractmethod
    def get_convergence_metrics(self) -> ConvergenceMetrics:
        """Get current convergence metrics."""
        pass


class ISampler(ABC):
    """Interface for permutation sampling strategies."""
    
    @abstractmethod
    def generate_permutation(self, items: List[TestId]) -> List[TestId]:
        """Generate a single permutation."""
        pass


class IVarianceReducer(ABC):
    """Interface for variance reduction techniques."""
    
    @abstractmethod
    def generate_samples(self, items: List[TestId], n_samples: int) -> List[List[TestId]]:
        """Generate variance-reduced samples."""
        pass


# Single Responsibility Principle (SRP) implementations
class VarianceBasedDetector(IConvergenceDetector):
    """Detects convergence based on variance stability."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.value_history: Dict[TestId, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.iteration_count = 0
    
    def update(self, values: Dict[TestId, float]) -> None:
        """Update value history."""
        self.iteration_count += 1
        for test_id, value in values.items():
            self.value_history[test_id].append(value)
    
    def has_converged(self, threshold: float) -> bool:
        """Check if variance has stabilized below threshold."""
        if self.iteration_count < self.window_size:
            return False
        
        # Calculate variance for each test over the window
        variances = []
        for test_id, history in self.value_history.items():
            if len(history) >= self.window_size:
                variance = np.var(list(history))
                variances.append(variance)
        
        if not variances:
            return False
        
        # Check if all variances are below threshold
        max_variance = max(variances)
        return max_variance < threshold
    
    def get_convergence_metrics(self) -> ConvergenceMetrics:
        """Calculate current convergence metrics."""
        if not self.value_history:
            return ConvergenceMetrics(
                iterations=self.iteration_count,
                mean_variance=0.0,
                max_variance=0.0,
                confidence_level=0.0
            )
        
        variances = []
        for history in self.value_history.values():
            if len(history) >= 2:
                variances.append(np.var(list(history)))
        
        if not variances:
            return ConvergenceMetrics(
                iterations=self.iteration_count,
                mean_variance=0.0,
                max_variance=0.0,
                confidence_level=0.0
            )
        
        mean_var = np.mean(variances)
        max_var = np.max(variances)
        
        # No confidence if we have insufficient data
        if self.iteration_count < self.window_size:
            confidence = 0.0
        else:
            # Simple confidence based on iterations and variance
            confidence = min(1.0, self.iteration_count / (self.window_size * 2))
            if max_var > 0.1:
                confidence *= 0.5
        
        return ConvergenceMetrics(
            iterations=self.iteration_count,
            mean_variance=mean_var,
            max_variance=max_var,
            confidence_level=confidence
        )


class AntitheticSampler(ISampler, IVarianceReducer):
    """Generates variance-reduced permutation samples using antithetic variates."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def generate_permutation(self, items: List[TestId]) -> List[TestId]:
        """Generate a single random permutation."""
        permutation = items.copy()
        self.rng.shuffle(permutation)
        return permutation
    
    def generate_complementary_permutations(self, items: List[TestId]) -> Tuple[List[TestId], List[TestId]]:
        """Generate antithetic (negatively correlated) permutation pairs."""
        n = len(items)
        
        # Generate first permutation
        perm1 = self.generate_permutation(items)
        
        # Generate complementary permutation by reversing the order
        # This ensures that if item A comes early in perm1, it comes late in perm2
        perm2 = perm1[::-1]  # Simple reversal for maximum negative correlation
        
        return perm1, perm2
    
    def generate_samples(self, items: List[TestId], n_samples: int) -> List[List[TestId]]:
        """Generate variance-reduced samples using antithetic pairs."""
        samples = []
        
        # Generate pairs of antithetic variates
        for _ in range(n_samples // 2):
            perm1, perm2 = self.generate_complementary_permutations(items)
            samples.extend([perm1, perm2])
        
        # Add one more if odd number requested
        if n_samples % 2 == 1:
            samples.append(self.generate_permutation(items))
        
        return samples


class StandardSampler(ISampler):
    """Standard random permutation sampler."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def generate_permutation(self, items: List[TestId]) -> List[TestId]:
        """Generate a random permutation."""
        permutation = items.copy()
        self.rng.shuffle(permutation)
        return permutation


# Dependency Inversion Principle (DIP) - Main calculator
class OptimizedShapleyCalculator:
    """
    Optimized Shapley value calculator with convergence detection and variance reduction.
    
    This class implements an efficient algorithm for computing Shapley values with:
    - Early stopping based on convergence detection
    - Variance reduction using antithetic variates
    - Confidence interval estimation
    - Progress tracking
    """
    
    def __init__(
        self,
        convergence_threshold: Optional[float] = 0.01,
        max_iterations: int = 1000,
        use_antithetic_variates: bool = True,
        convergence_detector: Optional[IConvergenceDetector] = None,
        sampler: Optional[ISampler] = None,
        window_size: int = 10,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Initialize the optimized Shapley calculator.
        
        Args:
            convergence_threshold: Threshold for convergence detection (None to disable)
            max_iterations: Maximum number of permutations to sample
            use_antithetic_variates: Whether to use variance reduction
            convergence_detector: Custom convergence detector (optional)
            sampler: Custom sampler (optional)
            window_size: Window size for convergence detection
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            seed: Random seed for reproducibility
            progress_callback: Callback for progress updates
        """
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.use_antithetic_variates = use_antithetic_variates
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.progress_callback = progress_callback
        
        # Initialize components
        self.convergence_detector = convergence_detector or VarianceBasedDetector(window_size)
        
        if sampler:
            self.sampler = sampler
        elif use_antithetic_variates:
            self.sampler = AntitheticSampler(seed)
        else:
            self.sampler = StandardSampler(seed)
        
        # State tracking
        self.shapley_values: Dict[TestId, float] = defaultdict(float)
        self.contribution_counts: Dict[TestId, int] = defaultdict(int)
        self.variance_tracker: Dict[TestId, List[float]] = defaultdict(list)
        self.convergence_info: Dict[str, Any] = {}
        self.iteration_values: List[Dict[TestId, float]] = []
    
    def calculate_shapley_values(
        self,
        test_ids: List[TestId],
        metric_evaluator_func: Callable[[List[TestId]], float]
    ) -> Dict[TestId, float]:
        """
        Calculate Shapley values with optimizations.
        
        Args:
            test_ids: List of test identifiers
            metric_evaluator_func: Function that evaluates a subset of tests
        
        Returns:
            Dictionary mapping test IDs to their Shapley values
        """
        start_time = time.time()
        n = len(test_ids)
        
        if n == 0:
            return {}
        
        if n == 1:
            # Special case: single test
            test_id = test_ids[0]
            value = metric_evaluator_func(test_ids) - metric_evaluator_func([])
            return {test_id: value}
        
        # Reset state
        self.shapley_values = defaultdict(float)
        self.contribution_counts = defaultdict(int)
        self.variance_tracker = defaultdict(list)
        self.iteration_values = []
        
        # Get empty set score
        empty_score = metric_evaluator_func([])
        
        # Main computation loop
        iterations_used = 0
        converged = False
        
        for iteration in range(self.max_iterations):
            # Progress callback
            if self.progress_callback:
                self.progress_callback(iteration, self.max_iterations)
            
            # Generate permutation(s)
            if self.use_antithetic_variates and isinstance(self.sampler, AntitheticSampler):
                # Process antithetic pairs
                if iteration % 2 == 0:
                    perm1, perm2 = self.sampler.generate_complementary_permutations(test_ids)
                    permutations = [perm1, perm2]
                else:
                    continue  # Skip odd iterations for antithetic
            else:
                permutations = [self.sampler.generate_permutation(test_ids)]
            
            # Process each permutation
            for permutation in permutations:
                self._process_permutation(
                    permutation,
                    metric_evaluator_func,
                    empty_score
                )
                iterations_used += 1
            
            # Check convergence periodically
            if self.convergence_threshold and iteration > self.window_size and iteration % 5 == 0:
                current_values = self._get_current_estimates()
                self.convergence_detector.update(current_values)
                self.iteration_values.append(current_values.copy())
                
                if self.convergence_detector.has_converged(self.convergence_threshold):
                    converged = True
                    break
        
        # Calculate final estimates
        final_values = self._get_current_estimates()
        
        # Store convergence info
        self.convergence_info = {
            'converged': converged,
            'iterations_used': iterations_used,
            'max_iterations': self.max_iterations,
            'early_stopped': converged and iterations_used < self.max_iterations,
            'computation_time': time.time() - start_time,
            'convergence_metrics': self.convergence_detector.get_convergence_metrics()
        }
        
        logger.info(
            f"Calculated Shapley values for {n} tests in {iterations_used} iterations "
            f"({self.convergence_info['computation_time']:.2f}s)"
        )
        
        return dict(final_values)
    
    def _process_permutation(
        self,
        permutation: List[TestId],
        metric_evaluator_func: Callable[[List[TestId]], float],
        empty_score: float
    ) -> None:
        """Process a single permutation and update Shapley values."""
        current_subset: List[TestId] = []
        current_score = empty_score
        
        for test_id in permutation:
            # Score with test added
            new_subset = current_subset + [test_id]
            new_score = metric_evaluator_func(new_subset)
            
            # Marginal contribution
            marginal_contribution = new_score - current_score
            
            # Update Shapley value estimate
            self.shapley_values[test_id] += marginal_contribution
            self.contribution_counts[test_id] += 1
            self.variance_tracker[test_id].append(marginal_contribution)
            
            # Update for next iteration
            current_subset = new_subset
            current_score = new_score
    
    def _get_current_estimates(self) -> Dict[TestId, float]:
        """Get current Shapley value estimates."""
        estimates = {}
        
        for test_id, total_contribution in self.shapley_values.items():
            count = self.contribution_counts[test_id]
            if count > 0:
                estimates[test_id] = total_contribution / count
            else:
                estimates[test_id] = 0.0
        
        return estimates
    
    def get_confidence_intervals(self) -> Dict[TestId, Tuple[float, float]]:
        """
        Calculate confidence intervals for Shapley values.
        
        Returns:
            Dictionary mapping test IDs to (lower, upper) confidence bounds
        """
        intervals = {}
        z_score = 1.96  # 95% confidence
        
        current_estimates = self._get_current_estimates()
        
        for test_id in current_estimates:
            if test_id in self.variance_tracker and len(self.variance_tracker[test_id]) > 1:
                contributions = self.variance_tracker[test_id]
                mean = np.mean(contributions)
                std_error = np.std(contributions) / np.sqrt(len(contributions))
                
                lower = mean - z_score * std_error
                upper = mean + z_score * std_error
                
                intervals[test_id] = (lower, upper)
            else:
                # No variance info, use point estimate
                value = current_estimates[test_id]
                intervals[test_id] = (value, value)
        
        return intervals
    
    def get_variance_estimate(self) -> float:
        """Get average variance across all tests."""
        variances = []
        
        for contributions in self.variance_tracker.values():
            if len(contributions) > 1:
                variances.append(np.var(contributions))
        
        return np.mean(variances) if variances else 0.0
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get detailed convergence information."""
        return self.convergence_info.copy()


# Open/Closed Principle (OCP) - Strategy pattern extensions
class ConfidenceIntervalDetector(IConvergenceDetector):
    """Convergence based on confidence interval width."""
    
    def __init__(self, target_width: float = 0.05):
        self.target_width = target_width
        self.calculator: Optional[OptimizedShapleyCalculator] = None
    
    def set_calculator(self, calculator: OptimizedShapleyCalculator):
        """Set reference to calculator for CI computation."""
        self.calculator = calculator
    
    def has_converged(self, threshold: float) -> bool:
        """Check if confidence intervals are narrow enough."""
        if not self.calculator:
            return False
        
        intervals = self.calculator.get_confidence_intervals()
        
        if not intervals:
            return False
        
        # Check if all intervals are narrow enough
        for lower, upper in intervals.values():
            if upper - lower > self.target_width:
                return False
        
        return True
    
    def update(self, values: Dict[TestId, float]) -> None:
        """Update is handled by calculator."""
        pass
    
    def get_convergence_metrics(self) -> ConvergenceMetrics:
        """Get metrics based on confidence intervals."""
        if not self.calculator:
            return ConvergenceMetrics(0, 0.0, 0.0, 0.0)
        
        intervals = self.calculator.get_confidence_intervals()
        
        if not intervals:
            return ConvergenceMetrics(0, 0.0, 0.0, 0.0)
        
        widths = [upper - lower for lower, upper in intervals.values()]
        
        return ConvergenceMetrics(
            iterations=len(self.calculator.iteration_values),
            mean_variance=np.mean(widths),
            max_variance=np.max(widths),
            confidence_level=0.95
        )