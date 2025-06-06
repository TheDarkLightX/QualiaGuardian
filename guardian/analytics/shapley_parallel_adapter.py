"""
Adapter to integrate parallel Shapley calculation with the existing OptimizedShapleyCalculator.

This module provides a drop-in replacement that uses parallel computation for improved performance.

Author: DarkLightX/Dana Edwards
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from guardian.analytics.parallel_shapley import (
    ParallelShapleyCalculator,
    Backend,
    DistributionStrategy,
    ParallelConfig
)


class ParallelOptimizedShapleyCalculator:
    """
    Parallel Shapley value calculator for test importance analysis.
    
    This class provides a high-performance implementation using parallel computation
    to analyze test suite effectiveness and identify important tests.
    """
    
    def __init__(
        self,
        test_results: Dict[str, bool],
        test_execution_times: Optional[Dict[str, float]] = None,
        backend: Backend = Backend.THREADING,
        n_workers: Optional[int] = None,
        enable_monte_carlo: bool = True,
        monte_carlo_threshold: int = 15
    ):
        """
        Initialize the parallel Shapley calculator.
        
        Args:
            test_results: Dictionary mapping test names to pass/fail status
            test_execution_times: Optional dictionary of test execution times
            backend: Parallelization backend to use
            n_workers: Number of parallel workers (None for auto-detect)
            enable_monte_carlo: Use Monte Carlo for large test suites
            monte_carlo_threshold: Number of tests above which to use Monte Carlo
        """
        self.test_results = test_results
        self.test_execution_times = test_execution_times or {}
        self._shapley_cache = None
        
        self.backend = backend
        self.n_workers = n_workers
        self.enable_monte_carlo = enable_monte_carlo
        self.monte_carlo_threshold = monte_carlo_threshold
        
        # Configure parallel computation
        self.parallel_config = ParallelConfig(
            backend=backend,
            distribution_strategy=DistributionStrategy.DYNAMIC,
            n_workers=n_workers,
            enable_fault_tolerance=True,
            enable_progress_tracking=True,
            convergence_threshold=0.01,
            min_iterations=100,
            max_iterations=10000
        )
        
        self.parallel_calc = ParallelShapleyCalculator(
            backend=backend,
            n_workers=n_workers,
            config=self.parallel_config
        )
        
    def calculate_shapley_values(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> Dict[str, float]:
        """
        Calculate Shapley values for all tests using parallel computation.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping test names to their Shapley values
        """
        test_names = list(self.test_results.keys())
        n_tests = len(test_names)
        
        if n_tests == 0:
            return {}
            
        # Create value function for the test suite
        def test_value_function(coalition_indices: List[int]) -> float:
            # Get test names for this coalition
            coalition_tests = [test_names[i] for i in coalition_indices]
            
            # Compute test suite value
            return self._compute_test_suite_value(coalition_tests)
            
        # Choose computation method based on test suite size
        if self.enable_monte_carlo and n_tests > self.monte_carlo_threshold:
            # Use Monte Carlo for large test suites
            shapley_array, conv_info = self.parallel_calc.compute_shapley_values_monte_carlo(
                n_tests,
                test_value_function,
                progress_callback=progress_callback,
                return_convergence_info=True
            )
            
            if not conv_info['converged']:
                print(f"Warning: Monte Carlo did not converge after {conv_info['iterations']} iterations")
        else:
            # Use exact computation for smaller test suites
            shapley_array = self.parallel_calc.compute_shapley_values(
                n_tests,
                test_value_function,
                progress_callback=progress_callback
            )
            
        # Convert array to dictionary
        shapley_values = {
            test_name: float(shapley_array[i])
            for i, test_name in enumerate(test_names)
        }
        
        # Cache the results
        self._shapley_cache = shapley_values
        
        return shapley_values
        
    def _compute_test_suite_value(self, test_names: List[str]) -> float:
        """
        Compute the value of a test suite (coalition).
        
        This method can be overridden to implement different value functions.
        By default, it uses mutation score as the primary metric.
        
        Args:
            test_names: List of test names in the coalition
            
        Returns:
            Value of the test suite
        """
        if not test_names:
            return 0.0
            
        # Calculate mutation score for this subset of tests
        passing_tests = [t for t in test_names if self.test_results.get(t, False)]
        
        if not passing_tests:
            return 0.0
            
        # Simple approximation: assume each test kills unique mutants
        # In practice, this would query actual mutation testing results
        mutation_score = len(passing_tests) / len(self.test_results)
        
        # Apply diminishing returns for large coalitions
        efficiency_factor = 1.0 / (1.0 + 0.1 * len(test_names))
        
        return mutation_score * efficiency_factor
        
    def get_test_importance_ranking(
        self,
        top_k: Optional[int] = None
    ) -> List[tuple[str, float]]:
        """
        Get tests ranked by their Shapley values.
        
        Args:
            top_k: Return only top k tests (None for all)
            
        Returns:
            List of (test_name, shapley_value) tuples sorted by importance
        """
        if not hasattr(self, '_shapley_cache') or self._shapley_cache is None:
            self.calculate_shapley_values()
            
        ranked = sorted(
            self._shapley_cache.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if top_k is not None:
            ranked = ranked[:top_k]
            
        return ranked
        
    def suggest_test_removal_candidates(
        self,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Identify tests that contribute little value and could be removed.
        
        Args:
            threshold: Minimum Shapley value to keep a test
            
        Returns:
            List of test names that could be removed
        """
        if not hasattr(self, '_shapley_cache') or self._shapley_cache is None:
            self.calculate_shapley_values()
            
        candidates = [
            test_name
            for test_name, value in self._shapley_cache.items()
            if value < threshold
        ]
        
        return sorted(candidates)
        
    def analyze_test_interactions(
        self,
        test1: str,
        test2: str
    ) -> Dict[str, float]:
        """
        Analyze interaction effects between two tests.
        
        Args:
            test1: First test name
            test2: Second test name
            
        Returns:
            Dictionary with interaction metrics
        """
        # Calculate values for different combinations
        value_neither = self._compute_test_suite_value([])
        value_test1 = self._compute_test_suite_value([test1])
        value_test2 = self._compute_test_suite_value([test2])
        value_both = self._compute_test_suite_value([test1, test2])
        
        # Calculate interaction effects
        marginal_test1 = value_test1 - value_neither
        marginal_test2 = value_test2 - value_neither
        joint_effect = value_both - value_neither
        interaction = joint_effect - (marginal_test1 + marginal_test2)
        
        return {
            'marginal_test1': marginal_test1,
            'marginal_test2': marginal_test2,
            'joint_effect': joint_effect,
            'interaction': interaction,
            'synergy': interaction > 0
        }


def create_parallel_calculator(
    test_results: Dict[str, bool],
    backend: str = "threading",
    n_workers: Optional[int] = None
) -> ParallelOptimizedShapleyCalculator:
    """
    Convenience function to create a parallel Shapley calculator.
    
    Args:
        test_results: Test results dictionary
        backend: Backend name ("threading", "multiprocessing", or "ray")
        n_workers: Number of workers (None for auto)
        
    Returns:
        Configured parallel calculator
    """
    backend_map = {
        "threading": Backend.THREADING,
        "multiprocessing": Backend.MULTIPROCESSING,
        "ray": Backend.RAY
    }
    
    backend_enum = backend_map.get(backend.lower(), Backend.THREADING)
    
    return ParallelOptimizedShapleyCalculator(
        test_results=test_results,
        backend=backend_enum,
        n_workers=n_workers
    )