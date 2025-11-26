"""
Monte Carlo Uncertainty Propagation

Uses Monte Carlo simulation to propagate uncertainty through complex
quality metric calculations, providing more accurate uncertainty estimates.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results of Monte Carlo uncertainty propagation."""
    mean: float
    std: float
    percentiles: Dict[float, float]  # e.g., {0.025: lower, 0.975: upper}
    confidence_interval: Tuple[float, float]  # 95% CI
    samples: np.ndarray
    convergence_metric: float  # Coefficient of variation of mean estimate


class MonteCarloPropagator:
    """
    Propagates uncertainty through quality metric calculations using Monte Carlo.
    
    More accurate than analytical approximations for complex, non-linear functions.
    """
    
    @staticmethod
    def propagate_uncertainty(
        score_function: Callable,
        input_distributions: Dict[str, Callable],
        n_samples: int = 10000,
        confidence_level: float = 0.95,
        check_convergence: bool = True
    ) -> MonteCarloResult:
        """
        Propagate uncertainty through a function using Monte Carlo simulation.
        
        Args:
            score_function: Function that takes dict of inputs and returns score
            input_distributions: Dictionary mapping input names to distribution functions
                                (functions that return random samples)
            n_samples: Number of Monte Carlo samples
            confidence_level: Confidence level for intervals
            check_convergence: Whether to check for convergence
            
        Returns:
            MonteCarloResult with statistics
        """
        # Generate samples
        samples = []
        batch_size = min(1000, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch in range(n_batches):
            # Sample inputs
            inputs = {}
            for name, dist_func in input_distributions.items():
                inputs[name] = dist_func(batch_size)
            
            # Evaluate function
            if batch_size == 1:
                score = score_function(inputs)
                samples.append(score)
            else:
                # Vectorized evaluation if possible
                batch_scores = []
                for i in range(batch_size):
                    single_inputs = {name: values[i] if isinstance(values, np.ndarray) else values
                                     for name, values in inputs.items()}
                    batch_scores.append(score_function(single_inputs))
                samples.extend(batch_scores)
        
        samples = np.array(samples)
        
        # Calculate statistics
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        
        # Percentiles
        alpha = (1 - confidence_level) / 2
        percentiles = {
            0.01: np.percentile(samples, 1),
            0.05: np.percentile(samples, 5),
            0.25: np.percentile(samples, 25),
            0.50: np.percentile(samples, 50),
            0.75: np.percentile(samples, 75),
            0.95: np.percentile(samples, 95),
            0.99: np.percentile(samples, 99),
            alpha: np.percentile(samples, alpha * 100),
            1 - alpha: np.percentile(samples, (1 - alpha) * 100)
        }
        
        confidence_interval = (percentiles[alpha], percentiles[1 - alpha])
        
        # Convergence metric (coefficient of variation of mean)
        convergence = std / (np.sqrt(len(samples)) * mean) if mean != 0 else float('inf')
        
        return MonteCarloResult(
            mean=mean,
            std=std,
            percentiles=percentiles,
            confidence_interval=confidence_interval,
            samples=samples,
            convergence_metric=convergence
        )
    
    @staticmethod
    def create_beta_distribution(alpha: float, beta: float) -> Callable:
        """
        Create a Beta distribution sampler for bounded metrics [0, 1].
        
        Args:
            alpha: Beta distribution alpha parameter
            beta: Beta distribution beta parameter
            
        Returns:
            Function that returns random samples
        """
        def sampler(n: int = 1):
            return stats.beta.rvs(alpha, beta, size=n)
        return sampler
    
    @staticmethod
    def create_gamma_distribution(shape: float, rate: float) -> Callable:
        """
        Create a Gamma distribution sampler for positive metrics.
        
        Args:
            shape: Gamma distribution shape parameter
            rate: Gamma distribution rate parameter
            
        Returns:
            Function that returns random samples
        """
        def sampler(n: int = 1):
            return stats.gamma.rvs(shape, scale=1/rate, size=n)
        return sampler
    
    @staticmethod
    def create_normal_distribution(mean: float, std: float, bounds: Optional[Tuple[float, float]] = None) -> Callable:
        """
        Create a Normal distribution sampler (optionally truncated).
        
        Args:
            mean: Mean of distribution
            std: Standard deviation
            bounds: Optional (min, max) bounds for truncation
            
        Returns:
            Function that returns random samples
        """
        if bounds:
            from scipy.stats import truncnorm
            a, b = (bounds[0] - mean) / std, (bounds[1] - mean) / std
            def sampler(n: int = 1):
                return truncnorm.rvs(a, b, loc=mean, scale=std, size=n)
        else:
            def sampler(n: int = 1):
                return np.random.normal(mean, std, size=n)
        return sampler
    
    @staticmethod
    def propagate_betes_uncertainty(
        calculator,
        mutation_alpha: float,
        mutation_beta: float,
        flakiness_alpha: float,
        flakiness_beta: float,
        other_inputs: Dict[str, float],
        n_samples: int = 10000
    ) -> MonteCarloResult:
        """
        Propagate uncertainty through bE-TES calculation.
        
        Args:
            calculator: BETESCalculator instance
            mutation_alpha: Beta distribution alpha for mutation score
            mutation_beta: Beta distribution beta for mutation score
            flakiness_alpha: Beta distribution alpha for flakiness
            flakiness_beta: Beta distribution beta for flakiness
            other_inputs: Fixed values for other inputs
            n_samples: Number of Monte Carlo samples
            
        Returns:
            MonteCarloResult with uncertainty in final score
        """
        mutation_dist = MonteCarloPropagator.create_beta_distribution(mutation_alpha, mutation_beta)
        flakiness_dist = MonteCarloPropagator.create_beta_distribution(flakiness_alpha, flakiness_beta)
        
        def score_function(inputs):
            return calculator.calculate(
                raw_mutation_score=inputs['mutation_score'],
                raw_emt_gain=other_inputs.get('emt_gain', 0.0),
                raw_assertion_iq=other_inputs.get('assertion_iq', 1.0),
                raw_behaviour_coverage=other_inputs.get('behaviour_coverage', 0.0),
                raw_median_test_time_ms=other_inputs.get('test_time_ms', 1000.0),
                raw_flakiness_rate=inputs['flakiness_rate']
            ).betes_score
        
        input_distributions = {
            'mutation_score': mutation_dist,
            'flakiness_rate': flakiness_dist
        }
        
        return MonteCarloPropagator.propagate_uncertainty(
            score_function,
            input_distributions,
            n_samples
        )


class AdaptiveMonteCarlo:
    """
    Adaptive Monte Carlo that increases samples until convergence.
    """
    
    @staticmethod
    def propagate_with_adaptation(
        score_function: Callable,
        input_distributions: Dict[str, Callable],
        initial_samples: int = 1000,
        max_samples: int = 100000,
        convergence_threshold: float = 0.01,
        min_samples: int = 1000
    ) -> Tuple[MonteCarloResult, int]:
        """
        Run Monte Carlo with adaptive sample size.
        
        Increases samples until convergence metric is below threshold.
        
        Args:
            score_function: Function to evaluate
            input_distributions: Input distributions
            initial_samples: Initial number of samples
            max_samples: Maximum number of samples
            convergence_threshold: Threshold for convergence
            min_samples: Minimum samples before checking convergence
            
        Returns:
            Tuple of (MonteCarloResult, final_sample_count)
        """
        n_samples = initial_samples
        result = None
        
        while n_samples <= max_samples:
            result = MonteCarloPropagator.propagate_uncertainty(
                score_function,
                input_distributions,
                n_samples,
                check_convergence=True
            )
            
            if result.convergence_metric < convergence_threshold and n_samples >= min_samples:
                break
            
            # Double sample size
            n_samples = min(n_samples * 2, max_samples)
        
        return result, n_samples
