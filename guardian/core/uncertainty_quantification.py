"""
Uncertainty Quantification for Quality Metrics

Bayesian inference and uncertainty propagation for quality metrics.
Provides confidence intervals, credible intervals, and uncertainty-aware scoring.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.special import gammaln
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyInterval:
    """Represents an uncertainty interval for a metric."""
    mean: float
    lower_bound: float  # e.g., 2.5th percentile
    upper_bound: float  # e.g., 97.5th percentile
    confidence_level: float = 0.95
    distribution_type: str = "beta"  # beta, gamma, normal, etc.
    parameters: Dict[str, float] = None


@dataclass
class BayesianMetric:
    """Bayesian representation of a metric with uncertainty."""
    value: float  # Point estimate (posterior mean)
    uncertainty: UncertaintyInterval
    prior_parameters: Dict[str, float]
    posterior_parameters: Dict[str, float]
    sample_size: int
    effective_sample_size: float  # For weighted samples


class BayesianQualityCalculator:
    """
    Bayesian calculator for quality metrics with uncertainty quantification.
    
    Uses conjugate priors for efficient Bayesian updating:
    - Beta distribution for bounded metrics [0, 1]
    - Gamma distribution for positive metrics
    - Normal distribution for unbounded metrics
    """
    
    def __init__(self, use_jeffreys_prior: bool = True):
        """
        Initialize Bayesian calculator.
        
        Args:
            use_jeffreys_prior: Use Jeffreys non-informative prior (default: True)
        """
        self.use_jeffreys_prior = use_jeffreys_prior
    
    def estimate_bounded_metric(
        self,
        successes: int,
        trials: int,
        prior_alpha: Optional[float] = None,
        prior_beta: Optional[float] = None,
        confidence_level: float = 0.95
    ) -> BayesianMetric:
        """
        Estimate a bounded metric [0, 1] using Beta-Binomial conjugate prior.
        
        Args:
            successes: Number of successful outcomes
            trials: Total number of trials
            prior_alpha: Prior alpha parameter (default: Jeffreys prior = 0.5)
            prior_beta: Prior beta parameter (default: Jeffreys prior = 0.5)
            confidence_level: Confidence level for credible interval
            
        Returns:
            BayesianMetric with point estimate and uncertainty interval
        """
        if self.use_jeffreys_prior and prior_alpha is None:
            prior_alpha = 0.5
            prior_beta = 0.5
        
        if prior_alpha is None:
            prior_alpha = 1.0
        if prior_beta is None:
            prior_beta = 1.0
        
        # Posterior parameters (Beta-Binomial conjugate)
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + (trials - successes)
        
        # Point estimate (posterior mean)
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # Credible interval
        alpha_ci = (1 - confidence_level) / 2
        lower = stats.beta.ppf(alpha_ci, posterior_alpha, posterior_beta)
        upper = stats.beta.ppf(1 - alpha_ci, posterior_alpha, posterior_beta)
        
        uncertainty = UncertaintyInterval(
            mean=posterior_mean,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            distribution_type="beta",
            parameters={"alpha": posterior_alpha, "beta": posterior_beta}
        )
        
        return BayesianMetric(
            value=posterior_mean,
            uncertainty=uncertainty,
            prior_parameters={"alpha": prior_alpha, "beta": prior_beta},
            posterior_parameters={"alpha": posterior_alpha, "beta": posterior_beta},
            sample_size=trials,
            effective_sample_size=trials
        )
    
    def estimate_positive_metric(
        self,
        observations: List[float],
        prior_shape: Optional[float] = None,
        prior_rate: Optional[float] = None,
        confidence_level: float = 0.95
    ) -> BayesianMetric:
        """
        Estimate a positive metric using Gamma-Poisson/Gamma-Normal conjugate prior.
        
        Args:
            observations: List of positive observations
            prior_shape: Prior shape parameter
            prior_rate: Prior rate parameter
            confidence_level: Confidence level for credible interval
            
        Returns:
            BayesianMetric with point estimate and uncertainty interval
        """
        n = len(observations)
        if n == 0:
            raise ValueError("Need at least one observation")
        
        sample_mean = np.mean(observations)
        sample_sum = np.sum(observations)
        
        # Use non-informative prior if not specified
        if prior_shape is None:
            prior_shape = 0.001  # Very small for non-informative
        if prior_rate is None:
            prior_rate = 0.001
        
        # Posterior parameters (Gamma conjugate for exponential family)
        posterior_shape = prior_shape + n
        posterior_rate = prior_rate + sample_sum
        
        # Point estimate (posterior mean)
        posterior_mean = posterior_shape / posterior_rate
        
        # Credible interval
        alpha_ci = (1 - confidence_level) / 2
        lower = stats.gamma.ppf(alpha_ci, posterior_shape, scale=1/posterior_rate)
        upper = stats.gamma.ppf(1 - alpha_ci, posterior_shape, scale=1/posterior_rate)
        
        uncertainty = UncertaintyInterval(
            mean=posterior_mean,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            distribution_type="gamma",
            parameters={"shape": posterior_shape, "rate": posterior_rate}
        )
        
        return BayesianMetric(
            value=posterior_mean,
            uncertainty=uncertainty,
            prior_parameters={"shape": prior_shape, "rate": prior_rate},
            posterior_parameters={"shape": posterior_shape, "rate": posterior_rate},
            sample_size=n,
            effective_sample_size=n
        )
    
    def propagate_uncertainty_geometric_mean(
        self,
        metrics: List[BayesianMetric],
        weights: Optional[List[float]] = None
    ) -> BayesianMetric:
        """
        Propagate uncertainty through weighted geometric mean.
        
        Uses log-normal approximation for geometric mean of Beta/Gamma distributions.
        
        Args:
            metrics: List of Bayesian metrics
            weights: Optional weights for each metric
            
        Returns:
            BayesianMetric representing the aggregated result
        """
        if not metrics:
            raise ValueError("Need at least one metric")
        
        if weights is None:
            weights = [1.0] * len(metrics)
        
        if len(weights) != len(metrics):
            raise ValueError("Weights must match metrics length")
        
        # Convert to log space for geometric mean
        log_values = []
        log_variances = []
        
        for metric, weight in zip(metrics, weights):
            # Approximate variance using delta method
            if metric.uncertainty.distribution_type == "beta":
                alpha = metric.posterior_parameters["alpha"]
                beta = metric.posterior_parameters["beta"]
                # Variance of log(X) where X ~ Beta(alpha, beta)
                # Using approximation: Var(log(X)) ≈ Var(X) / E(X)^2
                mean = alpha / (alpha + beta)
                var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                log_var = var / (mean ** 2) if mean > 0 else 1.0
                log_mean = math.log(max(mean, 1e-10))
            else:
                # For other distributions, use approximation
                mean = metric.value
                # Approximate variance from credible interval
                interval_width = metric.uncertainty.upper_bound - metric.uncertainty.lower_bound
                # Approximate std from 95% CI: width ≈ 4 * std
                std = interval_width / 4.0
                var = std ** 2
                log_var = var / (mean ** 2) if mean > 0 else 1.0
                log_mean = math.log(max(mean, 1e-10))
            
            log_values.append(weight * log_mean)
            log_variances.append((weight ** 2) * log_var)
        
        # Weighted sum in log space
        sum_weights = sum(weights)
        log_sum = sum(log_values) / sum_weights
        log_var_sum = sum(log_variances) / (sum_weights ** 2)
        
        # Convert back to original space (log-normal)
        geometric_mean = math.exp(log_sum)
        log_std = math.sqrt(log_var_sum)
        
        # Approximate credible interval using log-normal
        alpha_ci = (1 - metric.uncertainty.confidence_level) / 2
        z_score = stats.norm.ppf(1 - alpha_ci)
        
        lower = geometric_mean * math.exp(-z_score * log_std)
        upper = geometric_mean * math.exp(z_score * log_std)
        
        uncertainty = UncertaintyInterval(
            mean=geometric_mean,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=metric.uncertainty.confidence_level,
            distribution_type="lognormal",
            parameters={"log_mean": log_sum, "log_var": log_var_sum}
        )
        
        return BayesianMetric(
            value=geometric_mean,
            uncertainty=uncertainty,
            prior_parameters={},
            posterior_parameters={"log_mean": log_sum, "log_var": log_var_sum},
            sample_size=min(m.sample_size for m in metrics),
            effective_sample_size=min(m.effective_sample_size for m in metrics)
        )
    
    def calculate_uncertainty_aware_score(
        self,
        metric: BayesianMetric,
        risk_aversion: float = 0.5
    ) -> float:
        """
        Calculate uncertainty-aware score using risk-adjusted expectation.
        
        Args:
            metric: Bayesian metric with uncertainty
            risk_aversion: Risk aversion parameter (0 = risk-neutral, 1 = very risk-averse)
            
        Returns:
            Risk-adjusted score
        """
        # Use lower bound for risk-averse scoring
        if risk_aversion == 0:
            return metric.value
        elif risk_aversion == 1:
            return metric.uncertainty.lower_bound
        else:
            # Interpolate between mean and lower bound
            return (1 - risk_aversion) * metric.value + risk_aversion * metric.uncertainty.lower_bound


def estimate_mutation_score_uncertainty(
    killed_mutants: int,
    total_mutants: int,
    confidence_level: float = 0.95
) -> BayesianMetric:
    """
    Estimate mutation score with uncertainty quantification.
    
    Args:
        killed_mutants: Number of killed mutants
        total_mutants: Total number of mutants
        confidence_level: Confidence level for credible interval
        
    Returns:
        BayesianMetric for mutation score
    """
    calculator = BayesianQualityCalculator()
    return calculator.estimate_bounded_metric(
        successes=killed_mutants,
        trials=total_mutants,
        confidence_level=confidence_level
    )


def estimate_flakiness_uncertainty(
    flaky_runs: int,
    total_runs: int,
    confidence_level: float = 0.95
) -> BayesianMetric:
    """
    Estimate flakiness rate with uncertainty quantification.
    
    Args:
        flaky_runs: Number of flaky test runs
        total_runs: Total number of test runs
        confidence_level: Confidence level for credible interval
        
    Returns:
        BayesianMetric for flakiness rate
    """
    calculator = BayesianQualityCalculator()
    return calculator.estimate_bounded_metric(
        successes=flaky_runs,
        trials=total_runs,
        confidence_level=confidence_level
    )


def estimate_execution_time_uncertainty(
    execution_times_ms: List[float],
    confidence_level: float = 0.95
) -> BayesianMetric:
    """
    Estimate test execution time with uncertainty quantification.
    
    Args:
        execution_times_ms: List of execution times in milliseconds
        confidence_level: Confidence level for credible interval
        
    Returns:
        BayesianMetric for execution time
    """
    calculator = BayesianQualityCalculator()
    return calculator.estimate_positive_metric(
        observations=execution_times_ms,
        confidence_level=confidence_level
    )
