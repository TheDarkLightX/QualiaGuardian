"""
UCB1-Thompson Hybrid Selector for Test Importance with Cold Start Handling

This module implements a hybrid multi-armed bandit approach that:
- Uses UCB1 for initial exploration when confidence is low
- Transitions to Thompson Sampling for exploitation as confidence grows
- Handles cold start problems effectively
- Adapts to distribution shifts

Author: DarkLightX/Dana Edwards
"""

import logging
import math
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Type aliases
TestId = Union[Path, str]


@dataclass
class TestStats:
    """Statistics for a single test."""
    success_count: int = 0
    failure_count: int = 0
    total_attempts: int = 0
    mean_reward: float = 0.0
    variance: float = 0.0
    last_rewards: List[float] = field(default_factory=list)
    
    def update(self, reward: float, window_size: int = 20):
        """Update statistics with new reward observation."""
        self.total_attempts += 1
        
        if reward > 0.5:  # Binary success/failure threshold
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update running mean
        old_mean = self.mean_reward
        self.mean_reward = ((self.total_attempts - 1) * old_mean + reward) / self.total_attempts
        
        # Keep recent rewards for variance estimation
        self.last_rewards.append(reward)
        if len(self.last_rewards) > window_size:
            self.last_rewards.pop(0)
        
        # Update variance estimate
        if len(self.last_rewards) > 1:
            self.variance = np.var(self.last_rewards)


class SelectionStrategy(Enum):
    """Available selection strategies."""
    UCB1 = "ucb1"
    THOMPSON = "thompson"
    HYBRID = "hybrid"


@dataclass
class TestSelectionResult:
    """Result of test selection with metadata."""
    selected_test: TestId
    strategy_used: SelectionStrategy
    confidence_level: float
    exploration_bonus: float = 0.0
    selection_probabilities: Optional[Dict[TestId, float]] = None


# Interface Segregation Principle (ISP)
class IExplorationStrategy(ABC):
    """Interface for exploration strategies."""
    
    @abstractmethod
    def select_test(
        self, 
        available_tests: List[TestId],
        test_stats: Dict[TestId, TestStats],
        total_iterations: int,
        **kwargs
    ) -> TestId:
        """Select a test using exploration strategy."""
        pass
    
    @abstractmethod
    def get_confidence_level(self, stats: TestStats) -> float:
        """Get confidence level for a test."""
        pass


class IExploitationStrategy(ABC):
    """Interface for exploitation strategies."""
    
    @abstractmethod
    def select_test(
        self,
        available_tests: List[TestId],
        test_stats: Dict[TestId, TestStats],
        **kwargs
    ) -> TestId:
        """Select a test using exploitation strategy."""
        pass
    
    @abstractmethod
    def sample_from_posterior(self, stats: TestStats) -> float:
        """Sample from posterior distribution."""
        pass


class ITransitionStrategy(ABC):
    """Interface for transition strategies."""
    
    @abstractmethod
    def select_strategy(
        self,
        test_stats: Dict[TestId, TestStats],
        iteration: int,
        **kwargs
    ) -> SelectionStrategy:
        """Select which strategy to use."""
        pass


# Single Responsibility Principle (SRP) implementations
class UCB1Calculator(IExplorationStrategy):
    """UCB1 exploration strategy implementation."""
    
    def __init__(self, exploration_constant: float = 2.0):
        self.exploration_constant = exploration_constant
    
    def calculate_ucb_score(
        self,
        stats: TestStats,
        total_iterations: int
    ) -> float:
        """Calculate UCB1 score for a test."""
        if stats.total_attempts == 0:
            return float('inf')  # Untested items have infinite score
        
        # UCB1 formula: mean + c * sqrt(ln(N) / n)
        exploitation_term = stats.mean_reward
        exploration_term = self.exploration_constant * math.sqrt(
            math.log(total_iterations) / stats.total_attempts
        )
        
        return exploitation_term + exploration_term
    
    def select_test(
        self,
        available_tests: List[TestId],
        test_stats: Dict[TestId, TestStats],
        total_iterations: int,
        **kwargs
    ) -> TestId:
        """Select test with highest UCB score."""
        if not available_tests:
            raise ValueError("No tests available for selection")
        
        # Calculate UCB scores
        scores = {}
        for test_id in available_tests:
            stats = test_stats.get(test_id, TestStats())
            scores[test_id] = self.calculate_ucb_score(stats, total_iterations)
        
        # Select test with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def get_confidence_level(self, stats: TestStats) -> float:
        """Calculate confidence level based on sample size."""
        if stats.total_attempts == 0:
            return 0.0
        
        # Confidence increases with sample size
        # Using a simple logarithmic scale
        confidence = min(1.0, math.log(stats.total_attempts + 1) / math.log(100))
        
        # Adjust for high variance
        if stats.variance > 0.25:  # High variance threshold
            confidence *= 0.7
        
        return confidence


class ThompsonSampler(IExploitationStrategy):
    """Thompson Sampling exploitation strategy."""
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.rng = np.random.RandomState()
    
    def sample_from_posterior(self, stats: TestStats) -> float:
        """Sample from Beta posterior distribution."""
        # Beta-Bernoulli conjugate prior
        alpha = self.prior_alpha + stats.success_count
        beta = self.prior_beta + stats.failure_count
        
        return self.rng.beta(alpha, beta)
    
    def select_test(
        self,
        available_tests: List[TestId],
        test_stats: Dict[TestId, TestStats],
        **kwargs
    ) -> TestId:
        """Select test using Thompson sampling."""
        if not available_tests:
            raise ValueError("No tests available for selection")
        
        # Sample from posterior for each test
        samples = {}
        for test_id in available_tests:
            stats = test_stats.get(test_id, TestStats())
            samples[test_id] = self.sample_from_posterior(stats)
        
        # Select test with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]


class ConfidenceBasedTransition(ITransitionStrategy):
    """Transition based on overall confidence level."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        min_samples_per_test: int = 5
    ):
        self.confidence_threshold = confidence_threshold
        self.min_samples_per_test = min_samples_per_test
    
    def select_strategy(
        self,
        test_stats: Dict[TestId, TestStats],
        iteration: int,
        **kwargs
    ) -> SelectionStrategy:
        """Select strategy based on confidence."""
        if not test_stats:
            return SelectionStrategy.UCB1
        
        # Calculate overall confidence
        confidences = []
        for stats in test_stats.values():
            if stats.total_attempts >= self.min_samples_per_test:
                # Use logarithmic scale with base 50 for faster confidence growth
                confidence = min(1.0, math.log(stats.total_attempts + 1) / math.log(50))
                
                # Adjust for variance - low variance increases confidence
                if stats.variance < 0.1:  # Low variance bonus
                    confidence = min(1.0, confidence * 1.2)
                elif stats.variance > 0.25:  # High variance penalty
                    confidence *= 0.8
                    
                confidences.append(confidence)
        
        if not confidences:
            return SelectionStrategy.UCB1
        
        avg_confidence = np.mean(confidences)
        
        return (SelectionStrategy.THOMPSON 
                if avg_confidence >= self.confidence_threshold
                else SelectionStrategy.UCB1)


class SmoothTransition(ITransitionStrategy):
    """Smooth transition between strategies."""
    
    def __init__(self, transition_start: int = 50, transition_end: int = 150):
        self.transition_start = transition_start
        self.transition_end = transition_end
        self.rng = np.random.RandomState()
    
    def get_thompson_probability(self, iteration: int) -> float:
        """Get probability of using Thompson sampling."""
        if iteration < self.transition_start:
            return 0.0
        elif iteration > self.transition_end:
            return 1.0
        else:
            # Linear interpolation
            progress = (iteration - self.transition_start) / (
                self.transition_end - self.transition_start
            )
            return progress
    
    def select_strategy(
        self,
        test_stats: Dict[TestId, TestStats],
        iteration: int,
        **kwargs
    ) -> SelectionStrategy:
        """Select strategy with smooth transition."""
        thompson_prob = self.get_thompson_probability(iteration)
        
        if self.rng.random() < thompson_prob:
            return SelectionStrategy.THOMPSON
        else:
            return SelectionStrategy.UCB1


class AdaptiveTransition(ITransitionStrategy):
    """Adaptive transition based on performance variance."""
    
    def __init__(self, base_threshold: float = 0.7):
        self.base_threshold = base_threshold
    
    def get_adaptive_threshold(
        self,
        test_stats: Dict[TestId, TestStats]
    ) -> float:
        """Calculate adaptive threshold based on variance."""
        if not test_stats:
            return self.base_threshold
        
        # Higher variance requires higher confidence
        variances = [s.variance for s in test_stats.values() if s.total_attempts > 0]
        
        if not variances:
            return self.base_threshold
        
        avg_variance = np.mean(variances)
        
        # Adjust threshold based on variance
        # High variance (0.25) -> threshold = 0.9
        # Low variance (0.05) -> threshold = 0.6
        adjusted_threshold = self.base_threshold + 0.8 * avg_variance
        
        return min(0.95, adjusted_threshold)
    
    def select_strategy(
        self,
        test_stats: Dict[TestId, TestStats],
        iteration: int,
        **kwargs
    ) -> SelectionStrategy:
        """Select strategy with adaptive threshold."""
        threshold = self.get_adaptive_threshold(test_stats)
        
        # Use same confidence calculation as ConfidenceBasedTransition
        transition = ConfidenceBasedTransition(confidence_threshold=threshold)
        return transition.select_strategy(test_stats, iteration)


# Dependency Inversion Principle (DIP) - Main hybrid selector
class UCB1ThompsonHybrid:
    """
    Hybrid selector combining UCB1 and Thompson Sampling.
    
    This class implements an adaptive multi-armed bandit approach that:
    - Starts with UCB1 for exploration
    - Transitions to Thompson Sampling as confidence grows
    - Handles cold start effectively
    - Adapts to distribution shifts
    """
    
    def __init__(
        self,
        exploration_constant: float = 2.0,
        transition_threshold: float = 0.8,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        window_size: int = 20,
        exploration_strategy: Optional[IExplorationStrategy] = None,
        exploitation_strategy: Optional[IExploitationStrategy] = None,
        transition_strategy: Optional[ITransitionStrategy] = None
    ):
        """
        Initialize hybrid selector.
        
        Args:
            exploration_constant: UCB1 exploration parameter
            transition_threshold: Confidence threshold for strategy transition
            prior_alpha: Beta prior alpha parameter
            prior_beta: Beta prior beta parameter
            window_size: Window size for variance estimation
            exploration_strategy: Custom exploration strategy
            exploitation_strategy: Custom exploitation strategy
            transition_strategy: Custom transition strategy
        """
        self.window_size = window_size
        
        # Initialize strategies (Dependency Injection)
        self.exploration_strategy = (
            exploration_strategy or 
            UCB1Calculator(exploration_constant)
        )
        self.exploitation_strategy = (
            exploitation_strategy or
            ThompsonSampler(prior_alpha, prior_beta)
        )
        self.transition_strategy = (
            transition_strategy or
            ConfidenceBasedTransition(transition_threshold)
        )
        
        # State tracking
        self.test_stats: Dict[TestId, TestStats] = defaultdict(TestStats)
        self.total_iterations = 0
        self.total_reward = 0.0
        self.strategy_usage = defaultdict(int)
        
        # Thread safety
        self.lock = threading.Lock()
    
    def select_next_test(self, available_tests: List[TestId]) -> TestId:
        """Select next test to run."""
        result = self.select_next_test_with_info(available_tests)
        return result.selected_test
    
    def select_next_test_with_info(
        self,
        available_tests: List[TestId],
        test_characteristics: Optional[Dict[TestId, Dict[str, Any]]] = None
    ) -> TestSelectionResult:
        """
        Select next test with detailed information.
        
        Args:
            available_tests: List of available test IDs
            test_characteristics: Optional test characteristics (exec_time, etc.)
        
        Returns:
            TestSelectionResult with selection details
        """
        with self.lock:
            if not available_tests:
                raise ValueError("No tests available for selection")
            
            # Determine which strategy to use
            strategy = self.transition_strategy.select_strategy(
                self.test_stats,
                self.total_iterations
            )
            
            # Update strategy usage
            self.strategy_usage[strategy] += 1
            
            # Select test based on strategy
            if strategy == SelectionStrategy.UCB1:
                selected = self.exploration_strategy.select_test(
                    available_tests,
                    self.test_stats,
                    max(1, self.total_iterations),
                    test_characteristics=test_characteristics
                )
                
                # Calculate exploration bonus
                stats = self.test_stats.get(selected, TestStats())
                if hasattr(self.exploration_strategy, 'calculate_ucb_score'):
                    score = self.exploration_strategy.calculate_ucb_score(
                        stats, max(1, self.total_iterations)
                    )
                    exploration_bonus = score - stats.mean_reward
                else:
                    exploration_bonus = 0.0
                    
            else:  # Thompson Sampling
                selected = self.exploitation_strategy.select_test(
                    available_tests,
                    self.test_stats,
                    test_characteristics=test_characteristics
                )
                exploration_bonus = 0.0
            
            # Calculate confidence level
            selected_stats = self.test_stats.get(selected, TestStats())
            confidence = self.exploration_strategy.get_confidence_level(selected_stats)
            
            # Calculate selection probabilities (optional)
            if len(available_tests) <= 20:  # Only for small sets
                probs = self._calculate_selection_probabilities(
                    available_tests, strategy
                )
            else:
                probs = None
            
            return TestSelectionResult(
                selected_test=selected,
                strategy_used=strategy,
                confidence_level=confidence,
                exploration_bonus=exploration_bonus,
                selection_probabilities=probs
            )
    
    def update(self, test_id: TestId, reward: float):
        """Update statistics with test execution result."""
        with self.lock:
            self.test_stats[test_id].update(reward, self.window_size)
            self.total_iterations += 1
            self.total_reward += reward
    
    def get_selection_probabilities(
        self,
        available_tests: List[TestId]
    ) -> Dict[TestId, float]:
        """Get selection probability for each test."""
        with self.lock:
            strategy = self.transition_strategy.select_strategy(
                self.test_stats,
                self.total_iterations
            )
            return self._calculate_selection_probabilities(available_tests, strategy)
    
    def _calculate_selection_probabilities(
        self,
        available_tests: List[TestId],
        strategy: SelectionStrategy
    ) -> Dict[TestId, float]:
        """Calculate selection probabilities for all tests."""
        if strategy == SelectionStrategy.UCB1:
            # For UCB1, use softmax on scores
            scores = {}
            for test_id in available_tests:
                stats = self.test_stats.get(test_id, TestStats())
                if hasattr(self.exploration_strategy, 'calculate_ucb_score'):
                    scores[test_id] = self.exploration_strategy.calculate_ucb_score(
                        stats, max(1, self.total_iterations)
                    )
                else:
                    scores[test_id] = float('inf') if stats.total_attempts == 0 else stats.mean_reward
            
            # Handle infinite scores
            inf_tests = [t for t, s in scores.items() if s == float('inf')]
            if inf_tests:
                # Equal probability for untested items
                probs = {t: 1.0/len(inf_tests) if t in inf_tests else 0.0 
                        for t in available_tests}
            else:
                # Softmax for finite scores
                max_score = max(scores.values())
                exp_scores = {t: math.exp(s - max_score) for t, s in scores.items()}
                total = sum(exp_scores.values())
                probs = {t: exp_scores[t] / total for t in available_tests}
                
        else:  # Thompson Sampling
            # Sample multiple times to estimate probabilities
            n_samples = 1000
            counts = defaultdict(int)
            
            for _ in range(n_samples):
                samples = {}
                for test_id in available_tests:
                    stats = self.test_stats.get(test_id, TestStats())
                    samples[test_id] = self.exploitation_strategy.sample_from_posterior(stats)
                
                selected = max(samples.items(), key=lambda x: x[1])[0]
                counts[selected] += 1
            
            probs = {t: counts[t] / n_samples for t in available_tests}
        
        return probs
    
    def get_all_stats(self) -> Dict[TestId, Optional[TestStats]]:
        """Get statistics for all tests."""
        with self.lock:
            return {test: stats if stats.total_attempts > 0 else None
                   for test, stats in self.test_stats.items()}
    
    def reset(self):
        """Reset all statistics."""
        with self.lock:
            self.test_stats.clear()
            self.total_iterations = 0
            self.total_reward = 0.0
            self.strategy_usage.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        with self.lock:
            metrics = {
                'total_iterations': self.total_iterations,
                'average_reward': (self.total_reward / self.total_iterations 
                                 if self.total_iterations > 0 else 0.0),
                'strategy_distribution': dict(self.strategy_usage),
                'tests_explored': len([s for s in self.test_stats.values() 
                                     if s.total_attempts > 0]),
                'total_tests': len(self.test_stats)
            }
            
            # Add convergence info
            if self.total_iterations > 0:
                ucb1_usage = self.strategy_usage.get(SelectionStrategy.UCB1, 0)
                thompson_usage = self.strategy_usage.get(SelectionStrategy.THOMPSON, 0)
                
                metrics['exploration_ratio'] = ucb1_usage / self.total_iterations
                metrics['exploitation_ratio'] = thompson_usage / self.total_iterations
            
            return metrics


# Open/Closed Principle (OCP) - Extension classes
class HybridSelector(UCB1ThompsonHybrid):
    """Extended hybrid selector with additional features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_history = []
    
    def update(self, test_id: TestId, reward: float):
        """Update with performance tracking."""
        super().update(test_id, reward)
        
        # Track performance over time
        self.performance_history.append({
            'iteration': self.total_iterations,
            'test': test_id,
            'reward': reward,
            'cumulative_reward': self.total_reward
        })
    
    def detect_distribution_shift(self, window: int = 50) -> bool:
        """Detect if there's been a distribution shift."""
        if len(self.performance_history) < window * 2:
            return False
        
        # Compare recent performance to older performance
        recent = self.performance_history[-window:]
        older = self.performance_history[-2*window:-window]
        
        recent_mean = np.mean([h['reward'] for h in recent])
        older_mean = np.mean([h['reward'] for h in older])
        
        # Significant change indicates shift
        return abs(recent_mean - older_mean) > 0.2
    
    def adapt_to_shift(self):
        """Adapt parameters when distribution shift is detected."""
        # Increase exploration temporarily
        if hasattr(self.exploration_strategy, 'exploration_constant'):
            self.exploration_strategy.exploration_constant *= 1.5
        
        # Reset transition threshold to delay exploitation
        if hasattr(self.transition_strategy, 'confidence_threshold'):
            self.transition_strategy.confidence_threshold = min(
                0.95, 
                self.transition_strategy.confidence_threshold * 1.2
            )