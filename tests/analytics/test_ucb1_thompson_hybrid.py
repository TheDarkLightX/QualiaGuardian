"""
TDD Tests for UCB1-Thompson Hybrid Selector

Author: DarkLightX/Dana Edwards
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from typing import Dict, List, Optional

from guardian.analytics.ucb1_thompson_hybrid import (
    UCB1ThompsonHybrid,
    UCB1Calculator,
    ThompsonSampler,
    HybridSelector,
    TestStats,
    SelectionStrategy,
    TestSelectionResult,
    IExplorationStrategy,
    IExploitationStrategy,
    ITransitionStrategy
)


class TestUCB1Calculator:
    """TDD tests for UCB1 exploration strategy."""
    
    def test_calculate_ucb_score_untested_item(self):
        """RED: Calculate UCB score for untested item."""
        calculator = UCB1Calculator(exploration_constant=2.0)
        
        stats = TestStats(
            success_count=0,
            failure_count=0,
            total_attempts=0,
            mean_reward=0.0,
            variance=0.0
        )
        
        score = calculator.calculate_ucb_score(stats, total_iterations=10)
        
        # Untested items should have infinite score for exploration
        assert score == float('inf')
    
    def test_calculate_ucb_score_tested_item(self):
        """RED: Calculate UCB score for tested item."""
        calculator = UCB1Calculator(exploration_constant=2.0)
        
        stats = TestStats(
            success_count=3,
            failure_count=2,
            total_attempts=5,
            mean_reward=0.6,
            variance=0.24
        )
        
        score = calculator.calculate_ucb_score(stats, total_iterations=100)
        
        # UCB = mean + c * sqrt(ln(N) / n)
        expected_confidence_radius = 2.0 * np.sqrt(np.log(100) / 5)
        expected_score = 0.6 + expected_confidence_radius
        
        assert abs(score - expected_score) < 1e-6
    
    def test_exploration_bonus_decreases_with_attempts(self):
        """RED: Exploration bonus should decrease as item is tested more."""
        calculator = UCB1Calculator(exploration_constant=2.0)
        
        scores = []
        for n_attempts in [1, 5, 10, 50, 100]:
            stats = TestStats(
                success_count=n_attempts // 2,
                failure_count=n_attempts - n_attempts // 2,
                total_attempts=n_attempts,
                mean_reward=0.5,
                variance=0.25
            )
            
            score = calculator.calculate_ucb_score(stats, total_iterations=1000)
            scores.append(score)
        
        # Scores should decrease as confidence increases
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]
    
    def test_select_with_ucb1(self):
        """RED: Select item with highest UCB score."""
        calculator = UCB1Calculator(exploration_constant=1.0)
        
        test_stats = {
            'test1': TestStats(0, 0, 0, 0.0, 0.0),  # Untested
            'test2': TestStats(5, 5, 10, 0.5, 0.25),  # Average
            'test3': TestStats(8, 2, 10, 0.8, 0.16),  # Good
            'test4': TestStats(2, 8, 10, 0.2, 0.16),  # Poor
        }
        
        selected = calculator.select_test(
            available_tests=['test1', 'test2', 'test3', 'test4'],
            test_stats=test_stats,
            total_iterations=50
        )
        
        # Should select untested item first
        assert selected == 'test1'
    
    def test_confidence_level_calculation(self):
        """RED: Calculate confidence level based on sample size."""
        calculator = UCB1Calculator()
        
        # Low samples = low confidence
        stats_low = TestStats(2, 3, 5, 0.4, 0.24)
        confidence_low = calculator.get_confidence_level(stats_low)
        
        # High samples = high confidence  
        stats_high = TestStats(40, 60, 100, 0.4, 0.24)
        confidence_high = calculator.get_confidence_level(stats_high)
        
        assert 0 <= confidence_low <= 1
        assert 0 <= confidence_high <= 1
        assert confidence_high > confidence_low


class TestThompsonSampler:
    """TDD tests for Thompson Sampling exploitation strategy."""
    
    def test_sample_from_posterior(self):
        """RED: Sample from Beta posterior distribution."""
        sampler = ThompsonSampler(prior_alpha=1.0, prior_beta=1.0)
        
        stats = TestStats(
            success_count=7,
            failure_count=3,
            total_attempts=10,
            mean_reward=0.7,
            variance=0.21
        )
        
        # Sample multiple times to test distribution
        samples = []
        for _ in range(1000):
            sample = sampler.sample_from_posterior(stats)
            samples.append(sample)
            assert 0 <= sample <= 1  # Beta samples in [0,1]
        
        # Mean should be close to (alpha / (alpha + beta))
        # With prior (1,1) and data (7,3): alpha=8, beta=4
        expected_mean = 8 / (8 + 4)
        assert abs(np.mean(samples) - expected_mean) < 0.05
    
    def test_thompson_selection(self):
        """RED: Select based on Thompson sampling."""
        sampler = ThompsonSampler()
        
        test_stats = {
            'test1': TestStats(2, 8, 10, 0.2, 0.16),  # Poor
            'test2': TestStats(5, 5, 10, 0.5, 0.25),  # Average
            'test3': TestStats(8, 2, 10, 0.8, 0.16),  # Good
        }
        
        # Run multiple selections to verify probabilistic behavior
        selection_counts = {'test1': 0, 'test2': 0, 'test3': 0}
        
        for _ in range(1000):
            selected = sampler.select_test(
                available_tests=['test1', 'test2', 'test3'],
                test_stats=test_stats
            )
            selection_counts[selected] += 1
        
        # test3 should be selected most often
        assert selection_counts['test3'] > selection_counts['test2']
        assert selection_counts['test2'] > selection_counts['test1']
    
    def test_handle_no_data(self):
        """RED: Handle selection with no prior data."""
        sampler = ThompsonSampler(prior_alpha=1.0, prior_beta=1.0)
        
        stats = TestStats(0, 0, 0, 0.0, 0.0)
        
        # Should sample from prior Beta(1,1) = Uniform(0,1)
        samples = [sampler.sample_from_posterior(stats) for _ in range(100)]
        
        # All samples should be valid probabilities
        assert all(0 <= s <= 1 for s in samples)
        # Mean should be around 0.5 for uniform prior
        assert 0.3 < np.mean(samples) < 0.7
    
    def test_update_posterior(self):
        """RED: Update posterior with new observations."""
        sampler = ThompsonSampler()
        
        # Initial stats
        stats = TestStats(3, 2, 5, 0.6, 0.24)
        
        # Get initial sample
        initial_samples = [sampler.sample_from_posterior(stats) for _ in range(100)]
        initial_mean = np.mean(initial_samples)
        
        # Update with success
        stats.success_count += 1
        stats.total_attempts += 1
        stats.mean_reward = stats.success_count / stats.total_attempts
        
        # Get updated samples
        updated_samples = [sampler.sample_from_posterior(stats) for _ in range(100)]
        updated_mean = np.mean(updated_samples)
        
        # Mean should increase after success
        assert updated_mean > initial_mean


class TestHybridTransition:
    """TDD tests for transition between strategies."""
    
    def test_confidence_based_transition(self):
        """RED: Transition based on confidence threshold."""
        from guardian.analytics.ucb1_thompson_hybrid import ConfidenceBasedTransition
        
        transition = ConfidenceBasedTransition(
            confidence_threshold=0.8,
            min_samples_per_test=5
        )
        
        # Low confidence state
        test_stats_low = {
            'test1': TestStats(1, 1, 2, 0.5, 0.25),
            'test2': TestStats(2, 1, 3, 0.67, 0.22),
        }
        
        strategy_low = transition.select_strategy(test_stats_low, iteration=10)
        assert strategy_low == SelectionStrategy.UCB1
        
        # High confidence state
        test_stats_high = {
            'test1': TestStats(20, 10, 30, 0.67, 0.22),
            'test2': TestStats(15, 15, 30, 0.5, 0.25),
        }
        
        strategy_high = transition.select_strategy(test_stats_high, iteration=100)
        assert strategy_high == SelectionStrategy.THOMPSON
    
    def test_smooth_transition(self):
        """RED: Ensure smooth transition between strategies."""
        from guardian.analytics.ucb1_thompson_hybrid import SmoothTransition
        
        transition = SmoothTransition(
            transition_start=50,
            transition_end=150
        )
        
        # Before transition
        assert transition.get_thompson_probability(iteration=30) == 0.0
        
        # During transition
        prob_75 = transition.get_thompson_probability(iteration=75)
        prob_100 = transition.get_thompson_probability(iteration=100)
        prob_125 = transition.get_thompson_probability(iteration=125)
        
        assert 0 < prob_75 < prob_100 < prob_125 < 1
        
        # After transition
        assert transition.get_thompson_probability(iteration=200) == 1.0
    
    def test_adaptive_transition(self):
        """RED: Adapt transition based on performance variance."""
        from guardian.analytics.ucb1_thompson_hybrid import AdaptiveTransition
        
        transition = AdaptiveTransition(base_threshold=0.7)
        
        # High variance should delay transition
        high_var_stats = {
            'test1': TestStats(10, 10, 20, 0.5, 0.25),  # High variance
            'test2': TestStats(12, 8, 20, 0.6, 0.24),
        }
        
        # Low variance should allow earlier transition
        low_var_stats = {
            'test1': TestStats(18, 2, 20, 0.9, 0.09),  # Low variance
            'test2': TestStats(17, 3, 20, 0.85, 0.13),
        }
        
        threshold_high_var = transition.get_adaptive_threshold(high_var_stats)
        threshold_low_var = transition.get_adaptive_threshold(low_var_stats)
        
        assert threshold_high_var > threshold_low_var


class TestUCB1ThompsonHybrid:
    """TDD tests for the main hybrid selector."""
    
    @pytest.fixture
    def hybrid_selector(self):
        """Create a hybrid selector instance."""
        return UCB1ThompsonHybrid(
            exploration_constant=2.0,
            transition_threshold=0.8,
            prior_alpha=1.0,
            prior_beta=1.0
        )
    
    @pytest.fixture
    def test_suite(self):
        """Create a test suite."""
        return [Path(f"test_{i}.py") for i in range(10)]
    
    def test_initial_exploration_phase(self, hybrid_selector, test_suite):
        """RED: Verify initial exploration with UCB1."""
        # First selections should use UCB1
        results = []
        
        for _ in range(len(test_suite)):
            result = hybrid_selector.select_next_test_with_info(test_suite)
            results.append(result)
            
            # Update with random reward
            hybrid_selector.update(result.selected_test, np.random.random())
        
        # All tests should be selected once (exploration)
        selected_tests = [r.selected_test for r in results]
        assert set(selected_tests) == set(test_suite)
        
        # Should use UCB1 strategy
        strategies = [r.strategy_used for r in results]
        assert all(s == SelectionStrategy.UCB1 for s in strategies)
    
    def test_transition_to_exploitation(self, hybrid_selector, test_suite):
        """RED: Test transition from exploration to exploitation."""
        # Build up confidence with many samples
        for _ in range(50):
            for test in test_suite:
                reward = np.random.beta(2, 3)  # Some randomness
                hybrid_selector.update(test, reward)
        
        # Now selections should use Thompson more
        thompson_count = 0
        for _ in range(20):
            result = hybrid_selector.select_next_test_with_info(test_suite)
            if result.strategy_used == SelectionStrategy.THOMPSON:
                thompson_count += 1
            
            hybrid_selector.update(result.selected_test, np.random.random())
        
        # Should have transitioned to Thompson
        assert thompson_count >= 10
    
    def test_cold_start_all_untested(self, hybrid_selector, test_suite):
        """RED: Handle cold start with all untested items."""
        result = hybrid_selector.select_next_test_with_info(test_suite)
        
        assert result.selected_test in test_suite
        assert result.strategy_used == SelectionStrategy.UCB1
        assert result.confidence_level == 0.0  # No data yet
    
    def test_get_selection_probabilities(self, hybrid_selector, test_suite):
        """RED: Get selection probabilities for all tests."""
        # Add some data
        for i, test in enumerate(test_suite[:5]):
            for _ in range(i + 1):
                hybrid_selector.update(test, i / 10.0)
        
        probs = hybrid_selector.get_selection_probabilities(test_suite)
        
        assert len(probs) == len(test_suite)
        assert all(0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-6  # Should sum to 1
    
    def test_reset_state(self, hybrid_selector, test_suite):
        """RED: Test resetting selector state."""
        # Add some data
        for test in test_suite[:3]:
            hybrid_selector.update(test, 0.5)
        
        # Verify data exists
        stats = hybrid_selector.get_all_stats()
        assert any(s.total_attempts > 0 for s in stats.values() if s)
        
        # Reset
        hybrid_selector.reset()
        
        # Verify clean state
        stats_after = hybrid_selector.get_all_stats()
        assert all(s is None or s.total_attempts == 0 
                  for s in stats_after.values())
    
    def test_handle_distribution_shift(self, hybrid_selector, test_suite):
        """RED: Adapt to distribution shifts."""
        # Phase 1: Establish baseline
        for _ in range(30):
            test = test_suite[0]  # Focus on one test
            hybrid_selector.update(test, 0.8)  # High reward
        
        # Check confidence is high
        result1 = hybrid_selector.select_next_test_with_info([test_suite[0]])
        initial_confidence = result1.confidence_level
        
        # Phase 2: Distribution shift - sudden poor performance
        for _ in range(10):
            hybrid_selector.update(test_suite[0], 0.1)  # Low reward
        
        # Confidence should drop, exploration should increase
        result2 = hybrid_selector.select_next_test_with_info(test_suite)
        
        assert result2.confidence_level < initial_confidence
    
    def test_multi_objective_selection(self, hybrid_selector):
        """RED: Handle multi-objective test selection."""
        # Tests with characteristics
        test_characteristics = {
            Path("fast_test.py"): {"exec_time": 1, "coverage": 0.8},
            Path("slow_test.py"): {"exec_time": 10, "coverage": 0.9},
            Path("medium_test.py"): {"exec_time": 5, "coverage": 0.85},
        }
        
        result = hybrid_selector.select_next_test_with_info(
            list(test_characteristics.keys()),
            test_characteristics=test_characteristics
        )
        
        assert result.selected_test in test_characteristics
    
    def test_parallel_update_safety(self, hybrid_selector, test_suite):
        """RED: Ensure thread-safe updates."""
        import threading
        
        def update_worker(test_id, n_updates):
            for _ in range(n_updates):
                reward = np.random.random()
                hybrid_selector.update(test_id, reward)
        
        # Parallel updates
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=update_worker,
                args=(test_suite[i], 100)
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all updates were recorded
        stats = hybrid_selector.get_all_stats()
        for i in range(5):
            assert stats[test_suite[i]].total_attempts == 100
    
    def test_custom_strategy_injection(self):
        """RED: Support custom strategy injection."""
        # Create custom exploration strategy
        class CustomExploration(IExplorationStrategy):
            def select_test(self, available_tests, test_stats, total_iterations, **kwargs):
                return available_tests[0]  # Always first
            
            def get_confidence_level(self, stats):
                return 0.5
        
        custom_exploration = CustomExploration()
        
        hybrid = UCB1ThompsonHybrid(
            exploration_strategy=custom_exploration
        )
        
        result = hybrid.select_next_test_with_info(['test1', 'test2'])
        assert result.selected_test == 'test1'
    
    def test_performance_metrics(self, hybrid_selector, test_suite):
        """RED: Track performance metrics."""
        # Run selection and updates
        rewards = []
        for _ in range(100):
            result = hybrid_selector.select_next_test_with_info(test_suite)
            reward = np.random.random()
            rewards.append(reward)
            hybrid_selector.update(result.selected_test, reward)
        
        metrics = hybrid_selector.get_performance_metrics()
        
        assert 'total_iterations' in metrics
        assert metrics['total_iterations'] == 100
        assert 'average_reward' in metrics
        assert abs(metrics['average_reward'] - np.mean(rewards)) < 0.1
        assert 'strategy_distribution' in metrics


class TestIntegration:
    """Integration tests for complete system."""
    
    def test_end_to_end_mab_problem(self):
        """Test on a standard multi-armed bandit problem."""
        # Create arms with different success rates
        true_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        arms = [f"arm_{i}" for i in range(len(true_rates))]
        
        selector = UCB1ThompsonHybrid(
            exploration_constant=2.0,
            transition_threshold=0.8
        )
        
        cumulative_reward = 0
        regrets = []
        
        for t in range(500):
            # Select arm
            result = selector.select_next_test_with_info(arms)
            selected_idx = arms.index(result.selected_test)
            
            # Get reward (Bernoulli)
            reward = 1 if np.random.random() < true_rates[selected_idx] else 0
            cumulative_reward += reward
            
            # Calculate regret
            best_rate = max(true_rates)
            instant_regret = best_rate - true_rates[selected_idx]
            regrets.append(instant_regret)
            
            # Update
            selector.update(result.selected_test, reward)
        
        # Performance checks
        avg_reward = cumulative_reward / 500
        assert avg_reward > 0.6  # Should learn to pick good arms
        
        # Regret should decrease over time
        early_regret = np.mean(regrets[:100])
        late_regret = np.mean(regrets[-100:])
        assert late_regret < early_regret * 0.5
    
    def test_with_shapley_values(self):
        """Test integration with Shapley value calculation."""
        from guardian.analytics.shapley_convergence import OptimizedShapleyCalculator
        
        # Mock test suite with interactions
        test_suite = [Path(f"test_{i}.py") for i in range(5)]
        
        def mock_evaluator(subset):
            # Create interaction effects
            value = len(subset) * 0.1
            if len(subset) > 1:
                value += 0.05 * len(subset)  # Synergy bonus
            return value
        
        # Use hybrid to select subset for Shapley calculation
        selector = UCB1ThompsonHybrid()
        
        # Build up some history
        for _ in range(20):
            test = selector.select_next_test(test_suite)
            reward = mock_evaluator([test])
            selector.update(test, reward)
        
        # Select promising subset based on history
        stats = selector.get_all_stats()
        promising_tests = [
            test for test in test_suite
            if stats.get(test) and stats[test].mean_reward > 0.08
        ]
        
        # Calculate Shapley values for promising subset
        if promising_tests:
            calculator = OptimizedShapleyCalculator()
            shapley_values = calculator.calculate_shapley_values(
                promising_tests,
                mock_evaluator
            )
            
            assert len(shapley_values) == len(promising_tests)
            assert all(v >= 0 for v in shapley_values.values())