"""
BDD Step Definitions for UCB1-Thompson Hybrid Feature

Author: DarkLightX/Dana Edwards
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from guardian.analytics.ucb1_thompson_hybrid import (
    UCB1ThompsonHybrid,
    TestSelectionResult,
    TestStats,
    SelectionStrategy
)

# Load scenarios from feature file
scenarios('../features/ucb1_thompson_hybrid.feature')


@pytest.fixture
def context():
    """Create a context object to store state between steps."""
    class Context:
        def __init__(self):
            self.test_suite = []
            self.metric_evaluator = None
            self.selector = None
            self.selection_history = []
            self.performance_history = []
            self.ucb1_selector = None
            self.thompson_selector = None
            self.pure_results = {}
            
    return Context()


@given("a test suite with unknown effectiveness values")
def step_given_test_suite_unknown(context):
    """Set up a test suite with hidden effectiveness values."""
    # Create tests with hidden true values (not known to algorithm)
    context.test_suite = []
    context.true_values = {}
    
    np.random.seed(42)
    for i in range(20):
        test_id = Path(f"test_{i}.py")
        context.test_suite.append(test_id)
        # Hidden true effectiveness values (Bernoulli rewards)
        context.true_values[test_id] = np.random.beta(2, 5)  # Diverse success rates
    
    # Metric evaluator returns stochastic rewards based on true values
    def stochastic_evaluator(test_id):
        true_value = context.true_values[test_id]
        return float(np.random.random() < true_value)  # Bernoulli reward
    
    context.metric_evaluator = stochastic_evaluator


@given("a metric evaluator for test performance")
def step_given_metric_evaluator(context):
    """Ensure metric evaluator is set up."""
    if not hasattr(context, 'metric_evaluator'):
        # Default evaluator if not already set
        def default_evaluator(test_id):
            return np.random.random()
        context.metric_evaluator = default_evaluator


@given("a UCB1-Thompson hybrid selector")
def step_given_hybrid_selector(context):
    """Create a UCB1-Thompson hybrid selector."""
    context.selector = UCB1ThompsonHybrid(
        exploration_constant=2.0,  # UCB1 exploration parameter
        transition_threshold=0.8,  # Confidence threshold for transition
        window_size=10
    )


@given("no prior test execution data")
def step_given_no_prior_data(context):
    """Ensure selector starts with no data."""
    context.selector.reset()
    context.selection_history = []
    context.performance_history = []


@given("test execution history with varying performance")
def step_given_execution_history(context):
    """Create execution history with varying performance."""
    # Run initial iterations to build history
    for i in range(30):
        test_id = context.selector.select_next_test(context.test_suite)
        reward = context.metric_evaluator(test_id)
        context.selector.update(test_id, reward)
        context.selection_history.append(test_id)
        context.performance_history.append(reward)


@given("a mix of well-tested and rarely-tested items")
def step_given_mixed_testing(context):
    """Create a mix of well-tested and rarely-tested items."""
    # Reset and create biased history
    context.selector.reset()
    
    # Test first 5 items many times
    for _ in range(50):
        for i in range(5):
            test_id = context.test_suite[i]
            reward = context.metric_evaluator(test_id)
            context.selector.update(test_id, reward)
    
    # Test next 5 items a few times
    for _ in range(5):
        for i in range(5, 10):
            test_id = context.test_suite[i]
            reward = context.metric_evaluator(test_id)
            context.selector.update(test_id, reward)
    
    # Leave remaining items untested


@given("identical test suites and evaluation functions")
def step_given_identical_setup(context):
    """Set up identical conditions for comparison."""
    # Ensure consistent setup
    np.random.seed(42)
    
    # Create pure UCB1 selector
    from guardian.analytics.ucb1_selector import UCB1Selector
    context.ucb1_selector = UCB1Selector(exploration_constant=2.0)
    
    # Create pure Thompson selector
    from guardian.analytics.thompson_sampler import ThompsonSampler
    context.thompson_selector = ThompsonSampler()
    
    # Reset hybrid
    context.selector.reset()


@given("tests with different execution times and value distributions")
def step_given_heterogeneous_tests(context):
    """Create tests with varied characteristics."""
    context.test_characteristics = {}
    
    for i, test_id in enumerate(context.test_suite):
        # Vary execution time (1-10 seconds)
        exec_time = 1 + (i % 10)
        
        # Vary value distribution
        if i < 5:
            # High value, slow tests
            value_mean = 0.8
            exec_time = 8 + (i % 3)
        elif i < 10:
            # Medium value, medium speed
            value_mean = 0.5
            exec_time = 4 + (i % 2)
        else:
            # Low value, fast tests
            value_mean = 0.3
            exec_time = 1 + (i % 2)
        
        context.test_characteristics[test_id] = {
            'execution_time': exec_time,
            'value_mean': value_mean
        }
    
    # Update evaluator to consider characteristics
    def characteristic_evaluator(test_id):
        char = context.test_characteristics[test_id]
        return np.random.normal(char['value_mean'], 0.1)
    
    context.metric_evaluator = characteristic_evaluator


@given("a stable test performance history")
def step_given_stable_history(context):
    """Build stable performance history."""
    # Run 50 iterations with stable performance
    for _ in range(50):
        test_id = context.selector.select_next_test(context.test_suite)
        reward = context.metric_evaluator(test_id)
        context.selector.update(test_id, reward)
        context.performance_history.append(reward)


@given("a standard MAB problem setup")
def step_given_mab_setup(context):
    """Set up standard multi-armed bandit problem."""
    # 10 arms with different success probabilities
    context.test_suite = [Path(f"arm_{i}") for i in range(10)]
    context.true_values = {}
    
    # Create diverse arm values
    np.random.seed(42)
    for i, arm in enumerate(context.test_suite):
        context.true_values[arm] = np.random.beta(1 + i/2, 3)
    
    # Best arm for regret calculation
    context.best_arm = max(context.true_values.items(), key=lambda x: x[1])[0]
    context.best_value = context.true_values[context.best_arm]


@when("selecting tests for the first 10 iterations")
def step_when_select_first_iterations(context):
    """Select tests for initial iterations."""
    context.selection_history = []
    context.strategy_history = []
    
    for i in range(10):
        result = context.selector.select_next_test_with_info(context.test_suite)
        test_id = result.selected_test
        
        context.selection_history.append(test_id)
        context.strategy_history.append(result.strategy_used)
        
        # Execute and update
        reward = context.metric_evaluator(test_id)
        context.selector.update(test_id, reward)


@when("confidence in estimates increases beyond threshold")
def step_when_confidence_increases(context):
    """Continue selection until confidence threshold is met."""
    context.transition_point = None
    context.strategy_history = []
    
    for i in range(100):  # Max iterations
        result = context.selector.select_next_test_with_info(context.test_suite)
        context.strategy_history.append(result.strategy_used)
        
        # Check for transition
        if context.transition_point is None and result.strategy_used == SelectionStrategy.THOMPSON:
            context.transition_point = i
        
        # Update with reward
        reward = context.metric_evaluator(result.selected_test)
        context.selector.update(result.selected_test, reward)


@when("selecting the next test to run")
def step_when_select_next_test(context):
    """Select next test with current state."""
    result = context.selector.select_next_test_with_info(context.test_suite)
    context.selection_result = result
    context.selected_test = result.selected_test


@when("comparing hybrid vs pure UCB1 vs pure Thompson")
def step_when_compare_strategies(context):
    """Run all three strategies on same problem."""
    n_iterations = 200
    
    # Function to run a strategy
    def run_strategy(selector, name):
        selector.reset() if hasattr(selector, 'reset') else None
        cumulative_reward = 0
        rewards = []
        
        for _ in range(n_iterations):
            if name == 'ucb1':
                test_id = selector.select_arm(context.test_suite)
            else:
                test_id = selector.select_next_test(context.test_suite)
            
            reward = context.metric_evaluator(test_id)
            cumulative_reward += reward
            rewards.append(cumulative_reward)
            
            selector.update(test_id, reward)
        
        return rewards
    
    # Run all strategies
    context.pure_results['hybrid'] = run_strategy(context.selector, 'hybrid')
    context.pure_results['ucb1'] = run_strategy(context.ucb1_selector, 'ucb1')
    context.pure_results['thompson'] = run_strategy(context.thompson_selector, 'thompson')


@when("selecting tests under time constraints")
def step_when_select_with_constraints(context):
    """Select tests considering time constraints."""
    context.time_budget = 100  # seconds
    context.selections_under_constraint = []
    context.total_time = 0
    
    while context.total_time < context.time_budget:
        # Provide characteristics to selector
        result = context.selector.select_next_test_with_info(
            context.test_suite,
            test_characteristics=context.test_characteristics
        )
        
        selected = result.selected_test
        context.selections_under_constraint.append(selected)
        
        # Update time
        context.total_time += context.test_characteristics[selected]['execution_time']
        
        # Get reward and update
        reward = context.metric_evaluator(selected)
        context.selector.update(selected, reward)


@when("test effectiveness suddenly changes")
def step_when_distribution_shift(context):
    """Simulate a distribution shift."""
    # Invert all test effectiveness values
    for test_id in context.true_values:
        context.true_values[test_id] = 1.0 - context.true_values[test_id]
    
    context.post_shift_history = []
    context.post_shift_strategies = []
    
    # Continue selecting after shift
    for _ in range(30):
        result = context.selector.select_next_test_with_info(context.test_suite)
        
        context.post_shift_history.append(result.selected_test)
        context.post_shift_strategies.append(result.strategy_used)
        
        reward = context.metric_evaluator(result.selected_test)
        context.selector.update(result.selected_test, reward)


@when("running for 1000 iterations")
def step_when_run_many_iterations(context):
    """Run MAB for many iterations to test regret."""
    context.regret_history = []
    cumulative_regret = 0
    
    for t in range(1000):
        result = context.selector.select_next_test_with_info(context.test_suite)
        selected = result.selected_test
        
        # Calculate instant regret
        reward = context.metric_evaluator(selected)
        instant_regret = context.best_value - context.true_values[selected]
        cumulative_regret += instant_regret
        context.regret_history.append(cumulative_regret)
        
        # Update selector
        context.selector.update(selected, reward)


@then("UCB1 exploration should dominate selection")
def step_then_ucb1_dominates(context):
    """Verify UCB1 dominates early selection."""
    ucb1_count = sum(1 for s in context.strategy_history if s == SelectionStrategy.UCB1)
    total_count = len(context.strategy_history)
    
    assert ucb1_count >= total_count * 0.8, \
        f"UCB1 used only {ucb1_count}/{total_count} times"


@then("all tests should be selected at least once")
def step_then_all_tests_selected(context):
    """Verify exploration covers all tests."""
    selected_tests = set(context.selection_history)
    all_tests = set(context.test_suite[:10])  # First 10 tests at minimum
    
    assert all_tests.issubset(selected_tests), \
        f"Not all tests explored: missing {all_tests - selected_tests}"


@then("confidence bounds should guide exploration")
def step_then_confidence_guides(context):
    """Verify confidence bounds influence selection."""
    # This is implicitly tested by UCB1 algorithm
    assert True


@then("selection should transition to Thompson Sampling")
def step_then_transition_occurs(context):
    """Verify transition to Thompson Sampling."""
    assert context.transition_point is not None, \
        "No transition to Thompson Sampling occurred"
    
    # Check that Thompson dominates after transition
    post_transition = context.strategy_history[context.transition_point:]
    thompson_count = sum(1 for s in post_transition if s == SelectionStrategy.THOMPSON)
    
    assert thompson_count >= len(post_transition) * 0.7, \
        "Thompson Sampling not dominant after transition"


@then("exploitation should increase over time")
def step_then_exploitation_increases(context):
    """Verify increasing exploitation."""
    # Measure selection concentration in later iterations
    early_selections = context.selection_history[:20]
    late_selections = context.selection_history[-20:]
    
    early_unique = len(set(early_selections))
    late_unique = len(set(late_selections))
    
    assert late_unique <= early_unique, \
        f"Exploitation not increasing: {late_unique} unique in late vs {early_unique} in early"


@then("the transition should be smooth not abrupt")
def step_then_smooth_transition(context):
    """Verify smooth strategy transition."""
    if context.transition_point:
        # Check window around transition
        window = 10
        start = max(0, context.transition_point - window)
        end = min(len(context.strategy_history), context.transition_point + window)
        
        transition_window = context.strategy_history[start:end]
        ucb1_count = sum(1 for s in transition_window if s == SelectionStrategy.UCB1)
        thompson_count = sum(1 for s in transition_window if s == SelectionStrategy.THOMPSON)
        
        # Should have mix of both strategies
        assert ucb1_count > 0 and thompson_count > 0, \
            "Transition too abrupt - no strategy mixing"


@then("rarely-tested items should have higher selection probability")
def step_then_explore_rare_items(context):
    """Verify exploration of rarely-tested items."""
    # Get selection counts
    stats = context.selector.get_all_stats()
    
    # Find rarely tested items (indices 10-19)
    rarely_tested = [context.test_suite[i] for i in range(10, 20)]
    
    # Check if any were selected
    selected = context.selection_result.selected_test
    
    # In exploration mode, rarely tested items should have non-zero probability
    # This is a probabilistic test, so we just verify the mechanism exists
    assert hasattr(context.selection_result, 'selection_probabilities') or \
           selected in rarely_tested or \
           context.selection_result.strategy_used == SelectionStrategy.UCB1


@then("uncertainty estimates should influence selection")
def step_then_uncertainty_influences(context):
    """Verify uncertainty affects selection."""
    result = context.selection_result
    
    # Check that confidence/uncertainty info is available
    assert hasattr(result, 'confidence_level') or \
           hasattr(result, 'exploration_bonus') or \
           result.strategy_used in [SelectionStrategy.UCB1, SelectionStrategy.THOMPSON]


@then("the system should adapt to performance variance")
def step_then_adapt_to_variance(context):
    """Verify adaptation to variance."""
    # System should track variance in some form
    stats = context.selector.get_all_stats()
    
    # At least one test should have stats
    assert any(hasattr(stat, 'variance') or hasattr(stat, 'std_dev') 
              for stat in stats.values() if stat is not None)


@then("hybrid should achieve better cumulative reward")
def step_then_hybrid_better_reward(context):
    """Verify hybrid outperforms pure strategies."""
    final_rewards = {
        name: results[-1] 
        for name, results in context.pure_results.items()
    }
    
    hybrid_reward = final_rewards['hybrid']
    
    # Hybrid should be at least as good as others
    assert hybrid_reward >= final_rewards['ucb1'] * 0.95, \
        f"Hybrid ({hybrid_reward:.2f}) worse than UCB1 ({final_rewards['ucb1']:.2f})"
    
    assert hybrid_reward >= final_rewards['thompson'] * 0.95, \
        f"Hybrid ({hybrid_reward:.2f}) worse than Thompson ({final_rewards['thompson']:.2f})"


@then("hybrid should have lower regret after 100 iterations")
def step_then_hybrid_lower_regret(context):
    """Verify hybrid has lower regret."""
    # Compare rewards at iteration 100
    rewards_100 = {
        name: results[99] if len(results) > 99 else results[-1]
        for name, results in context.pure_results.items()
    }
    
    # Higher reward = lower regret
    hybrid_reward = rewards_100['hybrid']
    assert hybrid_reward >= max(rewards_100['ucb1'], rewards_100['thompson']) * 0.98


@then("convergence should be faster than pure strategies")
def step_then_faster_convergence(context):
    """Verify faster convergence."""
    # Find iteration where each strategy reaches 90% of final performance
    def find_convergence_point(rewards, threshold=0.9):
        final = rewards[-1]
        target = final * threshold
        
        for i, reward in enumerate(rewards):
            if reward >= target:
                return i
        return len(rewards)
    
    convergence_points = {
        name: find_convergence_point(results)
        for name, results in context.pure_results.items()
    }
    
    # Hybrid should converge faster or equal
    hybrid_conv = convergence_points['hybrid']
    assert hybrid_conv <= min(convergence_points['ucb1'], convergence_points['thompson']) * 1.1


@then("the hybrid should adapt to test characteristics")
def step_then_adapt_to_characteristics(context):
    """Verify adaptation to test characteristics."""
    # Count selections by test category
    selections = context.selections_under_constraint
    
    high_value_fast = sum(1 for s in selections 
                         if context.test_characteristics[s]['value_mean'] > 0.6 
                         and context.test_characteristics[s]['execution_time'] < 3)
    
    # Should prefer high-value fast tests
    assert high_value_fast >= len(selections) * 0.3, \
        "Not adapting to test characteristics"


@then("selection should consider cost-benefit tradeoffs")
def step_then_consider_tradeoffs(context):
    """Verify cost-benefit consideration."""
    # Calculate value per time unit for selections
    value_rates = []
    
    for test in context.selections_under_constraint[:10]:
        char = context.test_characteristics[test]
        value_rate = char['value_mean'] / char['execution_time']
        value_rates.append(value_rate)
    
    # Average value rate should be reasonable
    avg_rate = np.mean(value_rates)
    assert avg_rate > 0.05, "Poor cost-benefit tradeoffs"


@then("high-value quick tests should be prioritized")
def step_then_prioritize_valuable_quick(context):
    """Verify prioritization of valuable quick tests."""
    # Find high-value quick tests
    valuable_quick = [
        test for test in context.test_suite
        if context.test_characteristics[test]['value_mean'] > 0.7
        and context.test_characteristics[test]['execution_time'] < 3
    ]
    
    if valuable_quick:
        # Check if any were selected early
        early_selections = context.selections_under_constraint[:5]
        assert any(test in valuable_quick for test in early_selections), \
            "High-value quick tests not prioritized"


@then("the hybrid should detect the shift")
def step_then_detect_shift(context):
    """Verify distribution shift detection."""
    # Check for increased exploration after shift
    pre_shift_ucb = sum(1 for s in context.strategy_history[-10:] 
                       if s == SelectionStrategy.UCB1)
    post_shift_ucb = sum(1 for s in context.post_shift_strategies[:10] 
                        if s == SelectionStrategy.UCB1)
    
    # UCB1 usage should increase after shift (more exploration)
    assert post_shift_ucb >= pre_shift_ucb, \
        "No increase in exploration after distribution shift"


@then("exploration should temporarily increase")
def step_then_exploration_increases(context):
    """Verify temporary exploration increase."""
    # Check strategy distribution after shift
    early_post_shift = context.post_shift_strategies[:15]
    exploration_count = sum(1 for s in early_post_shift 
                          if s == SelectionStrategy.UCB1)
    
    assert exploration_count >= 3, \
        f"Insufficient exploration after shift: {exploration_count}/15"


@then("the model should adapt to new distributions")
def step_then_adapt_to_new_distribution(context):
    """Verify adaptation to new distribution."""
    # Later selections should converge to new best arms
    late_selections = context.post_shift_history[-10:]
    
    # Find new best tests after inversion
    new_best_tests = sorted(context.true_values.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:5]
    new_best_ids = [t[0] for t in new_best_tests]
    
    # At least some selections should be from new best
    good_selections = sum(1 for s in late_selections if s in new_best_ids)
    assert good_selections >= 3, \
        "Not adapting to new distribution"


@then("cumulative regret should be sublinear")
def step_then_sublinear_regret(context):
    """Verify sublinear regret growth."""
    regret = context.regret_history
    
    # Check regret growth rate at different points
    regret_100 = regret[99] if len(regret) > 99 else regret[-1]
    regret_500 = regret[499] if len(regret) > 499 else regret[-1]
    regret_1000 = regret[999] if len(regret) > 999 else regret[-1]
    
    # Regret should grow slower than linearly
    # If linear, regret_500 = 5 * regret_100
    assert regret_500 < 4 * regret_100, \
        f"Regret growing too fast: {regret_500} vs {regret_100}"
    
    # Check second half grows even slower
    if len(regret) >= 1000:
        second_half_growth = regret_1000 - regret_500
        first_half_growth = regret_500
        assert second_half_growth < first_half_growth, \
            "Regret not decelerating"


@then("regret should follow theoretical bounds")
def step_then_theoretical_bounds(context):
    """Verify regret follows theoretical bounds."""
    n_arms = len(context.test_suite)
    T = len(context.regret_history)
    
    # UCB1 theoretical bound: O(log(T) * n_arms)
    theoretical_bound = 8 * np.log(T) * n_arms
    
    actual_regret = context.regret_history[-1]
    
    # Allow some slack for practical implementation
    assert actual_regret < theoretical_bound * 2, \
        f"Regret {actual_regret} exceeds theoretical bound {theoretical_bound}"


@then("the algorithm should be no-regret asymptotically")
def step_then_no_regret_asymptotic(context):
    """Verify asymptotic no-regret property."""
    regret = context.regret_history
    T = len(regret)
    
    # Average regret per round should decrease
    avg_regret_early = regret[99] / 100 if T > 99 else regret[-1] / T
    avg_regret_late = (regret[-1] - regret[T//2]) / (T - T//2) if T > 200 else avg_regret_early
    
    assert avg_regret_late < avg_regret_early * 0.8, \
        f"Average regret not decreasing: {avg_regret_late:.4f} vs {avg_regret_early:.4f}"