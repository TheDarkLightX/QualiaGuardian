import pytest
import math

# Ensure these imports work. This depends on:
# 1. scheduler.py having been created by you in guardian/core/
# 2. Your Python environment/PYTHONPATH being set up to find the 'guardian' package
#    (e.g., running pytest from the root of the 'Qualia' project or having 'Qualia' in PYTHONPATH)
from guardian.core.actions import ActionStats
from guardian.actions.mock_actions import MockFixedImprovementAction, MockRiskyAction, NoOpAction
from guardian.core.scheduler import EQRAScheduler

# --- Tests for ActionStats ---

def test_action_stats_initialization_default():
    """Tests ActionStats initialization with default prior values."""
    stats = ActionStats()
    assert stats.n_a == 0, "n_a should be 0 initially"
    assert stats.m2 == 0.0, "m2 should be 0.0 initially"
    assert stats.mu0 == 0.02, "Default prior mean mu0 is incorrect"
    assert stats.sigma0_sq == 0.0004, "Default prior variance sigma0_sq is incorrect"
    
    # delta_q_bar should be initialized to the prior mean mu0
    assert stats.delta_q_bar == stats.mu0, "delta_q_bar should be mu0 initially"
    
    # get_expected_delta_q() should return mu0 when n_a is 0
    assert stats.get_expected_delta_q() == stats.mu0, "E[ΔQ] should be mu0 for n_a=0"
    
    # get_variance_of_delta_q_mean() should return sigma0_sq (prior variance of an observation, treated as variance of mean for n_a=0)
    assert stats.get_variance_of_delta_q_mean() == stats.sigma0_sq, "Var[E[ΔQ]] should be sigma0_sq for n_a=0"

def test_action_stats_initialization_custom_priors():
    """Tests ActionStats initialization with custom prior values."""
    custom_mu0 = 0.1
    custom_sigma0_sq = 0.001
    stats = ActionStats(mu0=custom_mu0, sigma0_sq=custom_sigma0_sq)
    
    assert stats.n_a == 0
    assert stats.m2 == 0.0
    assert stats.mu0 == custom_mu0
    assert stats.sigma0_sq == custom_sigma0_sq
    assert stats.delta_q_bar == custom_mu0
    assert stats.get_expected_delta_q() == custom_mu0
    assert stats.get_variance_of_delta_q_mean() == custom_sigma0_sq

def test_action_stats_update_single_observation():
    """Tests updating ActionStats with a single observation."""
    stats = ActionStats(mu0=0.02, sigma0_sq=0.0004)
    observed_delta_q = 0.05
    
    stats.update_stats(observed_delta_q)
    
    assert stats.n_a == 1
    # For n_a = 1, delta_q_bar becomes the first observation
    assert stats.delta_q_bar == observed_delta_q
    assert stats.get_expected_delta_q() == observed_delta_q
    
    # For n_a = 1, m2 is 0.0 as per Welford's algorithm initialization for S_k
    assert stats.m2 == 0.0
    
    # Variance of the mean for n_a = 1: var_hat is sigma0_sq, so Var[E[ΔQ]] = sigma0_sq / 1
    assert stats.get_variance_of_delta_q_mean() == stats.sigma0_sq 

# We will add more tests for ActionStats.update_stats with multiple observations,
# tests for mock ActionProviders, and tests for EQRAScheduler in subsequent steps.

# Placeholder for further tests to be added:
# def test_action_stats_update_multiple_observations():
#     pass

# def test_mock_fixed_improvement_action():
#     pass

# def test_mock_risky_action():
#     pass

# def test_no_op_action():
#     pass

# def test_eqra_scheduler_initialization():
#     pass

# def test_eqra_scheduler_calculate_eqra():
#     pass

# def test_eqra_scheduler_select_best_action():
#     pass

# def test_eqra_scheduler_run_cycle():
#     pass
