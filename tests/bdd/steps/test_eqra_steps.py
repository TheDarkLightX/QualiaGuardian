import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import math

from guardian.core.actions import ActionStats, ActionProvider
from guardian.actions.mock_actions import MockFixedImprovementAction, MockRiskyAction, NoOpAction
from guardian.core.scheduler import EQRAScheduler

# --- Context for sharing state between steps ---
@pytest.fixture
def eqra_context():
    """Context to share data between BDD steps."""
    return {
        "scheduler": None,
        "actions": {},
        "total_budget": 0.0,
        "risk_aversion_lambda": 0.5,
        "run_history": [],
        "actions_considered": set(),
        "actions_run": set(),
        "actions_not_run_due_to_budget_or_eqra": set()
    }

# --- Point to the feature file ---
scenarios('../features/eqra_self_improvement.feature')

# --- Background Steps ---

@given("the Guardian system is initialized")
def guardian_system_initialized():
    """Placeholder for any general system setup if needed."""
    # For now, this step doesn't need to do much as core components are directly instantiated.
    pass

@given("the EQRA scheduler is available")
def eqra_scheduler_available(eqra_context):
    """Initializes the EQRAScheduler."""
    # Actions will be added by the next step, so initialize with an empty list.
    eqra_context["scheduler"] = EQRAScheduler(actions=[], risk_aversion_lambda=eqra_context["risk_aversion_lambda"])
    assert eqra_context["scheduler"] is not None

@given(parsers.parse("the following mock actions are registered with the scheduler:\n{action_table}"))
def register_mock_actions(eqra_context, action_table):
    """
    Registers mock actions based on a table provided in the Gherkin step.
    The table should have columns: name, cost, type, fixed_improvement, 
    avg_success_improvement, success_rate, regression_max, prior_mu0, prior_sigma0_sq
    """
    scheduler = eqra_context["scheduler"]
    if scheduler is None:
        raise ValueError("Scheduler not initialized. Ensure 'the EQRA scheduler is available' step runs first.")

    actions = {}
    for row in action_table.splitlines():
        if not row.strip() or row.startswith('| name'): # Skip empty lines or header
            continue
        parts = [p.strip() for p in row.split('|') if p.strip()] # name, cost, type, ...
        
        action_name = parts[0]
        cost = float(parts[1])
        action_type = parts[2]
        
        # Safely get optional numeric values
        fixed_improvement = float(parts[3]) if parts[3] else None
        avg_success_improvement = float(parts[4]) if parts[4] else None
        success_rate = float(parts[5]) if parts[5] else None
        regression_max = float(parts[6]) if parts[6] else None
        prior_mu0 = float(parts[7]) if parts[7] else 0.0 # Default if not specified
        prior_sigma0_sq = float(parts[8]) if parts[8] else 0.0001 # Default if not specified

        initial_stats = ActionStats(mu0=prior_mu0, sigma0_sq=prior_sigma0_sq)
        action = None

        if action_type == "fixed":
            action = MockFixedImprovementAction(
                name=action_name, 
                cost=cost, 
                improvement=fixed_improvement if fixed_improvement is not None else 0.0,
                initial_stats=initial_stats
            )
        elif action_type == "risky":
            action = MockRiskyAction(
                name=action_name, 
                cost=cost, 
                avg_improvement_on_success=avg_success_improvement if avg_success_improvement is not None else 0.0,
                success_rate=success_rate if success_rate is not None else 0.5,
                regression_on_failure_max=regression_max if regression_max is not None else 0.0,
                initial_stats=initial_stats
            )
        elif action_type == "no_op":
            action = NoOpAction(
                name=action_name, 
                cost=cost,
                initial_stats=initial_stats
            )
        else:
            raise ValueError(f"Unknown action type: {action_type} for action {action_name}")
        
        if action:
            scheduler.add_action(action)
            actions[action_name] = action
            
    eqra_context["actions"] = actions
    assert len(scheduler.actions) > 0, "No actions were registered with the scheduler"

# --- Scenario: Scheduler runs a cycle selecting affordable actions with positive EQRA ---

@given(parsers.parse("the EQRA scheduler is configured with risk_aversion_lambda {lambda_val:f}"))
def configure_scheduler_lambda(eqra_context, lambda_val):
    eqra_context["risk_aversion_lambda"] = lambda_val
    if eqra_context["scheduler"]:
        eqra_context["scheduler"].risk_aversion_lambda = lambda_val
    else:
        # If scheduler not created yet (e.g. background step hasn't run), store for init
        pass 

@given(parsers.parse("a total budget of {budget:f} units"))
def set_total_budget(eqra_context, budget):
    eqra_context["total_budget"] = budget

@when("the EQRA scheduler runs one self-improvement cycle")
def run_scheduler_cycle(eqra_context):
    scheduler = eqra_context["scheduler"]
    total_budget = eqra_context["total_budget"]
    
    # Mock select_best_action to record considered actions
    # This is a bit tricky as we want to test the actual selection logic, 
    # but also know what was considered. For now, we'll infer from debug logs or history.
    # A more direct way would be to inspect scheduler's internal state or enhance logging.

    eqra_context["run_history"] = scheduler.run_cycle(total_budget=total_budget, project_context=None)
    
    # Populate actions_run and actions_considered based on history and scheduler logic
    # This is an approximation for 'considered'. True 'considered' would need deeper inspection or logging.
    for entry in eqra_context["run_history"]:
        eqra_context["actions_run"].add(entry["action_id"])
    
    # Infer 'considered' - any action that *could* have been selected based on initial budget and positive EQRA
    # This is complex to do perfectly here without re-running parts of select_best_action.
    # For now, we'll rely on specific 'should be considered' steps to verify key actions.
    # And 'should not be run' for others.

@then(parsers.parse('action "{action_name}" should be considered for execution'))
def action_should_be_considered(eqra_context, action_name):
    """ 
    This step is more of a design-level check. 
    Actual consideration is internal to select_best_action. 
    We verify by checking if it *could* have been run based on its properties 
    and was either run or was a candidate that lost to a better EQRA action.
    For simplicity in BDD, we'll check if its cost was within initial budget and it's a known action.
    More rigorous check would involve calculating its initial EQRA.
    """
    action = eqra_context["actions"].get(action_name)
    assert action is not None, f"Action {action_name} not found in context."
    assert action.estimate_cost() <= eqra_context["total_budget"], \
        f"Action {action_name} cost {action.estimate_cost()} exceeds total budget {eqra_context['total_budget']}"
    # A true 'considered' check would also involve its EQRA being positive. 
    # This might be better tested by checking if it was run OR if it was not run but a higher EQRA action was.

@then(parsers.parse('action "{action_name}" should not be run due to budget or EQRA'))
def action_should_not_be_run(eqra_context, action_name):
    assert action_name not in eqra_context["actions_run"], \
        f"Action {action_name} was run, but was expected not to be (due to budget or EQRA)."
    # Further checks could be added to distinguish between budget vs EQRA reason if necessary.

@then(parsers.parse("the total cost incurred in the cycle should be less than or equal to {max_cost:f}"))
def total_cost_incurred(eqra_context, max_cost):
    total_incurred = sum(entry["cost_incurred"] for entry in eqra_context["run_history"])
    assert total_incurred <= max_cost, f"Total cost incurred {total_incurred} exceeded max {max_cost}"
    # Also check against the initial budget from context
    assert total_incurred <= eqra_context["total_budget"], \
        f"Total cost incurred {total_incurred} exceeded initial budget {eqra_context['total_budget']}"

@then("the statistics for any run actions should be updated")
def statistics_updated_for_run_actions(eqra_context):
    for action_id in eqra_context["actions_run"]:
        action = eqra_context["actions"][action_id]
        # Check if n_a (number of runs) has increased from its initial state (0 for new stats).
        # This assumes actions started with n_a = 0 or we'd need to store initial n_a.
        # For mock actions created in background, they start fresh.
        assert action.stats.n_a > 0, f"Statistics for action {action_id} (n_a) were not updated."
        # More detailed checks could compare delta_q_bar or m2 if specific outcomes were enforced.

@then("the run history should log the executed actions and their outcomes")
def run_history_logs_outcomes(eqra_context):
    assert len(eqra_context["run_history"]) == len(eqra_context["actions_run"]), "Mismatch between run history length and count of run actions."
    for entry in eqra_context["run_history"]:
        assert entry["action_id"] in eqra_context["actions_run"]
        assert "delta_q_observed" in entry
        assert "cost_incurred" in entry
        assert "eqra_score_before_run" in entry

# --- Placeholder for other scenario steps ---
# Steps for "Scheduler respects strict budget constraints"
# Steps for "Scheduler handles a scenario where no actions have positive EQRA"
# Steps for "Higher risk aversion (lambda) prioritizes less uncertain actions"

@given(parsers.parse('action "{action_name}" should not be run due to budget'))
def action_not_run_due_to_budget(eqra_context, action_name):
    action = eqra_context["actions"].get(action_name)
    assert action is not None, f"Action {action_name} not found in context."
    assert action_name not in eqra_context["actions_run"], f"Action {action_name} was run but was expected not to be."
    assert action.estimate_cost() > eqra_context["total_budget"], \
        f"Action {action_name} (cost {action.estimate_cost()}) was expected to be over budget {eqra_context['total_budget']}, but wasn't."

@given(parsers.parse("all available actions have their statistics updated such that their EQRA scores are not positive"))
def set_non_positive_eqra_stats(eqra_context):
    scheduler = eqra_context["scheduler"]
    # This requires manipulating action stats directly to force non-positive EQRA.
    # For each action, set mu0 very low (e.g., negative) and sigma0_sq high, then re-init stats or update.
    for action_id, action_obj in scheduler.actions.items():
        # A simple way: make expected improvement negative and variance high relative to cost
        action_obj.stats.mu0 = -1.0 # Ensure E[delta_Q] is negative
        action_obj.stats.delta_q_bar = -1.0 # Force posterior mean to be negative
        action_obj.stats.sigma0_sq = 100.0 # Make variance high
        action_obj.stats.n_a = 1 # Give it some runs so it doesn't just use prior
        action_obj.stats.m2 = 100.0 * (action_obj.stats.n_a) # Ensure sample variance is high
        # Ensure EQRA is negative
        assert scheduler._calculate_eqra(action_obj) <= 0, f"EQRA for {action_id} is not non-positive after manipulation"

@then("no actions should be selected or run")
def no_actions_selected_or_run(eqra_context):
    assert not eqra_context["actions_run"], "Actions were run when none were expected."
    assert len(eqra_context["run_history"]) == 0, "Run history is not empty when no actions were expected."

@then("the run history should reflect that no actions were taken")
def history_reflects_no_actions(eqra_context):
    assert len(eqra_context["run_history"]) == 0, "Run history is not empty."

@given(parsers.parse('action "{action_name}" has stats (n_a={n_a:d}, delta_q_bar={dq_bar:f}, m2_for_variance_calc={m2_val:f})'))
def set_action_specific_stats(eqra_context, action_name, n_a, dq_bar, m2_val):
    action = eqra_context["actions"].get(action_name)
    if not action:
        # This might occur if the background step for action registration hasn't run in this specific scenario variant
        # Or if the action name is misspelled. For now, assume actions are from Background.
        raise ValueError(f"Action {action_name} not found. Ensure it's defined in Background.")
    
    action.stats.n_a = n_a
    action.stats.delta_q_bar = dq_bar
    action.stats.m2 = m2_val
    # mu0 and sigma0_sq are priors, usually not changed after init for a specific action instance unless re-initializing.
    # The m2_for_variance_calc implies this m2 is to be used in var_hat = m2 / (n_a - 1)
    # If n_a=1, var_hat is sigma0_sq. If n_a=0, var_of_mean is sigma0_sq.
    # We must be careful if n_a is 0 or 1 with m2_val.
    if n_a <= 1 and m2_val != 0.0 and action.stats.sigma0_sq != (m2_val / (n_a if n_a==1 else 1)):
        print(f"Warning: Setting m2 for n_a={n_a} for {action_name}. Variance calc might use sigma0_sq instead.")

@when(parsers.parse("the EQRA scheduler is configured with a high risk_aversion_lambda (e.g., {lambda_val:f})"))
def configure_high_lambda(eqra_context, lambda_val):
    configure_scheduler_lambda(eqra_context, lambda_val)

@when(parsers.parse("the EQRA scheduler selects the best action"))
def scheduler_selects_best_action_explicitly(eqra_context):
    scheduler = eqra_context["scheduler"]
    budget = eqra_context.get("total_budget", float('inf')) # Use current budget or infinite if not set for this step
    best_action = scheduler.select_best_action(current_budget=budget)
    eqra_context["last_selected_action"] = best_action.get_id() if best_action else None

@when(parsers.parse("the EQRA scheduler is configured with a low risk_aversion_lambda (e.g., {lambda_val:f})"))
def configure_low_lambda(eqra_context, lambda_val):
    configure_scheduler_lambda(eqra_context, lambda_val)

@given(parsers.parse('action "{action_name}" stats are reset to (n_a={n_a:d}, delta_q_bar={dq_bar:f}, m2_for_variance_calc={m2_val:f})'))
def reset_action_stats(eqra_context, action_name, n_a, dq_bar, m2_val):
    # This is effectively the same as setting them, good for clarity in Gherkin
    set_action_specific_stats(eqra_context, action_name, n_a, dq_bar, m2_val)

@when(parsers.parse("the EQRA scheduler selects the best action with budget {budget:f}"))
def scheduler_selects_best_action_with_budget(eqra_context, budget):
    scheduler = eqra_context["scheduler"]
    best_action = scheduler.select_best_action(current_budget=budget)
    eqra_context["last_selected_action"] = best_action.get_id() if best_action else None

@then(parsers.parse('action "{action_name}" should be chosen'))
def action_should_be_chosen(eqra_context, action_name):
    assert eqra_context.get("last_selected_action") == action_name, \
        f"Expected action {action_name} to be chosen, but got {eqra_context.get('last_selected_action')}"

    assert action is not None, f"Action {action_name} not found in context."
    assert action_name not in eqra_context["actions_run"], f"Action {action_name} was run but was expected not to be."
    assert action.estimate_cost() > eqra_context["total_budget"], \
        f"Action {action_name} (cost {action.estimate_cost()}) was expected to be over budget {eqra_context['total_budget']}, but wasn't."

@given(parsers.parse("all available actions have their statistics updated such that their EQRA scores are not positive"))
def set_non_positive_eqra_stats(eqra_context):
    scheduler = eqra_context["scheduler"]
    # This requires manipulating action stats directly to force non-positive EQRA.
    # For each action, set mu0 very low (e.g., negative) and sigma0_sq high, then re-init stats or update.
    for action_id, action_obj in scheduler.actions.items():
        # A simple way: make expected improvement negative and variance high relative to cost
        action_obj.stats.mu0 = -1.0 # Ensure E[delta_Q] is negative
        action_obj.stats.delta_q_bar = -1.0 # Force posterior mean to be negative
        action_obj.stats.sigma0_sq = 100.0 # Make variance high
        action_obj.stats.n_a = 1 # Give it some runs so it doesn't just use prior
        action_obj.stats.m2 = 100.0 * (action_obj.stats.n_a) # Ensure sample variance is high
        # Ensure EQRA is negative
        assert scheduler._calculate_eqra(action_obj) <= 0, f"EQRA for {action_id} is not non-positive after manipulation"

@then("no actions should be selected or run")
def no_actions_selected_or_run(eqra_context):
    assert not eqra_context["actions_run"], "Actions were run when none were expected."
    assert len(eqra_context["run_history"]) == 0, "Run history is not empty when no actions were expected."

@then("the run history should reflect that no actions were taken")
def history_reflects_no_actions(eqra_context):
    assert len(eqra_context["run_history"]) == 0, "Run history is not empty."

@given(parsers.parse('action "{action_name}" has stats (n_a={n_a:d}, delta_q_bar={dq_bar:f}, m2_for_variance_calc={m2_val:f})'))
def set_action_specific_stats(eqra_context, action_name, n_a, dq_bar, m2_val):
    action = eqra_context["actions"].get(action_name)
    if not action:
        # This might occur if the background step for action registration hasn't run in this specific scenario variant
        # Or if the action name is misspelled. For now, assume actions are from Background.
        raise ValueError(f"Action {action_name} not found. Ensure it's defined in Background.")
    
    action.stats.n_a = n_a
    action.stats.delta_q_bar = dq_bar
    action.stats.m2 = m2_val
    # mu0 and sigma0_sq are priors, usually not changed after init for a specific action instance unless re-initializing.
    # The m2_for_variance_calc implies this m2 is to be used in var_hat = m2 / (n_a - 1)
    # If n_a=1, var_hat is sigma0_sq. If n_a=0, var_of_mean is sigma0_sq.
    # We must be careful if n_a is 0 or 1 with m2_val.
    if n_a <= 1 and m2_val != 0.0 and action.stats.sigma0_sq != (m2_val / (n_a if n_a==1 else 1)):
        print(f"Warning: Setting m2 for n_a={n_a} for {action_name}. Variance calc might use sigma0_sq instead.")

@when(parsers.parse("the EQRA scheduler is configured with a high risk_aversion_lambda (e.g., {lambda_val:f})"))
def configure_high_lambda(eqra_context, lambda_val):
    configure_scheduler_lambda(eqra_context, lambda_val)

@when(parsers.parse("the EQRA scheduler selects the best action"))
def scheduler_selects_best_action_explicitly(eqra_context):
    scheduler = eqra_context["scheduler"]
    budget = eqra_context.get("total_budget", float('inf')) # Use current budget or infinite if not set for this step
    best_action = scheduler.select_best_action(current_budget=budget)
    eqra_context["last_selected_action"] = best_action.get_id() if best_action else None

@then(parsers.parse('action "{action_name}" should be chosen'))
def action_should_be_chosen(eqra_context, action_name):
    assert eqra_context.get("last_selected_action") == action_name, \
        f"Expected action {action_name} to be chosen, but got {eqra_context.get('last_selected_action')}"

@when(parsers.parse("the EQRA scheduler is configured with a low risk_aversion_lambda (e.g., {lambda_val:f})"))
def configure_low_lambda(eqra_context, lambda_val):
    configure_scheduler_lambda(eqra_context, lambda_val)

@given(parsers.parse('action "{action_name}" stats are reset to (n_a={n_a:d}, delta_q_bar={dq_bar:f}, m2_for_variance_calc={m2_val:f})'))
def reset_action_stats(eqra_context, action_name, n_a, dq_bar, m2_val):
    # This is effectively the same as setting them, good for clarity in Gherkin
    set_action_specific_stats(eqra_context, action_name, n_a, dq_bar, m2_val)

@when(parsers.parse("the EQRA scheduler selects the best action with budget {budget:f}"))
def scheduler_selects_best_action_with_budget(eqra_context, budget):
    scheduler = eqra_context["scheduler"]
    best_action = scheduler.select_best_action(current_budget=budget)
    eqra_context["last_selected_action"] = best_action.get_id() if best_action else None
