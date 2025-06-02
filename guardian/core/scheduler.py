import math
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from .actions import ActionProvider, ActionStats

# Configure logging for the scheduler
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class EQRAScheduler:
    """
    EQRA (Expected Quality Return on Action) Scheduler.
    Selects and executes actions to improve system quality based on the EQRA formula,
    considering expected quality uplift, uncertainty, and cost.
    """

    def __init__(self, actions: List[ActionProvider], risk_aversion_lambda: float = 0.5):
        """
        Initializes the EQRAScheduler.

        Args:
            actions: A list of ActionProvider instances available to the scheduler.
            risk_aversion_lambda: The risk aversion parameter (lambda). Higher values
                                  make the scheduler more averse to uncertainty.
        """
        self.actions: Dict[str, ActionProvider] = {action.get_id(): action for action in actions}
        self.risk_aversion_lambda: float = risk_aversion_lambda
        self.run_history: List[Dict[str, Any]] = []
        logger.info(f"EQRAScheduler initialized with {len(actions)} actions and lambda={risk_aversion_lambda}.")

    def add_action(self, action: ActionProvider):
        """Adds a new action to the scheduler's available actions."""
        if action.get_id() in self.actions:
            logger.warning(f"Action with ID '{action.get_id()}' already exists. Overwriting.")
        self.actions[action.get_id()] = action
        logger.info(f"Action '{action.get_id()}' added to scheduler.")

    def _calculate_eqra(self, action: ActionProvider) -> float:
        """
        Calculates the EQRA score for a given action.
        EQRA(a) = (E[ΔQ|a] - λ * Var[E[ΔQ|a]]) / C(a)
        where:
        - E[ΔQ|a] is the expected quality uplift from action 'a'.
        - Var[E[ΔQ|a]] is the variance of the mean expected quality uplift.
        - λ is the risk aversion parameter.
        - C(a) is the estimated cost of action 'a'.

        Returns:
            The EQRA score. Returns -infinity if cost is zero or negative to avoid division by zero
            and to de-prioritize free or negatively costed actions if they don't offer improvement.
        """
        expected_delta_q = action.stats.get_expected_delta_q()
        variance_of_mean_delta_q = action.stats.get_variance_of_delta_q_mean()
        cost = action.estimate_cost()

        if cost <= 0:
            # Actions with zero or negative cost are problematic for EQRA if they don't guarantee improvement.
            # If they have positive expected_delta_q, this could be infinite.
            # We assign a very low score unless they have positive uplift, then a high score.
            # However, to keep it simple and avoid gaming, let's make it -inf unless positive expected_delta_q.
            # A more sophisticated handling might be needed for "free" beneficial actions.
            logger.warning(f"Action '{action.get_id()}' has cost {cost}. EQRA calculation might be unstable.")
            return float('-inf') # Penalize zero/negative cost actions unless they are purely beneficial

        eqra_score = (expected_delta_q - self.risk_aversion_lambda * variance_of_mean_delta_q) / cost
        logger.debug(f"EQRA for {action.get_id()}: E[ΔQ]={expected_delta_q:.4f}, Var[E[ΔQ]]={variance_of_mean_delta_q:.6f}, Cost={cost:.2f}, EQRA_Score={eqra_score:.4f}")
        return eqra_score

    def select_best_action(self, current_budget: float) -> Optional[ActionProvider]:
        """
        Selects the best action to take based on EQRA scores and budget.

        Args:
            current_budget: The currently available budget.

        Returns:
            The selected ActionProvider instance, or None if no suitable action is found.
        """
        best_action: Optional[ActionProvider] = None
        max_eqra: float = float('-inf') # Initialize with a very low EQRA score

        eligible_actions = []
        for action_id, action in self.actions.items():
            cost = action.estimate_cost()
            if cost > current_budget:
                logger.debug(f"Action '{action_id}' (cost: {cost}) exceeds current budget ({current_budget}). Skipping.")
                continue

            eqra_score = self._calculate_eqra(action)
            if eqra_score > 0: # Only consider actions with positive EQRA
                eligible_actions.append((eqra_score, action))
            else:
                logger.debug(f"Action '{action_id}' has non-positive EQRA score ({eqra_score:.4f}). Skipping.")

        if not eligible_actions:
            logger.info("No actions with positive EQRA found within budget.")
            return None

        # Sort by EQRA score in descending order.
        # If EQRA scores are identical, could add tie-breaking (e.g., lower cost, lower variance)
        eligible_actions.sort(key=lambda x: x[0], reverse=True)
        
        best_eqra_score, best_action_candidate = eligible_actions[0]
        
        logger.info(f"Selected action '{best_action_candidate.get_id()}' with EQRA score: {best_eqra_score:.4f} (Cost: {best_action_candidate.estimate_cost()})")
        return best_action_candidate


    def run_cycle(self, total_budget: float, project_context: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Runs a self-improvement cycle.
        The scheduler will attempt to select and run actions until the budget is exhausted
        or no more beneficial actions (positive EQRA) can be found.

        Args:
            total_budget: The total budget available for this cycle.
            project_context: Optional project-specific context to pass to actions.

        Returns:
            A list of dictionaries, where each dictionary contains details of an executed action.
        """
        current_budget = total_budget
        executed_actions_in_cycle: List[Dict[str, Any]] = []
        cycle_start_time = time.time()
        logger.info(f"Starting EQRA self-improvement cycle with budget: {total_budget:.2f}")

        while current_budget > 0:
            logger.info(f"Current budget: {current_budget:.2f}")
            best_action_to_run = self.select_best_action(current_budget)

            if best_action_to_run is None:
                logger.info("No suitable action found to run. Ending cycle.")
                break

            action_id = best_action_to_run.get_id()
            estimated_cost = best_action_to_run.estimate_cost()
            eqra_score_before_run = self._calculate_eqra(best_action_to_run) # Recalculate for logging, select_best_action already did it

            logger.info(f"Attempting to run action: '{action_id}' (Estimated Cost: {estimated_cost:.2f}, EQRA: {eqra_score_before_run:.4f})")

            # --- Execute the action ---
            # In a real system, this might involve more complex pre/post quality measurement
            # For now, the action's run() method simulates this and returns observed_delta_q
            try:
                action_start_time = time.time()
                # Pass project_context if action's run method expects it.
                # For now, assuming run() doesn't strictly need it or handles its absence.
                observed_delta_q = best_action_to_run.run(context=project_context)
                action_duration = time.time() - action_start_time
                logger.info(f"Action '{action_id}' executed. Observed ΔQ: {observed_delta_q:.4f}, Duration: {action_duration:.2f}s")

                # --- Update action statistics (Bayesian posterior update) ---
                best_action_to_run.update_stats(observed_delta_q)
                logger.info(f"Stats updated for action '{action_id}': New E[ΔQ]={best_action_to_run.stats.get_expected_delta_q():.4f}, New Var[E[ΔQ]]={best_action_to_run.stats.get_variance_of_delta_q_mean():.6f}")

                # --- Log and record ---
                # For simplicity, assume actual cost is same as estimated for mock actions
                cost_incurred = estimated_cost # In a real system, this might differ
                
                action_record = {
                    "action_id": action_id,
                    "estimated_cost": estimated_cost,
                    "cost_incurred": cost_incurred, # Actual cost might differ
                    "observed_delta_q": observed_delta_q,
                    "eqra_score_before_run": eqra_score_before_run,
                    "timestamp": time.time(),
                    "duration_seconds": action_duration,
                    "stats_after_run": best_action_to_run.stats.__dict__.copy() # Log current state of stats
                }
                self.run_history.append(action_record)
                executed_actions_in_cycle.append(action_record)

                current_budget -= cost_incurred

            except Exception as e:
                logger.error(f"Error running action '{action_id}': {e}", exc_info=True)
                # Decide if this error should consume budget or halt the cycle.
                # For now, let's assume it consumes the estimated cost and continue if budget allows.
                current_budget -= estimated_cost # Penalize for failed action
                action_record = { # Log the failure
                    "action_id": action_id,
                    "estimated_cost": estimated_cost,
                    "cost_incurred": estimated_cost, # Or 0 if we don't penalize budget for failures
                    "observed_delta_q": None,
                    "eqra_score_before_run": eqra_score_before_run,
                    "timestamp": time.time(),
                    "status": "FAILED",
                    "error": str(e)
                }
                self.run_history.append(action_record)
                executed_actions_in_cycle.append(action_record)
                # Potentially break or add a cooldown for this action if it fails repeatedly.

            if current_budget <= 0:
                logger.info("Budget exhausted.")
                break
        
        cycle_duration = time.time() - cycle_start_time
        logger.info(f"EQRA self-improvement cycle finished. Total actions executed: {len(executed_actions_in_cycle)}. Total duration: {cycle_duration:.2f}s.")
        return executed_actions_in_cycle

    def get_action_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Returns the current statistics for all actions."""
        return {
            action_id: action.stats.__dict__
            for action_id, action in self.actions.items()
        }

    def get_run_history(self) -> List[Dict[str, Any]]:
        """Returns the history of all actions run by the scheduler."""
        return self.run_history
