"""
Autonomous Quality-Optimizer Agent main loop and logic.
"""
import logging
import json
import time # For potential timeouts or delays
from pathlib import Path # Import Path
from typing import Dict, Any, Optional, List # Import List

# Guardian components
from guardian.metrics.api import get_latest_metrics, get_available_actions
from guardian.history import HistoryManager
# get_action_posteriors is in HistoryManager, so an instance will provide it.
from guardian.agent_interface import apply_patch, verify_patch
from guardian.agent.llm_wrapper import (
    LLMWrapper,
    DECISION_AGENT_SYSTEM_PROMPT,
    DECISION_AGENT_TOOLS,
    AUTO_TEST_IMPL_AGENT_SYSTEM_PROMPT, # Example, will need a map for action_id -> prompt
    FLAKE_HEAL_IMPL_AGENT_SYSTEM_PROMPT, # Example
    IMPLEMENTATION_AGENT_TOOLS
)

logger = logging.getLogger(__name__)

# Define a mapping from action_id to its specific implementation system prompt
IMPLEMENTATION_PROMPT_MAP = {
    "auto_test": AUTO_TEST_IMPL_AGENT_SYSTEM_PROMPT,
    "flake_heal": FLAKE_HEAL_IMPL_AGENT_SYSTEM_PROMPT,
    # Add other actions here
}

class OptimizerAgent:
    def __init__(self,
                 project_path: str,
                 initial_budget_cpu_minutes: float,
                 target_quality_metric: str = "betes_score", # e.g., bE-TES or OSQI
                 target_quality_threshold: Optional[float] = None, # Optional: stop if this quality is reached
                 eqra_epsilon: float = 0.0001, # Stop if best EQRA is below this
                 max_iterations: Optional[int] = None, # Optional: max agent iterations
                 player_id: str = "optimizer_agent_default_player"): # Agent's own player ID for history
        
        self.project_path = project_path
        self.budget_left_cpu_minutes = initial_budget_cpu_minutes
        self.target_quality_metric = target_quality_metric
        self.target_quality_threshold = target_quality_threshold
        self.eqra_epsilon = eqra_epsilon
        self.max_iterations = max_iterations
        self.player_id = player_id # The agent itself can be a "player" for tracking its actions

        self.llm_wrapper = LLMWrapper() # Uses default model
        self.history_manager = HistoryManager() # Uses default DB path
        
        logger.info(f"OptimizerAgent initialized for project: {project_path}")
        logger.info(f"Initial Budget: {initial_budget_cpu_minutes} CPU minutes.")
        logger.info(f"EQRA Epsilon: {eqra_epsilon}")

    def _get_surviving_mutants_data(self) -> Optional[List[Dict]]:
        """
        Placeholder: Fetches surviving mutants data.
        In a real system, this would call a Guardian API or run a subprocess.
        """
        logger.info("Fetching surviving mutants data (simulated)...")
        # Simulate some data for auto_test
        return [
            {"id": "m1", "file_path": "src/calculator.py", "line_number": 10, "original_code_snippet": "return a + b", "mutated_code_snippet": "return a - b", "operator_type": "BINARY_OPERATOR_REPLACEMENT"},
            {"id": "m2", "file_path": "src/utils.py", "line_number": 25, "original_code_snippet": "if x > 0:", "mutated_code_snippet": "if x < 0:", "operator_type": "CONDITION_REPLACEMENT"}
        ]

    def _get_flaky_test_data(self) -> Optional[Dict]:
        """
        Placeholder: Fetches data for a known flaky test.
        """
        logger.info("Fetching flaky test data (simulated)...")
        return {
            "test_code_snippet": "def test_sometimes_fails():\n    import random\n    assert random.choice([True, False, False])",
            "stack_trace": "AssertionError: assert False",
            "file_path": "tests/test_flaky.py"
        }

    def _should_stop_iteration(self, iterations: int) -> bool:
        """Check if optimization loop should stop based on iteration limits."""
        if self.max_iterations is not None and iterations >= self.max_iterations:
            logger.info(f"Reached maximum iterations ({self.max_iterations}). Stopping.")
            return True
        if self.budget_left_cpu_minutes <= 0:
            logger.info("CPU budget exhausted. Stopping.")
            return True
        return False

    def _is_target_quality_reached(self, current_metrics: Dict[str, Any]) -> bool:
        """Check if target quality threshold has been reached."""
        if not self.target_quality_threshold:
            return False
        if self.target_quality_metric not in current_metrics:
            return False
        
        current_quality = current_metrics.get(self.target_quality_metric)
        if current_quality is not None and current_quality >= self.target_quality_threshold:
            logger.info(
                f"Target quality ({self.target_quality_metric} >= {self.target_quality_threshold}) reached. Stopping."
            )
            return True
        return False

    def _get_decision_action(self, current_metrics: Dict[str, Any], 
                            available_actions: Dict[str, Any],
                            action_posteriors: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get action decision from LLM decision agent."""
        decision_context = {
            "current_metrics": current_metrics,
            "available_actions": available_actions,
            "action_posteriors": action_posteriors,
            "budget_left_cpu_minutes": self.budget_left_cpu_minutes
        }
        
        logger.info("Calling Decision-Agent...")
        decision_response_msg = self.llm_wrapper.call_llm_with_function_calling(
            system_prompt=DECISION_AGENT_SYSTEM_PROMPT,
            user_context=decision_context,
            tools=DECISION_AGENT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "pick_action"}}
        )

        if not (decision_response_msg and decision_response_msg.get("tool_calls")):
            logger.error("Decision-Agent did not return a valid tool call.")
            return None
        
        tool_call_args_str = decision_response_msg["tool_calls"][0]["function"]["arguments"]
        try:
            decision_args = json.loads(tool_call_args_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Decision-Agent arguments: {tool_call_args_str}.")
            return None

        action_id = decision_args.get("action_id")
        if action_id == "NO_ACTION" or not action_id:
            logger.info("Decision-Agent chose NO_ACTION.")
            return None

        action_params = decision_args.get("params", {})
        rationale = decision_args.get("rationale", "No rationale provided.")
        logger.info(f"Decision-Agent chose action: '{action_id}' with params: {action_params}. Rationale: {rationale}")

        return {
            "action_id": action_id,
            "action_params": action_params,
            "rationale": rationale
        }

    def _validate_action(self, action_id: str, available_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate selected action and check EQRA threshold."""
        selected_action_profile = available_actions.get(action_id)
        if not selected_action_profile:
            logger.error(f"Chosen action_id '{action_id}' not found in available actions.")
            return None
        
        best_eqra = selected_action_profile.get("eqra_score", 0.0)
        if best_eqra < self.eqra_epsilon:
            logger.info(f"Best EQRA score ({best_eqra}) is below epsilon ({self.eqra_epsilon}).")
            return None

        return selected_action_profile

    def _prepare_implementation_context(self, action_id: str, action_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare context for implementation agent based on action type."""
        impl_system_prompt = IMPLEMENTATION_PROMPT_MAP.get(action_id)
        if not impl_system_prompt:
            logger.error(f"No implementation prompt found for action '{action_id}'.")
            return None

        impl_context = {}
        if action_id == "auto_test":
            impl_context["surviving_mutants_data"] = self._get_surviving_mutants_data()
            impl_context["focus_modules"] = action_params.get("focus_modules", [])
            impl_context["target_mutant_ids"] = action_params.get("target_mutant_ids")
        elif action_id == "flake_heal":
            impl_context = self._get_flaky_test_data()
            impl_context.update(action_params)

        if not impl_context:
            logger.warning(f"No context could be prepared for implementation agent for action '{action_id}'.")
            return None

        return {"context": impl_context, "system_prompt": impl_system_prompt}

    def _get_patch_proposal(self, action_id: str, impl_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get patch proposal from implementation agent."""
        logger.info(f"Calling Implementation-Agent for action '{action_id}'...")
        impl_response_msg = self.llm_wrapper.call_llm_with_function_calling(
            system_prompt=impl_context["system_prompt"],
            user_context=impl_context["context"],
            tools=IMPLEMENTATION_AGENT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "propose_patch"}}
        )

        if not (impl_response_msg and impl_response_msg.get("tool_calls")):
            logger.error(f"Implementation-Agent for '{action_id}' did not return a valid patch proposal.")
            return None
        
        patch_tool_call_args_str = impl_response_msg["tool_calls"][0]["function"]["arguments"]
        try:
            patch_args = json.loads(patch_tool_call_args_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Implementation-Agent arguments: {patch_tool_call_args_str}.")
            return None

        patch_diff = patch_args.get("diff")
        patch_comment = patch_args.get("comment", "No comment from LLM.")
        
        if not patch_diff:
            logger.warning(f"Implementation-Agent for '{action_id}' proposed an empty diff.")
            return None

        logger.info(f"Implementation-Agent for '{action_id}' proposed patch. Comment: {patch_comment}")
        return {"diff": patch_diff, "comment": patch_comment}

    def _apply_and_verify_patch(self, action_id: str, patch_diff: str, 
                                patch_comment: str) -> Dict[str, Any]:
        """Apply patch and verify results."""
        project_path_obj = Path(self.project_path)
        patch_applied_successfully = apply_patch(patch_diff, project_path_obj)

        verification_details = {"apply_status": patch_applied_successfully}
        actual_delta_q = 0.0
        actual_cost_cpu = 0.2  # Default cost for failed apply
        verification_passed = False

        if patch_applied_successfully:
            actual_delta_q, actual_cost_cpu, verification_passed = verify_patch(
                action_id=action_id,
                project_root_path=project_path_obj,
                applied_patch_details={"comment": patch_comment}
            )
            verification_details["verify_status"] = verification_passed
            verification_details["measured_delta_q"] = actual_delta_q
            verification_details["measured_cost_cpu"] = actual_cost_cpu
        else:
            logger.warning(f"Failed to apply patch for action '{action_id}'.")

        return {
            "delta_q": actual_delta_q,
            "cost_cpu": actual_cost_cpu,
            "verification_passed": verification_passed,
            "verification_details": verification_details
        }

    def _record_failed_action(self, action_id: str, action_params: Dict[str, Any], 
                              error_message: str, cost: float = 0.1):
        """Record a failed action attempt."""
        self.history_manager.record_action_outcome(
            player_id=self.player_id,
            action_id=action_id,
            params_used=action_params,
            delta_q_achieved=0.0,
            cost_cpu_minutes_incurred=cost,
            success=False,
            verification_details_json=json.dumps({"error": error_message})
        )
        self.budget_left_cpu_minutes -= cost

    def _award_xp_if_successful(self, action_id: str, selected_action_profile: Dict[str, Any]):
        """Award XP if action was successful and has positive credible gain."""
        credible_gain = selected_action_profile.get("estimated_delta_q_credible_lower_bound", 0.0)
        if credible_gain > 0:
            xp_from_action = int(1000 * credible_gain)
            logger.info(
                f"Action '{action_id}' successful with credible gain {credible_gain:.4f}. "
                f"Awarding {xp_from_action} XP."
            )
            self.history_manager.add_xp_for_agent_action(
                player_id=self.player_id,
                xp_to_add=xp_from_action,
                action_id_for_context=action_id
            )
        else:
            logger.info(
                f"Action '{action_id}' successful, but credible gain ({credible_gain:.4f}) "
                "not positive. No XP awarded from this rule."
            )

    def run_optimization_loop(self):
        """Main optimization loop with reduced complexity."""
        logger.info("Starting optimization loop...")
        iterations = 0

        while True:
            # Check iteration and budget limits
            if self._should_stop_iteration(iterations):
                break
            iterations += 1
            logger.info(f"\n--- Iteration {iterations} ---")

            # Get current state
            current_metrics = get_latest_metrics(username=self.player_id)
            available_actions = get_available_actions(username=self.player_id)
            action_posteriors = self.history_manager.get_action_posteriors(player_id=self.player_id)

            logger.debug(f"Current Metrics: {current_metrics}")
            logger.debug(f"Available Actions: {available_actions}")
            logger.debug(f"Action Posteriors: {action_posteriors}")

            # Check if target quality reached
            if self._is_target_quality_reached(current_metrics):
                break

            # Get decision from LLM
            decision = self._get_decision_action(current_metrics, available_actions, action_posteriors)
            if not decision:
                break

            action_id = decision["action_id"]
            action_params = decision["action_params"]

            # Validate action
            selected_action_profile = self._validate_action(action_id, available_actions)
            if not selected_action_profile:
                break

            # Prepare implementation context
            impl_prep = self._prepare_implementation_context(action_id, action_params)
            if not impl_prep:
                self._record_failed_action(action_id, action_params, "No implementation context available")
                continue

            # Get patch proposal
            patch_proposal = self._get_patch_proposal(action_id, impl_prep)
            if not patch_proposal:
                self._record_failed_action(action_id, action_params, "LLM failed to propose patch")
                continue

            # Apply and verify patch
            patch_result = self._apply_and_verify_patch(
                action_id, patch_proposal["diff"], patch_proposal["comment"]
            )

            # Update history and budget
            self.history_manager.record_action_outcome(
                player_id=self.player_id,
                action_id=action_id,
                params_used=action_params,
                delta_q_achieved=patch_result["delta_q"] if patch_result["verification_passed"] else 0.0,
                cost_cpu_minutes_incurred=patch_result["cost_cpu"],
                success=patch_result["verification_passed"],
                verification_details_json=json.dumps(patch_result["verification_details"])
            )
            self.budget_left_cpu_minutes -= patch_result["cost_cpu"]

            # Award XP if successful
            if patch_result["verification_passed"]:
                self._award_xp_if_successful(action_id, selected_action_profile)

            logger.info(f"Budget left: {self.budget_left_cpu_minutes:.2f} CPU minutes.")

        logger.info("Optimization loop finished.")
        self.history_manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Create a dummy project structure for the agent to "work on"
    # In a real scenario, this would be an actual project path.
    dummy_project_for_agent = Path("./temp_optimizer_project")
    dummy_project_for_agent.mkdir(parents=True, exist_ok=True)
    (dummy_project_for_agent / "src").mkdir(exist_ok=True)
    (dummy_project_for_agent / "tests").mkdir(exist_ok=True)
    (dummy_project_for_agent / "src" / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (dummy_project_for_agent / "src" / "utils.py").write_text("def helper():\n    pass\n")
    (dummy_project_for_agent / "tests" / "test_calculator.py").write_text("from src.calculator import add\ndef test_add():\n    assert add(1,1) == 2\n")

    agent = OptimizerAgent(
        project_path=str(dummy_project_for_agent.resolve()),
        initial_budget_cpu_minutes=10.0, # Small budget for testing
        eqra_epsilon=0.00001, # Low epsilon for testing
        max_iterations=5 
    )
    agent.run_optimization_loop()

    # Basic cleanup (optional)
    # import shutil
    # shutil.rmtree(dummy_project_for_agent)
    # print(f"Cleaned up dummy project: {dummy_project_for_agent}")