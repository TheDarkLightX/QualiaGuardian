"""
API for retrieving consolidated metrics from Guardian.
"""
import logging
from typing import Optional, Dict, Any
import json

from guardian.history import HistoryManager

logger = logging.getLogger(__name__)

def get_latest_metrics(username: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves the latest available quality metrics for a given user.

    This function fetches the most recent run from the history and extracts
    bE-TES score, OSQI score, and potentially other relevant metrics.

    Args:
        username: The username of the player. If None, uses the default
                  player resolution logic in HistoryManager.

    Returns:
        A dictionary containing the latest metrics.
        Example:
        {
            "player_id": "some_player_id",
            "latest_run_timestamp": 1678886400,
            "betes_score": 0.75,
            "osqi_score": 0.80,
            "raw_metrics": {
                "mutation_score": 0.88,
                # Potentially other components if stored/retrievable
            },
            "actions_taken_in_last_run": ["cmd_quality_analysis", "mode_betes_v3.1"]
        }
        Returns a dictionary with "error" key if no runs are found or an issue occurs.
    """
    history_manager = HistoryManager()
    player_id = history_manager._get_player_id(username)

    if player_id == "fallback_error_player_id":
        history_manager.close()
        return {"error": "Could not retrieve or create player."}

    latest_run_query = """
    SELECT timestamp, betes_score, osqi_score, actions_taken
    FROM runs
    WHERE player_id = ?
    ORDER BY timestamp DESC
    LIMIT 1
    """
    latest_run_data = history_manager._execute_query(latest_run_query, (player_id,), fetch_one=True)
    history_manager.close()

    if not latest_run_data:
        return {
            "player_id": player_id,
            "error": "No runs found for this player.",
            "latest_run_timestamp": None,
            "betes_score": None,
            "osqi_score": None,
            "raw_metrics": {},
            "actions_taken_in_last_run": []
        }

    actions_taken_list = []
    if latest_run_data["actions_taken"]:
        try:
            actions_taken_list = json.loads(latest_run_data["actions_taken"])
        except json.JSONDecodeError:
            logger.warning(f"Could not parse actions_taken JSON for player {player_id}, run timestamp {latest_run_data['timestamp']}")
            actions_taken_list = [latest_run_data["actions_taken"]] # Store as raw string if not parsable

    # For now, raw_metrics will be a placeholder.
    # A more advanced version might try to deserialize components if they are stored as JSON
    # in the 'runs' table or fetch them from related tables if the schema evolves.
    # The current `runs` table only has betes_score and osqi_score directly.
    # The `current_metrics` passed to `record_run` in `cli.py` includes `mutation_score`.
    # This is not directly stored in a separate column in `runs` table.
    # For the agent, we might need to enhance `record_run` to store more detailed `current_metrics` as JSON.
    # For now, we'll focus on what's directly available.
    
    # Placeholder for extracting detailed raw metrics if they were stored as JSON in a 'current_metrics_json' column
    # For now, we'll simulate that `betes_score` implies some underlying raw metrics.
    # The `quality_analysis.components` in `cli.py`'s `results` dict has detailed components.
    # This would ideally be stored in the `runs` table.
    
    # Let's assume for the agent, the primary scores are most important initially.
    # We can enhance this later to store and retrieve full component breakdowns.

    return {
        "player_id": player_id,
        "latest_run_timestamp": latest_run_data["timestamp"],
        "betes_score": latest_run_data["betes_score"],
        "osqi_score": latest_run_data["osqi_score"],
        "raw_metrics": {
            # This section would be populated if detailed components were stored.
            # Example: "mutation_score": latest_run_data.get("mutation_score_from_json_column")
        },
        "actions_taken_in_last_run": actions_taken_list
    }

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    
    # Test with a user who has runs (e.g., from previous tests)
    # Ensure you have run some commands for 'quest_tester_1' or 'default_guardian_user'
    # For example, run: python guardian_ai_tool/guardian/cli.py gamify status --user test_metrics_user
    # then: python guardian_ai_tool/guardian/cli.py temp_guardian_project_for_api_test --run-quality --quality-mode betes_v3.1 --user test_metrics_user

    metrics_user = "quest_tester_1" # A user likely to have data
    print(f"Fetching latest metrics for user: {metrics_user}")
    latest_metrics = get_latest_metrics(username=metrics_user)
    print(json.dumps(latest_metrics, indent=2))

    print(f"\nFetching latest metrics for default user:")
    latest_metrics_default = get_latest_metrics()
    print(json.dumps(latest_metrics_default, indent=2))

    print(f"\nFetching latest metrics for a new user (should show no runs):")
    latest_metrics_new = get_latest_metrics(username="new_metrics_user_test")
    print(json.dumps(latest_metrics_new, indent=2))


def get_available_actions(username: Optional[str] = None) -> Dict[str, Any]:
    """
    Lists available improvement actions the agent can take.
    EQRA scores, estimated costs, and benefits are now derived from posteriors.

    Args:
        username: Optional username. The agent's `player_id` will be used to fetch posteriors.

    Returns:
        A dictionary where keys are action_ids and values are dicts
        containing action metadata including calculated EQRA.
    """
    history_manager = HistoryManager()
    # The agent uses its own player_id for its operational history / posteriors.
    # If no specific agent player_id is set, HistoryManager's default will be used.
    agent_player_id = username # Or a dedicated agent ID if `username` is for project context
    
    posteriors_data = history_manager.get_action_posteriors(player_id=agent_player_id)
    history_manager.close() # Close connection after fetching posteriors

    action_posteriors = posteriors_data.get("actions", {})
    
    # Define available actions and their basic info.
    # Manifest path and description would ideally come from a registry or action.yaml files.
    defined_actions_info = {
        "auto_test": {
            "description": "Automatically generate new tests to kill surviving mutants.",
            "manifest_path": "guardian/actions/auto_test/action.yaml"
        },
        "flake_heal": {
            "description": "Attempt to automatically fix flaky tests.",
            "manifest_path": "guardian/actions/flake_heal/action.yaml"
        }
        # Add more actions here as they are defined
    }

    output_actions = {}
    for action_id, base_info in defined_actions_info.items():
        posterior = action_posteriors.get(action_id)
        
        mean_delta_q = 0.0
        mean_cost_cpu = 1.0 # Default to 1 to avoid division by zero if no posterior
        eqra_score = 0.0

        if posterior:
            mean_delta_q = posterior.get("delta_q", {}).get("mean", 0.0)
            mean_cost_cpu = posterior.get("cost_cpu_minutes", {}).get("mean", 1.0)
            if mean_cost_cpu <= 0: # Avoid division by zero or negative cost issues
                mean_cost_cpu = 1.0 # Or handle as an error/very low EQRA
        
        if mean_cost_cpu > 0:
            eqra_score = mean_delta_q / mean_cost_cpu
        else:
            eqra_score = -float('inf') # Or some other indicator of problematic cost

        output_actions[action_id] = {
            "description": base_info["description"],
            "eqra_score": eqra_score,
            "estimated_delta_q": mean_delta_q,
            "estimated_cost_cpu_minutes": mean_cost_cpu,
            "manifest_path": base_info["manifest_path"]
            # Could also include variance here if Decision-Agent needs it
        }
        
    # TODO: Discover actions from a registry or manifests instead of hardcoding `defined_actions_info`.
    return output_actions


if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    
    # Test with a user who has runs (e.g., from previous tests)
    # Ensure you have run some commands for 'quest_tester_1' or 'default_guardian_user'
    # For example, run: python guardian_ai_tool/guardian/cli.py gamify status --user test_metrics_user
    # then: python guardian_ai_tool/guardian/cli.py temp_guardian_project_for_api_test --run-quality --quality-mode betes_v3.1 --user test_metrics_user

    metrics_user = "quest_tester_1" # A user likely to have data
    print(f"Fetching latest metrics for user: {metrics_user}")
    latest_metrics = get_latest_metrics(username=metrics_user)
    print(json.dumps(latest_metrics, indent=2))

    print(f"\nFetching latest metrics for default user:")
    latest_metrics_default = get_latest_metrics()
    print(json.dumps(latest_metrics_default, indent=2))

    print(f"\nFetching latest metrics for a new user (should show no runs):")
    latest_metrics_new = get_latest_metrics(username="new_metrics_user_test")
    print(json.dumps(latest_metrics_new, indent=2))

    print(f"\nFetching available actions:")
    available_actions = get_available_actions()
    print(json.dumps(available_actions, indent=2))