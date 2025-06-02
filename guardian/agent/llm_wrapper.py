"""
LLM Wrapper for the Autonomous Quality-Optimizer Agent.
Handles interactions with the language model, including system prompts,
user messages, and function/tool calling.
"""
import json
import logging
import os # For API Key
from typing import Dict, Any, Optional, List

# Actual OpenAI client
from openai import OpenAI, APIError, RateLimitError, AuthenticationError, APITimeoutError

logger = logging.getLogger(__name__)

# Default timeout for LLM API calls
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0

class LLMWrapper:
    """
    A wrapper class to manage interactions with an LLM.
    This is a placeholder and would need to be implemented with a specific
    LLM provider's API (e.g., OpenAI, Anthropic, Gemini).
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview", api_key: Optional[str] = None, timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS, dry_run: bool = False):
        self.model_name = model_name
        self.timeout = timeout
        self.dry_run = dry_run
        self.client = None
        self.is_functional = False

        if self.dry_run:
            logger.info(f"LLMWrapper initialized in DRY RUN mode with model: {self.model_name}. API calls will be simulated.")
            self.is_functional = True # Functional in the sense that it can provide simulated responses
            # No actual client needed for dry run, but we won't try to init one if no key.
            # If a key IS provided, we could still init the client for consistency, but it won't be used in dry_run.
            # For simplicity, let's skip client init if dry_run is True and no key is explicitly forced.
            return

        effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not effective_api_key:
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or pass it to LLMWrapper. Real API calls will fail.")
            # self.client remains None, self.is_functional remains False
        else:
            try:
                self.client = OpenAI(api_key=effective_api_key)
                self.is_functional = True
                logger.info(f"LLMWrapper initialized with model: {self.model_name} for REAL API calls.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                # self.client remains None, self.is_functional remains False

    def _construct_messages(self, system_prompt: str, user_context: Any) -> List[Dict[str, str]]:
        """Helper to construct the messages list for the LLM API."""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        if isinstance(user_context, str):
            messages.append({"role": "user", "content": user_context})
        elif isinstance(user_context, dict) or isinstance(user_context, list):
            # Serialize complex context (like JSON metrics) into a string for the user message
            try:
                messages.append({"role": "user", "content": json.dumps(user_context, indent=2)})
            except TypeError:
                logger.error("User context for LLM is not JSON serializable, converting to string.")
                messages.append({"role": "user", "content": str(user_context)})
        else:
             messages.append({"role": "user", "content": str(user_context)})
        return messages

    def call_llm_with_function_calling(
        self,
        system_prompt: str,
        user_context: Any,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = "auto" # Can be "auto", "none", or {"type": "function", "function": {"name": "my_function"}}
    ) -> Optional[Dict[str, Any]]:
        """
        Simulates a call to an LLM that supports function/tool calling.

        Args:
            system_prompt: The system prompt defining the LLM's role and instructions.
            user_context: The user message or context (e.g., metrics, code snippets).
            tools: A list of tool definitions for the LLM to use.
                   Example: [{"type": "function", "function": {"name": "pick_action", "parameters": {...}}}]
            tool_choice: How the model should use tools.

        Returns:
            A dictionary representing the LLM's response, typically indicating
            a tool call and its arguments, or None if an error occurs.
            Example of a tool call response part:
            {
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "pick_action", "arguments": '{"action_id": "auto_test", "params": {}}'}
                }]
            }
        """
        if self.dry_run:
            logger.info(f"DRY RUN: Simulating LLM call for model '{self.model_name}' with tool choice '{tool_choice}'.")
            return self._simulate_llm_call(system_prompt, user_context, tools, tool_choice)

        if not self.is_functional or not self.client:
            logger.error("LLMWrapper is not functional (OpenAI client not initialized or API key missing). Cannot make real API call.")
            return None

        messages = self._construct_messages(system_prompt, user_context)
        logger.info(f"Calling REAL LLM '{self.model_name}' with tool choice '{tool_choice}'. Timeout: {self.timeout}s.")
        logger.debug(f"System Prompt: {system_prompt}")
        # logger.debug(f"User Context: {json.dumps(user_context, indent=2) if isinstance(user_context, (dict, list)) else str(user_context)}")
        logger.debug(f"Tools: {json.dumps(tools)}")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools if tools else None, # Pass None if tools list is empty
                tool_choice=tool_choice if tools else None, # Pass None if no tools
                timeout=self.timeout
            )
            
            response_message = response.choices[0].message

            if response_message.tool_calls:
                parsed_tool_calls = []
                for tool_call in response_message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        parsed_tool_calls.append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": arguments # Now a dict
                            }
                        })
                        logger.info(f"LLM called tool: {tool_call.function.name} with args: {arguments}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON arguments for tool {tool_call.function.name}: {tool_call.function.arguments}. Error: {e}")
                        # Optionally, could return the raw string or skip this tool call
                        parsed_tool_calls.append({ # Store with raw arguments if parsing fails
                             "id": tool_call.id,
                             "type": "function",
                             "function": {
                                 "name": tool_call.function.name,
                                 "arguments": tool_call.function.arguments,
                                 "error": "Failed to parse arguments as JSON"
                             }
                        })

                return {"tool_calls": parsed_tool_calls, "content": response_message.content}
            else:
                logger.info(f"LLM response did not include tool calls. Content: {response_message.content}")
                return {"content": response_message.content, "tool_calls": None}

        except APITimeoutError:
            logger.error(f"OpenAI API request timed out after {self.timeout} seconds.")
            return None
        except AuthenticationError as e:
            logger.error(f"OpenAI API authentication error: {e}. Check your API key and organization.")
            return None
        except RateLimitError as e:
            logger.error(f"OpenAI API rate limit exceeded: {e}")
            # TODO: Implement retry logic here if desired
            return None
        except APIError as e:
            logger.error(f"OpenAI API returned an API Error: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM call: {e}", exc_info=True)
            return None

    def _simulate_llm_call(
        self,
        system_prompt: str,
        user_context: Any,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] # Can be "auto", "none", or {"type": "function", "function": {"name": "my_function"}}
    ) -> Optional[Dict[str, Any]]:
        """Simulates an LLM call for dry_run mode."""
        logger.debug(f"DRY RUN - System Prompt: {system_prompt[:200]}...") # Log snippet
        logger.debug(f"DRY RUN - User Context: {str(user_context)[:200]}...") # Log snippet
        logger.debug(f"DRY RUN - Tools: {json.dumps(tools)}")

        # Default response: no tool call, just some content
        simulated_response_content = "This is a simulated response from the LLM in dry run mode."
        simulated_tool_calls = None

        # Try to be a bit more intelligent for specific tools if requested
        forced_tool_name = None
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            forced_tool_name = tool_choice.get("function", {}).get("name")
        
        if forced_tool_name == "pick_action" or (tool_choice == "auto" and any(t["function"]["name"] == "pick_action" for t in tools)):
            action_id_to_pick = "NO_ACTION"
            params = {}
            rationale = "Simulated: No suitable action found or default NO_ACTION."

            if isinstance(user_context, dict) and "available_actions" in user_context:
                available_actions = user_context["available_actions"]
                if isinstance(available_actions, dict) and available_actions:
                    # Pick the first available action for simplicity in dry run
                    first_action_id = next(iter(available_actions))
                    action_details = available_actions[first_action_id]
                    
                    # Basic budget check
                    budget = user_context.get("budget_left_cpu_minutes", float('inf'))
                    cost = action_details.get("estimated_cost_cpu_minutes", 0)

                    if action_details.get("eqra_score", -1) > 0 and cost <= budget :
                        action_id_to_pick = first_action_id
                        # Example: if auto_test, add dummy params
                        if action_id_to_pick == "auto_test":
                             params = {"focus_modules": ["simulated_module.py"], "target_mutant_ids": ["sim_mut_1"]}
                        rationale = f"Simulated: Picked first available action '{action_id_to_pick}' as it has positive EQRA and fits budget."
                    else:
                        rationale = f"Simulated: First action '{first_action_id}' not suitable (EQRA <=0 or over budget). Defaulting to NO_ACTION."


            simulated_tool_calls = [{
                "id": "call_sim_decision",
                "type": "function",
                "function": {
                    "name": "pick_action",
                    "arguments": {"action_id": action_id_to_pick, "params": params, "rationale": rationale}
                }
            }]
            simulated_response_content = None # Tool call implies no direct content usually
            logger.info(f"DRY RUN: Simulated 'pick_action' call. Action: {action_id_to_pick}, Rationale: {rationale}")

        elif forced_tool_name == "propose_patch" or (tool_choice == "auto" and any(t["function"]["name"] == "propose_patch" for t in tools)):
            # Simulate a patch proposal
            dummy_diff = """--- a/dummy_file.py
+++ b/dummy_file.py
@@ -1,1 +1,2 @@
 def hello():
+    print("Hello from simulated patch!")
     return "hello"
"""
            simulated_tool_calls = [{
                "id": "call_sim_impl",
                "type": "function",
                "function": {
                    "name": "propose_patch",
                    "arguments": {"diff": dummy_diff, "comment": "Simulated patch from dry run."}
                }
            }]
            simulated_response_content = None
            logger.info("DRY RUN: Simulated 'propose_patch' call with a dummy diff.")
        
        # Ensure arguments are JSON strings if they were dicts, like the real API
        if simulated_tool_calls:
            for tc in simulated_tool_calls:
                if isinstance(tc["function"]["arguments"], dict):
                    tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])

        return {"tool_calls": simulated_tool_calls, "content": simulated_response_content}

# --- Agent Persona Definitions (Prompts & Tool Schemas) ---

DECISION_AGENT_SYSTEM_PROMPT = """
You are a highly efficient Quality-Optimizer agent.
Your goal is to improve code quality by selecting the best next action based on provided metrics and action profiles.
Input will be a JSON object containing:
1.  `current_metrics`: The latest quality scores (e.g., bE-TES, OSQI, pillar scores).
2.  `available_actions`: A dictionary of actions, each with:
    *   `description`: What the action does.
    *   `eqra_score`: Host-calculated EQRA score (Expected Quality Return per unit of Action cost).
    *   `estimated_cost_cpu_minutes`: Estimated CPU cost.
    *   `estimated_delta_q`: Expected quality improvement.
    *   `manifest_path`: Path to its configuration manifest.
3.  `action_posteriors`: Historical performance (mean, variance for delta_q and cost) for actions.
4.  `budget_left_cpu_minutes`: Remaining CPU budget.

Your task is to select the action with the highest positive `eqra_score`.
Constraints:
- Do NOT select an action if its `estimated_delta_q` is less than 0.002, unless no other actions have a positive EQRA.
- Ensure the `estimated_cost_cpu_minutes` of the selected action is less than or equal to `budget_left_cpu_minutes`.
- If multiple actions have similar high EQRA scores, you may prioritize based on other factors like lower cost or higher `estimated_delta_q`.
- You may suggest combining two cheap actions if their combined cost is within budget (this is an advanced consideration, for now focus on single best action).

Output your decision by calling the `pick_action` function with the chosen `action_id`, any necessary `params` for that action (consult action description/manifest if needed, for now, params can be simple e.g. for `auto_test` it might be `{"focus_modules": ["module_to_improve"]}` or empty if not specified), and a brief `rationale` for your choice.
If no suitable action can be taken (e.g., all EQRA scores are too low, or budget constraints cannot be met), call `pick_action` with `action_id` as "NO_ACTION".
"""

DECISION_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pick_action",
            "description": "Selects the next quality improvement action to take.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "The ID of the chosen action (e.g., 'auto_test', 'flake_heal', or 'NO_ACTION')."
                    },
                    "params": {
                        "type": "object",
                        "description": "A dictionary of parameters required for the chosen action. Can be empty.",
                        "additionalProperties": True # Allows flexible parameters
                    },
                    "rationale": {
                        "type": "string",
                        "description": "A brief explanation for why this action was chosen."
                    }
                },
                "required": ["action_id", "rationale"]
            }
        }
    }
]

# --- Implementation Agent Prompts & Schemas (Examples) ---

# impl_auto_test
AUTO_TEST_IMPL_AGENT_SYSTEM_PROMPT = """
You are `impl_auto_test`, a specialized agent for generating PyTest test cases.
Your goal is to write a new test that specifically kills one or more of the provided surviving mutants.
Input will be:
1.  `surviving_mutants_data`: A list of dictionaries, each representing a surviving mutant with details like `file_path`, `line_number`, `original_code_snippet`, `mutated_code_snippet`, `operator_type`.
2.  `focus_modules`: A list of module or file paths that are the current focus for improvement. Prioritize mutants in these modules.
3.  `target_mutant_ids` (optional): Specific IDs of mutants to target if provided.

Instructions:
- Analyze the provided surviving mutants, especially those in `focus_modules` or matching `target_mutant_ids`.
- Select one or a small group of related mutants.
- Write a concise and effective PyTest test function that asserts the correct behavior, thereby killing the selected mutant(s).
- Use descriptive test function names (e.g., `test_calculator_handles_division_by_zero_for_add_mutant`).
- Use `pytest` fixtures if appropriate.
- Employ property-based testing with `hypothesis` if it makes the test more robust and general for the type of bug the mutant represents.
- Ensure the test is self-contained or uses standard, readily available fixtures.
- The output must be a patch in the unified diff format.
- Only modify or create files within a standard test directory structure (e.g., `tests/`).
- Do not include setup for the entire test file if only adding one function; assume standard imports like `pytest` are present.
"""

# impl_flake_heal
FLAKE_HEAL_IMPL_AGENT_SYSTEM_PROMPT = """
You are `impl_flake_heal`, a specialized agent for fixing flaky tests.
Input will be:
1.  `test_code_snippet`: The source code of the flaky test function.
2.  `stack_trace`: The stack trace or error message associated with the flakiness.
3.  `file_path`: The path to the test file.

Instructions:
- Analyze the test code and stack trace to understand the cause of flakiness (e.g., race conditions, timing issues, reliance on unstable external state).
- Modify the test code to make it more robust. This might involve:
    - Adding explicit waits or polling loops with timeouts (e.g., using `time.sleep()` judiciously or a library like `tenacity`).
    - Using mocking to control dependencies.
    - Refactoring the test to be more deterministic.
    - As a last resort, if the flakiness is hard to eliminate, add a `@pytest.mark.flaky(reruns=N, reruns_delay=M)` decorator (or equivalent for other test frameworks).
- The output must be a patch in the unified diff format, modifying only the provided test code snippet within its original file.
"""

IMPLEMENTATION_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_patch",
            "description": "Proposes a code modification as a diff patch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diff": {
                        "type": "string",
                        "description": "The proposed code changes in unified diff format."
                    },
                    "comment": {
                        "type": "string",
                        "description": "A brief comment explaining the patch and its intent."
                    }
                },
                "required": ["diff", "comment"]
            }
        }
    }
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Initialize in dry_run mode for testing without API key
    wrapper = LLMWrapper(dry_run=True)
    # To test with real API (ensure OPENAI_API_KEY is set):
    # wrapper = LLMWrapper()


    # Simulate Decision Agent call
    print("\n--- Simulating Decision Agent ---")
    decision_context = {
        "current_metrics": {"betes_score": 0.65, "osqi_score": 0.7},
        "available_actions": {
            "auto_test": {"eqra_score": 0.0008, "estimated_cost_cpu_minutes": 5, "estimated_delta_q": 0.02},
            "flake_heal": {"eqra_score": 0.0005, "estimated_cost_cpu_minutes": 1, "estimated_delta_q": 0.005}
        },
        "action_posteriors": { # Simplified for this example
            "auto_test": {"delta_q": {"mean": 0.02}, "cost_cpu_minutes": {"mean": 5.0}},
            "flake_heal": {"delta_q": {"mean": 0.005}, "cost_cpu_minutes": {"mean": 1.0}}
        },
        "budget_left_cpu_minutes": 10.0
    }
    decision_response = wrapper.call_llm_with_function_calling(
        system_prompt=DECISION_AGENT_SYSTEM_PROMPT,
        user_context=decision_context,
        tools=DECISION_AGENT_TOOLS,
        tool_choice={"type": "function", "function": {"name": "pick_action"}} # Force tool for simulation
    )
    if decision_response and decision_response.get("tool_calls"):
        tool_call = decision_response["tool_calls"][0]["function"]
        print(f"Decision Agent chose: {tool_call['name']}")
        print(f"Arguments: {json.loads(tool_call['arguments'])}")

    # Simulate Implementation Agent (auto_test) call
    print("\n--- Simulating Auto-Test Implementation Agent ---")
    auto_test_context = {
        "surviving_mutants_data": [
            {"id": "m1", "file_path": "src/calculator.py", "line_number": 10, "original_code_snippet": "return a + b", "mutated_code_snippet": "return a - b", "operator_type": "BINARY_OPERATOR_REPLACEMENT"}
        ],
        "focus_modules": ["src/calculator.py"]
    }
    impl_response = wrapper.call_llm_with_function_calling(
        system_prompt=AUTO_TEST_IMPL_AGENT_SYSTEM_PROMPT,
        user_context=auto_test_context,
        tools=IMPLEMENTATION_AGENT_TOOLS,
        tool_choice={"type": "function", "function": {"name": "propose_patch"}} # Force tool
    )
    if impl_response and impl_response.get("tool_calls"):
        tool_call = impl_response["tool_calls"][0]["function"]
        print(f"Implementation Agent proposed patch via: {tool_call['name']}")
        # Args are JSON string
        # print(f"Arguments: {json.loads(tool_call['arguments'])}")
        # For brevity, just print the raw string for diff
        print(f"Arguments (raw): {tool_call['arguments']}")