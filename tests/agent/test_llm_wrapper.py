import unittest
from unittest.mock import patch, MagicMock
import os
import json

from guardian.agent.llm_wrapper import LLMWrapper, DECISION_AGENT_TOOLS, IMPLEMENTATION_AGENT_TOOLS, DEFAULT_LLM_TIMEOUT_SECONDS
# Assuming OpenAI errors are imported in llm_wrapper or we mock them appropriately
# from openai import APIError, RateLimitError, AuthenticationError, APITimeoutError # Not needed directly in test if mocking client methods

class TestLLMWrapper(unittest.TestCase):

    def setUp(self):
        # Ensure a dummy API key is set for tests that might try to init client
        # but we will mostly mock the client itself.
        self.original_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test_api_key_for_unit_tests"

    def tearDown(self):
        if self.original_api_key is None:
            del os.environ["OPENAI_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = self.original_api_key

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_init_success_with_api_key_env(self, MockOpenAI):
        """Test successful initialization when API key is in env."""
        mock_client_instance = MockOpenAI.return_value
        wrapper = LLMWrapper(model_name="test-model")
        self.assertTrue(wrapper.is_functional)
        self.assertEqual(wrapper.model_name, "test-model")
        self.assertIsNotNone(wrapper.client)
        MockOpenAI.assert_called_once_with(api_key="test_api_key_for_unit_tests")

    @patch('guardian.agent.llm_wrapper.os.getenv')
    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_init_failure_no_api_key(self, MockOpenAI, mock_getenv):
        """Test initialization failure when no API key is found."""
        mock_getenv.return_value = None
        wrapper = LLMWrapper()
        self.assertFalse(wrapper.is_functional)
        self.assertIsNone(wrapper.client)
        MockOpenAI.assert_not_called()

    @patch('guardian.agent.llm_wrapper.OpenAI') # Mock OpenAI even for dry_run to ensure it's NOT called
    def test_init_dry_run_mode_no_api_key(self, MockOpenAI):
        """Test initialization in dry_run mode without an API key."""
        with patch('guardian.agent.llm_wrapper.os.getenv', return_value=None):
            wrapper = LLMWrapper(dry_run=True)
            self.assertTrue(wrapper.is_functional)
            self.assertTrue(wrapper.dry_run)
            self.assertIsNone(wrapper.client) # No client should be initialized
            MockOpenAI.assert_not_called() # OpenAI client init should be skipped

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_dry_run_simulates_pick_action(self, MockOpenAI):
        """Test dry_run mode simulates a pick_action call."""
        wrapper = LLMWrapper(dry_run=True)
        # Minimal context for pick_action simulation
        decision_context = {
            "available_actions": {"action1": {"eqra_score": 0.1, "estimated_cost_cpu_minutes": 1}},
            "budget_left_cpu_minutes": 5
        }
        response = wrapper.call_llm_with_function_calling(
            system_prompt="Decision prompt",
            user_context=decision_context,
            tools=DECISION_AGENT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "pick_action"}}
        )
        self.assertIsNotNone(response)
        self.assertIn("tool_calls", response)
        self.assertIsNotNone(response["tool_calls"])
        self.assertEqual(len(response["tool_calls"]), 1)
        tool_call = response["tool_calls"][0]["function"]
        self.assertEqual(tool_call["name"], "pick_action")
        # Arguments are now dicts directly from _simulate_llm_call before json.dumps
        # but the final response structure from call_llm_with_function_calling
        # (which calls _simulate_llm_call) should have them as JSON strings.
        # Let's re-check llm_wrapper.py _simulate_llm_call, it does json.dumps at the end.
        args = json.loads(tool_call["arguments"])
        self.assertIn("action_id", args)
        self.assertIn("rationale", args)
        MockOpenAI.assert_not_called() # Ensure no real API call

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_dry_run_simulates_propose_patch(self, MockOpenAI):
        """Test dry_run mode simulates a propose_patch call."""
        wrapper = LLMWrapper(dry_run=True)
        response = wrapper.call_llm_with_function_calling(
            system_prompt="Implementation prompt",
            user_context={"data": "some_mutant_info"},
            tools=IMPLEMENTATION_AGENT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "propose_patch"}}
        )
        self.assertIsNotNone(response)
        self.assertIn("tool_calls", response)
        self.assertIsNotNone(response["tool_calls"])
        tool_call = response["tool_calls"][0]["function"]
        self.assertEqual(tool_call["name"], "propose_patch")
        args = json.loads(tool_call["arguments"])
        self.assertIn("diff", args)
        self.assertIn("comment", args)
        self.assertIn("dummy_file.py", args["diff"]) # Check for part of the dummy diff
        MockOpenAI.assert_not_called()

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_dry_run_simulates_no_tool_call(self, MockOpenAI):
        """Test dry_run mode can simulate a response with no tool call if tool_choice is 'none'."""
        wrapper = LLMWrapper(dry_run=True)
        response = wrapper.call_llm_with_function_calling(
            system_prompt="Generic prompt",
            user_context="Some query",
            tools=DECISION_AGENT_TOOLS, # Tools available
            tool_choice="none" # But explicitly told not to use them
        )
        self.assertIsNotNone(response)
        self.assertIsNone(response.get("tool_calls"))
        self.assertIn("This is a simulated response", response.get("content", ""))
        MockOpenAI.assert_not_called()

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_dry_run_pick_action_with_context_logic(self, MockOpenAI):
        """Test the dry_run pick_action logic based on context."""
        wrapper = LLMWrapper(dry_run=True)
        
        # Case 1: Action available and fits budget
        context1 = {
            "available_actions": {
                "test_action_1": {"eqra_score": 0.5, "estimated_cost_cpu_minutes": 2},
                "test_action_2": {"eqra_score": 0.1, "estimated_cost_cpu_minutes": 1}
            },
            "budget_left_cpu_minutes": 5
        }
        response1 = wrapper.call_llm_with_function_calling(system_prompt="Test", user_context=context1, tools=DECISION_AGENT_TOOLS, tool_choice="auto")
        args1 = json.loads(response1["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args1["action_id"], "test_action_1")

        # Case 2: Action available but too expensive
        context2 = {
            "available_actions": {"test_action_3": {"eqra_score": 0.5, "estimated_cost_cpu_minutes": 10}},
            "budget_left_cpu_minutes": 5
        }
        response2 = wrapper.call_llm_with_function_calling(system_prompt="Test", user_context=context2, tools=DECISION_AGENT_TOOLS, tool_choice="auto")
        args2 = json.loads(response2["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args2["action_id"], "NO_ACTION")
        self.assertIn("over budget", args2["rationale"])


        # Case 3: Action available but negative EQRA
        context3 = {
            "available_actions": {"test_action_4": {"eqra_score": -0.1, "estimated_cost_cpu_minutes": 2}},
            "budget_left_cpu_minutes": 5
        }
        response3 = wrapper.call_llm_with_function_calling(system_prompt="Test", user_context=context3, tools=DECISION_AGENT_TOOLS, tool_choice="auto")
        args3 = json.loads(response3["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args3["action_id"], "NO_ACTION")
        self.assertIn("EQRA <=0", args3["rationale"])

        # Case 4: No actions available
        context4 = {"available_actions": {}, "budget_left_cpu_minutes": 5}
        response4 = wrapper.call_llm_with_function_calling(system_prompt="Test", user_context=context4, tools=DECISION_AGENT_TOOLS, tool_choice="auto")
        args4 = json.loads(response4["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args4["action_id"], "NO_ACTION")
        
        MockOpenAI.assert_not_called()


    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_call_llm_successful_tool_call(self, MockOpenAI):
        """Test a successful LLM call that results in a tool_call."""
        mock_openai_client = MockOpenAI.return_value
        mock_completion_message = MagicMock()
        mock_tool_call_function = MagicMock()
        mock_tool_call_function.name = "pick_action"
        mock_tool_call_function.arguments = json.dumps({"action_id": "auto_test", "params": {}})
        
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_tool_call_function
        
        mock_completion_message.tool_calls = [mock_tool_call]
        mock_completion_message.content = None # Explicitly None if only tool_calls

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_completion_message)]
        mock_openai_client.chat.completions.create.return_value = mock_response

        wrapper = LLMWrapper()
        self.assertTrue(wrapper.is_functional) # Assuming API key is set by setUp

        response = wrapper.call_llm_with_function_calling(
            system_prompt="Test system prompt",
            user_context={"data": "test_data"},
            tools=DECISION_AGENT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "pick_action"}}
        )

        self.assertIsNotNone(response)
        self.assertIn("tool_calls", response)
        self.assertEqual(len(response["tool_calls"]), 1)
        tool_call_resp = response["tool_calls"][0]
        self.assertEqual(tool_call_resp["function"]["name"], "pick_action")
        self.assertEqual(tool_call_resp["function"]["arguments"]["action_id"], "auto_test")
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs['model'], wrapper.model_name)
        self.assertEqual(call_args.kwargs['timeout'], DEFAULT_LLM_TIMEOUT_SECONDS)


    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_call_llm_no_tool_call_text_response(self, MockOpenAI):
        """Test LLM call resulting in a text response, not a tool call."""
        mock_openai_client = MockOpenAI.return_value
        mock_completion_message = MagicMock()
        mock_completion_message.tool_calls = None
        mock_completion_message.content = "This is a text response."
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_completion_message)]
        mock_openai_client.chat.completions.create.return_value = mock_response

        wrapper = LLMWrapper()
        response = wrapper.call_llm_with_function_calling(
            system_prompt="Test system prompt",
            user_context="Just a question",
            tools=DECISION_AGENT_TOOLS, # Tools provided
            tool_choice="auto" # Model might choose not to use them
        )
        self.assertIsNotNone(response)
        self.assertIsNone(response.get("tool_calls"))
        self.assertEqual(response.get("content"), "This is a text response.")

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_call_llm_api_error_handling(self, MockOpenAI):
        """Test handling of a generic APIError."""
        mock_openai_client = MockOpenAI.return_value
        # Dynamically get APIError from the openai module if available, or mock it
        try:
            from openai import APIError
        except ImportError:
            APIError = ConnectionError # Fallback if openai not fully installed for test env

        mock_openai_client.chat.completions.create.side_effect = APIError("Simulated API Error", request=None, body=None)

        wrapper = LLMWrapper()
        response = wrapper.call_llm_with_function_calling(
            system_prompt="Test", user_context="Test", tools=[], tool_choice="none"
        )
        self.assertIsNone(response)

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_call_llm_timeout_error_handling(self, MockOpenAI):
        """Test handling of APITimeoutError."""
        mock_openai_client = MockOpenAI.return_value
        try:
            from openai import APITimeoutError
        except ImportError:
            APITimeoutError = TimeoutError # Fallback

        mock_openai_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        
        wrapper = LLMWrapper()
        response = wrapper.call_llm_with_function_calling(
            system_prompt="Test", user_context="Test", tools=[], tool_choice="none"
        )
        self.assertIsNone(response)

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_llm_wrapper_not_functional(self, MockOpenAI):
        """Test that calls are not made if wrapper is not functional."""
        # Simulate API key not being found during __init__
        with patch('guardian.agent.llm_wrapper.os.getenv', return_value=None):
            wrapper = LLMWrapper() # This will set is_functional to False
        
        self.assertFalse(wrapper.is_functional)
        response = wrapper.call_llm_with_function_calling(
            system_prompt="Test", user_context="Test", tools=[], tool_choice="none"
        )
        self.assertIsNone(response)
        MockOpenAI.assert_not_called() # Client shouldn't even be attempted to be created

    @patch('guardian.agent.llm_wrapper.OpenAI')
    def test_tool_argument_parsing_failure(self, MockOpenAI):
        """Test handling of malformed JSON in tool call arguments."""
        mock_openai_client = MockOpenAI.return_value
        mock_completion_message = MagicMock()
        mock_tool_call_function = MagicMock()
        mock_tool_call_function.name = "pick_action"
        mock_tool_call_function.arguments = "this is not valid json" # Malformed
        
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_malformed"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_tool_call_function
        
        mock_completion_message.tool_calls = [mock_tool_call]
        mock_completion_message.content = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_completion_message)]
        mock_openai_client.chat.completions.create.return_value = mock_response

        wrapper = LLMWrapper()
        response = wrapper.call_llm_with_function_calling(
            system_prompt="Test", user_context="Test", tools=DECISION_AGENT_TOOLS, tool_choice="auto"
        )
        self.assertIsNotNone(response)
        self.assertIn("tool_calls", response)
        self.assertEqual(len(response["tool_calls"]), 1)
        tool_call_resp = response["tool_calls"][0]
        self.assertEqual(tool_call_resp["function"]["name"], "pick_action")
        self.assertEqual(tool_call_resp["function"]["arguments"], "this is not valid json")
        self.assertIn("error", tool_call_resp["function"])
        self.assertEqual(tool_call_resp["function"]["error"], "Failed to parse arguments as JSON")


if __name__ == '__main__':
    unittest.main()