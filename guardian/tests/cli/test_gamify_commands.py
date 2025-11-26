import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from typer.testing import CliRunner

# Import the cli module and access app from it
from guardian_ai_tool.guardian import cli as guardian_cli_module
app = guardian_cli_module.app # The Typer app instance

class TestGamifyCrownCommand(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()
        # Create a dummy project structure for these tests to run against
        self.test_project_dir = Path("temp_test_project_for_cli_gamify")
        self.test_project_dir.mkdir(exist_ok=True)
        
        self.tests_sub_dir = self.test_project_dir / "tests"
        self.tests_sub_dir.mkdir(exist_ok=True)

        self.test_file_a_path = self.tests_sub_dir / "test_a.py"
        self.test_file_b_path = self.tests_sub_dir / "test_b.py"
        self.test_file_c_path = self.tests_sub_dir / "test_c.py"

        with open(self.test_file_a_path, "w") as f:
            f.write("def test_alpha(): assert True")
        with open(self.test_file_b_path, "w") as f:
            f.write("def test_beta(): assert True")
        with open(self.test_file_c_path, "w") as f:
            f.write("def test_gamma(): assert True")

    def tearDown(self):
        # Clean up dummy project
        import shutil
        if self.test_project_dir.exists():
            shutil.rmtree(self.test_project_dir)

    @patch('guardian_ai_tool.guardian.core.api.evaluate_subset')
    def test_gamify_crown_simple_run(self, mock_evaluate_subset: MagicMock):
        """Test basic execution of gamify crown with mocked evaluation."""

        # Define how evaluate_subset should behave
        # evaluate_subset(project_path: Path, selected_tests: List[Path]) -> float
        def side_effect_evaluate_subset(project_path, selected_tests, mutant_cache=None): # Add mutant_cache if still in signature
            # Note: evaluate_subset in core.api was refactored and no longer takes mutant_cache.
            # The patch target 'guardian_ai_tool.guardian.cli.evaluate_subset' implies it's imported there.
            # If gamify_crown calls core.api.evaluate_subset, the patch target should be 'guardian_ai_tool.guardian.core.api.evaluate_subset'
            
            # For this test, let's assume the patch target is correct as 'guardian_ai_tool.guardian.cli.evaluate_subset'
            # (meaning it's imported directly into cli.py or the wrapper metric_eval_for_shapley calls it from there)
            # If it's actually 'guardian.core.api.evaluate_subset', the test will fail to mock.
            # Based on cli.py, it's `from guardian.core.api import evaluate_subset`, so patch target should be `guardian.core.api.evaluate_subset`

            if not selected_tests: return 0.0
            # Make scores dependent on which tests are in the subset
            score = 0.0
            # Use resolved paths for comparison as discovered_tests in CLI are resolved
            resolved_a = self.test_file_a_path.resolve()
            resolved_b = self.test_file_b_path.resolve()
            resolved_c = self.test_file_c_path.resolve()

            # Convert selected_tests (which are Path objects) to their resolved string forms for comparison
            selected_test_paths_resolved = [p.resolve() for p in selected_tests]

            if resolved_a in selected_test_paths_resolved: score += 0.5
            if resolved_b in selected_test_paths_resolved: score += 0.3
            if resolved_c in selected_test_paths_resolved: score += 0.1 # C contributes less
            
            # Simulate some interaction to make scores non-additive for Shapley
            if resolved_a in selected_test_paths_resolved and resolved_b in selected_test_paths_resolved:
                score += 0.1 # Synergy
            return round(score, 2)

        mock_evaluate_subset.side_effect = side_effect_evaluate_subset

        result = self.runner.invoke(app, [
            "gamify", "crown",
            "--project-root", str(self.test_project_dir), # Use the temp project
            "--test-dir", "tests", # Relative to project_root
            "--top", "2",
            "--permutations", "20" # Low permutations for faster test
        ])

        self.assertEqual(result.exit_code, 0, f"CLI Error: {result.stdout}")
        
        # Expected Shapley values (approximate due to low permutations, but order should hold)
        # A: (0.5) + (0.5+0.3+0.1 - (0.3+0.1)) / 2 = 0.5 + (0.9 - 0.4)/2 = 0.5 + 0.25 = 0.75 (Incorrect manual calc)
        # Let's re-evaluate based on the side_effect:
        # v({}) = 0
        # v({A}) = 0.5
        # v({B}) = 0.3
        # v({C}) = 0.1
        # v({A,B}) = 0.5 + 0.3 + 0.1 = 0.9
        # v({A,C}) = 0.5 + 0.1 = 0.6
        # v({B,C}) = 0.3 + 0.1 = 0.4
        # v({A,B,C}) = 0.5 + 0.3 + 0.1 + 0.1 = 1.0

        # Shapley(A):
        # Permutations: ABC, ACB, BAC, BCA, CAB, CBA
        # A is first (2/6): v(A)-v({}) = 0.5
        # A is second (2/6): (v(AB)-v(B) + v(AC)-v(C)) / 2 = ( (0.9-0.3) + (0.6-0.1) )/2 = (0.6+0.5)/2 = 0.55
        # A is third (2/6): (v(ABC)-v(BC)) = (1.0 - 0.4) = 0.6
        # Avg for A = (0.5 * 2 + 0.55 * 2 + 0.6 * 2) / 6 = (1 + 1.1 + 1.2)/6 = 3.3/6 = 0.55

        # Shapley(B):
        # B is first (2/6): v(B)-v({}) = 0.3
        # B is second (2/6): (v(AB)-v(A) + v(CB)-v(C))/2 = ((0.9-0.5) + (0.4-0.1))/2 = (0.4+0.3)/2 = 0.35
        # B is third (2/6): (v(ABC)-v(AC)) = (1.0 - 0.6) = 0.4
        # Avg for B = (0.3*2 + 0.35*2 + 0.4*2)/6 = (0.6 + 0.7 + 0.8)/6 = 2.1/6 = 0.35
        
        # Shapley(C):
        # C is first (2/6): v(C)-v({}) = 0.1
        # C is second (2/6): (v(AC)-v(A) + v(BC)-v(B))/2 = ((0.6-0.5) + (0.4-0.3))/2 = (0.1+0.1)/2 = 0.1
        # C is third (2/6): (v(ABC)-v(AB)) = (1.0 - 0.9) = 0.1
        # Avg for C = (0.1*2 + 0.1*2 + 0.1*2)/6 = 0.6/6 = 0.1

        # Expected order: A > B > C
        # With --top 2, we expect A and B.

        output = result.stdout
        self.assertIn("Top 2 Most Valuable Tests", output)
        self.assertIn(str(self.test_file_a_path.resolve()), output) # A should be there
        self.assertIn(str(self.test_file_b_path.resolve()), output) # B should be there
        self.assertNotIn(str(self.test_file_c_path.resolve()), output) # C should not be in top 2

        # Verify evaluate_subset was called multiple times (due to permutations)
        self.assertTrue(mock_evaluate_subset.call_count > 0)


    @patch('guardian_ai_tool.guardian.core.api.evaluate_subset')
    def test_gamify_crown_no_tests_found(self, mock_evaluate_subset: MagicMock):
        """Test gamify crown when no tests are discovered."""
        # Create an empty test directory
        empty_test_dir = self.test_project_dir / "empty_tests"
        empty_test_dir.mkdir(exist_ok=True)

        result = self.runner.invoke(app, [
            "gamify", "crown",
            "--project-root", str(self.test_project_dir),
            "--test-dir", "empty_tests" 
        ])
        
        self.assertEqual(result.exit_code, 0) # Command should exit cleanly
        self.assertIn("No test files (test_*.py) found", result.stdout)
        mock_evaluate_subset.assert_not_called() # evaluate_subset shouldn't be called if no tests

if __name__ == '__main__':
    unittest.main()