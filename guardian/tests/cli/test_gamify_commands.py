import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from guardian.cli import _root_cli_module as guardian_cli_module

app = guardian_cli_module.app


class TestGamifyCrownCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.test_project_dir = Path("temp_test_project_for_cli_gamify")
        self.test_project_dir.mkdir(exist_ok=True)

        self.tests_sub_dir = self.test_project_dir / "tests"
        self.tests_sub_dir.mkdir(exist_ok=True)

        self.test_file_a_path = self.tests_sub_dir / "test_a.py"
        self.test_file_b_path = self.tests_sub_dir / "test_b.py"
        self.test_file_c_path = self.tests_sub_dir / "test_c.py"

        self.test_file_a_path.write_text("def test_alpha(): assert True", encoding="utf-8")
        self.test_file_b_path.write_text("def test_beta(): assert True", encoding="utf-8")
        self.test_file_c_path.write_text("def test_gamma(): assert True", encoding="utf-8")

    def tearDown(self):
        if self.test_project_dir.exists():
            shutil.rmtree(self.test_project_dir)

    @patch("guardian.cli._root_cli_module.evaluate_subset")
    def test_gamify_crown_simple_run(self, mock_evaluate_subset: MagicMock):
        def side_effect_evaluate_subset(project_path, selected_tests, mutant_cache=None):
            if not selected_tests:
                return 0.0

            resolved_a = self.test_file_a_path.resolve()
            resolved_b = self.test_file_b_path.resolve()
            resolved_c = self.test_file_c_path.resolve()
            selected_test_paths = [p.resolve() for p in selected_tests]

            score = 0.0
            if resolved_a in selected_test_paths:
                score += 0.5
            if resolved_b in selected_test_paths:
                score += 0.3
            if resolved_c in selected_test_paths:
                score += 0.1
            if resolved_a in selected_test_paths and resolved_b in selected_test_paths:
                score += 0.1
            return round(score, 2)

        mock_evaluate_subset.side_effect = side_effect_evaluate_subset

        result = self.runner.invoke(
            app,
            [
                "gamify",
                "crown",
                "--project-root",
                str(self.test_project_dir),
                "--test-dir",
                "tests",
                "--top",
                "2",
                "--permutations",
                "20",
            ],
        )

        self.assertEqual(result.exit_code, 0, f"CLI Error: {result.stdout}")
        self.assertIn("Top 2 Most Valuable Tests", result.stdout)
        self.assertIn("tests/test_a.py", result.stdout)
        self.assertIn("tests/test_b.py", result.stdout)
        self.assertNotIn("tests/test_c.py", result.stdout)
        self.assertGreater(mock_evaluate_subset.call_count, 0)

    @patch("guardian.cli._root_cli_module.evaluate_subset")
    def test_gamify_crown_no_tests_found(self, mock_evaluate_subset: MagicMock):
        empty_test_dir = self.test_project_dir / "empty_tests"
        empty_test_dir.mkdir(exist_ok=True)

        result = self.runner.invoke(
            app,
            [
                "gamify",
                "crown",
                "--project-root",
                str(self.test_project_dir),
                "--test-dir",
                "empty_tests",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("No test files (test_*.py) found", result.stdout)
        mock_evaluate_subset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
