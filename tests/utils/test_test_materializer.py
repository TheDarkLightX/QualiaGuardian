import unittest
from pathlib import Path
import tempfile
import shutil
import time # For TestIndividual ID generation if not mocked

from guardian_ai_tool.guardian.evolution.types import TestIndividual
from guardian_ai_tool.guardian.utils.test_materializer import materialize_tests

class TestTestMaterializer(unittest.TestCase):

    def setUp(self):
        self.temp_output_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_output_dir)

    def test_materialize_single_test(self):
        """Test materializing a single TestIndividual."""
        test_code_content = "def test_sample_one():\n    assert True\n"
        # TestIndividual's __post_init__ uses time.time() for ID.
        # For predictable filenames, we can either mock time.time() during TestIndividual creation
        # or accept the generated ID and check for file existence based on prefix.
        # Let's create it and then find the file.
        
        individual1 = TestIndividual(
            test_code=test_code_content,
            assertions=[{"type": "assert", "code": "assert True"}]
            # setup_code and teardown_code are empty by default
        )
        # Manually set a predictable ID part for easier filename checking in test
        # The ID format is "test_{hash_value}". We need the hash_value part.
        # For simplicity in test, let's mock the ID or make it predictable.
        individual1.id = "test_single_abcdef"


        created_files = materialize_tests([individual1], self.temp_output_dir, base_filename_prefix="test_gen_")
        
        self.assertEqual(len(created_files), 1)
        expected_filename = self.temp_output_dir / "test_gen_abcdef_0.py"
        self.assertEqual(created_files[0], expected_filename)
        self.assertTrue(expected_filename.exists())

        with open(expected_filename, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, individual1.code()) # .code() adds a newline

    def test_materialize_multiple_tests(self):
        """Test materializing multiple TestIndividuals."""
        ind1_code = "def test_multi_one():\n    assert 1 == 1\n"
        ind1 = TestIndividual(test_code=ind1_code, assertions=[])
        ind1.id = "test_multi_111111"

        ind2_code = "def test_multi_two(param):\n    assert param > 0\n"
        ind2_setup = "import pytest"
        ind2 = TestIndividual(test_code=ind2_code, assertions=[], setup_code=ind2_setup)
        ind2.id = "test_multi_222222"
        
        individuals = [ind1, ind2]
        created_files = materialize_tests(individuals, self.temp_output_dir) # Use default prefix

        self.assertEqual(len(created_files), 2)

        expected_file1 = self.temp_output_dir / f"test_auto_generated_{ind1.id.split('_')[-1][:8]}_0.py"
        expected_file2 = self.temp_output_dir / f"test_auto_generated_{ind2.id.split('_')[-1][:8]}_1.py"

        self.assertIn(expected_file1, created_files)
        self.assertIn(expected_file2, created_files)

        self.assertTrue(expected_file1.exists())
        with open(expected_file1, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, ind1.code())

        self.assertTrue(expected_file2.exists())
        with open(expected_file2, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, ind2.code()) # Should include setup_code

    def test_materialize_empty_list(self):
        """Test materializing an empty list of individuals."""
        created_files = materialize_tests([], self.temp_output_dir)
        self.assertEqual(len(created_files), 0)
        # Check if directory is empty (or only contains .gitkeep if that's a pattern)
        self.assertEqual(len(list(self.temp_output_dir.iterdir())), 0)

    def test_materialize_individual_with_empty_code(self):
        """Test skipping an individual if its .code() method returns empty/whitespace."""
        ind_valid_code = "def test_valid():\n    pass\n"
        ind_valid = TestIndividual(test_code=ind_valid_code, assertions=[])
        ind_valid.id = "test_valid_abc"

        ind_empty_code = "" # TestIndividual.code() will strip this
        ind_empty = TestIndividual(test_code=ind_empty_code, assertions=[])
        ind_empty.id = "test_empty_def"
        
        # Mock .code() for ind_empty to return empty string after strip
        # Or ensure TestIndividual.code() handles it.
        # The current TestIndividual.code() returns "\n" for empty self.test_code.
        # The materializer checks `if not test_code_content.strip():`
        # So, an individual with test_code="" will result in .code() returning "\n",
        # and .strip() on that will be empty, so it should be skipped.

        individuals = [ind_valid, ind_empty]
        created_files = materialize_tests(individuals, self.temp_output_dir)

        self.assertEqual(len(created_files), 1)
        expected_filename = self.temp_output_dir / f"test_auto_generated_{ind_valid.id.split('_')[-1][:8]}_0.py"
        self.assertEqual(created_files[0], expected_filename)
        self.assertTrue(expected_filename.exists())

    def test_output_directory_creation(self):
        """Test that the output directory is created if it doesn't exist."""
        non_existent_dir = self.temp_output_dir / "new_subdir"
        self.assertFalse(non_existent_dir.exists())

        ind1_code = "def test_for_new_dir():\n    assert True\n"
        ind1 = TestIndividual(test_code=ind1_code, assertions=[])
        ind1.id = "test_newdir_123"
        
        created_files = materialize_tests([ind1], non_existent_dir)
        
        self.assertTrue(non_existent_dir.exists())
        self.assertEqual(len(created_files), 1)
        self.assertTrue((non_existent_dir / f"test_auto_generated_{ind1.id.split('_')[-1][:8]}_0.py").exists())

if __name__ == '__main__':
    unittest.main()