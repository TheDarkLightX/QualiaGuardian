import unittest
from unittest.mock import MagicMock, patch
import time
from pathlib import Path
import os # For creating dummy files/dirs if needed by SmartMutator

from guardian_ai_tool.guardian.evolution.adaptive_emt import AdaptiveEMT
from guardian_ai_tool.guardian.budget import Budget
from guardian_ai_tool.guardian.utils.pid_controller import PIDController
from guardian_ai_tool.guardian.evolution.fitness import FitnessEvaluator # For type hinting
from guardian_ai_tool.guardian.evolution.types import TestIndividual # For return type

# Minimal FitnessEvaluator mock that can be instantiated
class MockFitnessEvaluator:
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path

    def evaluate_individual(self, individual, mutants):
        # Return a dummy FitnessVector or whatever is expected
        return MagicMock() 

class TestAdaptiveEMTEvolveLoop(unittest.TestCase):

    def setUp(self):
        self.code_root = Path("dummy_code_root")
        # SmartMutator expects a string path, and it might try to access it.
        # Create a dummy directory for it if its __init__ tries to validate path.
        if not self.code_root.exists():
            self.code_root.mkdir(parents=True, exist_ok=True)
            
        self.existing_tests_paths = [self.code_root / "test_dummy.py"] # Dummy path
        # Create a dummy test file if _initialize_population tries to read it
        if not self.existing_tests_paths[0].exists():
             with open(self.existing_tests_paths[0], "w") as f:
                 f.write("def test_example(): assert True")

        self.mock_fitness_evaluator = MockFitnessEvaluator(str(self.code_root))

    def tearDown(self):
        # Clean up dummy directory and file
        if self.existing_tests_paths[0].exists():
            self.existing_tests_paths[0].unlink()
        if self.code_root.exists():
            # Check if directory is empty before removing, or handle rmdir error
            try:
                self.code_root.rmdir() 
            except OSError: # If other files were created by SmartMutator, this might fail
                pass # Or use shutil.rmtree if more robust cleanup is needed


    def test_evolve_stops_on_cpu_budget(self):
        """Test that evolve loop stops when CPU budget is exceeded."""
        budget = Budget(cpu_core_min=1.0, wall_min=None, target_delta_m=None)
        
        # Mock PID to return a fixed pop_size that will cause budget to be hit
        mock_pid = MagicMock(spec=PIDController)
        mock_pid.next.return_value = 10 # Pop size for _simulate_generation_cost

        emt = AdaptiveEMT(
            code_root=self.code_root,
            existing_tests=self.existing_tests_paths,
            fitness_evaluator=self.mock_fitness_evaluator,
            budget=budget,
            pid=mock_pid
        )

        # Patch _simulate_generation_cost to control CPU cost per generation
        # Each call to _run_one_generation will incur this cost
        # If pop_size=10, default sim cost is 10*0.05 + 0.1 = 0.6 CPU min
        # So, 2 generations (0.6 + 0.6 = 1.2) should exceed budget of 1.0
        
        # No need to patch _simulate_generation_cost if its default behavior is fine.
        # Let's verify:
        # Gen 1: pop=10, cost=0.6. cpu_used_total = 0.6. Budget 1.0. Continue.
        # Gen 2: pop=10, cost=0.6. cpu_used_total = 1.2. Budget 1.0. Stop.
        # So, it should run for 2 generations (0 and 1). current_generation_num will be 2 when loop breaks.

        result = emt.evolve()

        self.assertEqual(emt.current_generation_num, 2) # Ran gen 0, gen 1
        self.assertGreater(emt.cpu_used_total_min, budget.cpu_core_min)
        self.assertEqual(result, []) # M0 returns empty list
        mock_pid.reset_integral.assert_called_once()
        self.assertEqual(mock_pid.next.call_count, 2) # Called for gen 0 and gen 1

    def test_evolve_stops_on_target_delta_m(self):
        """Test that evolve loop stops when target_delta_m is achieved."""
        target_delta = 0.05
        budget = Budget(cpu_core_min=10.0, wall_min=None, target_delta_m=target_delta)
        mock_pid = MagicMock(spec=PIDController)
        mock_pid.next.return_value = 5 # Small pop_size to not hit CPU budget quickly

        emt = AdaptiveEMT(
            code_root=self.code_root,
            existing_tests=self.existing_tests_paths,
            fitness_evaluator=self.mock_fitness_evaluator,
            budget=budget,
            pid=mock_pid
        )

        # _run_one_generation simulates delta_m: self.current_delta_m += random.uniform(0.001, 0.005) * (pop_size / 10.0)
        # If pop_size = 5, improvement is random.uniform(0.0005, 0.0025)
        # To hit 0.05, it might take many generations. Let's control this.
        
        # We'll mock _run_one_generation's effect on current_delta_m
        # Side effect function to update current_delta_m
        delta_m_values = [0.01, 0.02, 0.03, 0.04, 0.055] # Will hit target on 5th call (gen 4)
        call_count = 0
        def mock_run_one_gen_effect(*args, **kwargs):
            nonlocal call_count
            if call_count < len(delta_m_values):
                emt.current_delta_m = delta_m_values[call_count]
            call_count +=1
            # Original _run_one_generation logic for pop adjustment (simplified)
            pop_size = kwargs.get('pop_size', 5)
            if len(emt.population) > pop_size: emt.population = emt.population[:pop_size]
            elif len(emt.population) < pop_size:
                for _ in range(pop_size - len(emt.population)): emt.population.append(emt._generate_random_test())


        with patch.object(emt, '_run_one_generation', side_effect=mock_run_one_gen_effect) as mock_method:
            result = emt.evolve()

        self.assertEqual(emt.current_generation_num, 4) # Gen 0,1,2,3,4. Loop breaks when current_generation_num is 4.
        self.assertGreaterEqual(emt.current_delta_m, target_delta)
        self.assertEqual(result, [])
        self.assertEqual(mock_method.call_count, 5)

    def test_evolve_stops_on_wall_time_budget(self):
        """Test that evolve loop stops when wall time budget is exceeded."""
        budget = Budget(cpu_core_min=100.0, wall_min=0.001, target_delta_m=None) # Very small wall time (0.001 min = 0.06s)
        mock_pid = MagicMock(spec=PIDController)
        mock_pid.next.return_value = 5 

        emt = AdaptiveEMT(
            code_root=self.code_root,
            existing_tests=self.existing_tests_paths,
            fitness_evaluator=self.mock_fitness_evaluator,
            budget=budget,
            pid=mock_pid
        )

        # Patch time.monotonic to control elapsed time
        # Start time will be captured. Then each check will show more time passed.
        start_time = time.monotonic() # Actual start time
        
        # Let the first check pass, second check exceed budget
        # Note: time.monotonic() returns seconds. budget.wall_min is minutes.
        # So, 0.001 min = 0.06 seconds.
        # If _run_one_generation takes any time, it should exceed quickly.
        
        # To make it more predictable, let's control the time progression
        # First call to monotonic (in evolve start): time_A
        # Second call (in loop check): time_A + 0.03 seconds (within budget)
        # Third call (in loop check): time_A + 0.07 seconds (exceeds budget of 0.06s)
        
        # This requires careful patching of time.monotonic.
        # A simpler way for M0 is to ensure _run_one_generation takes some "simulated" time
        # and the loop naturally breaks.
        # The current _simulate_generation_cost doesn't affect wall time used in the loop check.
        
        # Let's make _run_one_generation itself sleep a bit to consume wall time.
        def slow_run_one_generation(*args, **kwargs):
            time.sleep(0.04) # 40ms, should be enough for 2 gens to exceed 60ms
            # Original _run_one_generation logic for pop adjustment (simplified)
            pop_size = kwargs.get('pop_size', 5)
            if len(emt.population) > pop_size: emt.population = emt.population[:pop_size]
            elif len(emt.population) < pop_size:
                for _ in range(pop_size - len(emt.population)): emt.population.append(emt._generate_random_test())


        with patch.object(emt, '_run_one_generation', side_effect=slow_run_one_generation) as mock_run_gen:
            result = emt.evolve()
        
        # Expected:
        # Gen 0: PID -> pop=5. _run_one_generation sleeps 0.04s. CPU cost sim. Loop continues.
        # Gen 1: PID -> pop=5. _run_one_generation sleeps 0.04s. Total sleep = 0.08s.
        # Wall budget is 0.001 min = 0.06s. So after Gen 1, wall budget should be hit.
        # Loop for Gen 1 runs, then at start of Gen 2 check, wall budget exceeded.
        # So, current_generation_num should be 1 (meaning gen 0 completed).
        
        self.assertLessEqual(emt.current_generation_num, 2) # Should run at most 1-2 generations
        self.assertTrue((time.monotonic() - emt.wall_time_start) / 60.0 >= budget.wall_min)
        self.assertEqual(result, [])

    def test_evolve_runs_max_generations_if_no_other_limits_hit(self):
        """Test that evolve runs for max_generations_heuristic if no budget/target is hit."""
        # Large budget, no target delta
        budget = Budget(cpu_core_min=1000.0, wall_min=100.0, target_delta_m=None)
        mock_pid = MagicMock(spec=PIDController)
        # PID returns small pop_size to ensure CPU budget isn't hit quickly
        mock_pid.next.return_value = 1 

        emt = AdaptiveEMT(
            code_root=self.code_root,
            existing_tests=self.existing_tests_paths,
            fitness_evaluator=self.mock_fitness_evaluator,
            budget=budget,
            pid=mock_pid
        )
        # Access the heuristic from the instance if it's not hardcoded in evolve
        # It is hardcoded in evolve for M0, so we use that value.
        max_gens_heuristic = 100 
        
        # Ensure _run_one_generation doesn't make delta_m hit a potential target
        def no_delta_m_change_run_one_gen(*args, **kwargs):
            # emt.current_delta_m remains 0 or very small
            pop_size = kwargs.get('pop_size', 1) # Use the pop_size from PID
            if len(emt.population) > pop_size: emt.population = emt.population[:pop_size]
            elif len(emt.population) < pop_size:
                for _ in range(pop_size - len(emt.population)): emt.population.append(emt._generate_random_test())


        with patch.object(emt, '_run_one_generation', side_effect=no_delta_m_change_run_one_gen):
            result = emt.evolve()

        self.assertEqual(emt.current_generation_num, max_gens_heuristic)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()