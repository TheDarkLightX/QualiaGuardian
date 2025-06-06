"""
Test module for parallel Shapley value computation with multiple backends.

Author: DarkLightX/Dana Edwards
"""

import pytest
import numpy as np
import time
import psutil
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import queue

from guardian.analytics.parallel_shapley import (
    ParallelShapleyCalculator,
    Backend,
    DistributionStrategy,
    WorkItem,
    WorkerResult,
    ParallelConfig,
    ConvergenceChecker,
    FaultTolerantExecutor,
    MemoryEfficientDataSharer,
    ResourceAwareScheduler,
    ProgressTracker,
    ParallelizationError,
    WorkerError,
    ConvergenceError
)


class TestParallelShapleyCalculator:
    """Test the main ParallelShapleyCalculator class."""
    
    @pytest.fixture
    def mock_value_function(self):
        """Create a mock value function for testing."""
        def value_func(coalition):
            # Simple additive value function for testing
            return sum(coalition) * 0.1
        return value_func
    
    @pytest.fixture
    def simple_game(self):
        """Create a simple game setup for testing."""
        return {
            'n_players': 4,
            'players': list(range(4)),
            'value_function': lambda coalition: sum(coalition) * 0.1
        }
    
    def test_initialization_with_default_config(self):
        """Test calculator initialization with default configuration."""
        calc = ParallelShapleyCalculator()
        
        assert calc.backend == Backend.THREADING
        assert calc.distribution_strategy == DistributionStrategy.DYNAMIC
        assert calc.n_workers is None  # Auto-detect
        assert calc.config is not None
        assert calc.config.enable_fault_tolerance is True
        assert calc.config.enable_progress_tracking is True
    
    def test_initialization_with_custom_config(self):
        """Test calculator initialization with custom configuration."""
        config = ParallelConfig(
            backend=Backend.MULTIPROCESSING,
            distribution_strategy=DistributionStrategy.WORK_STEALING,
            n_workers=4,
            chunk_size=100,
            enable_fault_tolerance=False,
            max_retries=5,
            retry_delay=0.5,
            enable_progress_tracking=False,
            memory_limit_mb=2048,
            convergence_threshold=0.001,
            min_iterations=100
        )
        
        calc = ParallelShapleyCalculator(
            backend=Backend.MULTIPROCESSING,
            distribution_strategy=DistributionStrategy.WORK_STEALING,
            n_workers=4,
            config=config
        )
        
        assert calc.backend == Backend.MULTIPROCESSING
        assert calc.distribution_strategy == DistributionStrategy.WORK_STEALING
        assert calc.n_workers == 4
        assert calc.config.chunk_size == 100
        assert calc.config.enable_fault_tolerance is False
    
    def test_compute_shapley_values_threading(self, simple_game):
        """Test Shapley value computation using threading backend."""
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            n_workers=2
        )
        
        shapley_values = calc.compute_shapley_values(
            simple_game['n_players'],
            simple_game['value_function']
        )
        
        assert len(shapley_values) == simple_game['n_players']
        assert all(isinstance(v, float) for v in shapley_values)
        assert abs(sum(shapley_values) - simple_game['value_function'](simple_game['players'])) < 0.01
    
    def test_compute_shapley_values_multiprocessing(self, simple_game):
        """Test Shapley value computation using multiprocessing backend."""
        calc = ParallelShapleyCalculator(
            backend=Backend.MULTIPROCESSING,
            n_workers=2
        )
        
        shapley_values = calc.compute_shapley_values(
            simple_game['n_players'],
            simple_game['value_function']
        )
        
        assert len(shapley_values) == simple_game['n_players']
        assert all(isinstance(v, float) for v in shapley_values)
        assert abs(sum(shapley_values) - simple_game['value_function'](simple_game['players'])) < 0.01
    
    @pytest.mark.skipif(not pytest.importorskip("ray"), reason="Ray not installed")
    def test_compute_shapley_values_ray(self, simple_game):
        """Test Shapley value computation using Ray backend."""
        calc = ParallelShapleyCalculator(
            backend=Backend.RAY,
            n_workers=2
        )
        
        shapley_values = calc.compute_shapley_values(
            simple_game['n_players'],
            simple_game['value_function']
        )
        
        assert len(shapley_values) == simple_game['n_players']
        assert all(isinstance(v, float) for v in shapley_values)
        assert abs(sum(shapley_values) - simple_game['value_function'](simple_game['players'])) < 0.01
    
    def test_progress_tracking(self, simple_game):
        """Test progress tracking during computation."""
        progress_events = []
        
        def progress_callback(completed, total, elapsed_time):
            progress_events.append({
                'completed': completed,
                'total': total,
                'elapsed_time': elapsed_time
            })
        
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            n_workers=2
        )
        
        calc.compute_shapley_values(
            simple_game['n_players'],
            simple_game['value_function'],
            progress_callback=progress_callback
        )
        
        assert len(progress_events) > 0
        assert progress_events[-1]['completed'] == progress_events[-1]['total']
        assert all(e['elapsed_time'] >= 0 for e in progress_events)
    
    def test_cancellation(self, simple_game):
        """Test computation cancellation."""
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            n_workers=2
        )
        
        cancel_event = threading.Event()
        
        def slow_value_function(coalition):
            time.sleep(0.1)
            if cancel_event.is_set():
                raise KeyboardInterrupt("Cancelled")
            return sum(coalition) * 0.1
        
        # Start computation in a thread
        result = []
        error = []
        
        def compute():
            try:
                values = calc.compute_shapley_values(
                    simple_game['n_players'],
                    slow_value_function,
                    cancel_token=cancel_event
                )
                result.append(values)
            except Exception as e:
                error.append(e)
        
        thread = threading.Thread(target=compute)
        thread.start()
        
        # Cancel after a short delay
        time.sleep(0.05)
        cancel_event.set()
        calc.cancel()
        
        thread.join(timeout=2)
        
        assert len(error) > 0 or calc.is_cancelled()
    
    def test_fault_tolerance(self):
        """Test fault tolerance with worker failures."""
        failure_count = {'count': 0}
        
        def faulty_value_function(coalition):
            # Fail the first 2 times
            if failure_count['count'] < 2:
                failure_count['count'] += 1
                raise ValueError("Simulated failure")
            return sum(coalition) * 0.1
        
        config = ParallelConfig(
            enable_fault_tolerance=True,
            max_retries=3,
            retry_delay=0.01
        )
        
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            n_workers=2,
            config=config
        )
        
        shapley_values = calc.compute_shapley_values(
            4,
            faulty_value_function
        )
        
        assert len(shapley_values) == 4
        assert failure_count['count'] >= 2
    
    def test_memory_efficient_data_sharing(self):
        """Test memory-efficient data sharing across workers."""
        large_data = np.random.rand(1000, 1000)
        
        def value_function_with_data(coalition, shared_data):
            # Use shared data in computation
            return np.sum(shared_data[coalition, :])
        
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            n_workers=2
        )
        
        shapley_values = calc.compute_shapley_values(
            4,
            value_function_with_data,
            shared_data=large_data
        )
        
        assert len(shapley_values) == 4
    
    def test_convergence_checking(self):
        """Test convergence checking for Monte Carlo sampling."""
        config = ParallelConfig(
            convergence_threshold=0.01,
            min_iterations=10,
            max_iterations=1000
        )
        
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            n_workers=2,
            config=config
        )
        
        shapley_values, convergence_info = calc.compute_shapley_values_monte_carlo(
            10,
            lambda coalition: sum(coalition) * 0.1,
            return_convergence_info=True
        )
        
        assert len(shapley_values) == 10
        assert 'iterations' in convergence_info
        assert 'converged' in convergence_info
        assert convergence_info['iterations'] >= config.min_iterations


class TestDistributionStrategies:
    """Test different work distribution strategies."""
    
    def test_static_distribution(self):
        """Test static work distribution."""
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            distribution_strategy=DistributionStrategy.STATIC,
            n_workers=2
        )
        
        work_items = list(range(100))
        chunks = calc._distribute_work_static(work_items, 2)
        
        assert len(chunks) == 2
        assert len(chunks[0]) + len(chunks[1]) == 100
        assert abs(len(chunks[0]) - len(chunks[1])) <= 1
    
    def test_dynamic_distribution(self):
        """Test dynamic work distribution."""
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            distribution_strategy=DistributionStrategy.DYNAMIC,
            n_workers=2
        )
        
        work_queue = calc._create_work_queue(list(range(100)))
        
        assert work_queue.qsize() == 100
        
        # Simulate workers taking work
        work_taken = []
        while not work_queue.empty():
            try:
                work_taken.append(work_queue.get_nowait())
            except queue.Empty:
                break
        
        assert len(work_taken) == 100
    
    def test_work_stealing_distribution(self):
        """Test work-stealing distribution."""
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            distribution_strategy=DistributionStrategy.WORK_STEALING,
            n_workers=2
        )
        
        work_items = list(range(100))
        work_queues = calc._setup_work_stealing(work_items, 2)
        
        assert len(work_queues) == 2
        total_work = sum(q.qsize() for q in work_queues)
        assert total_work == 100


class TestResourceAwareScheduling:
    """Test resource-aware scheduling capabilities."""
    
    def test_cpu_aware_worker_count(self):
        """Test automatic worker count based on CPU availability."""
        scheduler = ResourceAwareScheduler()
        
        optimal_workers = scheduler.get_optimal_worker_count(
            task_count=1000,
            task_memory_mb=10
        )
        
        cpu_count = psutil.cpu_count()
        assert 1 <= optimal_workers <= cpu_count * 2
    
    def test_memory_aware_scheduling(self):
        """Test memory-aware task scheduling."""
        scheduler = ResourceAwareScheduler()
        
        # Test with limited memory
        scheduler.memory_limit_mb = 1024
        
        can_schedule = scheduler.can_schedule_task(task_memory_mb=100)
        assert can_schedule is True
        
        # Simulate memory usage
        scheduler.current_memory_usage_mb = 950
        
        can_schedule = scheduler.can_schedule_task(task_memory_mb=100)
        assert can_schedule is False
    
    def test_adaptive_chunk_size(self):
        """Test adaptive chunk size based on system resources."""
        scheduler = ResourceAwareScheduler()
        
        chunk_size = scheduler.get_adaptive_chunk_size(
            total_tasks=10000,
            n_workers=4,
            task_complexity='high'
        )
        
        assert isinstance(chunk_size, int)
        assert 1 <= chunk_size <= 10000


class TestFaultTolerance:
    """Test fault tolerance mechanisms."""
    
    def test_retry_logic(self):
        """Test retry logic for failed tasks."""
        executor = FaultTolerantExecutor(max_retries=3, retry_delay=0.01)
        
        attempt_count = {'count': 0}
        
        def failing_task():
            attempt_count['count'] += 1
            if attempt_count['count'] < 3:
                raise ValueError("Simulated failure")
            return "success"
        
        result = executor.execute_with_retry(failing_task)
        
        assert result == "success"
        assert attempt_count['count'] == 3
    
    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        executor = FaultTolerantExecutor(max_retries=2, retry_delay=0.01)
        
        def always_failing_task():
            raise ValueError("Always fails")
        
        with pytest.raises(WorkerError) as exc_info:
            executor.execute_with_retry(always_failing_task)
        
        assert "Max retries exceeded" in str(exc_info.value)
    
    def test_partial_results_recovery(self):
        """Test recovery of partial results after failure."""
        executor = FaultTolerantExecutor()
        
        partial_results = [1, 2, 3]
        failed_indices = [4, 5]
        
        def recovery_function(index):
            if index in failed_indices:
                return index * 10
            raise ValueError("Should not be called for completed work")
        
        recovered = executor.recover_partial_results(
            partial_results,
            failed_indices,
            recovery_function
        )
        
        assert len(recovered) == 5
        assert recovered[:3] == [1, 2, 3]
        assert recovered[3:] == [40, 50]


class TestConvergenceChecker:
    """Test convergence checking for iterative algorithms."""
    
    def test_convergence_detection(self):
        """Test convergence detection with stable values."""
        checker = ConvergenceChecker(threshold=0.01, window_size=5)
        
        # Simulate converging values
        values = [1.0, 0.99, 0.98, 0.985, 0.984, 0.983, 0.982]
        
        for i, val in enumerate(values):
            converged = checker.check_convergence(np.array([val]))
            if i >= checker.window_size:
                assert converged is True
                break
        else:
            assert False, "Should have converged"
    
    def test_no_convergence(self):
        """Test when values don't converge."""
        checker = ConvergenceChecker(threshold=0.01, window_size=3)
        
        # Simulate oscillating values
        values = [1.0, 2.0, 1.0, 2.0, 1.0]
        
        converged = False
        for val in values:
            converged = checker.check_convergence(np.array([val]))
        
        assert converged is False
    
    def test_multi_dimensional_convergence(self):
        """Test convergence with multi-dimensional values."""
        checker = ConvergenceChecker(threshold=0.01, window_size=3)
        
        # Simulate converging multi-dimensional values
        values = [
            np.array([1.0, 2.0, 3.0]),
            np.array([0.99, 2.01, 2.99]),
            np.array([0.985, 2.005, 2.995]),
            np.array([0.984, 2.004, 2.994])
        ]
        
        converged = False
        for val in values:
            converged = checker.check_convergence(val)
        
        assert converged is True


class TestMemoryEfficientDataSharing:
    """Test memory-efficient data sharing mechanisms."""
    
    def test_shared_memory_creation(self):
        """Test creation of shared memory for numpy arrays."""
        sharer = MemoryEfficientDataSharer()
        
        data = np.random.rand(100, 100)
        shared_name = sharer.create_shared_array(data, "test_array")
        
        assert shared_name is not None
        
        # Retrieve shared data
        retrieved = sharer.get_shared_array("test_array", data.shape, data.dtype)
        
        np.testing.assert_array_equal(data, retrieved)
        
        # Cleanup
        sharer.cleanup()
    
    def test_memory_mapped_files(self):
        """Test memory-mapped file sharing."""
        sharer = MemoryEfficientDataSharer()
        
        data = np.random.rand(1000, 1000)
        mmap_file = sharer.create_memory_mapped_file(data, "test_mmap")
        
        assert mmap_file is not None
        
        # Read from memory-mapped file
        retrieved = sharer.read_memory_mapped_file("test_mmap", data.shape, data.dtype)
        
        np.testing.assert_array_almost_equal(data, retrieved)
        
        # Cleanup
        sharer.cleanup()
    
    def test_reference_counting(self):
        """Test reference counting for shared resources."""
        sharer = MemoryEfficientDataSharer()
        
        data = np.array([1, 2, 3, 4, 5])
        
        # Create shared resource
        sharer.create_shared_array(data, "ref_test")
        
        # Simulate multiple references
        sharer.add_reference("ref_test")
        sharer.add_reference("ref_test")
        
        assert sharer.get_reference_count("ref_test") == 3
        
        # Release references
        sharer.release_reference("ref_test")
        assert sharer.get_reference_count("ref_test") == 2
        
        sharer.cleanup()


class TestProgressTracker:
    """Test progress tracking functionality."""
    
    def test_progress_updates(self):
        """Test progress update notifications."""
        tracker = ProgressTracker(total_tasks=100)
        
        updates = []
        
        def callback(completed, total, elapsed, rate):
            updates.append({
                'completed': completed,
                'total': total,
                'elapsed': elapsed,
                'rate': rate
            })
        
        tracker.set_callback(callback)
        
        # Simulate task completion
        for i in range(10):
            time.sleep(0.01)
            tracker.update(10)
        
        assert len(updates) > 0
        assert updates[-1]['completed'] == 100
        assert updates[-1]['rate'] > 0
    
    def test_eta_calculation(self):
        """Test estimated time of arrival calculation."""
        tracker = ProgressTracker(total_tasks=100)
        
        # Complete some tasks
        tracker.update(25)
        time.sleep(0.1)
        
        eta = tracker.get_eta()
        assert eta > 0
        
        # Complete more tasks
        tracker.update(25)
        new_eta = tracker.get_eta()
        
        assert new_eta < eta  # ETA should decrease
    
    def test_thread_safety(self):
        """Test thread-safe progress updates."""
        tracker = ProgressTracker(total_tasks=1000)
        
        def worker():
            for _ in range(100):
                tracker.update(1)
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert tracker.completed == 1000


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_shapley_calculation(self):
        """Test complete Shapley value calculation with all features."""
        # Define a cooperative game
        def characteristic_function(coalition):
            # Simple voting game
            if len(coalition) >= 3:
                return 1.0
            return 0.0
        
        config = ParallelConfig(
            backend=Backend.THREADING,
            distribution_strategy=DistributionStrategy.DYNAMIC,
            n_workers=4,
            enable_fault_tolerance=True,
            enable_progress_tracking=True,
            convergence_threshold=0.001
        )
        
        calc = ParallelShapleyCalculator(config=config)
        
        progress_updates = []
        
        def progress_callback(completed, total, elapsed):
            progress_updates.append(completed / total)
        
        shapley_values = calc.compute_shapley_values(
            n_players=5,
            value_function=characteristic_function,
            progress_callback=progress_callback
        )
        
        assert len(shapley_values) == 5
        assert all(0 <= v <= 1 for v in shapley_values)
        assert len(progress_updates) > 0
        assert progress_updates[-1] == 1.0
    
    def test_large_scale_computation(self):
        """Test computation with a large number of players."""
        n_players = 15
        
        def value_function(coalition):
            return len(coalition) ** 2 / n_players ** 2
        
        config = ParallelConfig(
            backend=Backend.MULTIPROCESSING,
            distribution_strategy=DistributionStrategy.WORK_STEALING,
            n_workers=mp.cpu_count(),
            chunk_size=100,
            memory_limit_mb=2048
        )
        
        calc = ParallelShapleyCalculator(config=config)
        
        # Use Monte Carlo for large games
        shapley_values, conv_info = calc.compute_shapley_values_monte_carlo(
            n_players=n_players,
            value_function=value_function,
            max_iterations=1000,
            return_convergence_info=True
        )
        
        assert len(shapley_values) == n_players
        assert conv_info['converged'] or conv_info['iterations'] == 1000
        assert abs(sum(shapley_values) - 1.0) < 0.1  # Efficiency property
    
    def test_error_handling_integration(self):
        """Test integrated error handling across components."""
        def problematic_value_function(coalition):
            if len(coalition) == 2:
                raise ValueError("Cannot compute value for pairs")
            return len(coalition)
        
        config = ParallelConfig(
            enable_fault_tolerance=True,
            max_retries=2
        )
        
        calc = ParallelShapleyCalculator(
            backend=Backend.THREADING,
            config=config
        )
        
        with pytest.raises(ParallelizationError):
            calc.compute_shapley_values(
                n_players=4,
                value_function=problematic_value_function
            )


class TestPerformance:
    """Performance benchmarks for parallel implementations."""
    
    def test_speedup_vs_sequential(self):
        """Test speedup compared to sequential computation."""
        n_players = 10
        
        def value_function(coalition):
            # Simulate some computation
            time.sleep(0.001)
            return sum(coalition)
        
        # Sequential baseline
        start = time.time()
        seq_calc = ParallelShapleyCalculator(n_workers=1)
        seq_values = seq_calc.compute_shapley_values(n_players, value_function)
        seq_time = time.time() - start
        
        # Parallel computation
        start = time.time()
        par_calc = ParallelShapleyCalculator(n_workers=4)
        par_values = par_calc.compute_shapley_values(n_players, value_function)
        par_time = time.time() - start
        
        # Should see some speedup
        speedup = seq_time / par_time
        assert speedup > 1.5  # At least 1.5x speedup
        
        # Results should be similar
        np.testing.assert_allclose(seq_values, par_values, rtol=0.01)
    
    def test_scaling_efficiency(self):
        """Test scaling efficiency with different worker counts."""
        n_players = 12
        iterations = 100
        
        def value_function(coalition):
            # Light computation
            return np.sum(coalition) * np.random.rand()
        
        times = {}
        
        for n_workers in [1, 2, 4, 8]:
            calc = ParallelShapleyCalculator(
                backend=Backend.MULTIPROCESSING,
                n_workers=n_workers
            )
            
            start = time.time()
            calc.compute_shapley_values_monte_carlo(
                n_players, 
                value_function,
                max_iterations=iterations
            )
            times[n_workers] = time.time() - start
        
        # Check that doubling workers provides meaningful speedup
        assert times[2] < times[1] * 0.7
        assert times[4] < times[2] * 0.7