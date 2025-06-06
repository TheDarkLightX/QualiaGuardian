"""
Parallel Shapley value computation with multiple backends and optimization strategies.

This module provides a high-performance parallel implementation of Shapley value
calculation with support for threading, multiprocessing, and Ray backends.

Author: DarkLightX/Dana Edwards
"""

import numpy as np
import time
import queue
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import itertools
import tempfile
import mmap
import os
import pickle
from pathlib import Path
import warnings
from math import comb

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None


class Backend(Enum):
    """Available parallelization backends."""
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    RAY = "ray"


class DistributionStrategy(Enum):
    """Work distribution strategies."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    WORK_STEALING = "work_stealing"


@dataclass
class WorkItem:
    """Represents a unit of work for Shapley computation."""
    player_id: int
    coalition_indices: List[int]
    permutation_id: Optional[int] = None
    priority: int = 0


@dataclass
class WorkerResult:
    """Result from a worker computation."""
    player_id: int
    contribution: float
    computation_time: float
    coalition_indices: List[int] = field(default_factory=list)
    worker_id: Optional[int] = None
    error: Optional[Exception] = None


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    backend: Backend = Backend.THREADING
    distribution_strategy: DistributionStrategy = DistributionStrategy.DYNAMIC
    n_workers: Optional[int] = None
    chunk_size: int = 10
    enable_fault_tolerance: bool = True
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_progress_tracking: bool = True
    memory_limit_mb: int = 4096
    convergence_threshold: float = 0.001
    min_iterations: int = 100
    max_iterations: int = 10000
    shared_memory_threshold_mb: float = 100
    enable_work_stealing: bool = True
    work_steal_threshold: float = 0.2


class ParallelizationError(Exception):
    """Base exception for parallelization errors."""
    pass


class WorkerError(ParallelizationError):
    """Exception raised when a worker encounters an error."""
    pass


class ConvergenceError(ParallelizationError):
    """Exception raised when convergence criteria are not met."""
    pass


class ProgressTracker:
    """Thread-safe progress tracker."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._callback = None
        self._update_interval = max(1, total_tasks // 100)  # Update every 1%
        
    def set_callback(self, callback: Callable[[int, int, float, float], None]):
        """Set progress callback function."""
        self._callback = callback
        
    def update(self, increment: int = 1):
        """Update progress by increment."""
        with self._lock:
            self.completed += increment
            
            if self._callback and self.completed % self._update_interval == 0:
                elapsed = time.time() - self.start_time
                rate = self.completed / elapsed if elapsed > 0 else 0
                self._callback(self.completed, self.total_tasks, elapsed, rate)
                
    def get_eta(self) -> float:
        """Get estimated time to completion in seconds."""
        with self._lock:
            if self.completed == 0:
                return float('inf')
                
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed
            remaining = self.total_tasks - self.completed
            
            return remaining / rate if rate > 0 else float('inf')


class ConvergenceChecker:
    """Check convergence of iterative algorithms."""
    
    def __init__(self, threshold: float = 0.001, window_size: int = 5):
        self.threshold = threshold
        self.window_size = window_size
        self.history = []
        
    def check_convergence(self, values: np.ndarray) -> bool:
        """Check if values have converged."""
        self.history.append(values.copy())
        
        if len(self.history) < self.window_size:
            return False
            
        # Keep only recent history
        self.history = self.history[-self.window_size:]
        
        # Check variance in recent values
        recent_values = np.array(self.history)
        std_dev = np.std(recent_values, axis=0)
        
        return np.all(std_dev < self.threshold)
        
    def reset(self):
        """Reset convergence history."""
        self.history = []


class FaultTolerantExecutor:
    """Execute tasks with fault tolerance."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    
        raise WorkerError(f"Max retries exceeded. Last error: {last_error}")
        
    def recover_partial_results(
        self,
        partial_results: List[Any],
        failed_indices: List[int],
        recovery_func: Callable[[int], Any]
    ) -> List[Any]:
        """Recover partial results by recomputing failed items."""
        results = partial_results.copy()
        
        for idx in failed_indices:
            result = self.execute_with_retry(recovery_func, idx)
            results.insert(idx, result)
            
        return results


class MemoryEfficientDataSharer:
    """Share large data structures efficiently across processes."""
    
    def __init__(self):
        self.shared_arrays = {}
        self.memory_maps = {}
        self.reference_counts = {}
        self._lock = threading.Lock()
        self.temp_dir = tempfile.mkdtemp()
        
    def create_shared_array(self, data: np.ndarray, name: str) -> str:
        """Create shared memory array."""
        with self._lock:
            # For multiprocessing, use shared memory
            if name not in self.shared_arrays:
                # Create shared memory
                shared_array = mp.Array('d', data.flatten())
                self.shared_arrays[name] = {
                    'array': shared_array,
                    'shape': data.shape,
                    'dtype': data.dtype
                }
                self.reference_counts[name] = 1
            else:
                self.reference_counts[name] += 1
                
            return name
            
    def get_shared_array(self, name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Get numpy array from shared memory."""
        with self._lock:
            if name in self.shared_arrays:
                shared_data = self.shared_arrays[name]
                return np.frombuffer(
                    shared_data['array'].get_obj(),
                    dtype=dtype
                ).reshape(shape)
            else:
                raise ValueError(f"Shared array '{name}' not found")
                
    def create_memory_mapped_file(self, data: np.ndarray, name: str) -> str:
        """Create memory-mapped file for very large data."""
        with self._lock:
            if name not in self.memory_maps:
                # Create memory-mapped file
                filepath = os.path.join(self.temp_dir, f"{name}.mmap")
                fp = np.memmap(filepath, dtype=data.dtype, mode='w+', shape=data.shape)
                fp[:] = data[:]
                fp.flush()
                
                self.memory_maps[name] = {
                    'filepath': filepath,
                    'shape': data.shape,
                    'dtype': data.dtype
                }
                self.reference_counts[name] = 1
            else:
                self.reference_counts[name] += 1
                
            return name
            
    def read_memory_mapped_file(self, name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Read from memory-mapped file."""
        with self._lock:
            if name in self.memory_maps:
                info = self.memory_maps[name]
                return np.memmap(
                    info['filepath'],
                    dtype=dtype,
                    mode='r',
                    shape=shape
                )
            else:
                raise ValueError(f"Memory-mapped file '{name}' not found")
                
    def add_reference(self, name: str):
        """Add reference to shared resource."""
        with self._lock:
            if name in self.reference_counts:
                self.reference_counts[name] += 1
                
    def release_reference(self, name: str):
        """Release reference to shared resource."""
        with self._lock:
            if name in self.reference_counts:
                self.reference_counts[name] -= 1
                if self.reference_counts[name] <= 0:
                    # Clean up resource
                    if name in self.shared_arrays:
                        del self.shared_arrays[name]
                    if name in self.memory_maps:
                        os.unlink(self.memory_maps[name]['filepath'])
                        del self.memory_maps[name]
                    del self.reference_counts[name]
                    
    def get_reference_count(self, name: str) -> int:
        """Get current reference count."""
        with self._lock:
            return self.reference_counts.get(name, 0)
            
    def cleanup(self):
        """Clean up all shared resources."""
        with self._lock:
            # Clean up memory-mapped files
            for info in self.memory_maps.values():
                try:
                    os.unlink(info['filepath'])
                except:
                    pass
                    
            # Clean up temp directory
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
                
            self.shared_arrays.clear()
            self.memory_maps.clear()
            self.reference_counts.clear()


class ResourceAwareScheduler:
    """Schedule tasks based on available system resources."""
    
    def __init__(self, memory_limit_mb: Optional[int] = None):
        self.memory_limit_mb = memory_limit_mb or self._get_available_memory_mb() * 0.8
        self.current_memory_usage_mb = 0
        self._lock = threading.Lock()
        
    def _get_available_memory_mb(self) -> int:
        """Get available system memory in MB."""
        return psutil.virtual_memory().available // (1024 * 1024)
        
    def get_optimal_worker_count(self, task_count: int, task_memory_mb: float) -> int:
        """Determine optimal number of workers based on resources."""
        cpu_count = psutil.cpu_count()
        
        # Consider CPU cores
        cpu_workers = min(cpu_count * 2, task_count)
        
        # Consider memory constraints
        memory_workers = int(self.memory_limit_mb / task_memory_mb) if task_memory_mb > 0 else cpu_workers
        
        # Use the more restrictive constraint
        optimal = max(1, min(cpu_workers, memory_workers))
        
        return optimal
        
    def can_schedule_task(self, task_memory_mb: float) -> bool:
        """Check if task can be scheduled given memory constraints."""
        with self._lock:
            return (self.current_memory_usage_mb + task_memory_mb) <= self.memory_limit_mb
            
    def allocate_memory(self, amount_mb: float):
        """Allocate memory for a task."""
        with self._lock:
            self.current_memory_usage_mb += amount_mb
            
    def release_memory(self, amount_mb: float):
        """Release memory after task completion."""
        with self._lock:
            self.current_memory_usage_mb = max(0, self.current_memory_usage_mb - amount_mb)
            
    def get_adaptive_chunk_size(
        self,
        total_tasks: int,
        n_workers: int,
        task_complexity: str = 'medium'
    ) -> int:
        """Get adaptive chunk size based on task characteristics."""
        complexity_factors = {
            'low': 100,
            'medium': 50,
            'high': 10
        }
        
        base_chunk = complexity_factors.get(task_complexity, 50)
        
        # Adjust based on total tasks and workers
        if total_tasks < n_workers * 10:
            # Few tasks, use smaller chunks
            chunk_size = max(1, total_tasks // (n_workers * 2))
        else:
            # Many tasks, use base chunk size
            chunk_size = base_chunk
            
        return chunk_size


class ParallelShapleyCalculator:
    """
    High-performance parallel Shapley value calculator.
    
    Supports multiple parallelization backends and optimization strategies
    for efficient computation of Shapley values in cooperative games.
    """
    
    def __init__(
        self,
        backend: Backend = Backend.THREADING,
        distribution_strategy: DistributionStrategy = DistributionStrategy.DYNAMIC,
        n_workers: Optional[int] = None,
        config: Optional[ParallelConfig] = None
    ):
        self.backend = backend
        self.distribution_strategy = distribution_strategy
        self.n_workers = n_workers or self._get_default_workers()
        self.config = config or ParallelConfig(
            backend=backend,
            distribution_strategy=distribution_strategy,
            n_workers=self.n_workers
        )
        
        self._executor = None
        self._progress_tracker = None
        self._convergence_checker = None
        self._fault_executor = None
        self._data_sharer = None
        self._resource_scheduler = None
        self._cancelled = False
        self._cancel_event = threading.Event()
        
        self._initialize_components()
        
    def _get_default_workers(self) -> int:
        """Get default number of workers based on system."""
        cpu_count = psutil.cpu_count()
        if self.backend == Backend.THREADING:
            # Use more threads than cores for I/O bound tasks
            return min(cpu_count * 2, 32)
        else:
            # Use number of cores for CPU bound tasks
            return cpu_count
            
    def _initialize_components(self):
        """Initialize required components based on configuration."""
        if self.config.enable_fault_tolerance:
            self._fault_executor = FaultTolerantExecutor(
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )
            
        if self.config.enable_progress_tracking:
            self._progress_tracker = ProgressTracker(0)  # Will be reset per computation
            
        self._convergence_checker = ConvergenceChecker(
            threshold=self.config.convergence_threshold
        )
        
        self._data_sharer = MemoryEfficientDataSharer()
        self._resource_scheduler = ResourceAwareScheduler(
            memory_limit_mb=self.config.memory_limit_mb
        )
        
    def compute_shapley_values(
        self,
        n_players: int,
        value_function: Callable[[List[int]], float],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_token: Optional[threading.Event] = None,
        shared_data: Optional[Any] = None
    ) -> np.ndarray:
        """
        Compute exact Shapley values using parallel computation.
        
        Args:
            n_players: Number of players in the game
            value_function: Characteristic function v(S) -> R
            progress_callback: Optional callback for progress updates
            cancel_token: Optional cancellation token
            shared_data: Optional shared data for value function
            
        Returns:
            Array of Shapley values for each player
        """
        self._cancelled = False
        self._cancel_event = cancel_token or threading.Event()
        
        # Prepare shared data if provided
        if shared_data is not None:
            self._prepare_shared_data(shared_data)
            
        # Initialize progress tracking
        total_evaluations = n_players * (2 ** (n_players - 1))
        if self._progress_tracker:
            self._progress_tracker = ProgressTracker(total_evaluations)
            if progress_callback:
                self._progress_tracker.set_callback(
                    lambda c, t, e, r: progress_callback(c, t, e)
                )
                
        # Create work items
        work_items = self._create_work_items(n_players)
        
        # Execute parallel computation
        if self.backend == Backend.THREADING:
            results = self._compute_threading(work_items, value_function)
        elif self.backend == Backend.MULTIPROCESSING:
            results = self._compute_multiprocessing(work_items, value_function)
        elif self.backend == Backend.RAY and HAS_RAY:
            results = self._compute_ray(work_items, value_function)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
            
        # Aggregate results
        shapley_values = self._aggregate_results(results, n_players)
        
        # Cleanup
        if self._data_sharer:
            self._data_sharer.cleanup()
            
        return shapley_values
        
    def compute_shapley_values_monte_carlo(
        self,
        n_players: int,
        value_function: Callable[[List[int]], float],
        max_iterations: int = 10000,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        return_convergence_info: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Compute approximate Shapley values using Monte Carlo sampling.
        
        Args:
            n_players: Number of players
            value_function: Characteristic function
            max_iterations: Maximum number of permutation samples
            progress_callback: Optional progress callback
            return_convergence_info: Whether to return convergence information
            
        Returns:
            Shapley values and optionally convergence information
        """
        shapley_values = np.zeros(n_players)
        self._convergence_checker.reset()
        
        # Initialize progress tracking
        if self._progress_tracker:
            self._progress_tracker = ProgressTracker(max_iterations)
            if progress_callback:
                self._progress_tracker.set_callback(
                    lambda c, t, e, r: progress_callback(c, t, e)
                )
                
        converged = False
        iteration = 0
        
        # Batch size for parallel permutation evaluation
        batch_size = min(self.config.chunk_size * self.n_workers, 1000)
        
        while iteration < max_iterations and not converged:
            if self._cancel_event.is_set():
                break
                
            # Generate batch of random permutations
            n_samples = min(batch_size, max_iterations - iteration)
            permutations = [
                np.random.permutation(n_players).tolist()
                for _ in range(n_samples)
            ]
            
            # Evaluate permutations in parallel
            batch_contributions = self._evaluate_permutations_parallel(
                permutations,
                value_function,
                n_players
            )
            
            # Update Shapley values
            for contrib in batch_contributions:
                shapley_values += contrib
                
            iteration += n_samples
            
            # Check convergence
            if iteration >= self.config.min_iterations:
                current_estimate = shapley_values / iteration
                converged = self._convergence_checker.check_convergence(current_estimate)
                
            if self._progress_tracker:
                self._progress_tracker.update(n_samples)
                
        # Normalize by number of samples
        shapley_values /= iteration
        
        if return_convergence_info:
            convergence_info = {
                'iterations': iteration,
                'converged': converged,
                'final_estimate': shapley_values
            }
            return shapley_values, convergence_info
        else:
            return shapley_values
            
    def cancel(self):
        """Cancel ongoing computation."""
        self._cancelled = True
        self._cancel_event.set()
        
    def is_cancelled(self) -> bool:
        """Check if computation was cancelled."""
        return self._cancelled
        
    def _prepare_shared_data(self, data: Any):
        """Prepare shared data for efficient access across workers."""
        if isinstance(data, np.ndarray):
            # Check if data is large enough to benefit from sharing
            data_size_mb = data.nbytes / (1024 * 1024)
            if data_size_mb > self.config.shared_memory_threshold_mb:
                # Use memory mapping for very large arrays
                self._data_sharer.create_memory_mapped_file(data, "shared_data")
            else:
                # Use shared memory for moderate arrays
                self._data_sharer.create_shared_array(data, "shared_data")
                
    def _create_work_items(self, n_players: int) -> List[WorkItem]:
        """Create work items for parallel computation."""
        work_items = []
        
        for player in range(n_players):
            # For exact computation, need to evaluate all coalitions containing player
            for size in range(n_players):
                for coalition in itertools.combinations(
                    [p for p in range(n_players) if p != player], 
                    size
                ):
                    work_items.append(WorkItem(
                        player_id=player,
                        coalition_indices=list(coalition),
                        priority=size  # Prioritize smaller coalitions
                    ))
                    
        return work_items
        
    def _distribute_work_static(self, work_items: List[WorkItem], n_workers: int) -> List[List[WorkItem]]:
        """Distribute work items statically among workers."""
        chunks = [[] for _ in range(n_workers)]
        
        for i, item in enumerate(work_items):
            chunks[i % n_workers].append(item)
            
        return chunks
        
    def _create_work_queue(self, work_items: List[WorkItem]) -> queue.Queue:
        """Create work queue for dynamic distribution."""
        work_queue = queue.Queue()
        
        # Sort by priority (smaller coalitions first)
        sorted_items = sorted(work_items, key=lambda x: x.priority)
        
        for item in sorted_items:
            work_queue.put(item)
            
        return work_queue
        
    def _setup_work_stealing(self, work_items: List[WorkItem], n_workers: int) -> List[queue.Queue]:
        """Set up work-stealing queues."""
        # Create per-worker queues
        work_queues = [queue.Queue() for _ in range(n_workers)]
        
        # Distribute initial work
        chunks = self._distribute_work_static(work_items, n_workers)
        
        for i, chunk in enumerate(chunks):
            for item in chunk:
                work_queues[i].put(item)
                
        return work_queues
        
    def _compute_threading(
        self,
        work_items: List[WorkItem],
        value_function: Callable
    ) -> List[WorkerResult]:
        """Compute using threading backend."""
        results = []
        results_lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            if self.distribution_strategy == DistributionStrategy.STATIC:
                # Static distribution
                chunks = self._distribute_work_static(work_items, self.n_workers)
                futures = []
                
                for worker_id, chunk in enumerate(chunks):
                    future = executor.submit(
                        self._worker_compute_batch,
                        chunk,
                        value_function,
                        worker_id,
                        results,
                        results_lock
                    )
                    futures.append(future)
                    
                # Wait for completion
                for future in as_completed(futures):
                    if self._cancel_event.is_set():
                        executor.shutdown(wait=False)
                        break
                        
            elif self.distribution_strategy == DistributionStrategy.DYNAMIC:
                # Dynamic distribution
                work_queue = self._create_work_queue(work_items)
                futures = []
                
                for worker_id in range(self.n_workers):
                    future = executor.submit(
                        self._worker_dynamic,
                        work_queue,
                        value_function,
                        worker_id,
                        results,
                        results_lock
                    )
                    futures.append(future)
                    
                # Wait for completion
                for future in as_completed(futures):
                    if self._cancel_event.is_set():
                        executor.shutdown(wait=False)
                        break
                        
            elif self.distribution_strategy == DistributionStrategy.WORK_STEALING:
                # Work-stealing distribution
                work_queues = self._setup_work_stealing(work_items, self.n_workers)
                futures = []
                
                for worker_id in range(self.n_workers):
                    future = executor.submit(
                        self._worker_work_stealing,
                        work_queues,
                        worker_id,
                        value_function,
                        results,
                        results_lock
                    )
                    futures.append(future)
                    
                # Wait for completion
                for future in as_completed(futures):
                    if self._cancel_event.is_set():
                        executor.shutdown(wait=False)
                        break
                        
        return results
        
    def _compute_multiprocessing(
        self,
        work_items: List[WorkItem],
        value_function: Callable
    ) -> List[WorkerResult]:
        """Compute using multiprocessing backend."""
        # Need to use manager for shared results in multiprocessing
        manager = mp.Manager()
        results = manager.list()
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            if self.distribution_strategy == DistributionStrategy.STATIC:
                chunks = self._distribute_work_static(work_items, self.n_workers)
                futures = []
                
                for worker_id, chunk in enumerate(chunks):
                    future = executor.submit(
                        _worker_process_batch,
                        chunk,
                        value_function,
                        worker_id
                    )
                    futures.append(future)
                    
                # Collect results
                for future in as_completed(futures):
                    if self._cancel_event.is_set():
                        executor.shutdown(wait=False)
                        break
                    try:
                        worker_results = future.result()
                        results.extend(worker_results)
                    except Exception as e:
                        if self.config.enable_fault_tolerance:
                            # Handle failed batch
                            warnings.warn(f"Worker failed: {e}")
                        else:
                            raise
                            
        return list(results)
        
    def _compute_ray(
        self,
        work_items: List[WorkItem],
        value_function: Callable
    ) -> List[WorkerResult]:
        """Compute using Ray backend."""
        if not HAS_RAY:
            raise RuntimeError("Ray is not installed")
            
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        try:
            # Put value function in object store
            value_func_ref = ray.put(value_function)
            
            # Create Ray tasks
            if self.distribution_strategy == DistributionStrategy.STATIC:
                chunks = self._distribute_work_static(work_items, self.n_workers)
                
                # Create remote tasks
                futures = []
                for worker_id, chunk in enumerate(chunks):
                    future = _ray_worker_batch.remote(
                        chunk,
                        value_func_ref,
                        worker_id
                    )
                    futures.append(future)
                    
                # Get results
                results = []
                ready_futures = futures
                
                while ready_futures and not self._cancel_event.is_set():
                    ready, not_ready = ray.wait(ready_futures, timeout=0.1)
                    
                    for future in ready:
                        worker_results = ray.get(future)
                        results.extend(worker_results)
                        
                    ready_futures = not_ready
                    
                return results
                
        finally:
            # Don't shutdown Ray as it might be used by other parts
            pass
            
    def _worker_compute_batch(
        self,
        work_items: List[WorkItem],
        value_function: Callable,
        worker_id: int,
        results: List[WorkerResult],
        results_lock: threading.Lock
    ):
        """Worker function for batch computation."""
        local_results = []
        
        for item in work_items:
            if self._cancel_event.is_set():
                break
                
            try:
                start_time = time.time()
                
                # Compute marginal contribution
                coalition_with = item.coalition_indices + [item.player_id]
                coalition_without = item.coalition_indices
                
                value_with = value_function(coalition_with)
                value_without = value_function(coalition_without)
                
                contribution = value_with - value_without
                
                compute_time = time.time() - start_time
                
                local_results.append(WorkerResult(
                    player_id=item.player_id,
                    contribution=contribution,
                    computation_time=compute_time,
                    coalition_indices=item.coalition_indices,
                    worker_id=worker_id
                ))
                
                if self._progress_tracker:
                    self._progress_tracker.update()
                    
            except Exception as e:
                if self.config.enable_fault_tolerance and self._fault_executor:
                    # Try to recover
                    try:
                        result = self._fault_executor.execute_with_retry(
                            self._compute_single_contribution,
                            item,
                            value_function,
                            worker_id
                        )
                        local_results.append(result)
                    except Exception as retry_error:
                        local_results.append(WorkerResult(
                            player_id=item.player_id,
                            contribution=0.0,
                            computation_time=0.0,
                            coalition_indices=item.coalition_indices,
                            worker_id=worker_id,
                            error=retry_error
                        ))
                else:
                    raise
                    
        # Add results to shared list
        with results_lock:
            results.extend(local_results)
            
    def _worker_dynamic(
        self,
        work_queue: queue.Queue,
        value_function: Callable,
        worker_id: int,
        results: List[WorkerResult],
        results_lock: threading.Lock
    ):
        """Worker function for dynamic work distribution."""
        local_results = []
        
        while not self._cancel_event.is_set():
            try:
                item = work_queue.get(timeout=0.1)
            except queue.Empty:
                break
                
            try:
                result = self._compute_single_contribution(item, value_function, worker_id)
                local_results.append(result)
                
                if self._progress_tracker:
                    self._progress_tracker.update()
                    
            except Exception as e:
                if self.config.enable_fault_tolerance:
                    # Add error result
                    local_results.append(WorkerResult(
                        player_id=item.player_id,
                        contribution=0.0,
                        computation_time=0.0,
                        coalition_indices=item.coalition_indices,
                        worker_id=worker_id,
                        error=e
                    ))
                else:
                    raise
                    
            work_queue.task_done()
            
        # Add results to shared list
        with results_lock:
            results.extend(local_results)
            
    def _worker_work_stealing(
        self,
        work_queues: List[queue.Queue],
        worker_id: int,
        value_function: Callable,
        results: List[WorkerResult],
        results_lock: threading.Lock
    ):
        """Worker function with work-stealing capability."""
        local_results = []
        own_queue = work_queues[worker_id]
        
        while not self._cancel_event.is_set():
            # Try own queue first
            try:
                item = own_queue.get(timeout=0.01)
            except queue.Empty:
                # Try stealing from other queues
                item = None
                for i, other_queue in enumerate(work_queues):
                    if i != worker_id:
                        try:
                            item = other_queue.get(timeout=0.01)
                            break
                        except queue.Empty:
                            continue
                            
                if item is None:
                    # No work available
                    break
                    
            # Process work item
            try:
                result = self._compute_single_contribution(item, value_function, worker_id)
                local_results.append(result)
                
                if self._progress_tracker:
                    self._progress_tracker.update()
                    
            except Exception as e:
                if self.config.enable_fault_tolerance:
                    local_results.append(WorkerResult(
                        player_id=item.player_id,
                        contribution=0.0,
                        computation_time=0.0,
                        coalition_indices=item.coalition_indices,
                        worker_id=worker_id,
                        error=e
                    ))
                else:
                    raise
                    
        # Add results to shared list
        with results_lock:
            results.extend(local_results)
            
    def _compute_single_contribution(
        self,
        item: WorkItem,
        value_function: Callable,
        worker_id: int
    ) -> WorkerResult:
        """Compute single marginal contribution."""
        start_time = time.time()
        
        # Compute marginal contribution
        coalition_with = item.coalition_indices + [item.player_id]
        coalition_without = item.coalition_indices
        
        value_with = value_function(coalition_with)
        value_without = value_function(coalition_without)
        
        contribution = value_with - value_without
        
        compute_time = time.time() - start_time
        
        return WorkerResult(
            player_id=item.player_id,
            contribution=contribution,
            computation_time=compute_time,
            coalition_indices=item.coalition_indices,
            worker_id=worker_id
        )
        
    def _evaluate_permutations_parallel(
        self,
        permutations: List[List[int]],
        value_function: Callable,
        n_players: int
    ) -> List[np.ndarray]:
        """Evaluate multiple permutations in parallel."""
        if self.backend == Backend.THREADING:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for perm in permutations:
                    future = executor.submit(
                        self._evaluate_single_permutation,
                        perm,
                        value_function,
                        n_players
                    )
                    futures.append(future)
                    
                results = []
                for future in as_completed(futures):
                    if self._cancel_event.is_set():
                        break
                    results.append(future.result())
                    
                return results
                
        elif self.backend == Backend.MULTIPROCESSING:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for perm in permutations:
                    future = executor.submit(
                        _evaluate_permutation_process,
                        perm,
                        value_function,
                        n_players
                    )
                    futures.append(future)
                    
                results = []
                for future in as_completed(futures):
                    if self._cancel_event.is_set():
                        break
                    results.append(future.result())
                    
                return results
                
        else:
            # Fallback to sequential
            return [
                self._evaluate_single_permutation(perm, value_function, n_players)
                for perm in permutations
            ]
            
    def _evaluate_single_permutation(
        self,
        permutation: List[int],
        value_function: Callable,
        n_players: int
    ) -> np.ndarray:
        """Evaluate Shapley contributions for a single permutation."""
        contributions = np.zeros(n_players)
        coalition = []
        prev_value = 0.0
        
        for player in permutation:
            coalition.append(player)
            curr_value = value_function(sorted(coalition))
            contributions[player] = curr_value - prev_value
            prev_value = curr_value
            
        return contributions
        
    def _aggregate_results(
        self,
        results: List[WorkerResult],
        n_players: int
    ) -> np.ndarray:
        """Aggregate worker results into final Shapley values."""
        shapley_values = np.zeros(n_players)
        
        # Group results by coalition size for proper weighting
        coalition_weights = {}
        for result in results:
            if result.error is None:
                coalition_size = len(result.coalition_indices)
                weight = 1.0 / (n_players * comb(n_players - 1, coalition_size))
                shapley_values[result.player_id] += result.contribution * weight
                
        return shapley_values


# Multiprocessing helper functions (must be at module level)
def _worker_process_batch(
    work_items: List[WorkItem],
    value_function: Callable,
    worker_id: int
) -> List[WorkerResult]:
    """Process batch of work items in a separate process."""
    results = []
    
    for item in work_items:
        try:
            start_time = time.time()
            
            coalition_with = item.coalition_indices + [item.player_id]
            coalition_without = item.coalition_indices
            
            value_with = value_function(coalition_with)
            value_without = value_function(coalition_without)
            
            contribution = value_with - value_without
            compute_time = time.time() - start_time
            
            results.append(WorkerResult(
                player_id=item.player_id,
                contribution=contribution,
                computation_time=compute_time,
                coalition_indices=item.coalition_indices,
                worker_id=worker_id
            ))
        except Exception as e:
            results.append(WorkerResult(
                player_id=item.player_id,
                contribution=0.0,
                computation_time=0.0,
                coalition_indices=item.coalition_indices,
                worker_id=worker_id,
                error=e
            ))
            
    return results


def _evaluate_permutation_process(
    permutation: List[int],
    value_function: Callable,
    n_players: int
) -> np.ndarray:
    """Evaluate permutation in a separate process."""
    contributions = np.zeros(n_players)
    coalition = []
    prev_value = 0.0
    
    for player in permutation:
        coalition.append(player)
        curr_value = value_function(sorted(coalition))
        contributions[player] = curr_value - prev_value
        prev_value = curr_value
        
    return contributions


# Ray remote functions
if HAS_RAY:
    @ray.remote
    def _ray_worker_batch(
        work_items: List[WorkItem],
        value_func_ref: ray.ObjectRef,
        worker_id: int
    ) -> List[WorkerResult]:
        """Ray remote function for batch processing."""
        value_function = ray.get(value_func_ref)
        results = []
        
        for item in work_items:
            try:
                start_time = time.time()
                
                coalition_with = item.coalition_indices + [item.player_id]
                coalition_without = item.coalition_indices
                
                value_with = value_function(coalition_with)
                value_without = value_function(coalition_without)
                
                contribution = value_with - value_without
                compute_time = time.time() - start_time
                
                results.append(WorkerResult(
                    player_id=item.player_id,
                    contribution=contribution,
                    computation_time=compute_time,
                    coalition_indices=item.coalition_indices,
                    worker_id=worker_id
                ))
            except Exception as e:
                results.append(WorkerResult(
                    player_id=item.player_id,
                    contribution=0.0,
                    computation_time=0.0,
                    coalition_indices=item.coalition_indices,
                    worker_id=worker_id,
                    error=e
                ))
                
        return results