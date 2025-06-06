"""
Progressive Approximation Framework for Quality Analysis

This module implements an anytime algorithm framework that provides
progressively refined results with quality bounds and user control.

Author: DarkLightX/Dana Edwards
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
import time
import threading
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import psutil
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ApproximationStage(Enum):
    """Stages of progressive approximation"""
    HEURISTIC = "heuristic"
    SAMPLING = "sampling"
    FULL = "full"
    REFINED = "refined"


class ResourceType(Enum):
    """Types of computational resources"""
    TIME = "time"
    MEMORY = "memory"
    CPU = "cpu"


@dataclass
class QualityBounds:
    """Quality bounds for approximation results"""
    lower: float
    upper: float
    confidence: float
    stage: ApproximationStage
    
    @property
    def range(self) -> float:
        """Get the range of bounds"""
        return self.upper - self.lower
    
    @property
    def midpoint(self) -> float:
        """Get the midpoint estimate"""
        return (self.lower + self.upper) / 2


@dataclass
class ApproximationResult(Generic[T]):
    """Result from progressive approximation"""
    value: T
    bounds: QualityBounds
    computation_time: float
    resources_used: Dict[ResourceType, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceBudget:
    """Resource budget constraints"""
    max_time: Optional[float] = None  # seconds
    max_memory: Optional[float] = None  # MB
    max_cpu_percent: Optional[float] = None  # percentage
    
    def is_exhausted(self, used: Dict[ResourceType, float]) -> bool:
        """Check if any resource budget is exhausted"""
        if self.max_time and used.get(ResourceType.TIME, 0) >= self.max_time:
            return True
        if self.max_memory and used.get(ResourceType.MEMORY, 0) >= self.max_memory:
            return True
        if self.max_cpu_percent and used.get(ResourceType.CPU, 0) >= self.max_cpu_percent:
            return True
        return False


class ProgressiveApproximator(ABC, Generic[T]):
    """
    Abstract base class for progressive approximation algorithms.
    Follows Single Responsibility Principle - focused on approximation logic.
    """
    
    @abstractmethod
    def heuristic_approximation(self, data: Any) -> ApproximationResult[T]:
        """Quick heuristic approximation"""
        pass
    
    @abstractmethod
    def sampling_approximation(self, data: Any, sample_size: int) -> ApproximationResult[T]:
        """Sampling-based approximation"""
        pass
    
    @abstractmethod
    def full_analysis(self, data: Any) -> ApproximationResult[T]:
        """Complete analysis"""
        pass
    
    @abstractmethod
    def refine_result(self, previous: ApproximationResult[T], data: Any) -> ApproximationResult[T]:
        """Refine previous result"""
        pass


class ResourceMonitor:
    """
    Monitors computational resource usage.
    Follows Single Responsibility Principle - only monitors resources.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def get_usage(self) -> Dict[ResourceType, float]:
        """Get current resource usage"""
        return {
            ResourceType.TIME: time.time() - self.start_time,
            ResourceType.MEMORY: (self.process.memory_info().rss / 1024 / 1024) - self.initial_memory,
            ResourceType.CPU: self.process.cpu_percent(interval=0.1)
        }


class UserControlInterface:
    """
    Interface for user control of approximation process.
    Follows Interface Segregation Principle - focused interface for control.
    """
    
    def __init__(self):
        self._stop_flag = threading.Event()
        self._skip_flag = threading.Event()
        self._refine_flag = threading.Event()
        self._callbacks: Dict[str, List[Callable]] = {
            'progress': [],
            'stage_complete': [],
            'result_ready': []
        }
    
    def stop(self):
        """Stop the approximation process"""
        self._stop_flag.set()
    
    def skip_stage(self):
        """Skip current stage"""
        self._skip_flag.set()
    
    def request_refinement(self):
        """Request additional refinement"""
        self._refine_flag.set()
    
    def should_stop(self) -> bool:
        """Check if stop was requested"""
        return self._stop_flag.is_set()
    
    def should_skip(self) -> bool:
        """Check if skip was requested"""
        if self._skip_flag.is_set():
            self._skip_flag.clear()
            return True
        return False
    
    def refinement_requested(self) -> bool:
        """Check if refinement was requested"""
        if self._refine_flag.is_set():
            self._refine_flag.clear()
            return True
        return False
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for events"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs):
        """Emit event to callbacks"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback: {e}")


class ResultInterpolator:
    """
    Interpolates results between approximation stages.
    Follows Single Responsibility Principle - only handles interpolation.
    """
    
    @staticmethod
    def interpolate_bounds(
        prev: QualityBounds,
        curr: QualityBounds,
        weight: float = 0.7
    ) -> QualityBounds:
        """Interpolate between two quality bounds"""
        # Weight newer results more heavily
        lower = prev.lower * (1 - weight) + curr.lower * weight
        upper = prev.upper * (1 - weight) + curr.upper * weight
        
        # Confidence is minimum of weighted average and current
        confidence = min(
            prev.confidence * (1 - weight) + curr.confidence * weight,
            curr.confidence
        )
        
        return QualityBounds(
            lower=lower,
            upper=upper,
            confidence=confidence,
            stage=curr.stage
        )
    
    @staticmethod
    def merge_results(
        results: List[ApproximationResult[T]],
        weights: Optional[List[float]] = None
    ) -> ApproximationResult[T]:
        """Merge multiple approximation results"""
        if not results:
            raise ValueError("No results to merge")
        
        if weights is None:
            # Default: exponentially increasing weights for later stages
            weights = [2 ** i for i in range(len(results))]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Merge bounds
        merged_lower = sum(r.bounds.lower * w for r, w in zip(results, weights))
        merged_upper = sum(r.bounds.upper * w for r, w in zip(results, weights))
        merged_confidence = sum(r.bounds.confidence * w for r, w in zip(results, weights))
        
        # Use latest result's value and stage
        latest = results[-1]
        
        return ApproximationResult(
            value=latest.value,
            bounds=QualityBounds(
                lower=merged_lower,
                upper=merged_upper,
                confidence=merged_confidence,
                stage=latest.bounds.stage
            ),
            computation_time=sum(r.computation_time for r in results),
            resources_used=latest.resources_used,
            metadata={'merged_from': len(results)}
        )


class ProgressVisualizer:
    """
    Visualizes approximation progress.
    Follows Single Responsibility Principle - only handles visualization.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.fig = None
        self.axes = None
        self.lines = {}
        self.animation = None
    
    def initialize_plot(self):
        """Initialize the plot"""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Progressive Approximation Progress')
        
        # Configure subplots
        self.axes[0, 0].set_title('Quality Bounds')
        self.axes[0, 0].set_ylabel('Value')
        self.axes[0, 0].set_xlabel('Iteration')
        
        self.axes[0, 1].set_title('Confidence Level')
        self.axes[0, 1].set_ylabel('Confidence')
        self.axes[0, 1].set_xlabel('Iteration')
        
        self.axes[1, 0].set_title('Resource Usage')
        self.axes[1, 0].set_ylabel('Usage')
        self.axes[1, 0].set_xlabel('Iteration')
        
        self.axes[1, 1].set_title('Convergence Rate')
        self.axes[1, 1].set_ylabel('Bound Range')
        self.axes[1, 1].set_xlabel('Iteration')
        
        plt.tight_layout()
    
    def update(self, result: ApproximationResult, iteration: int):
        """Update visualization with new result"""
        self.history.append({
            'iteration': iteration,
            'lower': result.bounds.lower,
            'upper': result.bounds.upper,
            'midpoint': result.bounds.midpoint,
            'confidence': result.bounds.confidence,
            'range': result.bounds.range,
            'time': result.resources_used.get(ResourceType.TIME, 0),
            'memory': result.resources_used.get(ResourceType.MEMORY, 0),
            'cpu': result.resources_used.get(ResourceType.CPU, 0)
        })
        
        if self.fig is None:
            self.initialize_plot()
        
        self._update_plots()
    
    def _update_plots(self):
        """Update all subplot data"""
        if not self.history:
            return
        
        data = list(self.history)
        iterations = [d['iteration'] for d in data]
        
        # Quality bounds
        self.axes[0, 0].clear()
        self.axes[0, 0].fill_between(
            iterations,
            [d['lower'] for d in data],
            [d['upper'] for d in data],
            alpha=0.3,
            label='Bounds'
        )
        self.axes[0, 0].plot(
            iterations,
            [d['midpoint'] for d in data],
            'b-',
            label='Midpoint'
        )
        self.axes[0, 0].legend()
        self.axes[0, 0].set_title('Quality Bounds')
        
        # Confidence
        self.axes[0, 1].clear()
        self.axes[0, 1].plot(iterations, [d['confidence'] for d in data], 'g-')
        self.axes[0, 1].set_ylim(0, 1.1)
        self.axes[0, 1].set_title('Confidence Level')
        
        # Resources
        self.axes[1, 0].clear()
        if len(data) > 1:
            self.axes[1, 0].plot(
                iterations,
                [d['time'] for d in data],
                label='Time (s)'
            )
            self.axes[1, 0].plot(
                iterations,
                [d['memory'] for d in data],
                label='Memory (MB)'
            )
            self.axes[1, 0].plot(
                iterations,
                [d['cpu'] for d in data],
                label='CPU %'
            )
            self.axes[1, 0].legend()
        self.axes[1, 0].set_title('Resource Usage')
        
        # Convergence
        self.axes[1, 1].clear()
        self.axes[1, 1].plot(iterations, [d['range'] for d in data], 'r-')
        self.axes[1, 1].set_title('Convergence Rate')
        
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        """Close the visualization"""
        if self.fig:
            plt.close(self.fig)


class ProgressiveAnalysisEngine:
    """
    Main engine for progressive approximation.
    Follows Open/Closed Principle - extensible through approximators.
    """
    
    def __init__(
        self,
        approximator: ProgressiveApproximator[T],
        budget: Optional[ResourceBudget] = None,
        enable_visualization: bool = False
    ):
        self.approximator = approximator
        self.budget = budget or ResourceBudget()
        self.control = UserControlInterface()
        self.interpolator = ResultInterpolator()
        self.visualizer = ProgressVisualizer() if enable_visualization else None
        
        self.results_history: List[ApproximationResult[T]] = []
        self.current_stage = ApproximationStage.HEURISTIC
        
    def analyze(
        self,
        data: Any,
        stages: Optional[List[ApproximationStage]] = None
    ) -> ApproximationResult[T]:
        """
        Perform progressive analysis with anytime properties.
        
        Args:
            data: Input data to analyze
            stages: Stages to execute (default: all)
            
        Returns:
            Best available approximation result
        """
        if stages is None:
            stages = [
                ApproximationStage.HEURISTIC,
                ApproximationStage.SAMPLING,
                ApproximationStage.FULL
            ]
        
        monitor = ResourceMonitor()
        iteration = 0
        
        try:
            for stage in stages:
                if self.control.should_stop():
                    break
                
                if self.control.should_skip():
                    continue
                
                # Check resource budget
                if self.budget.is_exhausted(monitor.get_usage()):
                    logger.info(f"Resource budget exhausted at stage {stage}")
                    break
                
                self.current_stage = stage
                self.control.emit('progress', stage, iteration)
                
                # Execute appropriate approximation
                result = self._execute_stage(stage, data, monitor)
                
                if result:
                    self.results_history.append(result)
                    
                    # Update visualization
                    if self.visualizer:
                        self.visualizer.update(result, iteration)
                    
                    # Emit events
                    self.control.emit('stage_complete', stage, result)
                    self.control.emit('result_ready', result)
                    
                    iteration += 1
            
            # Handle refinement requests
            while self.control.refinement_requested() and not self.control.should_stop():
                if self.budget.is_exhausted(monitor.get_usage()):
                    break
                
                if self.results_history:
                    refined = self._refine_last_result(data, monitor)
                    if refined:
                        self.results_history.append(refined)
                        if self.visualizer:
                            self.visualizer.update(refined, iteration)
                        iteration += 1
            
            # Return best result
            if self.results_history:
                return self.interpolator.merge_results(self.results_history)
            else:
                raise RuntimeError("No approximation results generated")
                
        finally:
            if self.visualizer:
                self.visualizer.close()
    
    def _execute_stage(
        self,
        stage: ApproximationStage,
        data: Any,
        monitor: ResourceMonitor
    ) -> Optional[ApproximationResult[T]]:
        """Execute a specific approximation stage"""
        try:
            if stage == ApproximationStage.HEURISTIC:
                return self.approximator.heuristic_approximation(data)
            elif stage == ApproximationStage.SAMPLING:
                # Adaptive sample size based on available resources
                sample_size = self._calculate_sample_size(data, monitor)
                return self.approximator.sampling_approximation(data, sample_size)
            elif stage == ApproximationStage.FULL:
                return self.approximator.full_analysis(data)
            else:
                logger.warning(f"Unknown stage: {stage}")
                return None
        except Exception as e:
            logger.error(f"Error in stage {stage}: {e}")
            return None
    
    def _refine_last_result(
        self,
        data: Any,
        monitor: ResourceMonitor
    ) -> Optional[ApproximationResult[T]]:
        """Refine the last result"""
        if not self.results_history:
            return None
        
        try:
            return self.approximator.refine_result(self.results_history[-1], data)
        except Exception as e:
            logger.error(f"Error in refinement: {e}")
            return None
    
    def _calculate_sample_size(self, data: Any, monitor: ResourceMonitor) -> int:
        """Calculate adaptive sample size based on resources"""
        # Start with 10% of data size
        if hasattr(data, '__len__'):
            base_size = max(10, len(data) // 10)
        else:
            base_size = 100
        
        # Adjust based on available resources
        usage = monitor.get_usage()
        
        if self.budget.max_time:
            time_factor = 1 - (usage.get(ResourceType.TIME, 0) / self.budget.max_time)
            base_size = int(base_size * time_factor)
        
        if self.budget.max_memory:
            mem_factor = 1 - (usage.get(ResourceType.MEMORY, 0) / self.budget.max_memory)
            base_size = int(base_size * mem_factor)
        
        return max(10, base_size)
    
    def get_current_best(self) -> Optional[ApproximationResult[T]]:
        """Get current best approximation"""
        if self.results_history:
            return self.interpolator.merge_results(self.results_history)
        return None
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of approximation progress"""
        if not self.results_history:
            return {'status': 'no_results'}
        
        latest = self.results_history[-1]
        return {
            'current_stage': self.current_stage.value,
            'stages_completed': len(self.results_history),
            'latest_bounds': {
                'lower': latest.bounds.lower,
                'upper': latest.bounds.upper,
                'confidence': latest.bounds.confidence
            },
            'total_time': sum(r.computation_time for r in self.results_history),
            'convergence_rate': self._calculate_convergence_rate()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate rate of convergence"""
        if len(self.results_history) < 2:
            return 0.0
        
        # Compare bound ranges
        first_range = self.results_history[0].bounds.range
        last_range = self.results_history[-1].bounds.range
        
        if first_range > 0:
            return 1.0 - (last_range / first_range)
        return 0.0


# Example implementation for quality score approximation
class QualityScoreApproximator(ProgressiveApproximator[float]):
    """
    Example approximator for software quality scores.
    Demonstrates the framework with concrete implementation.
    """
    
    def __init__(self):
        self.rng = np.random.RandomState(42)
    
    def heuristic_approximation(self, data: Dict[str, Any]) -> ApproximationResult[float]:
        """Quick heuristic based on file count and basic metrics"""
        start_time = time.time()
        
        # Simple heuristic: base score on file count and test ratio
        file_count = data.get('file_count', 10)
        test_ratio = data.get('test_file_ratio', 0.3)
        
        # Quick estimate
        base_score = 0.5 + 0.3 * test_ratio
        variance = 0.2 / np.sqrt(file_count)
        
        bounds = QualityBounds(
            lower=max(0, base_score - variance),
            upper=min(1, base_score + variance),
            confidence=0.3,  # Low confidence for heuristic
            stage=ApproximationStage.HEURISTIC
        )
        
        return ApproximationResult(
            value=base_score,
            bounds=bounds,
            computation_time=time.time() - start_time,
            resources_used={ResourceType.TIME: time.time() - start_time},
            metadata={'method': 'heuristic', 'file_count': file_count}
        )
    
    def sampling_approximation(
        self,
        data: Dict[str, Any],
        sample_size: int
    ) -> ApproximationResult[float]:
        """Sample-based approximation"""
        start_time = time.time()
        
        # Simulate sampling analysis
        files = data.get('files', [])
        if not files:
            files = [f'file_{i}.py' for i in range(100)]
        
        # Sample files
        sample_indices = self.rng.choice(
            len(files),
            size=min(sample_size, len(files)),
            replace=False
        )
        
        # Simulate quality analysis on sample
        scores = []
        for idx in sample_indices:
            # Simulate score calculation
            time.sleep(0.001)  # Simulate work
            score = 0.4 + 0.5 * self.rng.beta(2, 5)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Calculate bounds with better confidence
        confidence_factor = np.sqrt(sample_size / len(files))
        margin = 1.96 * std_score / np.sqrt(sample_size)  # 95% CI
        
        bounds = QualityBounds(
            lower=max(0, mean_score - margin),
            upper=min(1, mean_score + margin),
            confidence=0.5 + 0.3 * confidence_factor,
            stage=ApproximationStage.SAMPLING
        )
        
        return ApproximationResult(
            value=mean_score,
            bounds=bounds,
            computation_time=time.time() - start_time,
            resources_used={
                ResourceType.TIME: time.time() - start_time,
                ResourceType.MEMORY: sample_size * 0.001  # Simulated
            },
            metadata={
                'method': 'sampling',
                'sample_size': sample_size,
                'total_files': len(files)
            }
        )
    
    def full_analysis(self, data: Dict[str, Any]) -> ApproximationResult[float]:
        """Complete quality analysis"""
        start_time = time.time()
        
        # Simulate full analysis
        files = data.get('files', [])
        if not files:
            files = [f'file_{i}.py' for i in range(100)]
        
        # Analyze all files
        scores = []
        for i, file in enumerate(files):
            # Simulate detailed analysis
            time.sleep(0.002)  # Simulate work
            
            # More complex scoring
            complexity_score = 0.3 + 0.7 * self.rng.beta(3, 2)
            test_score = 0.2 + 0.8 * self.rng.beta(2, 3)
            doc_score = 0.4 + 0.6 * self.rng.beta(4, 2)
            
            file_score = 0.4 * complexity_score + 0.4 * test_score + 0.2 * doc_score
            scores.append(file_score)
        
        final_score = np.mean(scores)
        
        # Very tight bounds with high confidence
        bounds = QualityBounds(
            lower=max(0, final_score - 0.02),
            upper=min(1, final_score + 0.02),
            confidence=0.95,
            stage=ApproximationStage.FULL
        )
        
        return ApproximationResult(
            value=final_score,
            bounds=bounds,
            computation_time=time.time() - start_time,
            resources_used={
                ResourceType.TIME: time.time() - start_time,
                ResourceType.MEMORY: len(files) * 0.005,
                ResourceType.CPU: 45.0  # Simulated CPU usage
            },
            metadata={
                'method': 'full',
                'files_analyzed': len(files),
                'detailed_scores': {
                    'complexity': np.mean([s for s in scores]),
                    'testing': np.mean([s * 0.9 for s in scores]),
                    'documentation': np.mean([s * 0.8 for s in scores])
                }
            }
        )
    
    def refine_result(
        self,
        previous: ApproximationResult[float],
        data: Dict[str, Any]
    ) -> ApproximationResult[float]:
        """Refine previous result with additional analysis"""
        start_time = time.time()
        
        # Simulate refinement
        time.sleep(0.5)
        
        # Tighten bounds
        current_range = previous.bounds.range
        new_range = current_range * 0.7  # 30% improvement
        
        new_bounds = QualityBounds(
            lower=previous.value - new_range / 2,
            upper=previous.value + new_range / 2,
            confidence=min(0.99, previous.bounds.confidence * 1.1),
            stage=ApproximationStage.REFINED
        )
        
        return ApproximationResult(
            value=previous.value,
            bounds=new_bounds,
            computation_time=time.time() - start_time,
            resources_used={ResourceType.TIME: time.time() - start_time},
            metadata={
                'method': 'refinement',
                'improvement': f"{(1 - new_range/current_range) * 100:.1f}%"
            }
        )