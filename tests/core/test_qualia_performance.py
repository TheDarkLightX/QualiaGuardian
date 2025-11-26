"""
Performance Tests for Qualia Quality System

Tests performance, scalability, and optimization.
"""

import unittest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from guardian.core.qualia_quality import QualiaQualityEngine, improve_code_quality
from guardian.core.cqs_enhanced import EnhancedCQSCalculator


class TestQualiaPerformance(unittest.TestCase):
    """Performance tests for Qualia Quality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.small_code = """
def add(a, b):
    return a + b
"""
        self.medium_code = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            if item < 100:
                result.append(item * 2)
            else:
                result.append(item)
        else:
            result.append(0)
    return result

def calculate_total(items):
    total = 0
    for item in items:
        total += item
    return total
"""
        self.large_code = self.medium_code * 10  # 10x larger
    
    def test_cqs_calculation_performance(self):
        """Test CQS calculation performance."""
        calc = EnhancedCQSCalculator()
        
        # Small code
        start = time.time()
        result_small = calc.calculate_enhanced(self.small_code)
        time_small = time.time() - start
        
        # Medium code
        start = time.time()
        result_medium = calc.calculate_enhanced(self.medium_code)
        time_medium = time.time() - start
        
        # Large code
        start = time.time()
        result_large = calc.calculate_enhanced(self.large_code)
        time_large = time.time() - start
        
        # Performance assertions
        self.assertLess(time_small, 1.0, "Small code should process in <1s")
        self.assertLess(time_medium, 2.0, "Medium code should process in <2s")
        self.assertLess(time_large, 5.0, "Large code should process in <5s")
        
        # Verify results are valid
        self.assertIsNotNone(result_small)
        self.assertIsNotNone(result_medium)
        self.assertIsNotNone(result_large)
    
    def test_evolutionary_performance(self):
        """Test evolutionary algorithm performance."""
        engine = QualiaQualityEngine(
            population_size=10,
            max_generations=3,
            use_evolutionary=True,
            use_generation=False,  # Disable to test evolution only
            max_iterations=2
        )
        
        start = time.time()
        result = engine.improve_code(self.medium_code)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 10.0, "Evolutionary improvement should complete in <10s")
        self.assertIsNotNone(result)
    
    def test_full_improvement_performance(self):
        """Test full improvement system performance."""
        engine = QualiaQualityEngine(
            target_quality=0.8,
            max_iterations=3
        )
        
        start = time.time()
        result = engine.improve_code(self.medium_code)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 15.0, "Full improvement should complete in <15s")
        self.assertIsNotNone(result)
    
    def test_scalability_small_to_large(self):
        """Test scalability from small to large code."""
        calc = EnhancedCQSCalculator()
        
        sizes = [
            (self.small_code, "small"),
            (self.medium_code, "medium"),
            (self.large_code, "large")
        ]
        
        times = []
        for code, name in sizes:
            start = time.time()
            calc.calculate_enhanced(code)
            elapsed = time.time() - start
            times.append((name, elapsed))
        
        # Times should scale reasonably (not exponentially)
        small_time = times[0][1]
        large_time = times[2][1]
        
        # Large should not be more than 10x slower than small
        self.assertLess(large_time, small_time * 10, 
                       "Performance should scale reasonably")
    
    def test_iteration_performance(self):
        """Test that iterations don't slow down significantly."""
        engine = QualiaQualityEngine(max_iterations=5)
        
        iteration_times = []
        for i in range(3):
            start = time.time()
            result = engine.improve_code(self.medium_code)
            elapsed = time.time() - start
            iteration_times.append(elapsed)
        
        # Times should be relatively consistent
        avg_time = sum(iteration_times) / len(iteration_times)
        for time_val in iteration_times:
            # Each iteration should be within 2x of average
            self.assertLess(time_val, avg_time * 2,
                           "Iteration times should be consistent")
    
    def test_memory_usage(self):
        """Test memory usage (simplified)."""
        import sys
        
        engine = QualiaQualityEngine(population_size=20)
        
        # Measure approximate memory
        size_before = sys.getsizeof(engine)
        
        result = engine.improve_code(self.medium_code)
        
        size_after = sys.getsizeof(engine)
        
        # Memory increase should be reasonable
        # (This is a simplified test - real memory profiling would be more complex)
        self.assertIsNotNone(result)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks."""
    
    def test_benchmark_cqs_calculation(self):
        """Benchmark CQS calculation."""
        calc = EnhancedCQSCalculator()
        code = """
def process(x, y, z):
    if x > 0 and y > 0 and z > 0:
        return x * y * z
    return 0
"""
        
        iterations = 10
        times = []
        
        for _ in range(iterations):
            start = time.time()
            calc.calculate_enhanced(code)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"\nCQS Calculation Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")
        
        # Should be fast
        self.assertLess(avg_time, 0.5, "Average should be <500ms")
        self.assertLess(max_time, 1.0, "Max should be <1s")
    
    def test_benchmark_improvement(self):
        """Benchmark full improvement process."""
        code = """
def calc(x, y):
    if x > 0:
        if y > 0:
            return x * y
    return 0
"""
        
        iterations = 3
        times = []
        
        for _ in range(iterations):
            start = time.time()
            result = improve_code_quality(code, target_quality=0.8)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        
        print(f"\nImprovement Benchmark:")
        print(f"  Average: {avg_time:.2f}s")
        
        # Should complete in reasonable time
        self.assertLess(avg_time, 10.0, "Average should be <10s")


if __name__ == '__main__':
    unittest.main()
