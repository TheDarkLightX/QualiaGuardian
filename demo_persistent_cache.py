#!/usr/bin/env python3
"""
Demonstration of Persistent Result Cache

Shows how to use the cache system with different backends,
invalidation strategies, and monitoring capabilities.

Author: DarkLightX/Dana Edwards
"""

import time
import json
from pathlib import Path

from guardian.analytics.persistent_cache import (
    CacheFactory,
    ASTHashInvalidator,
    TTLInvalidator,
    VersionInvalidator,
    compute_ast_hash
)


def demo_basic_caching():
    """Demonstrate basic cache operations"""
    print("=== Basic Caching Demo ===\n")
    
    # Create a simple in-memory cache
    cache = CacheFactory.create_memory_cache()
    
    # Example: Caching test analysis results
    test_content = {
        "test_file": "test_example.py",
        "test_name": "test_calculation",
        "code": "assert calculate(5) == 25"
    }
    
    analysis_result = {
        "mutation_score": 0.85,
        "assertion_iq": 0.92,
        "complexity": 3,
        "vulnerabilities": []
    }
    
    # Cache the result
    print("Caching analysis result...")
    cache.set(test_content, analysis_result)
    
    # Retrieve from cache
    print("Retrieving from cache...")
    cached_result = cache.get(test_content)
    print(f"Cached result: {json.dumps(cached_result, indent=2)}")
    
    # Check statistics
    stats = cache.get_statistics()
    print(f"\nCache statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")


def demo_ttl_caching():
    """Demonstrate TTL-based expiration"""
    print("\n\n=== TTL Caching Demo ===\n")
    
    cache = CacheFactory.create_memory_cache()
    
    # Cache with 2-second TTL
    content = {"query": "temporary data"}
    result = {"status": "fresh", "timestamp": time.time()}
    
    print("Caching with 2-second TTL...")
    cache.set(content, result, ttl=2)
    
    # Immediate retrieval works
    print("Immediate retrieval:", cache.get(content) is not None)
    
    # Wait for expiration
    print("Waiting 2.5 seconds...")
    time.sleep(2.5)
    
    # Should be expired
    print("After expiration:", cache.get(content) is None)


def demo_filesystem_backend():
    """Demonstrate filesystem storage backend"""
    print("\n\n=== Filesystem Backend Demo ===\n")
    
    cache_dir = ".demo_cache"
    cache = CacheFactory.create_filesystem_cache(
        cache_dir=cache_dir,
        compression_threshold=50  # Compress anything over 50 bytes
    )
    
    # Store some data
    large_data = {
        "test_results": [
            {"name": f"test_{i}", "passed": True, "duration": 0.1}
            for i in range(100)
        ]
    }
    
    print(f"Storing large data in {cache_dir}/...")
    cache.set("large_test_results", large_data)
    
    # Show file structure
    print("\nCache directory structure:")
    for path in sorted(Path(cache_dir).rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            print(f"  {path.relative_to(cache_dir)} ({size} bytes)")
    
    # Retrieve and verify
    retrieved = cache.get("large_test_results")
    print(f"\nSuccessfully retrieved {len(retrieved['test_results'])} test results")
    
    # Cleanup
    cache.clear()
    Path(cache_dir).rmdir()


def demo_ast_invalidation():
    """Demonstrate AST-based cache invalidation"""
    print("\n\n=== AST Invalidation Demo ===\n")
    
    # Original test code
    test_code_v1 = """
def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
"""
    
    # Modified test code (same logic, different formatting)
    test_code_v2 = """
def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0  # Added comment
"""
    
    # Actually changed test code
    test_code_v3 = """
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0  # Changed test case
"""
    
    # Compute AST hashes
    hash_v1 = compute_ast_hash(test_code_v1)
    hash_v2 = compute_ast_hash(test_code_v2)
    hash_v3 = compute_ast_hash(test_code_v3)
    
    print("AST hashes:")
    print(f"  Version 1: {hash_v1[:16]}...")
    print(f"  Version 2: {hash_v2[:16]}... (same as v1: {hash_v1 == hash_v2})")
    print(f"  Version 3: {hash_v3[:16]}... (different: {hash_v1 != hash_v3})")
    
    # Create cache with AST invalidator
    test_hashes = {"test_example.py": hash_v3}
    cache = CacheFactory.create_memory_cache(
        invalidators=[ASTHashInvalidator(test_hashes)]
    )
    
    # Cache results for old version
    cache.set(
        "test_1",
        {"score": 0.9},
        metadata={"test_file": "test_example.py", "test_ast_hash": hash_v1}
    )
    
    # Cache results for current version
    cache.set(
        "test_2",
        {"score": 0.95},
        metadata={"test_file": "test_example.py", "test_ast_hash": hash_v3}
    )
    
    print("\nBefore invalidation:")
    print(f"  Old version cached: {cache.get('test_1') is not None}")
    print(f"  Current version cached: {cache.get('test_2') is not None}")
    
    # Run invalidation
    invalidated = cache.invalidate()
    
    print(f"\nAfter invalidation (removed {invalidated} entries):")
    print(f"  Old version cached: {cache.get('test_1') is not None}")
    print(f"  Current version cached: {cache.get('test_2') is not None}")


def demo_lru_eviction():
    """Demonstrate LRU eviction policy"""
    print("\n\n=== LRU Eviction Demo ===\n")
    
    # Create small cache for demo
    cache = CacheFactory.create_memory_cache(
        max_entries=5,
        max_size_bytes=1024
    )
    
    # Fill cache
    print("Filling cache with 5 entries...")
    for i in range(5):
        cache.set(f"key_{i}", f"value_{i}")
    
    # Access some entries to update LRU order
    print("\nAccessing key_0 and key_2 to move them to end of LRU...")
    cache.get("key_0")
    cache.get("key_2")
    
    # Add new entry - should evict least recently used
    print("\nAdding new entry (will trigger eviction)...")
    cache.set("key_new", "value_new")
    
    # Check what was evicted
    print("\nCache contents after eviction:")
    for i in range(5):
        exists = cache.get(f"key_{i}") is not None
        print(f"  key_{i}: {'exists' if exists else 'evicted'}")
    print(f"  key_new: exists")
    
    # Show eviction statistics
    stats = cache.get_statistics()
    print(f"\nEviction reasons: {stats['evictions']}")


def demo_monitoring_and_statistics():
    """Demonstrate comprehensive monitoring"""
    print("\n\n=== Monitoring Demo ===\n")
    
    cache = CacheFactory.create_memory_cache(
        compression_threshold=100
    )
    
    # Generate various cache operations
    print("Generating cache activity...")
    
    # Some hits
    cache.set("popular_key", {"data": "X" * 200})  # Will be compressed
    for _ in range(10):
        cache.get("popular_key")
    
    # Some misses
    for i in range(5):
        cache.get(f"missing_key_{i}")
    
    # Various sized entries
    cache.set("small", "tiny")
    cache.set("medium", "M" * 50)
    cache.set("large", "L" * 500)
    
    # Get comprehensive statistics
    stats = cache.get_statistics()
    
    print("\nCache Statistics:")
    print(f"  Total requests: {stats['hits'] + stats['misses']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Current entries: {stats['current_entries']}")
    print(f"  Current size: {stats['current_size_bytes']} bytes")
    print(f"  Usage: {stats['usage_percent']:.1f}%")
    print(f"  Compression active: Yes (for entries > 100 bytes)")


def demo_production_setup():
    """Demonstrate production-ready cache configuration"""
    print("\n\n=== Production Setup Demo ===\n")
    
    # Production cache with all features
    cache = CacheFactory.create_sqlite_cache(
        db_path=".guardian_production_cache.db",
        invalidators=[
            TTLInvalidator(),
            VersionInvalidator("2.0.0"),
            ASTHashInvalidator({})  # Would be populated with actual test hashes
        ],
        max_entries=50000,
        max_size_bytes=1024 * 1024 * 1024,  # 1GB
        compression_threshold=1024  # 1KB
    )
    
    print("Production cache configuration:")
    print("  Backend: SQLite (persistent)")
    print("  Max entries: 50,000")
    print("  Max size: 1GB")
    print("  Compression: Enabled for entries > 1KB")
    print("  Invalidation strategies:")
    print("    - TTL-based expiration")
    print("    - Algorithm version tracking")
    print("    - Test code AST change detection")
    
    # Example usage for Shapley value caching
    shapley_content = {
        "test_suite": "tests/",
        "target_metric": "mutation_score",
        "algorithm": "exact_shapley"
    }
    
    shapley_result = {
        "values": {
            "test_critical_1": 0.342,
            "test_critical_2": 0.287,
            "test_helper_1": 0.089,
            # ... more results
        },
        "computation_time": 45.3,
        "algorithm_version": "2.0.0"
    }
    
    # Cache with metadata for invalidation
    cache.set(
        shapley_content,
        shapley_result,
        metadata={
            "algorithm_version": "2.0.0",
            "test_ast_hash": compute_ast_hash("test code here"),
            "timestamp": time.time()
        },
        ttl=3600  # 1 hour TTL
    )
    
    print("\nCached Shapley computation result")
    print("  TTL: 1 hour")
    print("  Will invalidate if:")
    print("    - Algorithm version changes")
    print("    - Test code changes")
    print("    - TTL expires")
    
    # Cleanup
    Path(".guardian_production_cache.db").unlink(missing_ok=True)


if __name__ == "__main__":
    print("Guardian Persistent Cache Demonstration")
    print("=" * 50)
    
    demos = [
        demo_basic_caching,
        demo_ttl_caching,
        demo_filesystem_backend,
        demo_ast_invalidation,
        demo_lru_eviction,
        demo_monitoring_and_statistics,
        demo_production_setup
    ]
    
    for demo in demos:
        demo()
    
    print("\n\nDemo complete!")