"""
Tests for Persistent Result Cache

Comprehensive test suite validating all cache functionality including
storage backends, invalidation strategies, compression, and monitoring.

Author: DarkLightX/Dana Edwards
"""

import pytest
import tempfile
import shutil
import time
import threading
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from guardian.analytics.persistent_cache import (
    PersistentResultCache,
    CacheFactory,
    SHA256KeyGenerator,
    SQLiteStorage,
    FilesystemStorage,
    RedisStorage,
    PickleSerializer,
    ZlibCompressor,
    ASTHashInvalidator,
    TTLInvalidator,
    VersionInvalidator,
    DefaultCacheMonitor,
    CacheEntry,
    compute_ast_hash,
    create_default_cache
)


class TestSHA256KeyGenerator:
    """Test SHA256 key generation"""
    
    def test_generates_consistent_keys(self):
        """Should generate same key for same content"""
        generator = SHA256KeyGenerator()
        
        content = {"test": "data", "number": 42}
        metadata = {"version": "1.0"}
        
        key1 = generator.generate_key(content, metadata)
        key2 = generator.generate_key(content, metadata)
        
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length
    
    def test_different_content_different_keys(self):
        """Should generate different keys for different content"""
        generator = SHA256KeyGenerator()
        
        key1 = generator.generate_key({"a": 1}, {})
        key2 = generator.generate_key({"a": 2}, {})
        
        assert key1 != key2
    
    def test_handles_various_content_types(self):
        """Should handle strings, bytes, and objects"""
        generator = SHA256KeyGenerator()
        
        # String content
        key1 = generator.generate_key("test string", {})
        assert len(key1) == 64
        
        # Bytes content
        key2 = generator.generate_key(b"test bytes", {})
        assert len(key2) == 64
        
        # Object content
        key3 = generator.generate_key({"complex": ["data", 123]}, {})
        assert len(key3) == 64
        
        # All different
        assert key1 != key2 != key3


class TestSQLiteStorage:
    """Test SQLite storage backend"""
    
    @pytest.fixture
    def storage(self):
        """Create temporary SQLite storage"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        storage = SQLiteStorage(db_path)
        yield storage
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_basic_operations(self, storage):
        """Should support get, set, delete, exists"""
        key = "test_key"
        data = b"test data"
        
        # Initially doesn't exist
        assert not storage.exists(key)
        assert storage.get(key) is None
        
        # Set data
        assert storage.set(key, data)
        assert storage.exists(key)
        
        # Get data
        retrieved = storage.get(key)
        assert retrieved == data
        
        # Delete data
        assert storage.delete(key)
        assert not storage.exists(key)
    
    def test_ttl_expiration(self, storage):
        """Should expire entries based on TTL"""
        key = "ttl_key"
        data = b"expires soon"
        
        # Set with 1 second TTL
        storage.set(key, data, ttl=1)
        
        # Should exist immediately
        assert storage.get(key) == data
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert storage.get(key) is None
        assert not storage.exists(key)
    
    def test_access_tracking(self, storage):
        """Should track access time and count"""
        key = "tracked_key"
        data = b"track me"
        
        storage.set(key, data)
        
        # Access multiple times
        for _ in range(3):
            storage.get(key)
        
        # Check access count in database
        import sqlite3
        with sqlite3.connect(storage.db_path) as conn:
            cursor = conn.execute(
                "SELECT access_count FROM cache_entries WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            assert row[0] == 3
    
    def test_clear_all_entries(self, storage):
        """Should clear all entries"""
        # Add multiple entries
        for i in range(5):
            storage.set(f"key_{i}", f"data_{i}".encode())
        
        # Clear all
        count = storage.clear()
        assert count == 5
        
        # Verify all gone
        for i in range(5):
            assert not storage.exists(f"key_{i}")
    
    def test_thread_safety(self, storage):
        """Should handle concurrent operations safely"""
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(10):
                    key = f"thread_{thread_id}_key_{i}"
                    data = f"thread_{thread_id}_data_{i}".encode()
                    
                    storage.set(key, data)
                    retrieved = storage.get(key)
                    
                    if retrieved != data:
                        errors.append(f"Mismatch in thread {thread_id}")
                    
                    results.append(True)
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 operations


class TestFilesystemStorage:
    """Test filesystem storage backend"""
    
    @pytest.fixture
    def storage(self):
        """Create temporary filesystem storage"""
        cache_dir = tempfile.mkdtemp()
        storage = FilesystemStorage(cache_dir)
        yield storage
        
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)
    
    def test_basic_operations(self, storage):
        """Should support basic cache operations"""
        key = "a" * 64  # Ensure we have directory sharding
        data = b"filesystem data"
        
        # Set and get
        assert storage.set(key, data)
        assert storage.get(key) == data
        assert storage.exists(key)
        
        # Verify file structure
        expected_path = storage.cache_dir / key[:2] / key
        assert expected_path.exists()
        assert expected_path.with_suffix('.meta').exists()
        
        # Delete
        assert storage.delete(key)
        assert not storage.exists(key)
    
    def test_ttl_handling(self, storage):
        """Should handle TTL through metadata"""
        key = "ttl_test"
        data = b"expires"
        
        storage.set(key, data, ttl=1)
        
        # Should exist initially
        assert storage.get(key) == data
        
        # Wait and check expiration
        time.sleep(1.1)
        assert storage.get(key) is None
    
    def test_clear_recursive(self, storage):
        """Should clear all files recursively"""
        # Create entries with different prefixes for sharding
        keys = ["aa_test", "bb_test", "cc_test"]
        for key in keys:
            storage.set(key, b"data")
        
        count = storage.clear()
        assert count == 3
        
        for key in keys:
            assert not storage.exists(key)


class TestPickleSerializer:
    """Test pickle serialization"""
    
    def test_serialize_deserialize_objects(self):
        """Should handle complex Python objects"""
        serializer = PickleSerializer()
        
        # Test various object types
        test_cases = [
            {"dict": "data", "number": 42},
            ["list", "of", "items", 123],
            ("tuple", "data", 456),
            {"nested": {"structure": ["with", "lists"]}},
            set([1, 2, 3]),
        ]
        
        for obj in test_cases:
            serialized = serializer.serialize(obj)
            assert isinstance(serialized, bytes)
            
            deserialized = serializer.deserialize(serialized)
            assert deserialized == obj


class TestZlibCompressor:
    """Test zlib compression"""
    
    def test_compress_decompress(self):
        """Should compress and decompress data"""
        compressor = ZlibCompressor()
        
        # Create compressible data
        data = b"A" * 1000 + b"B" * 1000 + b"C" * 1000
        
        compressed = compressor.compress(data)
        assert len(compressed) < len(data)  # Should be smaller
        
        decompressed = compressor.decompress(compressed)
        assert decompressed == data
    
    def test_compression_levels(self):
        """Should support different compression levels"""
        data = b"X" * 10000
        
        # Test different levels
        compressor_fast = ZlibCompressor(level=1)
        compressor_best = ZlibCompressor(level=9)
        
        compressed_fast = compressor_fast.compress(data)
        compressed_best = compressor_best.compress(data)
        
        # Higher compression should be smaller (usually)
        assert len(compressed_best) <= len(compressed_fast)
        
        # Both should decompress correctly
        assert compressor_fast.decompress(compressed_fast) == data
        assert compressor_best.decompress(compressed_best) == data


class TestInvalidators:
    """Test cache invalidation strategies"""
    
    def test_ast_hash_invalidator(self):
        """Should invalidate when AST changes"""
        test_hashes = {
            "test.py": "hash123",
            "test2.py": "hash456"
        }
        
        invalidator = ASTHashInvalidator(test_hashes)
        
        # Entry with matching hash - should not invalidate
        entry1 = CacheEntry(
            key="key1",
            data=b"",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            metadata={
                "test_file": "test.py",
                "test_ast_hash": "hash123"
            }
        )
        assert not invalidator.should_invalidate(entry1)
        
        # Entry with different hash - should invalidate
        entry2 = CacheEntry(
            key="key2",
            data=b"",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            metadata={
                "test_file": "test.py",
                "test_ast_hash": "old_hash"
            }
        )
        assert invalidator.should_invalidate(entry2)
        
        # Entry with unknown file - should invalidate
        entry3 = CacheEntry(
            key="key3",
            data=b"",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            metadata={
                "test_file": "unknown.py",
                "test_ast_hash": "some_hash"
            }
        )
        assert invalidator.should_invalidate(entry3)
    
    def test_ttl_invalidator(self):
        """Should invalidate expired entries"""
        invalidator = TTLInvalidator()
        
        # Non-expired entry
        entry1 = CacheEntry(
            key="key1",
            data=b"",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            ttl_seconds=3600  # 1 hour
        )
        assert not invalidator.should_invalidate(entry1)
        
        # Expired entry
        entry2 = CacheEntry(
            key="key2",
            data=b"",
            created_at=datetime.now() - timedelta(hours=2),
            accessed_at=datetime.now(),
            ttl_seconds=3600  # 1 hour
        )
        assert invalidator.should_invalidate(entry2)
        
        # No TTL - should not invalidate
        entry3 = CacheEntry(
            key="key3",
            data=b"",
            created_at=datetime.now() - timedelta(days=365),
            accessed_at=datetime.now(),
            ttl_seconds=None
        )
        assert not invalidator.should_invalidate(entry3)
    
    def test_version_invalidator(self):
        """Should invalidate on version mismatch"""
        invalidator = VersionInvalidator("2.0.0")
        
        # Matching version
        entry1 = CacheEntry(
            key="key1",
            data=b"",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            metadata={"algorithm_version": "2.0.0"}
        )
        assert not invalidator.should_invalidate(entry1)
        
        # Different version
        entry2 = CacheEntry(
            key="key2",
            data=b"",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            metadata={"algorithm_version": "1.0.0"}
        )
        assert invalidator.should_invalidate(entry2)


class TestDefaultCacheMonitor:
    """Test cache monitoring"""
    
    def test_records_statistics(self):
        """Should track hits, misses, and evictions"""
        monitor = DefaultCacheMonitor()
        
        # Record events
        for _ in range(10):
            monitor.record_hit("key1")
        
        for _ in range(5):
            monitor.record_miss("key2")
        
        monitor.record_eviction("key3", "max_entries")
        monitor.record_eviction("key4", "max_entries")
        monitor.record_eviction("key5", "ttl_expired")
        
        # Get statistics
        stats = monitor.get_statistics()
        
        assert stats['hits'] == 10
        assert stats['misses'] == 5
        assert stats['hit_rate'] == 10 / 15  # 10 hits / 15 total
        assert stats['evictions']['max_entries'] == 2
        assert stats['evictions']['ttl_expired'] == 1
    
    def test_thread_safe_operations(self):
        """Should handle concurrent statistics updates"""
        monitor = DefaultCacheMonitor()
        
        def record_hits():
            for _ in range(100):
                monitor.record_hit("key")
        
        def record_misses():
            for _ in range(100):
                monitor.record_miss("key")
        
        # Run concurrent updates
        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=record_hits)
            t2 = threading.Thread(target=record_misses)
            threads.extend([t1, t2])
            t1.start()
            t2.start()
        
        for t in threads:
            t.join()
        
        stats = monitor.get_statistics()
        assert stats['hits'] == 500
        assert stats['misses'] == 500


class TestPersistentResultCache:
    """Test main cache manager"""
    
    @pytest.fixture
    def cache(self):
        """Create in-memory cache for testing"""
        return CacheFactory.create_memory_cache(
            max_entries=10,
            max_size_bytes=1024,
            compression_threshold=50
        )
    
    def test_basic_cache_operations(self, cache):
        """Should support get and set operations"""
        content = {"query": "test", "params": [1, 2, 3]}
        result = {"score": 0.95, "data": "result"}
        metadata = {"version": "1.0"}
        
        # Initially not cached
        assert cache.get(content, metadata) is None
        
        # Cache result
        assert cache.set(content, result, metadata)
        
        # Retrieve cached result
        cached = cache.get(content, metadata)
        assert cached == result
    
    def test_compression_for_large_data(self, cache):
        """Should compress data above threshold"""
        content = "test"
        # Create data larger than compression threshold (50 bytes)
        large_result = {"data": "X" * 100}
        
        assert cache.set(content, large_result)
        
        # Should retrieve correctly despite compression
        cached = cache.get(content)
        assert cached == large_result
    
    def test_lru_eviction_by_count(self, cache):
        """Should evict oldest entries when max_entries reached"""
        # Fill cache to max_entries (10)
        for i in range(10):
            cache.set(f"content_{i}", f"result_{i}")
        
        # Add one more - should evict oldest
        cache.set("content_new", "result_new")
        
        # First entry should be evicted
        assert cache.get("content_0") is None
        
        # Others should still exist
        assert cache.get("content_1") == "result_1"
        assert cache.get("content_new") == "result_new"
    
    def test_lru_eviction_by_size(self):
        """Should evict entries when max_size_bytes reached"""
        cache = CacheFactory.create_memory_cache(
            max_entries=1000,  # High limit
            max_size_bytes=200,  # Low size limit
            compression_threshold=1000  # Disable compression
        )
        
        # Add entries that together exceed size limit
        cache.set("key1", "X" * 80)
        cache.set("key2", "Y" * 80)
        cache.set("key3", "Z" * 80)  # Should trigger eviction
        
        # First entry should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "Y" * 80
        assert cache.get("key3") == "Z" * 80
    
    def test_lru_ordering(self, cache):
        """Should update LRU order on access"""
        # Fill cache
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Access early entry to move it to end
        cache.get("key_0")
        
        # Add new entry - should evict key_1, not key_0
        cache.set("key_new", "value_new")
        
        assert cache.get("key_0") == "value_0"  # Still exists
        assert cache.get("key_1") is None  # Evicted
    
    def test_invalidation_with_pattern(self, cache):
        """Should invalidate entries matching pattern"""
        # Add various entries
        cache.set("test_1", "data1")
        cache.set("test_2", "data2")
        cache.set("other_1", "data3")
        cache.set("other_2", "data4")
        
        # Invalidate by pattern
        count = cache.invalidate(pattern="test_")
        assert count == 2
        
        # Verify correct entries removed
        assert cache.get("test_1") is None
        assert cache.get("test_2") is None
        assert cache.get("other_1") == "data3"
        assert cache.get("other_2") == "data4"
    
    def test_invalidation_with_invalidators(self):
        """Should use invalidators to determine what to remove"""
        # Create cache with version invalidator
        cache = CacheFactory.create_memory_cache(
            invalidators=[VersionInvalidator("2.0")]
        )
        
        # Add entries with different versions
        cache.set("old", "data1", metadata={"algorithm_version": "1.0"})
        cache.set("current", "data2", metadata={"algorithm_version": "2.0"})
        
        # Run invalidation
        count = cache.invalidate()
        assert count == 1
        
        # Old version should be gone
        assert cache.get("old", metadata={"algorithm_version": "1.0"}) is None
        assert cache.get("current", metadata={"algorithm_version": "2.0"}) == "data2"
    
    def test_clear_cache(self, cache):
        """Should clear all entries"""
        # Add entries
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Clear
        count = cache.clear()
        assert count == 5
        
        # Verify all gone
        for i in range(5):
            assert cache.get(f"key_{i}") is None
    
    def test_statistics_tracking(self, cache):
        """Should provide comprehensive statistics"""
        # Generate some activity
        cache.set("key1", "value1")
        cache.set("key2", "X" * 100)  # Large enough to compress
        
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss
        
        stats = cache.get_statistics()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] > 0
        assert stats['current_entries'] == 2
        assert stats['current_size_bytes'] > 0
        assert stats['usage_percent'] > 0
    
    def test_thread_safety(self, cache):
        """Should handle concurrent operations"""
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(20):
                    key = f"thread_{thread_id}_item_{i}"
                    cache.set(key, f"data_{thread_id}_{i}")
                    
                    # Sometimes read
                    if i % 2 == 0:
                        result = cache.get(key)
                        if result != f"data_{thread_id}_{i}":
                            errors.append(f"Mismatch in thread {thread_id}")
                    
                    # Sometimes invalidate
                    if i % 5 == 0:
                        cache.invalidate(pattern=f"thread_{thread_id}")
                        
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestCacheFactory:
    """Test cache factory methods"""
    
    def test_create_sqlite_cache(self):
        """Should create SQLite-backed cache"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            cache = CacheFactory.create_sqlite_cache(db_path)
            assert isinstance(cache.storage, SQLiteStorage)
            
            # Should work
            cache.set("test", "data")
            assert cache.get("test") == "data"
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_create_filesystem_cache(self):
        """Should create filesystem-backed cache"""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = CacheFactory.create_filesystem_cache(cache_dir)
            assert isinstance(cache.storage, FilesystemStorage)
            
            # Should work
            cache.set("test", "data")
            assert cache.get("test") == "data"
    
    @patch('redis.Redis')
    def test_create_redis_cache(self, mock_redis):
        """Should create Redis-backed cache"""
        cache = CacheFactory.create_redis_cache()
        assert isinstance(cache.storage, RedisStorage)
    
    def test_create_memory_cache(self):
        """Should create in-memory cache"""
        cache = CacheFactory.create_memory_cache()
        assert isinstance(cache.storage, SQLiteStorage)
        
        # Should work
        cache.set("test", "data")
        assert cache.get("test") == "data"


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_compute_ast_hash(self):
        """Should compute consistent AST hashes"""
        # Same logic, different formatting
        code1 = """
def foo(x):
    return x + 1
"""
        
        code2 = """
def foo(x):
            return x + 1
"""
        
        # Should have same AST hash
        hash1 = compute_ast_hash(code1)
        hash2 = compute_ast_hash(code2)
        assert hash1 == hash2
        
        # Different logic
        code3 = """
def foo(x):
    return x + 2
"""
        
        hash3 = compute_ast_hash(code3)
        assert hash3 != hash1
    
    def test_compute_ast_hash_handles_errors(self):
        """Should handle invalid Python code"""
        invalid_code = "def foo( invalid syntax"
        
        # Should not raise, but return some hash
        hash_value = compute_ast_hash(invalid_code)
        assert len(hash_value) == 64  # SHA256 length
    
    def test_create_default_cache(self):
        """Should create cache with sensible defaults"""
        cache = create_default_cache()
        
        assert cache.max_entries == 10000
        assert cache.max_size_bytes == 1024 * 1024 * 1024  # 1GB
        assert cache.compression_threshold == 1024  # 1KB
        assert len(cache.invalidators) == 2  # TTL and Version
        
        # Should work
        cache.set("test", {"data": "value"})
        assert cache.get("test") == {"data": "value"}


class TestRedisStorageIntegration:
    """Integration tests for Redis storage (requires Redis)"""
    
    @pytest.mark.skipif(
        True,  # Skip Redis tests by default
        reason="Redis integration tests require Redis server"
    )
    def test_redis_operations(self):
        """Should work with real Redis instance"""
        storage = RedisStorage()
        
        # Basic operations
        key = "integration_test"
        data = b"redis data"
        
        assert storage.set(key, data)
        assert storage.exists(key)
        assert storage.get(key) == data
        assert storage.delete(key)
        assert not storage.exists(key)
        
        # TTL
        storage.set("ttl_key", b"expires", ttl=2)
        assert storage.exists("ttl_key")
        time.sleep(2.1)
        assert not storage.exists("ttl_key")