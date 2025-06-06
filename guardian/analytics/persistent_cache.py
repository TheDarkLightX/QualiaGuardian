"""
Persistent Result Cache Implementation

A SOLID-compliant caching system for Guardian analytics results.
Provides content-addressed storage with multiple backend support,
automatic invalidation, and comprehensive monitoring.

Author: DarkLightX/Dana Edwards
"""

import hashlib
import json
import time
import threading
import zlib
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import redis
from collections import OrderedDict
import ast
import logging

logger = logging.getLogger(__name__)


# ===== Interface Segregation: Define focused interfaces =====

class CacheKeyGenerator(ABC):
    """Interface for generating cache keys"""
    
    @abstractmethod
    def generate_key(self, content: Any, metadata: Dict[str, Any]) -> str:
        """Generate a unique cache key from content and metadata"""
        pass


class CacheStorage(ABC):
    """Interface for cache storage backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data by key"""
        pass
    
    @abstractmethod
    def set(self, key: str, data: bytes, ttl: Optional[int] = None) -> bool:
        """Store data with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """Clear all entries, return count deleted"""
        pass


class CacheInvalidator(ABC):
    """Interface for cache invalidation strategies"""
    
    @abstractmethod
    def should_invalidate(self, entry: 'CacheEntry') -> bool:
        """Check if cache entry should be invalidated"""
        pass


class CacheSerializer(ABC):
    """Interface for data serialization"""
    
    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data"""
        pass


class CacheCompressor(ABC):
    """Interface for data compression"""
    
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data"""
        pass
    
    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data"""
        pass


class CacheMonitor(ABC):
    """Interface for cache monitoring"""
    
    @abstractmethod
    def record_hit(self, key: str) -> None:
        """Record cache hit"""
        pass
    
    @abstractmethod
    def record_miss(self, key: str) -> None:
        """Record cache miss"""
        pass
    
    @abstractmethod
    def record_eviction(self, key: str, reason: str) -> None:
        """Record cache eviction"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


# ===== Data Models =====

@dataclass
class CacheEntry:
    """Represents a cached entry with metadata"""
    key: str
    data: bytes
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    entry_count: int = 0
    compression_ratio: float = 1.0
    last_cleared: Optional[datetime] = None


# ===== Single Responsibility: Focused implementations =====

class SHA256KeyGenerator(CacheKeyGenerator):
    """Generate cache keys using SHA256 hashing"""
    
    def generate_key(self, content: Any, metadata: Dict[str, Any]) -> str:
        """Generate SHA256 hash of content and metadata"""
        hasher = hashlib.sha256()
        
        # Hash content
        if isinstance(content, str):
            hasher.update(content.encode('utf-8'))
        elif isinstance(content, bytes):
            hasher.update(content)
        else:
            hasher.update(json.dumps(content, sort_keys=True).encode('utf-8'))
        
        # Hash metadata
        if metadata:
            hasher.update(json.dumps(metadata, sort_keys=True).encode('utf-8'))
        
        return hasher.hexdigest()


class SQLiteStorage(CacheStorage):
    """SQLite-based cache storage"""
    
    def __init__(self, db_path: str = ".guardian_cache.db"):
        self.db_path = db_path
        self._init_db()
        self._lock = threading.Lock()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER DEFAULT 0,
                    metadata TEXT,
                    ttl_seconds INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)")
    
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data by key"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data, created_at, ttl_seconds FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                data, created_at, ttl_seconds = row
                
                # Check TTL
                if ttl_seconds:
                    age = time.time() - created_at
                    if age > ttl_seconds:
                        self.delete(key)
                        return None
                
                # Update access time and count
                conn.execute(
                    "UPDATE cache_entries SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                    (time.time(), key)
                )
                
                return data
    
    def set(self, key: str, data: bytes, ttl: Optional[int] = None) -> bool:
        """Store data with optional TTL"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    now = time.time()
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, data, created_at, accessed_at, size_bytes, ttl_seconds)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (key, data, now, now, len(data), ttl))
                return True
            except Exception as e:
                logger.error(f"Failed to set cache entry: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete data by key"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                return cursor.rowcount > 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM cache_entries WHERE key = ? LIMIT 1", (key,))
            return cursor.fetchone() is not None
    
    def clear(self) -> int:
        """Clear all entries"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM cache_entries")
                return cursor.rowcount


class FilesystemStorage(CacheStorage):
    """Filesystem-based cache storage"""
    
    def __init__(self, cache_dir: str = ".guardian_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
    
    def _key_to_path(self, key: str) -> Path:
        """Convert key to filesystem path"""
        # Use first 2 chars for directory sharding
        return self.cache_dir / key[:2] / key
    
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data by key"""
        path = self._key_to_path(key)
        if not path.exists():
            return None
        
        try:
            with self._lock:
                # Check metadata for TTL
                meta_path = path.with_suffix('.meta')
                if meta_path.exists():
                    metadata = json.loads(meta_path.read_text())
                    if 'ttl_seconds' in metadata and metadata['ttl_seconds']:
                        age = time.time() - metadata['created_at']
                        if age > metadata['ttl_seconds']:
                            self.delete(key)
                            return None
                
                return path.read_bytes()
        except Exception as e:
            logger.error(f"Failed to read cache entry: {e}")
            return None
    
    def set(self, key: str, data: bytes, ttl: Optional[int] = None) -> bool:
        """Store data with optional TTL"""
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with self._lock:
                path.write_bytes(data)
                
                # Write metadata
                metadata = {
                    'created_at': time.time(),
                    'size_bytes': len(data),
                    'ttl_seconds': ttl
                }
                meta_path = path.with_suffix('.meta')
                meta_path.write_text(json.dumps(metadata))
                
            return True
        except Exception as e:
            logger.error(f"Failed to write cache entry: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete data by key"""
        path = self._key_to_path(key)
        meta_path = path.with_suffix('.meta')
        
        with self._lock:
            existed = path.exists()
            if existed:
                path.unlink(missing_ok=True)
            if meta_path.exists():
                meta_path.unlink()
            return existed
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self._key_to_path(key).exists()
    
    def clear(self) -> int:
        """Clear all entries"""
        count = 0
        with self._lock:
            for path in self.cache_dir.rglob('*'):
                if path.is_file() and not path.suffix == '.meta':
                    path.unlink()
                    count += 1
                    meta_path = path.with_suffix('.meta')
                    if meta_path.exists():
                        meta_path.unlink()
        return count


class RedisStorage(CacheStorage):
    """Redis-based cache storage"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, prefix: str = 'guardian:'):
        self.prefix = prefix
        self.client = redis.Redis(host=host, port=port, db=db)
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data by key"""
        try:
            return self.client.get(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, data: bytes, ttl: Optional[int] = None) -> bool:
        """Store data with optional TTL"""
        try:
            redis_key = self._make_key(key)
            if ttl:
                return bool(self.client.setex(redis_key, ttl, data))
            else:
                return bool(self.client.set(redis_key, data))
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete data by key"""
        try:
            return bool(self.client.delete(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(self.client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def clear(self) -> int:
        """Clear all entries"""
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0


class PickleSerializer(CacheSerializer):
    """Pickle-based serialization"""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize using pickle"""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize using pickle"""
        return pickle.loads(data)


class ZlibCompressor(CacheCompressor):
    """Zlib-based compression"""
    
    def __init__(self, level: int = 6):
        self.level = level
    
    def compress(self, data: bytes) -> bytes:
        """Compress using zlib"""
        return zlib.compress(data, self.level)
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress using zlib"""
        return zlib.decompress(data)


class ASTHashInvalidator(CacheInvalidator):
    """Invalidate based on test code AST changes"""
    
    def __init__(self, test_files: Dict[str, str]):
        """Initialize with mapping of test files to their AST hashes"""
        self.test_hashes = test_files
    
    def should_invalidate(self, entry: CacheEntry) -> bool:
        """Check if test code has changed"""
        if 'test_file' not in entry.metadata:
            return False
        
        test_file = entry.metadata['test_file']
        if test_file not in self.test_hashes:
            return True
        
        stored_hash = entry.metadata.get('test_ast_hash')
        current_hash = self.test_hashes[test_file]
        
        return stored_hash != current_hash


class TTLInvalidator(CacheInvalidator):
    """Invalidate based on time-to-live"""
    
    def should_invalidate(self, entry: CacheEntry) -> bool:
        """Check if entry has expired"""
        return entry.is_expired()


class VersionInvalidator(CacheInvalidator):
    """Invalidate based on algorithm version changes"""
    
    def __init__(self, current_version: str):
        self.current_version = current_version
    
    def should_invalidate(self, entry: CacheEntry) -> bool:
        """Check if algorithm version has changed"""
        stored_version = entry.metadata.get('algorithm_version')
        return stored_version != self.current_version


class DefaultCacheMonitor(CacheMonitor):
    """Default implementation of cache monitoring"""
    
    def __init__(self):
        self.stats = CacheStatistics()
        self._lock = threading.Lock()
    
    def record_hit(self, key: str) -> None:
        """Record cache hit"""
        with self._lock:
            self.stats.hits += 1
    
    def record_miss(self, key: str) -> None:
        """Record cache miss"""
        with self._lock:
            self.stats.misses += 1
    
    def record_eviction(self, key: str, reason: str) -> None:
        """Record cache eviction"""
        with self._lock:
            if reason not in self.stats.evictions:
                self.stats.evictions[reason] = 0
            self.stats.evictions[reason] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = self.stats.hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': hit_rate,
                'evictions': dict(self.stats.evictions),
                'total_size_bytes': self.stats.total_size_bytes,
                'entry_count': self.stats.entry_count,
                'compression_ratio': self.stats.compression_ratio,
                'last_cleared': self.stats.last_cleared.isoformat() if self.stats.last_cleared else None
            }


# ===== Open/Closed: Extensible cache manager =====

class PersistentResultCache:
    """
    Main cache manager implementing LRU eviction and coordinating all components.
    
    This class follows the Dependency Inversion Principle by depending on
    abstractions rather than concrete implementations.
    """
    
    def __init__(
        self,
        storage: CacheStorage,
        key_generator: Optional[CacheKeyGenerator] = None,
        serializer: Optional[CacheSerializer] = None,
        compressor: Optional[CacheCompressor] = None,
        monitor: Optional[CacheMonitor] = None,
        invalidators: Optional[List[CacheInvalidator]] = None,
        max_entries: int = 10000,
        max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB
        compression_threshold: int = 1024  # 1KB
    ):
        self.storage = storage
        self.key_generator = key_generator or SHA256KeyGenerator()
        self.serializer = serializer or PickleSerializer()
        self.compressor = compressor or ZlibCompressor()
        self.monitor = monitor or DefaultCacheMonitor()
        self.invalidators = invalidators or [TTLInvalidator()]
        
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self.compression_threshold = compression_threshold
        
        # LRU tracking
        self._lru = OrderedDict()
        self._lock = threading.Lock()
        self._total_size = 0
    
    def get(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Retrieve cached result"""
        metadata = metadata or {}
        key = self.key_generator.generate_key(content, metadata)
        
        # Check storage
        compressed_data = self.storage.get(key)
        if compressed_data is None:
            self.monitor.record_miss(key)
            return None
        
        # Decompress if needed
        try:
            if len(compressed_data) > 0 and compressed_data[0] == 0x78:  # zlib magic number
                data = self.compressor.decompress(compressed_data)
            else:
                data = compressed_data
            
            # Deserialize
            result = self.serializer.deserialize(data)
            
            # Update LRU
            with self._lock:
                if key in self._lru:
                    self._lru.move_to_end(key)
            
            self.monitor.record_hit(key)
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve cache entry: {e}")
            self.storage.delete(key)
            return None
    
    def set(
        self,
        content: Any,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Store result in cache"""
        metadata = metadata or {}
        key = self.key_generator.generate_key(content, metadata)
        
        try:
            # Serialize
            data = self.serializer.serialize(result)
            
            # Compress if above threshold
            if len(data) > self.compression_threshold:
                compressed_data = self.compressor.compress(data)
                compression_ratio = len(data) / len(compressed_data)
            else:
                compressed_data = data
                compression_ratio = 1.0
            
            # Check size limits and evict if necessary
            self._evict_if_needed(len(compressed_data))
            
            # Store
            success = self.storage.set(key, compressed_data, ttl)
            
            if success:
                # Update LRU and size tracking
                with self._lock:
                    self._lru[key] = {
                        'size': len(compressed_data),
                        'metadata': metadata
                    }
                    self._total_size += len(compressed_data)
                
                # Update monitor stats
                if hasattr(self.monitor, 'stats'):
                    self.monitor.stats.compression_ratio = compression_ratio
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store cache entry: {e}")
            return False
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries based on pattern or invalidators"""
        count = 0
        
        with self._lock:
            keys_to_delete = []
            
            for key, info in list(self._lru.items()):
                # Check pattern match
                if pattern and pattern not in key:
                    continue
                
                # Check invalidators
                entry = CacheEntry(
                    key=key,
                    data=b'',  # We don't need actual data for invalidation
                    created_at=datetime.fromtimestamp(info['metadata'].get('created_at', 0)),
                    accessed_at=datetime.now(),
                    metadata=info['metadata']
                )
                
                should_invalidate = any(
                    inv.should_invalidate(entry) for inv in self.invalidators
                )
                
                if should_invalidate or pattern:
                    keys_to_delete.append(key)
            
            # Delete invalidated entries
            for key in keys_to_delete:
                if self.storage.delete(key):
                    self._total_size -= self._lru[key]['size']
                    del self._lru[key]
                    count += 1
                    self.monitor.record_eviction(key, 'invalidation')
        
        return count
    
    def clear(self) -> int:
        """Clear all cache entries"""
        count = self.storage.clear()
        
        with self._lock:
            self._lru.clear()
            self._total_size = 0
        
        if hasattr(self.monitor, 'stats'):
            self.monitor.stats.last_cleared = datetime.now()
        
        return count
    
    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries if size or count limits exceeded"""
        with self._lock:
            # Evict by count
            while len(self._lru) >= self.max_entries:
                key, info = self._lru.popitem(last=False)  # Remove oldest
                self.storage.delete(key)
                self._total_size -= info['size']
                self.monitor.record_eviction(key, 'max_entries')
            
            # Evict by size
            while self._total_size + new_size > self.max_size_bytes and self._lru:
                key, info = self._lru.popitem(last=False)  # Remove oldest
                self.storage.delete(key)
                self._total_size -= info['size']
                self.monitor.record_eviction(key, 'max_size')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = self.monitor.get_statistics()
        
        with self._lock:
            stats.update({
                'current_entries': len(self._lru),
                'current_size_bytes': self._total_size,
                'max_entries': self.max_entries,
                'max_size_bytes': self.max_size_bytes,
                'usage_percent': (self._total_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0
            })
        
        return stats


# ===== Factory for easy cache creation =====

class CacheFactory:
    """Factory for creating cache instances with different configurations"""
    
    @staticmethod
    def create_sqlite_cache(
        db_path: str = ".guardian_cache.db",
        **kwargs
    ) -> PersistentResultCache:
        """Create cache with SQLite backend"""
        return PersistentResultCache(
            storage=SQLiteStorage(db_path),
            **kwargs
        )
    
    @staticmethod
    def create_filesystem_cache(
        cache_dir: str = ".guardian_cache",
        **kwargs
    ) -> PersistentResultCache:
        """Create cache with filesystem backend"""
        return PersistentResultCache(
            storage=FilesystemStorage(cache_dir),
            **kwargs
        )
    
    @staticmethod
    def create_redis_cache(
        host: str = 'localhost',
        port: int = 6379,
        **kwargs
    ) -> PersistentResultCache:
        """Create cache with Redis backend"""
        return PersistentResultCache(
            storage=RedisStorage(host, port),
            **kwargs
        )
    
    @staticmethod
    def create_memory_cache(**kwargs) -> PersistentResultCache:
        """Create in-memory cache (using SQLite :memory:)"""
        return PersistentResultCache(
            storage=SQLiteStorage(":memory:"),
            **kwargs
        )


# ===== Utility functions =====

def compute_ast_hash(code: str) -> str:
    """Compute hash of Python code AST"""
    try:
        tree = ast.parse(code)
        # Remove line numbers and other position info
        for node in ast.walk(tree):
            for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
                if hasattr(node, attr):
                    delattr(node, attr)
        
        # Convert to string representation
        ast_str = ast.dump(tree, indent=2)
        return hashlib.sha256(ast_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute AST hash: {e}")
        # Fall back to simple hash
        return hashlib.sha256(code.encode()).hexdigest()


def create_default_cache() -> PersistentResultCache:
    """Create a cache instance with sensible defaults"""
    return CacheFactory.create_sqlite_cache(
        invalidators=[
            TTLInvalidator(),
            VersionInvalidator("1.0.0")
        ],
        max_entries=10000,
        max_size_bytes=1024 * 1024 * 1024,  # 1GB
        compression_threshold=1024  # 1KB
    )