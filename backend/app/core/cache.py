"""
Simple In-Memory Cache for FitWhisperer

Provides LRU caching for measurement results and other expensive operations.
For production, consider using Redis (already in requirements).
"""

import hashlib
import time
from typing import Any, Optional, Dict
from collections import OrderedDict
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Thread-safe LRU Cache implementation.

    Features:
    - Configurable max size
    - TTL (time-to-live) support
    - Thread-safe operations
    - Hit/miss statistics
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments"""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, returns None if not found or expired"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry['timestamp'] > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry['value']

    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }

            # Evict oldest if over size limit
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate * 100, 2),
                "ttl_seconds": self.ttl_seconds
            }


# Create cache instances for different purposes
measurement_cache = LRUCache(max_size=500, ttl_seconds=300)  # 5 min TTL
size_recommendation_cache = LRUCache(max_size=1000, ttl_seconds=600)  # 10 min TTL


def cache_key_for_image(image_bytes: bytes, height_cm: Optional[float] = None) -> str:
    """
    Generate a cache key for an image.

    Uses MD5 hash of image bytes + optional height parameter.
    """
    h = hashlib.md5()
    h.update(image_bytes)
    if height_cm:
        h.update(str(height_cm).encode())
    return h.hexdigest()


def cache_key_for_recommendation(
    measurements: Dict[str, float],
    product_id: str,
    fit_preference: str = "regular"
) -> str:
    """Generate a cache key for size recommendation"""
    key_str = f"{product_id}:{fit_preference}:{sorted(measurements.items())}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_all_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches"""
    return {
        "measurement_cache": measurement_cache.stats(),
        "size_recommendation_cache": size_recommendation_cache.stats()
    }


def clear_all_caches() -> None:
    """Clear all caches"""
    measurement_cache.clear()
    size_recommendation_cache.clear()
    logger.info("All caches cleared")
