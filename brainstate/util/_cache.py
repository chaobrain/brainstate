# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import threading
from collections import OrderedDict
from typing import Any, Dict

__all__ = [
    'BoundedCache',
]

class BoundedCache:
    """
    A thread-safe LRU cache with bounded size.

    This cache stores a limited number of items and evicts the least recently used item
    when the cache reaches its maximum size. All operations are thread-safe.

    Parameters
    ----------
    maxsize : int, default 128
        Maximum number of items to store in the cache.
    """

    def __init__(self, maxsize: int = 128):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @property
    def maxsize(self) -> int:
        """Get the maximum size of the cache."""
        return self._maxsize

    def get(
        self,
        key: Any,
        default: Any = None,
        raise_on_miss: bool = False,
        error_context: str = "item"
    ) -> Any:
        """
        Get an item from the cache.

        Parameters
        ----------
        key : Any
            The cache key.
        default : Any, optional
            The default value to return if the key is not found.
        raise_on_miss : bool, optional
            If True, raise a detailed ValueError when the key is not found.
        error_context : str, optional
            Context description for the error message (e.g., "Function", "JAX expression").

        Returns
        -------
        Any
            The cached value or the default value.

        Raises
        ------
        ValueError
            If raise_on_miss is True and the key is not found.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1

            if raise_on_miss:
                available_keys = list(self._cache.keys())
                error_msg = [
                    f"{error_context} not compiled for the requested cache key.",
                    f"",
                    f"Requested key:",
                    f"  {key}",
                    f"",
                    f"Available {{len(available_keys)}} keys:",
                ]
                if available_keys:
                    for i, k in enumerate(available_keys, 1):
                        error_msg.append(f"  [{i}] {k}")
                else:
                    error_msg.append("  (none - not compiled yet)")
                error_msg.append("")
                error_msg.append("Call make_jaxpr() first with matching arguments.")
                raise ValueError("\n".join(error_msg))

            return default

    def set(self, key: Any, value: Any) -> None:
        """
        Set an item in the cache.

        Parameters
        ----------
        key : Any
            The cache key.
        value : Any
            The value to cache.

        Raises
        ------
        ValueError
            If the key already exists in the cache.
        """
        with self._lock:
            if key in self._cache:
                raise ValueError(
                    f"Cache key already exists: {key}. "
                    f"Cannot overwrite existing cached value. "
                    f"Clear the cache first if you need to recompile."
                )
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def pop(self, key: Any, default: Any = None) -> Any:
        """
        Remove and return an item from the cache.

        Parameters
        ----------
        key : Any
            The cache key to remove.
        default : Any, optional
            The default value to return if the key is not found.

        Returns
        -------
        Any
            The cached value or the default value if the key is not found.
        """
        with self._lock:
            if key in self._cache:
                return self._cache.pop(key)
            return default

    def replace(self, key: Any, value: Any) -> None:
        """
        Replace an existing item in the cache.

        Parameters
        ----------
        key : Any
            The cache key to replace.
        value : Any
            The new value to cache.

        Raises
        ------
        KeyError
            If the key does not exist in the cache.
        """
        with self._lock:
            if key not in self._cache:
                raise KeyError(
                    f"Cache key does not exist: {key}. "
                    f"Cannot replace non-existent cached value. "
                    f"Use set() to add a new cache entry."
                )
            self._cache[key] = value
            self._cache.move_to_end(key)

    def __contains__(self, key: Any) -> bool:
        """
        Check if a key exists in the cache.

        Parameters
        ----------
        key : Any
            The cache key to check.

        Returns
        -------
        bool
            True if the key exists in the cache, False otherwise.
        """
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """
        Get the number of items in the cache.

        Returns
        -------
        int
            The number of items currently in the cache.
        """
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """
        Clear all items from the cache and reset statistics.

        This method removes all cached items and resets hit/miss counters to zero.
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def keys(self):
        """
        Return all keys in the cache.

        Returns
        -------
        list
            A list of all keys currently in the cache.
        """
        with self._lock:
            return list(self._cache.keys())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            A dictionary with cache statistics including:

            - 'size': Current number of items in cache
            - 'maxsize': Maximum cache size
            - 'hits': Number of cache hits
            - 'misses': Number of cache misses
            - 'hit_rate': Hit rate percentage (0-100)
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'maxsize': self._maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
            }
