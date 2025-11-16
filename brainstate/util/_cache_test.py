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
import unittest

import pytest

from brainstate.util._cache import BoundedCache


class TestBoundedCache(unittest.TestCase):
    """Test the BoundedCache class."""

    def test_cache_basic_operations(self):
        """Test basic get and set operations."""
        cache = BoundedCache(maxsize=3)

        # Test set and get
        cache.set('key1', 'value1')
        self.assertEqual(cache.get('key1'), 'value1')

        # Test default value
        self.assertIsNone(cache.get('nonexistent'))
        self.assertEqual(cache.get('nonexistent', 'default'), 'default')

        # Test __contains__
        self.assertIn('key1', cache)
        self.assertNotIn('key2', cache)

        # Test __len__
        self.assertEqual(len(cache), 1)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = BoundedCache(maxsize=3)

        # Fill cache
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')
        self.assertEqual(len(cache), 3)

        # Add one more, should evict key1 (least recently used)
        cache.set('key4', 'value4')
        self.assertEqual(len(cache), 3)
        self.assertNotIn('key1', cache)
        self.assertIn('key4', cache)

        # Access key2 to make it recently used
        cache.get('key2')

        # Add another key, should evict key3 (now least recently used)
        cache.set('key5', 'value5')
        self.assertNotIn('key3', cache)
        self.assertIn('key2', cache)

    def test_cache_update_existing(self):
        """Test updating an existing key."""
        cache = BoundedCache(maxsize=2)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        # Update key1 (should move it to end)
        cache.replace('key1', 'updated_value1')
        self.assertEqual(cache.get('key1'), 'updated_value1')

        # Add new key, should evict key2 (now LRU)
        cache.set('key3', 'value3')
        self.assertNotIn('key2', cache)
        self.assertIn('key1', cache)

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = BoundedCache(maxsize=5)

        # Initial stats
        stats = cache.get_stats()
        self.assertEqual(stats['size'], 0)
        self.assertEqual(stats['maxsize'], 5)
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['hit_rate'], 0.0)

        # Add items and test hits/misses
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        # Generate hits
        cache.get('key1')  # hit
        cache.get('key1')  # hit
        cache.get('key3')  # miss
        cache.get('key2')  # hit

        stats = cache.get_stats()
        self.assertEqual(stats['size'], 2)
        self.assertEqual(stats['hits'], 3)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hit_rate'], 75.0)

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = BoundedCache(maxsize=5)

        # Add items
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.get('key1')  # Generate a hit

        # Clear cache
        cache.clear()

        self.assertEqual(len(cache), 0)
        self.assertNotIn('key1', cache)

        # Check stats are reset
        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)

    def test_cache_keys(self):
        """Test getting all cache keys."""
        cache = BoundedCache(maxsize=5)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')

        keys = cache.keys()
        self.assertEqual(set(keys), {'key1', 'key2', 'key3'})

    def test_cache_set_duplicate_raises(self):
        """Test that setting an existing key raises ValueError."""
        cache = BoundedCache(maxsize=5)

        cache.set('key1', 'value1')

        # Attempting to set the same key should raise ValueError
        with pytest.raises(ValueError, match="Cache key already exists"):
            cache.set('key1', 'value2')

    def test_cache_pop(self):
        """Test pop method."""
        cache = BoundedCache(maxsize=5)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        # Pop existing key
        value = cache.pop('key1')
        self.assertEqual(value, 'value1')
        self.assertNotIn('key1', cache)
        self.assertEqual(len(cache), 1)

        # Pop non-existent key with default
        value = cache.pop('nonexistent', 'default')
        self.assertEqual(value, 'default')

        # Pop non-existent key without default
        value = cache.pop('nonexistent')
        self.assertIsNone(value)

    def test_cache_replace(self):
        """Test replace method."""
        cache = BoundedCache(maxsize=5)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        # Replace existing key
        cache.replace('key1', 'new_value1')
        self.assertEqual(cache.get('key1'), 'new_value1')

        # Replacing should move to end (most recently used)
        cache.set('key3', 'value3')
        cache.replace('key2', 'new_value2')

        # Add more items to test LRU behavior
        cache.set('key4', 'value4')
        cache.set('key5', 'value5')

        # Now when we add key6, key1 should be evicted (oldest after replace moved key2 to end)
        cache.set('key6', 'value6')

        # key2 should still be there because replace moved it to end
        self.assertIn('key2', cache)

    def test_cache_replace_nonexistent_raises(self):
        """Test that replacing a non-existent key raises KeyError."""
        cache = BoundedCache(maxsize=5)

        with pytest.raises(KeyError, match="Cache key does not exist"):
            cache.replace('nonexistent', 'value')

    def test_cache_get_with_raise_on_miss(self):
        """Test get method with raise_on_miss parameter."""
        cache = BoundedCache(maxsize=5)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        # Should work normally for existing key
        value = cache.get('key1', raise_on_miss=True)
        self.assertEqual(value, 'value1')

        # Should raise ValueError for missing key with raise_on_miss=True
        with pytest.raises(ValueError, match="not compiled for the requested cache key"):
            cache.get('nonexistent', raise_on_miss=True, error_context="Test item")

    def test_cache_detailed_error_message(self):
        """Test that error message shows available keys."""
        cache = BoundedCache(maxsize=5)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        # Error should include all available keys
        with pytest.raises(ValueError) as exc_info:
            cache.get('nonexistent', raise_on_miss=True, error_context="Test item")

        error_msg = str(exc_info.value)
        # Should show requested key
        self.assertIn('nonexistent', error_msg)
        # Should show available keys
        self.assertIn('key1', error_msg)
        self.assertIn('key2', error_msg)
        # Should have helpful message
        self.assertIn('make_jaxpr()', error_msg)

    def test_cache_error_message_no_keys(self):
        """Test error message when cache is empty."""
        cache = BoundedCache(maxsize=5)

        with pytest.raises(ValueError) as exc_info:
            cache.get('key', raise_on_miss=True, error_context="Empty cache")

        error_msg = str(exc_info.value)
        # Should indicate no keys available
        self.assertIn('none', error_msg.lower())

    def test_cache_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = BoundedCache(maxsize=100)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    key = f'key_{thread_id}_{i}'
                    cache.set(key, f'value_{thread_id}_{i}')
                    value = cache.get(key)
                    if value != f'value_{thread_id}_{i}':
                        errors.append(f'Mismatch in thread {thread_id}')
            except Exception as e:
                errors.append(f'Error in thread {thread_id}: {e}')

        # Create multiple threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check no errors occurred
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
