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

"""Tests for ParaM caching functionality."""

import logging
import threading
import time
import unittest

import brainstate
import jax.numpy as jnp
import numpy as np

from brainstate.nn import ParaM, IdentityT, SigmoidT, SoftplusT, Transform


class TestParamCachingBasic(unittest.TestCase):
    """Tests for basic caching functionality."""

    def test_caching_always_enabled(self):
        """Test that caching is always enabled."""
        param = ParaM(jnp.array([1.0, 2.0]))
        self.assertIsNotNone(param._cache_lock)
        # RLock type name can be '_RLock' or 'RLock' depending on Python version
        self.assertIn(type(param._cache_lock).__name__, ['_RLock', 'RLock'])
        # Cache stats should not have 'enabled' key anymore
        self.assertNotIn('enabled', param.cache_stats)

    def test_cache_miss_on_first_access(self):
        """Test cache miss on first value() access."""
        param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))
        stats_before = param.cache_stats
        self.assertFalse(stats_before['valid'])
        self.assertFalse(stats_before['has_cached_value'])

        # First access - cache miss
        value1 = param.value()

        stats_after = param.cache_stats
        self.assertTrue(stats_after['valid'])
        self.assertTrue(stats_after['has_cached_value'])
        np.testing.assert_allclose(value1, jnp.array([1.0, 2.0]), rtol=1e-5)

    def test_cache_hit_on_second_access(self):
        """Test cache hit on second value() access."""
        param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))

        value1 = param.value()
        self.assertTrue(param.cache_stats['valid'])

        # Second access - cache hit
        value2 = param.value()

        # Values should be identical
        np.testing.assert_allclose(value1, value2)
        self.assertTrue(param.cache_stats['valid'])

    def test_cache_invalidation_on_set_value(self):
        """Test cache invalidation when set_value() is called."""
        param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))

        # Populate cache
        value1 = param.value()
        self.assertTrue(param.cache_stats['valid'])

        # Update value - should invalidate cache
        param.set_value(jnp.array([3.0, 4.0]))
        self.assertFalse(param.cache_stats['valid'])

        # Next access should recompute
        value2 = param.value()
        self.assertTrue(param.cache_stats['valid'])
        np.testing.assert_allclose(value2, jnp.array([3.0, 4.0]), rtol=1e-5)

    def test_cache_invalidation_on_direct_state_write(self):
        """Test cache invalidation on direct ParamState write."""
        param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))

        # Populate cache
        value1 = param.value()
        self.assertTrue(param.cache_stats['valid'])

        # Direct state write - should trigger hook and invalidate cache
        new_unconstrained = param.t.inverse(jnp.array([3.0, 4.0]))
        param.val.value = new_unconstrained
        self.assertFalse(param.cache_stats['valid'])

        # Next access should recompute
        value2 = param.value()
        np.testing.assert_allclose(value2, jnp.array([3.0, 4.0]), rtol=1e-5)

    def test_manual_cache_clear(self):
        """Test manual cache clearing with clearCache()."""
        param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))

        # Populate cache
        param.value()
        self.assertTrue(param.cache_stats['valid'])

        # Manual clear
        param.clear_cache()
        self.assertFalse(param.cache_stats['valid'])

        # Next access should recompute
        param.value()
        self.assertTrue(param.cache_stats['valid'])

    def test_cache_stats_structure(self):
        """Test cache_stats structure."""
        param = ParaM(jnp.array([1.0, 2.0]))
        stats = param.cache_stats
        # Should have 'valid' and 'has_cached_value' keys, but not 'enabled'
        self.assertIn('valid', stats)
        self.assertIn('has_cached_value', stats)
        self.assertNotIn('enabled', stats)
        self.assertFalse(stats['valid'])
        self.assertFalse(stats['has_cached_value'])

    def test_non_trainable_param_no_hooks(self):
        """Test that non-trainable params don't register hooks."""
        param = ParaM(jnp.array([1.0, 2.0]), fit_par=False)
        # Should still cache, but no hooks (no ParamState)
        self.assertIsNotNone(param._cache_lock)
        self.assertIsNone(param._cache_invalidation_hook_handle)


class TestParamCachingWithTransforms(unittest.TestCase):
    """Tests for caching with various transforms."""

    def test_cache_with_identity_transform(self):
        """Test caching with identity transform."""
        param = ParaM(jnp.array([1.0, 2.0]), t=IdentityT())
        value1 = param.value()
        value2 = param.value()
        np.testing.assert_allclose(value1, value2)
        np.testing.assert_allclose(value1, jnp.array([1.0, 2.0]))

    def test_cache_with_sigmoid_transform(self):
        """Test caching with sigmoid transform."""
        param = ParaM(jnp.array([0.3, 0.7]), t=SigmoidT(0.0, 1.0))
        value1 = param.value()
        value2 = param.value()
        np.testing.assert_allclose(value1, value2)
        self.assertTrue(param.cache_stats['valid'])

    def test_cache_with_softplus_transform(self):
        """Test caching with softplus transform."""
        param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))
        value1 = param.value()
        value2 = param.value()
        np.testing.assert_allclose(value1, value2)


class TestParamCachingErrorHandling(unittest.TestCase):
    """Tests for caching error handling."""

    def test_transformation_error_doesnt_cache(self):
        """Test that transformation errors don't populate cache."""

        class FailingTransform(Transform):
            """Transform that always raises an error."""

            def forward(self, x):
                raise ValueError("Transformation failed")

            def inverse(self, y):
                return y

        param = ParaM(jnp.array([1.0]), t=FailingTransform())

        # First access should raise and not cache
        with self.assertRaises(ValueError):
            param.value()

        # Cache should remain invalid
        self.assertFalse(param.cache_stats['valid'])
        self.assertFalse(param.cache_stats['has_cached_value'])

        # Second access should also raise
        with self.assertRaises(ValueError):
            param.value()

    def test_successful_after_failed_transformation(self):
        """Test successful caching after fixing a failed transformation."""

        class ConditionalTransform(Transform):
            """Transform that fails based on external flag."""

            def __init__(self):
                self.should_fail = True

            def forward(self, x):
                if self.should_fail:
                    raise ValueError("Transformation failed")
                return x * 2

            def inverse(self, y):
                if self.should_fail:
                    # Make inverse also fail when should_fail is True
                    raise ValueError("Inverse transformation failed")
                return y / 2

        transform = ConditionalTransform()

        # Create param with should_fail=False first, then enable failing
        transform.should_fail = False
        param = ParaM(jnp.array([1.0]), t=transform)
        transform.should_fail = True

        # First access fails
        with self.assertRaises(ValueError):
            param.value()
        self.assertFalse(param.cache_stats['valid'])

        # Fix the transform
        transform.should_fail = False

        # Should now succeed and cache
        # Value should be the original 1.0 (goes through inverse(1.0)=0.5, then forward(0.5)=1.0)
        value = param.value()
        np.testing.assert_allclose(value, jnp.array([1.0]))
        self.assertTrue(param.cache_stats['valid'])


class TestParamCachingThreadSafety(unittest.TestCase):
    """Tests for thread safety of caching mechanism."""

    def test_concurrent_reads(self):
        """Test concurrent reads are thread-safe."""
        param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))
        num_threads = 20
        reads_per_thread = 100
        results = [None] * num_threads

        def reader_thread(thread_id):
            for _ in range(reads_per_thread):
                value = param.value()
                results[thread_id] = value

        threads = [threading.Thread(target=reader_thread, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be identical
        for result in results:
            np.testing.assert_allclose(result, results[0])

        # Cache should be valid
        self.assertTrue(param.cache_stats['valid'])

    def test_concurrent_writes(self):
        """Test concurrent writes are thread-safe."""
        param = ParaM(jnp.array([1.0]), t=SoftplusT(0.0))
        num_threads = 10
        writes_per_thread = 10

        def writer_thread(thread_id):
            for i in range(writes_per_thread):
                value = jnp.array([float(thread_id * 100 + i)])
                param.set_value(value)

        threads = [threading.Thread(target=writer_thread, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock or crash
        value = param.value()
        self.assertIsNotNone(value)

    def test_mixed_read_write(self):
        """Test mixed concurrent reads and writes."""
        param = ParaM(jnp.array([1.0]), t=SoftplusT(0.0))
        num_readers = 5
        num_writers = 3

        def reader_thread():
            for _ in range(50):
                try:
                    param.value()
                except Exception:
                    pass  # Ignore errors during concurrent access

        def writer_thread(thread_id):
            for i in range(20):
                param.set_value(jnp.array([float(thread_id * 100 + i)]))

        threads = []
        threads.extend([threading.Thread(target=reader_thread) for _ in range(num_readers)])
        threads.extend([threading.Thread(target=writer_thread, args=(i,)) for i in range(num_writers)])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
        self.assertIsNotNone(param.value())


class TestParamCachingLogging(unittest.TestCase):
    """Tests for caching logging functionality."""

    def test_logging_disabled_by_default(self):
        """Test that logging is disabled by default."""
        param = ParaM(jnp.array([1.0]))
        self.assertFalse(param._enable_cache_logging)

    def test_logging_enabled(self):
        """Test enabling cache logging."""
        param = ParaM(jnp.array([1.0]), enable_cache_logging=True)
        self.assertTrue(param._enable_cache_logging)

    def test_logger_lazy_initialization(self):
        """Test that logger is lazily initialized."""
        param = ParaM(jnp.array([1.0]), enable_cache_logging=True)
        self.assertIsNone(param._cache_logger)

        # Trigger logging
        param.value()

        # Logger should now be initialized
        self.assertIsNotNone(param._cache_logger)
        self.assertIsInstance(param._cache_logger, logging.Logger)

    def test_logging_captures_events(self):
        """Test that logging captures cache events."""
        param = ParaM(jnp.array([1.0]), t=SoftplusT(0.0), enable_cache_logging=True)

        # Trigger some cache events
        param.value()  # miss
        param.value()  # hit
        param.set_value(jnp.array([2.0]))  # invalidate
        param.clear_cache()  # manual clear

        # Logger should have been used
        self.assertIsNotNone(param._cache_logger)


class TestParamCachingPerformance(unittest.TestCase):
    """Tests for caching performance benefits."""

    def test_caching_improves_performance(self):
        """Test that caching actually improves performance for expensive transforms."""

        class SlowTransform(Transform):
            """Transform that simulates expensive computation."""

            def forward(self, x):
                time.sleep(0.01)  # Simulate expensive computation
                return x * 2

            def inverse(self, y):
                return y / 2

        param = ParaM(jnp.array([1.0]), t=SlowTransform())

        # First access - cache miss (slow)
        start = time.time()
        _ = param.value()
        first_access_time = time.time() - start

        # Subsequent accesses - cache hits (fast)
        start = time.time()
        for _ in range(100):
            _ = param.value()
        cached_time = time.time() - start

        # Cached access should be significantly faster than first access
        # (100 cached accesses should be faster than 1 uncached access)
        self.assertLess(cached_time, first_access_time)


if __name__ == '__main__':
    unittest.main()
