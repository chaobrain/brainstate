# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import jax
import jax.numpy as jnp
import pytest

import brainstate
from brainstate._compatible_import import jaxpr_as_fun
from brainstate.transform._make_jaxpr import _BoundedCache, make_hashable


class TestMakeJaxpr(unittest.TestCase):
    def test_compar_jax_make_jaxpr(self):
        def func4(arg):  # Arg is a pair
            temp = arg[0] + jnp.sin(arg[1]) * 3.
            c = brainstate.random.rand_like(arg[0])
            return jnp.sum(temp + c)

        key = brainstate.random.DEFAULT.value
        jaxpr = jax.make_jaxpr(func4)((jnp.zeros(8), jnp.ones(8)))
        print(jaxpr)
        self.assertTrue(len(jaxpr.in_avals) == 2)
        self.assertTrue(len(jaxpr.consts) == 1)
        self.assertTrue(len(jaxpr.out_avals) == 1)
        self.assertTrue(jnp.allclose(jaxpr.consts[0], key))

        brainstate.random.seed(1)
        print(brainstate.random.DEFAULT.value)

        jaxpr2, states = brainstate.transform.make_jaxpr(func4)((jnp.zeros(8), jnp.ones(8)))
        print(jaxpr2)
        self.assertTrue(len(jaxpr2.in_avals) == 3)
        self.assertTrue(len(jaxpr2.out_avals) == 2)
        self.assertTrue(len(jaxpr2.consts) == 0)
        print(brainstate.random.DEFAULT.value)

    def test_StatefulFunction_1(self):
        def func4(arg):  # Arg is a pair
            temp = arg[0] + jnp.sin(arg[1]) * 3.
            c = brainstate.random.rand_like(arg[0])
            return jnp.sum(temp + c)

        fun = brainstate.transform.StatefulFunction(func4).make_jaxpr((jnp.zeros(8), jnp.ones(8)))
        cache_key = fun.get_arg_cache_key((jnp.zeros(8), jnp.ones(8)))
        print(fun.get_states(cache_key))
        print(fun.get_jaxpr(cache_key))

    def test_StatefulFunction_2(self):
        st1 = brainstate.State(jnp.ones(10))

        def f1(x):
            st1.value = x + st1.value

        def f2(x):
            jaxpr = brainstate.transform.make_jaxpr(f1)(x)
            c = 1. + x
            return c

        def f3(x):
            jaxpr = brainstate.transform.make_jaxpr(f1)(x)
            c = 1.
            return c

        print()
        jaxpr = brainstate.transform.make_jaxpr(f1)(jnp.zeros(1))
        print(jaxpr)
        jaxpr = jax.make_jaxpr(f2)(jnp.zeros(1))
        print(jaxpr)
        jaxpr = jax.make_jaxpr(f3)(jnp.zeros(1))
        print(jaxpr)
        jaxpr, _ = brainstate.transform.make_jaxpr(f3)(jnp.zeros(1))
        print(jaxpr)
        self.assertTrue(jnp.allclose(jaxpr_as_fun(jaxpr)(jnp.zeros(1), st1.value)[0],
                                     f3(jnp.zeros(1))))

    def test_compare_jax_make_jaxpr2(self):
        st1 = brainstate.State(jnp.ones(10))

        def fa(x):
            st1.value = x + st1.value

        def ffa(x):
            jaxpr, states = brainstate.transform.make_jaxpr(fa)(x)
            c = 1. + x
            return c

        jaxpr, states = brainstate.transform.make_jaxpr(ffa)(jnp.zeros(1))
        print()
        print(jaxpr)
        print(states)
        print(jaxpr_as_fun(jaxpr)(jnp.zeros(1), st1.value))
        jaxpr = jax.make_jaxpr(ffa)(jnp.zeros(1))
        print(jaxpr)
        print(jaxpr_as_fun(jaxpr)(jnp.zeros(1)))

    def test_compare_jax_make_jaxpr3(self):
        def fa(x):
            return 1.

        jaxpr, states = brainstate.transform.make_jaxpr(fa)(jnp.zeros(1))
        print()
        print(jaxpr)
        print(states)
        # print(jaxpr_as_fun(jaxpr)(jnp.zeros(1)))
        jaxpr = jax.make_jaxpr(fa)(jnp.zeros(1))
        print(jaxpr)
        # print(jaxpr_as_fun(jaxpr)(jnp.zeros(1)))

    def test_static_argnames(self):
        def func4(a, b):  # Arg is a pair
            temp = a + jnp.sin(b) * 3.
            c = brainstate.random.rand_like(a)
            return jnp.sum(temp + c)

        jaxpr, states = brainstate.transform.make_jaxpr(func4, static_argnames='b')(jnp.zeros(8), 1.)
        print()
        print(jaxpr)
        print(states)

    def test_state_in(self):
        def f(a):
            return a.value

        with pytest.raises(ValueError):
            brainstate.transform.StatefulFunction(f).make_jaxpr(brainstate.State(1.))

    def test_state_out(self):
        def f(a):
            return brainstate.State(a)

        with pytest.raises(ValueError):
            brainstate.transform.StatefulFunction(f).make_jaxpr(1.)

    def test_return_states(self):
        a = brainstate.State(jnp.ones(3))

        @brainstate.transform.jit
        def f():
            return a

        with pytest.raises(ValueError):
            f()


class TestBoundedCache(unittest.TestCase):
    """Test the _BoundedCache class."""

    def test_cache_basic_operations(self):
        """Test basic get and set operations."""
        cache = _BoundedCache(maxsize=3)

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
        cache = _BoundedCache(maxsize=3)

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
        cache = _BoundedCache(maxsize=2)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        # Update key1 (should move it to end)
        cache.set('key1', 'updated_value1')
        self.assertEqual(cache.get('key1'), 'updated_value1')

        # Add new key, should evict key2 (now LRU)
        cache.set('key3', 'value3')
        self.assertNotIn('key2', cache)
        self.assertIn('key1', cache)

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = _BoundedCache(maxsize=5)

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
        cache = _BoundedCache(maxsize=5)

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
        cache = _BoundedCache(maxsize=5)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')

        keys = cache.keys()
        self.assertEqual(set(keys), {'key1', 'key2', 'key3'})

    def test_cache_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = _BoundedCache(maxsize=100)
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


class TestStatefulFunctionEnhancements(unittest.TestCase):
    """Test enhancements to StatefulFunction class."""

    def test_cache_stats(self):
        """Test get_cache_stats method."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value * 2

        sf = brainstate.transform.StatefulFunction(f)

        # Compile for different inputs
        x1 = jnp.array([0.5, 0.5])
        x2 = jnp.array([1.0, 1.0])

        sf.make_jaxpr(x1)
        sf.make_jaxpr(x2)

        # Get cache stats
        stats = sf.get_cache_stats()

        # Verify all cache types are present
        self.assertIn('jaxpr_cache', stats)
        self.assertIn('out_shapes_cache', stats)
        self.assertIn('jaxpr_out_tree_cache', stats)
        self.assertIn('state_trace_cache', stats)

        # Verify each cache has proper stats
        for cache_name, cache_stats in stats.items():
            self.assertIn('size', cache_stats)
            self.assertIn('maxsize', cache_stats)
            self.assertIn('hits', cache_stats)
            self.assertIn('misses', cache_stats)
            self.assertIn('hit_rate', cache_stats)

    def test_validate_states(self):
        """Test validate_states method."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([0.5, 0.5])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)

        # Should validate successfully
        result = sf.validate_states(cache_key)
        self.assertTrue(result)

    def test_validate_all_states(self):
        """Test validate_all_states method."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x, n):
            state.value += x
            return state.value * n

        # Use static_argnums to create different cache keys
        sf = brainstate.transform.StatefulFunction(f, static_argnums=(1,))

        # Compile for multiple inputs with different static args
        x = jnp.array([0.5, 0.5])

        sf.make_jaxpr(x, 1)
        sf.make_jaxpr(x, 2)

        # Validate all
        results = sf.validate_all_states()

        # Should have results for both cache keys
        self.assertEqual(len(results), 2)

        # All should be valid
        for result in results.values():
            self.assertTrue(result)

    def test_clear_cache(self):
        """Test clear_cache method."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([0.5, 0.5])
        sf.make_jaxpr(x)

        # Verify cache has entries
        stats = sf.get_cache_stats()
        self.assertGreater(stats['jaxpr_cache']['size'], 0)

        # Clear cache
        sf.clear_cache()

        # Verify all caches are empty
        stats = sf.get_cache_stats()
        self.assertEqual(stats['jaxpr_cache']['size'], 0)
        self.assertEqual(stats['out_shapes_cache']['size'], 0)
        self.assertEqual(stats['jaxpr_out_tree_cache']['size'], 0)
        self.assertEqual(stats['state_trace_cache']['size'], 0)

    def test_return_only_write_parameter(self):
        """Test return_only_write parameter."""
        read_state = brainstate.State(jnp.array([1.0, 2.0]))
        write_state = brainstate.State(jnp.array([3.0, 4.0]))

        def f(x):
            # Read from read_state, write to write_state
            _ = read_state.value + x
            write_state.value += x
            return write_state.value

        # Test with return_only_write=False (default)
        sf_all = brainstate.transform.StatefulFunction(f, return_only_write=False)
        sf_all.make_jaxpr(jnp.array([0.5, 0.5]))
        cache_key = sf_all.get_arg_cache_key(jnp.array([0.5, 0.5]))
        states_all = sf_all.get_states(cache_key)

        # Test with return_only_write=True
        sf_write_only = brainstate.transform.StatefulFunction(f, return_only_write=True)
        sf_write_only.make_jaxpr(jnp.array([0.5, 0.5]))
        cache_key_write = sf_write_only.get_arg_cache_key(jnp.array([0.5, 0.5]))
        states_write = sf_write_only.get_states(cache_key_write)

        # With return_only_write=True, should have fewer or equal states
        self.assertLessEqual(len(states_write), len(states_all))


class TestErrorHandling(unittest.TestCase):
    """Test error handling in StatefulFunction."""

    def test_get_jaxpr_not_compiled(self):
        """Test error when getting jaxpr for uncompiled function."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        # Create a fake cache key
        fake_key = ('fake', 'key')

        with pytest.raises(ValueError, match="Function not compiled"):
            sf.get_jaxpr(fake_key)

    def test_get_out_shapes_not_compiled(self):
        """Test error when getting shapes for uncompiled function."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)
        fake_key = ('fake', 'key')

        with pytest.raises(ValueError, match="Function not compiled"):
            sf.get_out_shapes(fake_key)

    def test_jaxpr_call_state_mismatch(self):
        """Test error when state values length doesn't match."""
        state1 = brainstate.State(jnp.array([1.0, 2.0]))
        state2 = brainstate.State(jnp.array([3.0, 4.0]))

        def f(x):
            state1.value += x
            state2.value += x
            return state1.value + state2.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([0.5, 0.5])
        sf.make_jaxpr(x)

        # Try to call with wrong number of state values (only 1 instead of 2)
        with pytest.raises(ValueError, match="State length mismatch"):
            sf.jaxpr_call([jnp.array([1.0, 1.0])], x)  # Only 1 state instead of 2


class TestMakeHashable(unittest.TestCase):
    """Test the make_hashable utility function."""

    def test_hashable_list(self):
        """Test converting list to hashable."""
        result = make_hashable([1, 2, 3])
        # Should return a tuple
        self.assertIsInstance(result, tuple)
        # Should be hashable
        hash(result)

    def test_hashable_dict(self):
        """Test converting dict to hashable."""
        result = make_hashable({'b': 2, 'a': 1})
        # Should return a tuple of sorted key-value pairs
        self.assertIsInstance(result, tuple)
        # Should be hashable
        hash(result)
        # Keys should be sorted
        keys = [item[0] for item in result]
        self.assertEqual(keys, ['a', 'b'])

    def test_hashable_set(self):
        """Test converting set to hashable."""
        result = make_hashable({1, 2, 3})
        # Should return a frozenset
        self.assertIsInstance(result, frozenset)
        # Should be hashable
        hash(result)

    def test_hashable_nested(self):
        """Test converting nested structures."""
        nested = {
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'set': {4, 5}
        }
        result = make_hashable(nested)
        # Should be hashable
        hash(result)  # Should not raise

    def test_hashable_tuple(self):
        """Test with tuples."""
        result = make_hashable((1, 2, 3))
        # Should return a tuple
        self.assertIsInstance(result, tuple)
        # Should be hashable
        hash(result)

    def test_hashable_idempotent(self):
        """Test that applying make_hashable twice gives consistent results."""
        original = {'a': [1, 2], 'b': {3, 4}}
        result1 = make_hashable(original)
        result2 = make_hashable(original)
        # Should be the same
        self.assertEqual(result1, result2)


class TestMakeJaxprReturnOnlyWrite(unittest.TestCase):
    """Test make_jaxpr with return_only_write parameter."""

    def test_make_jaxpr_return_only_write(self):
        """Test make_jaxpr function with return_only_write parameter."""
        read_state = brainstate.State(jnp.array([1.0]))
        write_state = brainstate.State(jnp.array([2.0]))

        def f(x):
            _ = read_state.value  # Read only
            write_state.value += x  # Write
            return x * 2

        # Test with return_only_write=True
        jaxpr_maker = brainstate.transform.make_jaxpr(f, return_only_write=True)
        jaxpr, states = jaxpr_maker(jnp.array([1.0]))

        # Should compile successfully
        self.assertIsNotNone(jaxpr)
        self.assertIsInstance(states, tuple)
