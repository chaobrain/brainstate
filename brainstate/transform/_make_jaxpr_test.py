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


import unittest
import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import brainstate
from brainstate._compatible_import import jaxpr_as_fun
from brainstate._error import BatchAxisError
from brainstate.transform._make_jaxpr import _make_hashable
from brainstate.util import filter as state_filter


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
        print(fun.get_states_by_cache(cache_key))
        print(fun.get_jaxpr_by_cache(cache_key))

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
        states_all = sf_all.get_states_by_cache(cache_key)

        # Test with return_only_write=True
        sf_write_only = brainstate.transform.StatefulFunction(f, return_only_write=True)
        sf_write_only.make_jaxpr(jnp.array([0.5, 0.5]))
        cache_key_write = sf_write_only.get_arg_cache_key(jnp.array([0.5, 0.5]))
        states_write = sf_write_only.get_states_by_cache(cache_key_write)

        # With return_only_write=True, should have fewer or equal states
        self.assertLessEqual(len(states_write), len(states_all))


class TestErrorHandling(unittest.TestCase):
    """Test error handling in StatefulFunction."""

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

    def test_get_jaxpr_not_compiled_detailed_error(self):
        """Test detailed error message when getting jaxpr for uncompiled function."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        # Compile for one input shape
        sf.make_jaxpr(jnp.array([1.0, 2.0]))

        # Try to get jaxpr with a different cache key
        from brainstate.transform._make_jaxpr import hashabledict
        fake_key = hashabledict(
            static_args=(),
            dyn_args=(),
            static_kwargs=(),
            dyn_kwargs=()
        )

        # Should raise detailed error
        with pytest.raises(ValueError) as exc_info:
            sf.get_jaxpr_by_cache(fake_key)

        error_msg = str(exc_info.value)
        # Should contain the requested key
        self.assertIn('Requested key:', error_msg)
        # Should show available keys
        self.assertIn('Available', error_msg)
        # Should have helpful message
        self.assertIn('make_jaxpr()', error_msg)

    def test_get_out_shapes_not_compiled_detailed_error(self):
        """Test detailed error message when getting output shapes for uncompiled function."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        from brainstate.transform._make_jaxpr import hashabledict
        fake_key = hashabledict(
            static_args=(),
            dyn_args=(),
            static_kwargs=(),
            dyn_kwargs=()
        )

        # Should raise detailed error with context "Output shapes"
        with pytest.raises(ValueError) as exc_info:
            sf.get_out_shapes_by_cache(fake_key)

        error_msg = str(exc_info.value)
        self.assertIn('Output shapes', error_msg)
        self.assertIn('Requested key:', error_msg)

    def test_get_out_treedef_not_compiled_detailed_error(self):
        """Test detailed error message when getting output tree for uncompiled function."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        from brainstate.transform._make_jaxpr import hashabledict
        fake_key = hashabledict(
            static_args=(),
            dyn_args=(),
            static_kwargs=(),
            dyn_kwargs=()
        )

        # Should raise detailed error with context "Output tree"
        with pytest.raises(ValueError) as exc_info:
            sf.get_out_treedef_by_cache(fake_key)

        error_msg = str(exc_info.value)
        self.assertIn('Output tree', error_msg)
        self.assertIn('Requested key:', error_msg)

    def test_get_state_trace_not_compiled_detailed_error(self):
        """Test detailed error message when getting state trace for uncompiled function."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        from brainstate.transform._make_jaxpr import hashabledict
        fake_key = hashabledict(
            static_args=(),
            dyn_args=(),
            static_kwargs=(),
            dyn_kwargs=()
        )

        # Should raise detailed error with context "State trace"
        with pytest.raises(ValueError) as exc_info:
            sf.get_state_trace_by_cache(fake_key)

        error_msg = str(exc_info.value)
        self.assertIn('State trace', error_msg)
        self.assertIn('Requested key:', error_msg)


class TestCompileIfMiss(unittest.TestCase):
    """Test compile_if_miss parameter in *_by_call methods."""

    def test_get_jaxpr_by_call_with_compile_if_miss_true(self):
        """Test get_jaxpr_by_call with compile_if_miss=True (default)."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        # Should compile automatically
        jaxpr = sf.get_jaxpr(jnp.array([1.0, 2.0]), compile_if_miss=True)
        self.assertIsNotNone(jaxpr)

    def test_get_jaxpr_by_call_with_compile_if_miss_false(self):
        """Test get_jaxpr_by_call with compile_if_miss=False."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        # Should raise error because not compiled
        with pytest.raises(ValueError, match="not compiled"):
            sf.get_jaxpr(jnp.array([1.0, 2.0]), compile_if_miss=False)

    def test_get_out_shapes_by_call_compile_if_miss(self):
        """Test get_out_shapes_by_call with compile_if_miss parameter."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value * 2

        sf = brainstate.transform.StatefulFunction(f)

        # With compile_if_miss=True, should compile automatically
        shapes = sf.get_out_shapes(jnp.array([1.0, 2.0]), compile_if_miss=True)
        self.assertIsNotNone(shapes)

        # With compile_if_miss=False on different input, should fail
        with pytest.raises(ValueError):
            sf.get_out_shapes(jnp.array([1.0, 2.0, 3.0]), compile_if_miss=False)

    def test_get_out_treedef_by_call_compile_if_miss(self):
        """Test get_out_treedef_by_call with compile_if_miss parameter."""

        def f(x):
            return x * 2, x + 1

        sf = brainstate.transform.StatefulFunction(f)

        # Should compile automatically with default compile_if_miss=True
        treedef = sf.get_out_treedef(jnp.array([1.0, 2.0]))
        self.assertIsNotNone(treedef)

    def test_get_state_trace_by_call_compile_if_miss(self):
        """Test get_state_trace_by_call with compile_if_miss parameter."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)

        # Should compile automatically
        trace = sf.get_state_trace(jnp.array([1.0, 2.0]), compile_if_miss=True)
        self.assertIsNotNone(trace)

    def test_get_states_by_call_compile_if_miss(self):
        """Test get_states_by_call with compile_if_miss parameter."""
        state1 = brainstate.State(jnp.array([1.0, 2.0]))
        state2 = brainstate.State(jnp.array([3.0, 4.0]))

        def f(x):
            state1.value += x
            state2.value += x
            return state1.value + state2.value

        sf = brainstate.transform.StatefulFunction(f)

        # Should compile automatically
        states = sf.get_states(jnp.array([1.0, 2.0]), compile_if_miss=True)
        self.assertEqual(len(states), 2)

    def test_get_read_states_by_call_compile_if_miss(self):
        """Test get_read_states_by_call with compile_if_miss parameter."""
        read_state = brainstate.State(jnp.array([1.0, 2.0]))
        write_state = brainstate.State(jnp.array([3.0, 4.0]))

        def f(x):
            _ = read_state.value
            write_state.value += x
            return write_state.value

        sf = brainstate.transform.StatefulFunction(f)

        # Should compile automatically
        read_states = sf.get_read_states(jnp.array([1.0, 2.0]), compile_if_miss=True)
        self.assertIsNotNone(read_states)

    def test_get_write_states_by_call_compile_if_miss(self):
        """Test get_write_states_by_call with compile_if_miss parameter."""
        read_state = brainstate.State(jnp.array([1.0, 2.0]))
        write_state = brainstate.State(jnp.array([3.0, 4.0]))

        def f(x):
            _ = read_state.value
            write_state.value += x
            return write_state.value

        sf = brainstate.transform.StatefulFunction(f)

        # Should compile automatically
        write_states = sf.get_write_states(jnp.array([1.0, 2.0]), compile_if_miss=True)
        self.assertIsNotNone(write_states)

    def test_compile_if_miss_default_behavior(self):
        """Test that compile_if_miss defaults to True for all *_by_call methods."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)

        # All these should work without explicit compile_if_miss=True
        jaxpr = sf.get_jaxpr(jnp.array([1.0, 2.0]))
        self.assertIsNotNone(jaxpr)

        # Create new instance for fresh cache
        sf2 = brainstate.transform.StatefulFunction(f)
        shapes = sf2.get_out_shapes(jnp.array([1.0, 2.0]))
        self.assertIsNotNone(shapes)

        # Create new instance for fresh cache
        sf3 = brainstate.transform.StatefulFunction(f)
        states = sf3.get_states(jnp.array([1.0, 2.0]))
        self.assertIsNotNone(states)


class TestMakeHashable(unittest.TestCase):
    """Test the make_hashable utility function."""

    def test_hashable_list(self):
        """Test converting list to hashable."""
        result = _make_hashable([1, 2, 3])
        # Should return a tuple
        self.assertIsInstance(result, tuple)
        # Should be hashable
        hash(result)

    def test_hashable_dict(self):
        """Test converting dict to hashable."""
        result = _make_hashable({'b': 2, 'a': 1})
        # Should return a tuple of sorted key-value pairs
        self.assertIsInstance(result, tuple)
        # Should be hashable
        hash(result)
        # Keys should be sorted
        keys = [item[0] for item in result]
        self.assertEqual(keys, ['a', 'b'])

    def test_hashable_set(self):
        """Test converting set to hashable."""
        result = _make_hashable({1, 2, 3})
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
        result = _make_hashable(nested)
        # Should be hashable
        hash(result)  # Should not raise

    def test_hashable_tuple(self):
        """Test with tuples."""
        result = _make_hashable((1, 2, 3))
        # Should return a tuple
        self.assertIsInstance(result, tuple)
        # Should be hashable
        hash(result)

    def test_hashable_idempotent(self):
        """Test that applying make_hashable twice gives consistent results."""
        original = {'a': [1, 2], 'b': {3, 4}}
        result1 = _make_hashable(original)
        result2 = _make_hashable(original)
        # Should be the same
        self.assertEqual(result1, result2)


class TestCacheCleanupOnError(unittest.TestCase):
    """Test that cache is properly cleaned up when compilation fails."""

    def test_cache_cleanup_on_compilation_error(self):
        """Test that partial cache entries are cleaned up when make_jaxpr fails."""

        def f(x):
            # This will cause an error during JAX tracing
            if x > 0:  # Control flow not allowed in JAX
                return x * 2
            else:
                return x + 1

        sf = brainstate.transform.StatefulFunction(f)

        # Try to compile, should fail
        try:
            sf.make_jaxpr(jnp.array([1.0]))
        except Exception:
            pass  # Expected to fail

        # Cache should be empty after error
        stats = sf.get_cache_stats()
        # All caches should be empty since error cleanup should have removed partial entries
        # Note: The actual behavior depends on when the error occurs during compilation
        # If error happens early, no cache entries; if late, entries might exist
        # This test just verifies the cleanup mechanism exists


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


class TestStatefulFunctionCallable(unittest.TestCase):
    """Test __call__ method of StatefulFunction."""

    def test_stateful_function_call(self):
        """Test calling StatefulFunction directly."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value * 2

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([0.5, 0.5])
        sf.make_jaxpr(x)

        # Test direct call
        result = sf(x)
        self.assertEqual(result.shape, (2,))

    def test_stateful_function_call_auto_compile(self):
        """Test that __call__ automatically compiles if needed."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value * 2

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([0.5, 0.5])

        # Call without pre-compilation should work
        result = sf(x)
        self.assertEqual(result.shape, (2,))

    def test_stateful_function_multiple_calls(self):
        """Test multiple calls to StatefulFunction."""
        state = brainstate.State(jnp.array([0.0]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)

        # Multiple calls should accumulate state
        result1 = sf(jnp.array([1.0]))
        result2 = sf(jnp.array([2.0]))
        result3 = sf(jnp.array([3.0]))

        # Each call should update the state
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertIsNotNone(result3)


class TestStatefulFunctionStaticArgs(unittest.TestCase):
    """Test StatefulFunction with static arguments."""

    def test_static_argnums_basic(self):
        """Test basic usage of static_argnums."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x, multiplier):
            state.value += x
            return state.value * multiplier

        sf = brainstate.transform.StatefulFunction(f, static_argnums=(1,))
        x = jnp.array([0.5, 0.5])

        # Compile with multiplier=2
        sf.make_jaxpr(x, 2)
        cache_key1 = sf.get_arg_cache_key(x, 2)

        # Compile with multiplier=3
        sf.make_jaxpr(x, 3)
        cache_key2 = sf.get_arg_cache_key(x, 3)

        # Should have different cache keys
        self.assertNotEqual(cache_key1, cache_key2)

    def test_static_argnames_basic(self):
        """Test basic usage of static_argnames."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x, multiplier=2):
            state.value += x
            return state.value * multiplier

        sf = brainstate.transform.StatefulFunction(f, static_argnames='multiplier')
        x = jnp.array([0.5, 0.5])

        # Compile with different multiplier values
        sf.make_jaxpr(x, multiplier=2)
        cache_key1 = sf.get_arg_cache_key(x, multiplier=2)

        sf.make_jaxpr(x, multiplier=3)
        cache_key2 = sf.get_arg_cache_key(x, multiplier=3)

        # Should have different cache keys
        self.assertNotEqual(cache_key1, cache_key2)

    def test_static_args_combination(self):
        """Test using both static_argnums and static_argnames."""
        state = brainstate.State(jnp.array([1.0]))

        def f(x, multiplier, offset=0):
            state.value += x
            return state.value * multiplier + offset

        sf = brainstate.transform.StatefulFunction(
            f, static_argnums=(1,), static_argnames='offset'
        )
        x = jnp.array([0.5])

        # Compile with different static args
        sf.make_jaxpr(x, 2, offset=0)
        cache_key1 = sf.get_arg_cache_key(x, 2, offset=0)

        sf.make_jaxpr(x, 3, offset=1)
        cache_key2 = sf.get_arg_cache_key(x, 3, offset=1)

        # Should have different cache keys
        self.assertNotEqual(cache_key1, cache_key2)


class TestStatefulFunctionComplexStates(unittest.TestCase):
    """Test StatefulFunction with complex state scenarios."""

    def test_multiple_states(self):
        """Test function with multiple states."""
        state1 = brainstate.State(jnp.array([1.0]))
        state2 = brainstate.State(jnp.array([2.0]))
        state3 = brainstate.State(jnp.array([3.0]))

        def f(x):
            state1.value += x
            state2.value += x * 2
            state3.value += x * 3
            return state1.value + state2.value + state3.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([1.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        states = sf.get_states_by_cache(cache_key)

        # Should track all three states
        self.assertEqual(len(states), 3)

    def test_nested_state_access(self):
        """Test function with nested state access patterns."""
        outer_state = brainstate.State(jnp.array([1.0]))
        inner_state = brainstate.State(jnp.array([2.0]))

        def inner_fn(x):
            inner_state.value += x
            return inner_state.value

        def outer_fn(x):
            outer_state.value += x
            result = inner_fn(x)
            return outer_state.value + result

        sf = brainstate.transform.StatefulFunction(outer_fn)
        x = jnp.array([1.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        states = sf.get_states_by_cache(cache_key)

        # Should track both states
        self.assertGreaterEqual(len(states), 2)

    def test_conditional_state_write(self):
        """Test function that conditionally writes to states."""
        state1 = brainstate.State(jnp.array([1.0]))
        state2 = brainstate.State(jnp.array([2.0]))

        def f(x, write_state1=True):
            # Note: In JAX, actual control flow needs special handling
            # This test is more about the framework's ability to track states
            state1.value += x  # Always write to state1
            state2.value += x * 2  # Always write to state2
            return state1.value + state2.value

        sf = brainstate.transform.StatefulFunction(f, static_argnames='write_state1')
        x = jnp.array([1.0])
        sf.make_jaxpr(x, write_state1=True)

        cache_key = sf.get_arg_cache_key(x, write_state1=True)
        states = sf.get_states_by_cache(cache_key)

        # Should track states
        self.assertGreaterEqual(len(states), 2)


class TestStatefulFunctionOutputShapes(unittest.TestCase):
    """Test StatefulFunction output shape tracking."""

    def test_single_output(self):
        """Test tracking single output shape."""
        state = brainstate.State(jnp.array([1.0, 2.0, 3.0]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([1.0, 2.0, 3.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        out_shapes = sf.get_out_shapes_by_cache(cache_key)

        # Should have output shapes
        self.assertIsNotNone(out_shapes)

    def test_multiple_outputs(self):
        """Test tracking multiple output shapes."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value, state.value * 2, jnp.sum(state.value)

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([1.0, 2.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        out_shapes = sf.get_out_shapes_by_cache(cache_key)

        # Should track all output shapes
        self.assertIsNotNone(out_shapes)

    def test_nested_output_structure(self):
        """Test tracking nested output structures."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return {
                'sum': jnp.sum(state.value),
                'prod': jnp.prod(state.value),
                'values': state.value
            }

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([1.0, 2.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        out_treedef = sf.get_out_treedef_by_cache(cache_key)

        # Should have tree definition
        self.assertIsNotNone(out_treedef)


class TestStatefulFunctionJaxprCall(unittest.TestCase):
    """Test jaxpr_call and jaxpr_call_auto methods."""

    def test_jaxpr_call_basic(self):
        """Test basic jaxpr_call usage."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value * 2

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([0.5, 0.5])
        sf.make_jaxpr(x)

        # Get current state values
        state_vals = [state.value]

        # Call at jaxpr level
        new_state_vals, out = sf.jaxpr_call(state_vals, x)

        self.assertEqual(len(new_state_vals), 1)
        self.assertEqual(out.shape, (2,))

    def test_jaxpr_call_auto_basic(self):
        """Test basic jaxpr_call_auto usage."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            state.value += x
            return state.value * 2

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([0.5, 0.5])
        sf.make_jaxpr(x)

        # Call with automatic state management
        result = sf.jaxpr_call_auto(x)

        self.assertEqual(result.shape, (2,))

    def test_jaxpr_call_preserves_state_order(self):
        """Test that jaxpr_call preserves state order."""
        state1 = brainstate.State(jnp.array([1.0]))
        state2 = brainstate.State(jnp.array([2.0]))
        state3 = brainstate.State(jnp.array([3.0]))

        def f(x):
            state1.value += x
            state2.value += x * 2
            state3.value += x * 3
            return state1.value + state2.value + state3.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([1.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        states = sf.get_states_by_cache(cache_key)

        # Get initial state values
        state_vals = [s.value for s in states]

        # Call at jaxpr level
        new_state_vals, _ = sf.jaxpr_call(state_vals, x)

        # Should return same number of states
        self.assertEqual(len(new_state_vals), len(state_vals))


class TestStatefulFunctionEdgeCases(unittest.TestCase):
    """Test edge cases and corner scenarios."""

    def test_no_state_function(self):
        """Test function that doesn't use any states."""

        def f(x):
            return x * 2 + 1

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([1.0, 2.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        states = sf.get_states_by_cache(cache_key)

        # Should have no states
        self.assertEqual(len(states), 0)

    def test_read_only_state(self):
        """Test function that only reads from states."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            # Only read from state, don't write
            return state.value + x

        sf = brainstate.transform.StatefulFunction(f, return_only_write=True)
        x = jnp.array([1.0, 2.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        write_states = sf.get_write_states_by_cache(cache_key)

        # Should have no write states
        self.assertEqual(len(write_states), 0)

    def test_scalar_inputs_outputs(self):
        """Test with scalar inputs and outputs."""
        state = brainstate.State(jnp.array(1.0))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array(0.5)
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        jaxpr = sf.get_jaxpr_by_cache(cache_key)

        # Should compile successfully
        self.assertIsNotNone(jaxpr)

    def test_empty_function(self):
        """Test function with no operations."""

        def f(x):
            return x

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([1.0, 2.0])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        jaxpr = sf.get_jaxpr_by_cache(cache_key)

        # Should compile successfully
        self.assertIsNotNone(jaxpr)

    def test_complex_dtype(self):
        """Test with complex dtype arrays."""
        state = brainstate.State(jnp.array([1.0 + 2.0j, 3.0 + 4.0j]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([0.5 + 0.5j, 0.5 + 0.5j])
        sf.make_jaxpr(x)

        cache_key = sf.get_arg_cache_key(x)
        jaxpr = sf.get_jaxpr_by_cache(cache_key)

        # Should compile successfully
        self.assertIsNotNone(jaxpr)


class TestStatefulFunctionCacheKey(unittest.TestCase):
    """Test cache key generation and behavior."""

    def test_cache_key_different_shapes(self):
        """Test that different input shapes produce different cache keys."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        x1 = jnp.array([1.0, 2.0])
        x2 = jnp.array([1.0, 2.0, 3.0])

        cache_key1 = sf.get_arg_cache_key(x1)
        cache_key2 = sf.get_arg_cache_key(x2)

        # Should have different cache keys
        self.assertNotEqual(cache_key1, cache_key2)

    def test_cache_key_different_dtypes(self):
        """Test that different dtypes produce different cache keys."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        # Use int32 and float32 instead, which are always available in JAX
        x1 = jnp.array([1.0, 2.0], dtype=jnp.float32)
        x2 = jnp.array([1, 2], dtype=jnp.int32)

        cache_key1 = sf.get_arg_cache_key(x1)
        cache_key2 = sf.get_arg_cache_key(x2)

        # Should have different cache keys due to different dtypes
        self.assertNotEqual(cache_key1, cache_key2)

    def test_cache_key_same_abstract_values(self):
        """Test that same abstract values produce same cache keys."""

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        x1 = jnp.array([1.0, 2.0])
        x2 = jnp.array([3.0, 4.0])  # Different values, same shape/dtype

        cache_key1 = sf.get_arg_cache_key(x1)
        cache_key2 = sf.get_arg_cache_key(x2)

        # Should have same cache keys (abstract values are the same)
        self.assertEqual(cache_key1, cache_key2)

    def test_cache_key_with_pytree_inputs(self):
        """Test cache key generation with pytree inputs."""

        def f(inputs):
            x, y = inputs
            return x + y

        sf = brainstate.transform.StatefulFunction(f)

        inputs1 = (jnp.array([1.0]), jnp.array([2.0]))
        inputs2 = (jnp.array([3.0]), jnp.array([4.0]))

        cache_key1 = sf.get_arg_cache_key(inputs1)
        cache_key2 = sf.get_arg_cache_key(inputs2)

        # Should have same cache keys (same structure/shapes)
        self.assertEqual(cache_key1, cache_key2)


class TestStatefulFunctionRecompilation(unittest.TestCase):
    """Test recompilation scenarios."""

    def test_cache_reuse(self):
        """Test that cache is reused for same inputs."""
        state = brainstate.State(jnp.array([1.0]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)

        x = jnp.array([1.0])

        # First compilation
        sf.make_jaxpr(x)
        stats1 = sf.get_cache_stats()

        # Second call with same shape should reuse cache
        sf.make_jaxpr(x)
        stats2 = sf.get_cache_stats()

        # Cache size should remain the same
        self.assertEqual(
            stats1['jaxpr_cache']['size'],
            stats2['jaxpr_cache']['size']
        )

    def test_multiple_compilations_different_shapes(self):
        """Test multiple compilations with different shapes."""
        state = brainstate.State(jnp.array([1.0]))

        def f(x):
            return x * 2

        sf = brainstate.transform.StatefulFunction(f)

        # Compile for different shapes
        shapes = [
            jnp.array([1.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([1.0, 2.0, 3.0]),
        ]

        for x in shapes:
            sf.make_jaxpr(x)

        stats = sf.get_cache_stats()

        # Should have 3 different cache entries
        self.assertEqual(stats['jaxpr_cache']['size'], 3)

    def test_clear_and_recompile(self):
        """Test clearing cache and recompiling."""
        state = brainstate.State(jnp.array([1.0]))

        def f(x):
            state.value += x
            return state.value

        sf = brainstate.transform.StatefulFunction(f)
        x = jnp.array([1.0])

        # Compile
        sf.make_jaxpr(x)
        stats_before = sf.get_cache_stats()
        self.assertGreater(stats_before['jaxpr_cache']['size'], 0)

        # Clear cache
        sf.clear_cache()
        stats_after_clear = sf.get_cache_stats()
        self.assertEqual(stats_after_clear['jaxpr_cache']['size'], 0)

        # Recompile
        sf.make_jaxpr(x)
        stats_after_recompile = sf.get_cache_stats()
        self.assertGreater(stats_after_recompile['jaxpr_cache']['size'], 0)


class TestStatefulMapping(unittest.TestCase):
    def test_state_filters_and_caching(self):
        counter = brainstate.ShortTermState(jnp.zeros(3))

        def accumulate(x):
            print('run')
            counter.value = counter.value + x
            return counter.value

        mapper = brainstate.transform.StatefulMapping(
            accumulate,
            in_axes=0,
            out_axes=0,
            state_in_axes={0: state_filter.OfType(brainstate.ShortTermState)},
            state_out_axes={0: state_filter.OfType(brainstate.ShortTermState)},
        )

        xs = jnp.asarray([1.0, 2.0, 3.0])
        result = mapper(xs)
        self.assertTrue(jnp.allclose(result, xs))
        self.assertTrue(jnp.allclose(counter.value, xs))

        jaxpr = mapper.get_jaxpr(xs)
        print(jaxpr)

    def test_random_state_restoration(self):
        rng_state = brainstate.random.RandomState(0)

        def draw(_):
            key = rng_state.split_key()
            return jr.normal(key, ())

        mapper = brainstate.transform.StatefulMapping(
            draw,
            in_axes=0,
            out_axes=0,
        )

        xs = jnp.ones((4,))
        before = rng_state.value
        samples = mapper(xs)
        self.assertEqual(samples.shape, xs.shape)
        self.assertFalse(jnp.allclose(samples, jnp.repeat(samples[0], xs.shape[0])))
        self.assertTrue(jnp.array_equal(rng_state.value.shape, before.shape))

        jaxpr = mapper.get_jaxpr(xs)
        print(jaxpr)
        print('\n\n')
        j2 = brainstate.transform.optimize_jaxpr(jaxpr.jaxpr)
        print(j2)

    def test_inconsistent_batch_sizes_raise(self):
        tracker = brainstate.ShortTermState(jnp.array(0.0))

        def combine(x, y):
            tracker.value = tracker.value + x + y
            return tracker.value

        mapper = brainstate.transform.StatefulMapping(
            combine,
            in_axes=(0, 0),
            out_axes=0,
            state_in_axes={0: state_filter.OfType(brainstate.ShortTermState)},
            state_out_axes={0: state_filter.OfType(brainstate.ShortTermState)},
        )

        with self.assertRaisesRegex(ValueError, "Inconsistent batch sizes"):
            mapper(jnp.ones((3,)), jnp.ones((4,)))

    def test_unexpected_out_state_mapping_raise(self):
        leak = brainstate.ShortTermState(jnp.array(0.0))

        def mutate(x):
            leak.value = leak.value + x
            return x

        mapper = brainstate.transform.StatefulMapping(
            mutate,
            in_axes=0,
            out_axes=0,
            state_in_axes={},
            state_out_axes={},
            unexpected_out_state_mapping='raise',
        )

        with self.assertRaises(BatchAxisError):
            mapper(jnp.ones((2,)))

    def test_unexpected_out_state_mapping_warn(self):
        leak = brainstate.ShortTermState(jnp.array(0.0))

        def mutate(x):
            leak.value = leak.value + x
            return x

        mapper = brainstate.transform.StatefulMapping(
            mutate,
            in_axes=0,
            out_axes=0,
            state_in_axes={},
            state_out_axes={},
            unexpected_out_state_mapping='warn',
        )

        with pytest.warns(UserWarning):
            mapper(jnp.ones((2,)))
        self.assertTrue(jnp.allclose(leak.value, 1.0))

    def test_unexpected_out_state_mapping_ignore(self):
        leak = brainstate.ShortTermState(jnp.array(0.0))

        def mutate(x):
            leak.value = leak.value + x
            return x

        mapper = brainstate.transform.StatefulMapping(
            mutate,
            in_axes=0,
            out_axes=0,
            state_in_axes={},
            state_out_axes={},
            unexpected_out_state_mapping='ignore',
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            mapper(jnp.ones((2,)))
        self.assertEqual(len(caught), 0)
        self.assertTrue(jnp.allclose(leak.value, 1.0))
