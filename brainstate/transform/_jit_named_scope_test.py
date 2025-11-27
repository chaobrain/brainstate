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

from __future__ import annotations

import unittest

import jax.numpy as jnp

import brainstate as bst
from brainstate._state import TRACE_CONTEXT, StateTraceStack
from brainstate.transform._jit_named_scope import jit_named_scope, fn_to_call


class TestJitNamedScopeBasic(unittest.TestCase):
    """Tests for basic jit_named_scope functionality."""

    def test_basic_decoration(self):
        """Test basic decoration with just a name."""

        @jit_named_scope(name='test_fn')
        def add(x, y):
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = add(x, y)
        expected = jnp.array([4.0, 6.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_fn_attribute(self):
        """Test that the wrapped function has the fn attribute."""

        def original_fn(x):
            return x * 2

        wrapped = jit_named_scope(name='test')(original_fn)
        self.assertTrue(hasattr(wrapped, 'fn'))
        self.assertIs(wrapped.fn, original_fn)

    def test_functools_wraps_preserved(self):
        """Test that functools.wraps preserves function metadata."""

        @jit_named_scope(name='documented_fn')
        def documented(x):
            """This is a documented function."""
            return x

        self.assertEqual(documented.__doc__, "This is a documented function.")
        self.assertEqual(documented.__name__, "documented")


class TestStaticArgnums(unittest.TestCase):
    """Tests for static_argnums parameter."""

    def test_single_static_argnum(self):
        """Test with a single static positional argument."""

        @jit_named_scope(name='power', static_argnums=1)
        def power(x, n):
            return x ** n

        x = jnp.array([2.0, 3.0])
        result = power(x, 2)
        expected = jnp.array([4.0, 9.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_multiple_static_argnums_as_tuple(self):
        """Test with multiple static positional arguments as a tuple."""

        @jit_named_scope(name='scaled_power', static_argnums=(1, 2))
        def scaled_power(x, n, scale):
            return (x ** n) * scale

        x = jnp.array([2.0, 3.0])
        result = scaled_power(x, 2, 10)
        expected = jnp.array([40.0, 90.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_multiple_static_argnums_as_list(self):
        """Test with multiple static positional arguments as a list."""

        @jit_named_scope(name='scaled_power', static_argnums=[1, 2])
        def scaled_power(x, n, scale):
            return (x ** n) * scale

        x = jnp.array([2.0, 3.0])
        result = scaled_power(x, 2, 10)
        expected = jnp.array([40.0, 90.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_negative_static_argnum(self):
        """Test with negative index for static argument."""

        @jit_named_scope(name='neg_index', static_argnums=-1)
        def neg_index_fn(x, y, mode):
            if mode == 'add':
                return x + y
            return x - y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result_add = neg_index_fn(x, y, 'add')
        result_sub = neg_index_fn(x, y, 'sub')
        self.assertTrue(jnp.allclose(result_add, jnp.array([4.0, 6.0])))
        self.assertTrue(jnp.allclose(result_sub, jnp.array([-2.0, -2.0])))

    def test_multiple_negative_static_argnums(self):
        """Test with multiple negative indices for static arguments."""

        @jit_named_scope(name='neg_indices', static_argnums=(-1, -2))
        def fn_with_neg_indices(x, n, scale):
            return (x ** n) * scale

        x = jnp.array([2.0, 3.0])
        result = fn_with_neg_indices(x, 2, 3)
        expected = jnp.array([12.0, 27.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestStaticArgnames(unittest.TestCase):
    """Tests for static_argnames parameter."""

    def test_single_static_argname(self):
        """Test with a single static keyword argument name."""

        @jit_named_scope(name='power_kw', static_argnames='n')
        def power(x, n=2):
            return x ** n

        x = jnp.array([2.0, 3.0])
        result = power(x, n=3)
        expected = jnp.array([8.0, 27.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_multiple_static_argnames_as_tuple(self):
        """Test with multiple static keyword argument names as tuple."""

        @jit_named_scope(name='multi_kw', static_argnames=('n', 'scale'))
        def scaled_power(x, n=2, scale=1):
            return (x ** n) * scale

        x = jnp.array([2.0, 3.0])
        result = scaled_power(x, n=2, scale=5)
        expected = jnp.array([20.0, 45.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_multiple_static_argnames_as_list(self):
        """Test with multiple static keyword argument names as list."""

        @jit_named_scope(name='multi_kw', static_argnames=['n', 'scale'])
        def scaled_power(x, n=2, scale=1):
            return (x ** n) * scale

        x = jnp.array([2.0, 3.0])
        result = scaled_power(x, n=2, scale=5)
        expected = jnp.array([20.0, 45.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_single_static_argname_as_string(self):
        """Test with static_argnames as a single string."""

        @jit_named_scope(name='str_argname', static_argnames='mode')
        def fn_with_mode(x, mode='default'):
            if mode == 'double':
                return x * 2
            return x

        x = jnp.array([1.0, 2.0])
        result = fn_with_mode(x, mode='double')
        expected = jnp.array([2.0, 4.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestCallableStaticArgs(unittest.TestCase):
    """Tests for callable static_argnums and static_argnames."""

    def test_callable_static_argnums(self):
        """Test with callable static_argnums that dynamically determines static args."""

        def determine_static(*args, **kwargs):
            # Make all args except the first one static
            return tuple(range(1, len(args)))

        @jit_named_scope(name='dynamic_static', static_argnums=determine_static)
        def dynamic_fn(x, n, scale):
            return (x ** n) * scale

        x = jnp.array([2.0, 3.0])
        result = dynamic_fn(x, 2, 3)
        expected = jnp.array([12.0, 27.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_callable_static_argnames(self):
        """Test with callable static_argnames that dynamically determines static kwargs."""

        def determine_static_names(*args, **kwargs):
            # Make 'mode' static if it's present
            return ('mode',) if 'mode' in kwargs else ()

        @jit_named_scope(name='dynamic_names', static_argnames=determine_static_names)
        def fn_with_dynamic_names(x, mode='add'):
            if mode == 'add':
                return x + 1
            return x - 1

        x = jnp.array([1.0, 2.0])
        result = fn_with_dynamic_names(x, mode='add')
        expected = jnp.array([2.0, 3.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_callable_static_argnums_returns_empty(self):
        """Test callable static_argnums that returns empty tuple."""

        def no_static(*args, **kwargs):
            return ()

        @jit_named_scope(name='no_static', static_argnums=no_static)
        def simple_fn(x, y):
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = simple_fn(x, y)
        expected = jnp.array([4.0, 6.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestCombinedStaticArgs(unittest.TestCase):
    """Tests for combining static_argnums and static_argnames."""

    def test_combined_static_args(self):
        """Test using both static_argnums and static_argnames."""

        @jit_named_scope(name='combined', static_argnums=1, static_argnames='scale')
        def combined_fn(x, n, scale=1):
            return (x ** n) * scale

        x = jnp.array([2.0, 3.0])
        result = combined_fn(x, 2, scale=3)
        expected = jnp.array([12.0, 27.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestTraceContext(unittest.TestCase):
    """Tests for trace context behavior."""

    def test_no_jit_outside_trace_context(self):
        """Test that function is called directly when outside trace context."""
        call_log = []

        @jit_named_scope(name='test_trace')
        def logged_fn(x):
            call_log.append('called')
            return x * 2

        # Outside trace context, should call original function directly
        self.assertEqual(TRACE_CONTEXT.get_trace_stack_level(), 0)
        x = jnp.array([1.0, 2.0])
        result = logged_fn(x)
        expected = jnp.array([2.0, 4.0])
        self.assertTrue(jnp.allclose(result, expected))
        self.assertEqual(len(call_log), 1)

    def test_jit_inside_trace_context(self):
        """Test that function is JIT compiled when inside trace context."""

        @jit_named_scope(name='inner_fn')
        def inner(x):
            return x * 2

        @bst.transform.jit
        def outer(x):
            # Inside outer's trace context, inner should be JIT compiled
            return inner(x) + 1

        x = jnp.array([1.0, 2.0])
        result = outer(x)
        expected = jnp.array([3.0, 5.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestMethodBinding(unittest.TestCase):
    """Tests for method binding support."""

    def test_as_instance_method(self):
        """Test jit_named_scope as a class instance method."""

        class MyModule:
            def __init__(self, scale):
                self.scale = scale

            @jit_named_scope(name='compute')
            def compute(self, x):
                return x * self.scale

        module = MyModule(scale=3.0)
        x = jnp.array([1.0, 2.0])
        result = module.compute(x)
        expected = jnp.array([3.0, 6.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_method_with_static_argnums(self):
        """Test method with static positional arguments."""

        class PowerModule:
            @jit_named_scope(name='power', static_argnums=1)
            def power(self, x, n):
                return x ** n

        module = PowerModule()
        x = jnp.array([2.0, 3.0])
        result = module.power(x, 3)
        expected = jnp.array([8.0, 27.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_method_with_static_argnames(self):
        """Test method with static keyword arguments."""

        class ModeModule:
            @jit_named_scope(name='process', static_argnames='mode')
            def process(self, x, mode='add'):
                if mode == 'add':
                    return x + 1
                return x - 1

        module = ModeModule()
        x = jnp.array([1.0, 2.0])
        result = module.process(x, mode='add')
        expected = jnp.array([2.0, 3.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestFunctionToCall(unittest.TestCase):
    """Tests for the function_to_call helper function."""

    def test_direct_use(self):
        """Test using function_to_call directly."""

        def original(x, y):
            return x + y

        wrapped = fn_to_call(original, name='direct_test')
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = wrapped(x, y)
        expected = jnp.array([4.0, 6.0])
        self.assertTrue(jnp.allclose(result, expected))
        self.assertIs(wrapped.fn, original)

    def test_with_all_options(self):
        """Test function_to_call with all options."""

        def original(x, n, scale=1):
            return (x ** n) * scale

        wrapped = fn_to_call(
            original,
            name='all_options',
            static_argnums=1,
            static_argnames='scale'
        )
        x = jnp.array([2.0, 3.0])
        result = wrapped(x, 2, scale=3)
        expected = jnp.array([12.0, 27.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestNormalizeHelpers(unittest.TestCase):
    """Tests for normalization of static arguments."""

    def test_none_static_argnums(self):
        """Test with None static_argnums."""

        @jit_named_scope(name='none_argnums', static_argnums=None)
        def fn(x, y):
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = fn(x, y)
        expected = jnp.array([4.0, 6.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_none_static_argnames(self):
        """Test with None static_argnames."""

        @jit_named_scope(name='none_argnames', static_argnames=None)
        def fn(x, y):
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = fn(x, y)
        expected = jnp.array([4.0, 6.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestWithBrainstateState(unittest.TestCase):
    """Tests using brainstate State objects."""

    def test_with_state_inside_jit(self):
        """Test jit_named_scope with brainstate State inside JIT context."""
        state = bst.State(jnp.array([1.0, 2.0]))

        @jit_named_scope(name='state_fn')
        def update_state(x):
            state.value = state.value + x
            return state.value

        @bst.transform.jit
        def outer(x):
            return update_state(x)

        result = outer(jnp.array([1.0, 1.0]))
        expected = jnp.array([2.0, 3.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_with_state_direct_call(self):
        """Test jit_named_scope with brainstate State in direct call (no outer JIT)."""
        state = bst.State(jnp.array([1.0, 2.0]))

        @jit_named_scope(name='state_direct')
        def update_state(x):
            state.value = state.value + x
            return state.value

        # Direct call outside trace context
        result = update_state(jnp.array([1.0, 1.0]))
        expected = jnp.array([2.0, 3.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_no_args_function(self):
        """Test function with no arguments."""

        @jit_named_scope(name='no_args')
        def constant():
            return jnp.array([1.0, 2.0])

        result = constant()
        expected = jnp.array([1.0, 2.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_kwargs_only(self):
        """Test function called with kwargs only."""

        @jit_named_scope(name='kwargs_only')
        def add(x, y):
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = add(x=x, y=y)
        expected = jnp.array([4.0, 6.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_mixed_args_kwargs(self):
        """Test function with mixed positional and keyword arguments."""

        @jit_named_scope(name='mixed', static_argnums=1, static_argnames='scale')
        def mixed_fn(x, n, scale=1):
            return (x ** n) * scale

        x = jnp.array([2.0, 3.0])
        result = mixed_fn(x, 2, scale=3)
        expected = jnp.array([12.0, 27.0])
        self.assertTrue(jnp.allclose(result, expected))

    def test_nested_jit_named_scope(self):
        """Test nested jit_named_scope calls."""

        @jit_named_scope(name='outer_scope')
        def outer(x):
            return inner(x) + 1

        @jit_named_scope(name='inner_scope')
        def inner(x):
            return x * 2

        @bst.transform.jit
        def main(x):
            return outer(x)

        x = jnp.array([1.0, 2.0])
        result = main(x)
        expected = jnp.array([3.0, 5.0])
        self.assertTrue(jnp.allclose(result, expected))


class TestStaticArgRecompilation(unittest.TestCase):
    """Tests for recompilation behavior with different static args."""

    def test_recompilation_with_different_static_values(self):
        """Test that different static argument values trigger recompilation."""
        compile_count = [0]

        @jit_named_scope(name='recompile_test', static_argnums=1)
        def fn_with_static(x, mode):
            compile_count[0] += 1
            if mode == 'double':
                return x * 2
            return x * 3

        # The outer JIT must also mark mode as static for it to pass through
        @bst.transform.jit(static_argnums=1)
        def outer(x, mode):
            return fn_with_static(x, mode)

        x = jnp.array([1.0, 2.0])

        # First call with mode='double'
        result1 = outer(x, 'double')
        expected1 = jnp.array([2.0, 4.0])
        self.assertTrue(jnp.allclose(result1, expected1))
        first_compile_count = compile_count[0]

        # Second call with same mode - should use cached compilation
        result2 = outer(x, 'double')
        self.assertTrue(jnp.allclose(result2, expected1))

        # Third call with different mode - should recompile
        result3 = outer(x, 'triple')
        expected3 = jnp.array([3.0, 6.0])
        self.assertTrue(jnp.allclose(result3, expected3))


if __name__ == '__main__':
    unittest.main()
