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


import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import make_jaxpr

from brainstate.transform._ir_inline import inline_jit


def has_primitive(jaxpr, primitive_name: str) -> bool:
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == primitive_name:
            return True
    return False


def count_equations(jaxpr) -> int:
    return len(jaxpr.eqns)


def expand_small_jits(max_eqns: int = 5):
    def predicate(eqn):
        call_jaxpr = eqn.params.get('call_jaxpr') or eqn.params.get('jaxpr')
        if call_jaxpr is None:
            return False
        return count_equations(call_jaxpr) <= max_eqns

    return predicate


def expand_without_primitive(primitive_name: str):
    def predicate(eqn):
        call_jaxpr = eqn.params.get('call_jaxpr') or eqn.params.get('jaxpr')
        if call_jaxpr is None:
            return False
        return not has_primitive(call_jaxpr, primitive_name)

    return predicate


def count_call_equations(jaxpr) -> int:
    return sum(
        1 for eqn in jaxpr.eqns
        if (eqn.params.get('call_jaxpr') is not None) or (eqn.params.get('jaxpr') is not None)
    )


def eval_jaxpr(jaxpr, *args):
    return jax.core.eval_jaxpr(jaxpr, [], *args)


def _as_jaxpr(maybe_closed):
    """Return the bare Jaxpr from a Jaxpr or ClosedJaxpr."""
    return maybe_closed.jaxpr if hasattr(maybe_closed, 'jaxpr') else maybe_closed


class TestInlineJit(unittest.TestCase):
    def test_expand_all_jits_preserves_value_and_removes_calls(self):
        @jax.jit
        def inner_func(x, y):
            return x + y * 2

        def outer_func(a, b, c):
            result1 = inner_func(a, b)
            result2 = jnp.sin(result1)
            result3 = inner_func(result2, c)
            return result3

        inputs = (1.0, 2.0, 3.0)
        jaxpr = make_jaxpr(outer_func)(*inputs)
        orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)

        expanded = inline_jit(jaxpr.jaxpr)
        exp_out = eval_jaxpr(_as_jaxpr(expanded), *inputs)

        assert np.allclose(np.array(orig_out), np.array(exp_out))
        # Expect no remaining call-style equations after full expansion
        assert count_call_equations(_as_jaxpr(expanded)) == 0

    def test_conditional_expansion_by_size(self):
        @jax.jit
        def small_func(x):
            return x + 1

        @jax.jit
        def large_func(x):
            x = x + 1
            x = x * 2
            x = jnp.sin(x)
            x = jnp.cos(x)
            x = x ** 2
            return x

        def outer_func(a, b):
            result1 = small_func(a)
            result2 = large_func(b)
            return result1 + result2

        inputs = (1.0, 2.0)
        jaxpr = make_jaxpr(outer_func)(*inputs)
        orig_call_count = count_call_equations(jaxpr.jaxpr)

        predicate = expand_small_jits(max_eqns=3)
        expanded = inline_jit(jaxpr.jaxpr, predicate)
        exp_call_count = count_call_equations(_as_jaxpr(expanded))

        # Values preserved
        orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)
        exp_out = eval_jaxpr(_as_jaxpr(expanded), *inputs)
        assert np.allclose(np.array(orig_out), np.array(exp_out))

        # Small jits should be expanded, so call count should decrease but not necessarily to zero
        assert exp_call_count <= orig_call_count

    def test_expand_without_sin_primitive(self):
        @jax.jit
        def func_with_sin(x):
            return jnp.sin(x) + 1

        @jax.jit
        def func_without_sin(x):
            return x * 2 + 1

        def outer_func(a, b):
            result1 = func_without_sin(a)
            result2 = func_with_sin(b)
            return result1 + result2

        inputs = (1.0, 2.0)
        jaxpr = make_jaxpr(outer_func)(*inputs)
        orig_call_count = count_call_equations(jaxpr.jaxpr)

        predicate = expand_without_primitive('sin')
        expanded = inline_jit(jaxpr.jaxpr, predicate)

        orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)
        exp_out = eval_jaxpr(_as_jaxpr(expanded), *inputs)
        assert np.allclose(np.array(orig_out), np.array(exp_out))

        # At least some calls (those without 'sin') should be expanded
        assert count_call_equations(_as_jaxpr(expanded)) <= orig_call_count

    def test_nested_jits_expand_recursively(self):
        @jax.jit
        def innermost(x):
            return x + 1

        @jax.jit
        def middle(x):
            return innermost(x) * 2

        def outer(x):
            return middle(x) + 3

        inputs = (1.0,)
        jaxpr = make_jaxpr(outer)(*inputs)
        orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)

        expanded = inline_jit(jaxpr.jaxpr)
        exp_out = eval_jaxpr(_as_jaxpr(expanded), *inputs)

        assert np.allclose(np.array(orig_out), np.array(exp_out))
        # Fully expanded nested jits should remove call-style equations
        assert count_call_equations(_as_jaxpr(expanded)) == 0

    def test_custom_predicate_expands_only_selected_jits(self):
        @jax.jit
        def func1(x):
            return x + 1

        @jax.jit
        def func2(x):
            return x * 2

        @jax.jit
        def func3(x):
            return x ** 2

        def outer(a, b, c):
            r1 = func1(a)
            r2 = func2(b)
            r3 = func3(c)
            return r1 + r2 + r3

        inputs = (1.0, 2.0, 3.0)
        jaxpr = make_jaxpr(outer)(*inputs)
        orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)

        def custom_predicate(eqn):
            call_jaxpr = eqn.params.get('call_jaxpr') or eqn.params.get('jaxpr')
            if call_jaxpr is None:
                return False
            return has_primitive(call_jaxpr, 'mul') or has_primitive(call_jaxpr, 'add')

        expanded = inline_jit(jaxpr.jaxpr, custom_predicate)
        exp_out = eval_jaxpr(_as_jaxpr(expanded), *inputs)

        assert np.allclose(np.array(orig_out), np.array(exp_out))
        # Some calls should be expanded according to the custom predicate
        assert count_call_equations(_as_jaxpr(expanded)) <= count_call_equations(jaxpr.jaxpr)

    def test_same_helper_inlined_twice_has_unique_binders(self):
        # Regression for the variable-collision bug: inlining the same helper
        # twice must not reuse inner binders.
        @jax.jit
        def helper(x):
            return x * x + 1.0

        def outer(x):
            return helper(x) + helper(x + 1.0)

        cj = jax.make_jaxpr(outer)(jnp.float32(2.0))
        expanded = _as_jaxpr(inline_jit(cj.jaxpr))
        binders = [v for e in expanded.eqns for v in e.outvars]
        ids = [id(v) for v in binders]
        self.assertEqual(len(ids), len(set(ids)),
                         "inlining the same helper twice produced duplicate binders")
        out = jax.core.eval_jaxpr(expanded, [], jnp.float32(2.0))
        self.assertTrue(np.allclose(np.asarray(out[0]), np.asarray(outer(jnp.float32(2.0)))))

    def test_input_validation(self):
        from brainstate.transform._ir_utils import IRValidationError
        with self.assertRaises(IRValidationError):
            inline_jit(123)

    def test_inline_closed_jaxpr_with_consts_runs(self):
        const = jnp.arange(3, dtype=jnp.float32)

        @jax.jit
        def helper(x):
            return x + const

        def outer(x):
            return helper(x) * 2.0

        cj = jax.make_jaxpr(outer)(jnp.zeros(3, jnp.float32))
        expanded = inline_jit(cj)  # pass the ClosedJaxpr
        # Must remain runnable and numerically correct.
        consts = expanded.consts if hasattr(expanded, 'consts') else []
        inner = _as_jaxpr(expanded)
        out = jax.core.eval_jaxpr(inner, consts, jnp.zeros(3, jnp.float32))
        self.assertTrue(np.allclose(np.asarray(out[0]), np.asarray(outer(jnp.zeros(3, jnp.float32)))))

    def test_should_expand_false_keeps_all_calls(self):
        """A predicate that always returns False leaves jit calls untouched."""
        @jax.jit
        def helper(x):
            return x * 3.0

        def outer(x):
            return helper(x) + 1.0

        cj = jax.make_jaxpr(outer)(jnp.float32(2.0))
        expanded = inline_jit(cj.jaxpr, should_expand=lambda eqn: False)
        # No expansion: the jit call survives.
        self.assertEqual(count_call_equations(_as_jaxpr(expanded)),
                         count_call_equations(cj.jaxpr))
        out = eval_jaxpr(_as_jaxpr(expanded), jnp.float32(2.0))
        self.assertTrue(np.allclose(np.array(out), np.array(outer(jnp.float32(2.0)))))


class TestInlineJitConstructedEqns(unittest.TestCase):
    """Cover the jaxpr-variant branches using directly constructed equations."""

    def _jit_primitive(self):
        cj = jax.make_jaxpr(jax.jit(lambda x: x + 1.0))(jnp.float32(1.0))
        return [e.primitive for e in cj.jaxpr.eqns if e.primitive.name == 'jit'][0]

    def _make_eqn(self, primitive, invars, outvars, params):
        from jax._src.core import JaxprEqnContext
        from jax.extend import source_info_util
        from brainstate._compatible_import import JaxprEqn
        try:
            ctx = JaxprEqnContext()
        except TypeError:
            ctx = JaxprEqnContext(None, True)
        return JaxprEqn(invars, outvars, primitive, params, set(),
                        source_info_util.new_source_info(), ctx)

    def test_jit_with_bare_jaxpr_param_inlines(self):
        """A jit equation whose 'jaxpr' param is a bare Jaxpr (no consts) inlines correctly."""
        import numpy as np
        from jax._src import core as jcore
        from brainstate._compatible_import import Jaxpr
        from brainstate.transform._ir_utils import make_var_factory

        inner_bare = jax.make_jaxpr(lambda a: a + 1.0)(jnp.float32(0.0)).jaxpr
        factory = make_var_factory()
        aval = jcore.ShapedArray((), np.float32)
        inv, outv = factory(aval), factory(aval)
        eqn = self._make_eqn(self._jit_primitive(), [inv], [outv], {'jaxpr': inner_bare})
        jpr = Jaxpr(constvars=[], invars=[inv], outvars=[outv], eqns=[eqn])

        res = inline_jit(jpr)
        inner = _as_jaxpr(res)
        # The body was inlined: the 'add' primitive now appears directly.
        self.assertTrue(has_primitive(inner, 'add'))
        out = jax.core.eval_jaxpr(inner, getattr(res, 'consts', []), jnp.float32(5.0))
        self.assertTrue(np.allclose(np.asarray(out[0]), 6.0))

    def test_jit_without_jaxpr_param_is_kept(self):
        """A jit equation that carries no jaxpr/call_jaxpr param is preserved verbatim."""
        import numpy as np
        from jax._src import core as jcore
        from brainstate._compatible_import import Jaxpr, is_jit_primitive
        from brainstate.transform._ir_utils import make_var_factory

        factory = make_var_factory()
        aval = jcore.ShapedArray((), np.float32)
        inv, outv = factory(aval), factory(aval)
        eqn = self._make_eqn(self._jit_primitive(), [inv], [outv], {})
        self.assertTrue(is_jit_primitive(eqn))
        jpr = Jaxpr(constvars=[], invars=[inv], outvars=[outv], eqns=[eqn])

        res = inline_jit(jpr)
        inner = _as_jaxpr(res)
        # The unexpandable jit equation is retained.
        self.assertEqual(len(inner.eqns), 1)
        self.assertEqual(inner.eqns[0].primitive.name, 'jit')


if __name__ == '__main__':
    unittest.main()
