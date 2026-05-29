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

import numpy as np
import jax
import jax.numpy as jnp

from brainstate._compatible_import import Literal, Var
from brainstate.transform._ir_utils import (
    IRError, IRValidationError, UnsupportedPrimitiveError,
    IdentitySet, IdentityMap,
    ensure_jaxpr, check_all_vars, is_scalar_literal_value, literal_with_dtype,
    make_var_factory, partial_eval_jaxpr, CONSTANT_FOLD_BLACKLIST,
)


class TestExceptionHierarchy(unittest.TestCase):
    def test_validation_error_is_value_error(self):
        self.assertTrue(issubclass(IRValidationError, ValueError))
        self.assertTrue(issubclass(IRValidationError, IRError))

    def test_unsupported_primitive_is_ir_error(self):
        self.assertTrue(issubclass(UnsupportedPrimitiveError, IRError))

    def test_ir_error_is_exception(self):
        self.assertTrue(issubclass(IRError, Exception))


class TestIdentitySet(unittest.TestCase):
    def test_membership_by_identity(self):
        s = IdentitySet()
        a = [1, 2, 3]
        b = [1, 2, 3]
        s.add(a)
        self.assertIn(a, s)
        self.assertNotIn(b, s)

    def test_len_and_iter(self):
        s = IdentitySet()
        a, b = object(), object()
        s.add(a); s.add(b); s.add(a)
        self.assertEqual(len(s), 2)
        self.assertEqual({id(x) for x in s}, {id(a), id(b)})

    def test_discard_and_update(self):
        s = IdentitySet()
        a, b = object(), object()
        s.update([a, b])
        s.discard(a)
        self.assertNotIn(a, s)
        self.assertIn(b, s)

    def test_repr_and_str(self):
        """__repr__ and __str__ both render the contained elements."""
        s = IdentitySet([1, 2])
        self.assertTrue(repr(s).startswith("IdentitySet("))
        self.assertTrue(str(s).startswith("IdentitySet("))


class TestIdentityMap(unittest.TestCase):
    def test_set_get_by_identity(self):
        m = IdentityMap()
        a = [1]
        b = [1]
        m[a] = "x"
        self.assertEqual(m[a], "x")
        self.assertNotIn(b, m)

    def test_len_iter_del(self):
        m = IdentityMap()
        a, b = object(), object()
        m[a] = 1
        m[b] = 2
        self.assertEqual(len(m), 2)
        self.assertEqual({id(k) for k in m}, {id(a), id(b)})
        del m[a]
        self.assertEqual(len(m), 1)

    def test_repr(self):
        """__repr__ renders key/value pairs."""
        m = IdentityMap()
        a = object()
        m[a] = 7
        self.assertTrue(repr(m).startswith("IdentityMap("))


class TestValidationHelpers(unittest.TestCase):
    def _make_jaxpr(self):
        return jax.make_jaxpr(lambda x: x + 1.0)(jnp.float32(1.0))

    def test_ensure_jaxpr_accepts_jaxpr(self):
        cj = self._make_jaxpr()
        jaxpr, consts, was_closed = ensure_jaxpr(cj)
        self.assertTrue(was_closed)
        self.assertEqual(list(consts), list(cj.consts))
        jaxpr2, consts2, was_closed2 = ensure_jaxpr(cj.jaxpr)
        self.assertFalse(was_closed2)
        self.assertEqual(consts2, [])

    def test_ensure_jaxpr_rejects_other(self):
        with self.assertRaises(IRValidationError):
            ensure_jaxpr(42)

    def test_check_all_vars(self):
        cj = self._make_jaxpr()
        check_all_vars(cj.jaxpr.invars, 'invars')  # no raise
        with self.assertRaises(IRValidationError):
            check_all_vars([123], 'invars')

    def test_is_scalar_literal_value(self):
        lit0 = Literal(np.float32(0.0), jax.core.ShapedArray((), np.float32))
        lit1 = Literal(np.float32(1.0), jax.core.ShapedArray((), np.float32))
        self.assertTrue(is_scalar_literal_value(lit0, 0))
        self.assertFalse(is_scalar_literal_value(lit0, 1))
        self.assertTrue(is_scalar_literal_value(lit1, 1))

    def test_literal_with_dtype_matches_aval(self):
        aval = jax.core.ShapedArray((), np.float32)
        lit = literal_with_dtype(0, aval)
        self.assertIsInstance(lit, Literal)
        self.assertEqual(np.asarray(lit.val).dtype, np.float32)

    def test_make_var_factory_unique(self):
        factory = make_var_factory()
        aval = jax.core.ShapedArray((), np.float32)
        v1 = factory(aval)
        v2 = factory(aval)
        self.assertIsNot(v1, v2)
        self.assertNotEqual(id(v1), id(v2))

    def test_make_var_factory_caches_working_form(self):
        """The factory caches the first constructor form that succeeds."""
        factory = make_var_factory()
        aval = jax.core.ShapedArray((), np.float32)
        vars_ = [factory(aval) for _ in range(3)]
        # All produced vars are distinct objects of the correct aval.
        self.assertEqual(len({id(v) for v in vars_}), 3)
        for v in vars_:
            self.assertEqual(v.aval, aval)

    def test_make_var_factory_suffix_aval_form(self):
        """A Var(suffix, aval) signature selects the suffix-first construction order."""
        import brainstate.transform._ir_utils as U

        class _SuffixVar:
            def __init__(self, suffix, aval):
                if not isinstance(suffix, str):
                    raise TypeError("suffix must be a string")
                self.suffix = suffix
                self.aval = aval

        orig = U.Var
        U.Var = _SuffixVar
        try:
            factory = U.make_var_factory()
            aval = jax.core.ShapedArray((), np.float32)
            v = factory(aval)
            self.assertIsInstance(v, _SuffixVar)
            self.assertEqual(v.aval, aval)
        finally:
            U.Var = orig

    def test_make_var_factory_count_suffix_aval_form(self):
        """A Var(count, suffix, aval) signature selects the count-first construction order."""
        import brainstate.transform._ir_utils as U

        class _CountVar:
            def __init__(self, count, suffix, aval):
                if not isinstance(count, int):
                    raise TypeError("count must be an int")
                self.count = count
                self.suffix = suffix
                self.aval = aval

        orig = U.Var
        U.Var = _CountVar
        try:
            factory = U.make_var_factory()
            aval = jax.core.ShapedArray((), np.float32)
            v1 = factory(aval)
            v2 = factory(aval)
            self.assertIsInstance(v1, _CountVar)
            # Counter increments across calls.
            self.assertNotEqual(v1.count, v2.count)
        finally:
            U.Var = orig

    def test_make_var_factory_fallback_on_typeerror(self):
        """When the preferred form raises TypeError, the factory falls back to the next form."""
        import brainstate.transform._ir_utils as U

        class _ModernOnlyVar:
            # Signature looks like the modern (aval, ...) form, so the modern
            # candidate is tried first -- but it raises, forcing the fallback
            # chain to the suffix/count forms.
            def __init__(self, aval, initial_qdd=None, final_qdd=None):
                # Reject the single-arg modern call to force the fallback path.
                if initial_qdd is None and final_qdd is None and not isinstance(aval, str) \
                        and not isinstance(aval, int):
                    raise TypeError("modern form unsupported")
                self.aval = aval

        orig = U.Var
        U.Var = _ModernOnlyVar
        try:
            factory = U.make_var_factory()
            aval = jax.core.ShapedArray((), np.float32)
            v = factory(aval)
            self.assertIsInstance(v, _ModernOnlyVar)
        finally:
            U.Var = orig

    def test_make_var_factory_raises_when_all_forms_fail(self):
        """If every construction form raises TypeError, the last error is re-raised."""
        import brainstate.transform._ir_utils as U

        class _AlwaysFailVar:
            def __init__(self, aval, initial_qdd=None, final_qdd=None):
                raise TypeError("never constructible")

        orig = U.Var
        U.Var = _AlwaysFailVar
        try:
            factory = U.make_var_factory()
            aval = jax.core.ShapedArray((), np.float32)
            with self.assertRaises(TypeError):
                factory(aval)
        finally:
            U.Var = orig

    def test_is_scalar_literal_value_non_literal(self):
        """A plain Var (not a Literal) is never a scalar literal value."""
        cj = self._make_jaxpr()
        self.assertFalse(is_scalar_literal_value(cj.jaxpr.invars[0], 0))

    def test_is_scalar_literal_value_python_scalar(self):
        """A Literal holding a plain python scalar compares directly."""
        aval = jax.core.ShapedArray((), np.float32)
        lit = Literal(5, aval)
        self.assertTrue(is_scalar_literal_value(lit, 5))
        self.assertFalse(is_scalar_literal_value(lit, 6))

    def test_is_scalar_literal_value_non_scalar_array(self):
        """A Literal holding a non-scalar array does not match a scalar value."""
        aval = jax.core.ShapedArray((2,), np.float32)
        lit = Literal(np.array([0.0, 0.0], dtype=np.float32), aval)
        self.assertFalse(is_scalar_literal_value(lit, 0))

    def test_is_scalar_literal_value_unconvertible_returns_false(self):
        """A Literal whose value raises during conversion is reported as a non-match."""
        aval = jax.core.ShapedArray((), np.float32)

        class _Boom:
            def __array__(self, *a, **k):
                raise RuntimeError("cannot convert")

        lit = Literal(_Boom(), aval)
        self.assertFalse(is_scalar_literal_value(lit, 0))

    def test_literal_with_dtype_without_dtype_attr(self):
        """When the aval has no dtype, literal_with_dtype wraps the raw value."""

        class _NoDtypeAval:
            pass

        lit = literal_with_dtype(3, _NoDtypeAval())
        self.assertIsInstance(lit, Literal)
        self.assertEqual(lit.val, 3)


class TestPartialEval(unittest.TestCase):
    def test_folds_pure_constants(self):
        cj = jax.make_jaxpr(lambda x: x + (jnp.float32(2.0) + jnp.float32(3.0)))(jnp.float32(1.0))
        folded = partial_eval_jaxpr(cj.jaxpr, {})
        n_add = sum(1 for e in folded.eqns if e.primitive.name == 'add')
        self.assertEqual(n_add, 1)

    def test_blacklist_contains_broadcast_and_control_flow(self):
        for name in ('broadcast_in_dim', 'broadcast', 'while', 'scan', 'cond'):
            self.assertIn(name, CONSTANT_FOLD_BLACKLIST)

    def test_no_constants_is_noop_shape(self):
        cj = jax.make_jaxpr(lambda x, y: x * y)(jnp.float32(2.0), jnp.float32(3.0))
        folded = partial_eval_jaxpr(cj.jaxpr, {})
        self.assertEqual(len(folded.eqns), len(cj.jaxpr.eqns))

    def test_env_var_substitution_keeps_var(self):
        """A blacklisted eqn whose invar maps to another Var keeps that Var (read_or_self)."""
        cj = jax.make_jaxpr(lambda x: jnp.broadcast_to(x, (3,)))(jnp.float32(1.0))
        x = cj.jaxpr.invars[0]
        factory = make_var_factory()
        replacement = factory(x.aval)
        folded = partial_eval_jaxpr(cj.jaxpr, {x: replacement})
        bcast = [e for e in folded.eqns if 'broadcast' in e.primitive.name][0]
        self.assertIs(bcast.invars[0], replacement)

    def test_env_literal_substitution_rewraps(self):
        """A blacklisted eqn whose invar maps to a Literal is re-wrapped with the var's aval."""
        cj = jax.make_jaxpr(lambda x: jnp.broadcast_to(x, (3,)))(jnp.float32(1.0))
        x = cj.jaxpr.invars[0]
        lit = Literal(np.float32(7.0), x.aval)
        folded = partial_eval_jaxpr(cj.jaxpr, {x: lit})
        bcast = [e for e in folded.eqns if 'broadcast' in e.primitive.name][0]
        self.assertIsInstance(bcast.invars[0], Literal)
        self.assertEqual(np.asarray(bcast.invars[0].val).item(), 7.0)

    def test_closed_call_is_folded_inline(self):
        """A closed_call equation with concrete inputs is evaluated and inlined."""
        from jax._src import core as jcore
        from jax._src.core import JaxprEqnContext
        from jax.extend import source_info_util
        from brainstate._compatible_import import Jaxpr, JaxprEqn

        inner_cj = jax.make_jaxpr(lambda a, b: a + b)(jnp.float32(0.0), jnp.float32(0.0))
        factory = make_var_factory()
        aval = jax.core.ShapedArray((), np.float32)
        out = factory(aval)
        lit2 = Literal(np.float32(2.0), aval)
        lit3 = Literal(np.float32(3.0), aval)
        try:
            ctx = JaxprEqnContext()
        except TypeError:
            ctx = JaxprEqnContext(None, True)
        eqn = JaxprEqn(
            [lit2, lit3], [out], jcore.closed_call_p,
            {'call_jaxpr': inner_cj}, set(),
            source_info_util.new_source_info(), ctx,
        )
        outer = Jaxpr(constvars=[], invars=[], outvars=[out], eqns=[eqn])
        folded = partial_eval_jaxpr(outer, {})
        # closed_call folds to the constant 5.0 (no surviving equations).
        self.assertEqual(len(folded.eqns), 0)
        self.assertIsInstance(folded.outvars[0], Literal)
        self.assertEqual(np.asarray(folded.outvars[0].val).item(), 5.0)


if __name__ == '__main__':
    unittest.main()
