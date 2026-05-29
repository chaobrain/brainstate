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

from brainstate._compatible_import import Literal
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


if __name__ == '__main__':
    unittest.main()
