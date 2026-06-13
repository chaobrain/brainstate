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

import jax
import jax.numpy as jnp

import brainstate
from brainstate._compatible_import import Tracer


class TestCond(unittest.TestCase):
    def test1(self):
        brainstate.random.seed(1)
        brainstate.transform.cond(True, lambda: brainstate.random.random(10), lambda: brainstate.random.random(10))
        brainstate.transform.cond(False, lambda: brainstate.random.random(10), lambda: brainstate.random.random(10))

    def test2(self):
        st1 = brainstate.State(brainstate.random.rand(10))
        st2 = brainstate.State(brainstate.random.rand(2))
        st3 = brainstate.State(brainstate.random.rand(5))
        st4 = brainstate.State(brainstate.random.rand(2, 10))

        def true_fun(x):
            st1.value = st2.value @ st4.value + x

        def false_fun(x):
            st3.value = (st3.value + 1.) * x

        brainstate.transform.cond(True, true_fun, false_fun, 2.)
        assert not isinstance(st1.value, Tracer)
        assert not isinstance(st2.value, Tracer)
        assert not isinstance(st3.value, Tracer)
        assert not isinstance(st4.value, Tracer)


class TestSwitch(unittest.TestCase):
    def testSwitch(self):
        def branch(x):
            y = jax.lax.mul(2, x)
            return y, jax.lax.mul(2, y)

        branches = [lambda x: (x, x),
                    branch,
                    lambda x: (x, -x)]

        def fun(x):
            if x <= 0:
                return branches[0](x)
            elif x == 1:
                return branches[1](x)
            else:
                return branches[2](x)

        def cfun(x):
            return brainstate.transform.switch(x, branches, x)

        self.assertEqual(fun(-1), cfun(-1))
        self.assertEqual(fun(0), cfun(0))
        self.assertEqual(fun(1), cfun(1))
        self.assertEqual(fun(2), cfun(2))
        self.assertEqual(fun(3), cfun(3))

        cfun = jax.jit(cfun)

        self.assertEqual(fun(-1), cfun(-1))
        self.assertEqual(fun(0), cfun(0))
        self.assertEqual(fun(1), cfun(1))
        self.assertEqual(fun(2), cfun(2))
        self.assertEqual(fun(3), cfun(3))

    def testSwitchMultiOperands(self):
        branches = [jax.lax.add, jax.lax.mul]

        def fun(x):
            i = 0 if x <= 0 else 1
            return branches[i](x, x)

        def cfun(x):
            return brainstate.transform.switch(x, branches, x, x)

        self.assertEqual(fun(-1), cfun(-1))
        self.assertEqual(fun(0), cfun(0))
        self.assertEqual(fun(1), cfun(1))
        self.assertEqual(fun(2), cfun(2))
        cfun = jax.jit(cfun)
        self.assertEqual(fun(-1), cfun(-1))
        self.assertEqual(fun(0), cfun(0))
        self.assertEqual(fun(1), cfun(1))
        self.assertEqual(fun(2), cfun(2))

    def testSwitchResidualsMerge(self):
        def get_conds(fun):
            jaxpr = jax.make_jaxpr(jax.grad(fun))(0., 0)
            return [eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive.name == 'cond']

        def branch_invars_len(cond_eqn):
            lens = [len(jaxpr.jaxpr.invars) for jaxpr in cond_eqn.params['branches']]
            assert len(set(lens)) == 1
            return lens[0]

        def branch_outvars_len(cond_eqn):
            lens = [len(jaxpr.jaxpr.outvars) for jaxpr in cond_eqn.params['branches']]
            assert len(set(lens)) == 1
            return lens[0]

        branches1 = [lambda x: jnp.sin(x),
                     lambda x: jnp.cos(x)]  # branch residuals overlap, should be reused
        branches2 = branches1 + [lambda x: jnp.sinh(x)]  # another overlapping residual, expect reuse
        branches3 = branches2 + [lambda x: jnp.sin(x) + jnp.cos(x)]  # requires one more residual slot

        def fun1(x, i):
            return brainstate.transform.switch(i + 1, branches1, x)

        def fun2(x, i):
            return brainstate.transform.switch(i + 1, branches2, x)

        def fun3(x, i):
            return brainstate.transform.switch(i + 1, branches3, x)

        fwd1, bwd1 = get_conds(fun1)
        fwd2, bwd2 = get_conds(fun2)
        fwd3, bwd3 = get_conds(fun3)

        fwd1_num_out = branch_outvars_len(fwd1)
        fwd2_num_out = branch_outvars_len(fwd2)
        fwd3_num_out = branch_outvars_len(fwd3)
        assert fwd1_num_out == fwd2_num_out
        assert fwd3_num_out == fwd2_num_out + 1

        bwd1_num_in = branch_invars_len(bwd1)
        bwd2_num_in = branch_invars_len(bwd2)
        bwd3_num_in = branch_invars_len(bwd3)
        assert bwd1_num_in == bwd2_num_in
        assert bwd3_num_in == bwd2_num_in + 1

    def testOneBranchSwitch(self):
        branch = lambda x: -x
        f = lambda i, x: brainstate.transform.switch(i, [branch], x)
        x = 7.
        self.assertEqual(f(-1, x), branch(x))
        self.assertEqual(f(0, x), branch(x))
        self.assertEqual(f(1, x), branch(x))
        cf = jax.jit(f)
        self.assertEqual(cf(-1, x), branch(x))
        self.assertEqual(cf(0, x), branch(x))
        self.assertEqual(cf(1, x), branch(x))
        cf = jax.jit(f, static_argnums=0)
        self.assertEqual(cf(-1, x), branch(x))
        self.assertEqual(cf(0, x), branch(x))
        self.assertEqual(cf(1, x), branch(x))


class TestIfElse(unittest.TestCase):
    def test1(self):
        def f(a):
            return brainstate.transform.ifelse(
                conditions=[a < 0,
                            a >= 0 and a < 2,
                            a >= 2 and a < 5,
                            a >= 5 and a < 10,
                            a >= 10],
                branches=[lambda: 1,
                          lambda: 2,
                          lambda: 3,
                          lambda: 4,
                          lambda: 5]
            )

        self.assertTrue(f(3) == 3)
        self.assertTrue(f(1) == 2)
        self.assertTrue(f(-1) == 1)

    def test_vmap(self):
        def f(operands):
            f = lambda a: brainstate.transform.ifelse(
                [a > 10,
                 jnp.logical_and(a <= 10, a > 5),
                 jnp.logical_and(a <= 5, a > 2),
                 jnp.logical_and(a <= 2, a > 0),
                 a <= 0],
                [lambda _: 1,
                 lambda _: 2,
                 lambda _: 3,
                 lambda _: 4,
                 lambda _: 5, ],
                a
            )
            return jax.vmap(f)(operands)

        r = f(brainstate.random.randint(-20, 20, 200))
        self.assertTrue(r.size == 200)

    def test_grad1(self):
        def F2(x):
            return brainstate.transform.ifelse(
                (x >= 10, x < 10),
                [lambda x: x, lambda x: x ** 2, ],
                x
            )

        self.assertTrue(jax.grad(F2)(9.0) == 18.)
        self.assertTrue(jax.grad(F2)(11.0) == 1.)

    def test_grad2(self):
        def F3(x):
            return brainstate.transform.ifelse(
                (x >= 10, jnp.logical_and(x >= 0, x < 10), x < 0),
                [lambda x: x,
                 lambda x: x ** 2,
                 lambda x: x ** 4, ],
                x
            )

        self.assertTrue(jax.grad(F3)(9.0) == 18.)
        self.assertTrue(jax.grad(F3)(11.0) == 1.)


class TestCondValidation(unittest.TestCase):
    """Argument validation and the disable-jit fast path for ``cond``."""

    def test_non_callable_branches_raise(self):
        """Non-callable branches raise ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.cond(True, 1, 2)

    def test_none_predicate_raises(self):
        """A ``None`` predicate raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.cond(None, lambda: 1, lambda: 2)

    def test_non_scalar_predicate_raises(self):
        """A non-scalar predicate raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.cond(jnp.array([True, False]), lambda: 1, lambda: 2)

    def test_sequence_predicate_raises(self):
        """A list predicate (a Sequence) raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.cond([True], lambda: 1, lambda: 2)

    def test_non_numeric_predicate_raises(self):
        """A predicate whose type is neither boolean nor number raises."""
        with self.assertRaises(TypeError):
            brainstate.transform.cond(object(), lambda: 1, lambda: 2)

    def test_complex_predicate_raises(self):
        """A complex predicate (kind 'c') raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.cond(1 + 2j, lambda: 1, lambda: 2)

    def test_integer_predicate_is_truthy(self):
        """A non-zero integer predicate selects the true branch."""
        self.assertEqual(int(brainstate.transform.cond(1, lambda: 10, lambda: 20)), 10)
        self.assertEqual(int(brainstate.transform.cond(0, lambda: 10, lambda: 20)), 20)

    def test_disable_jit_fast_path(self):
        """Under ``disable_jit`` a concrete predicate runs the branch directly."""
        with jax.disable_jit():
            self.assertEqual(brainstate.transform.cond(True, lambda: 1, lambda: 2), 1)
            self.assertEqual(brainstate.transform.cond(False, lambda: 1, lambda: 2), 2)

    def test_vmap_traced_predicate_under_disable_jit(self):
        """``vmap`` over ``cond`` with a traced predicate works under ``disable_jit``.

        The eager fast-path must be gated on the *predicate* being concrete, not
        on ``disable_jit`` alone; otherwise a traced ``pred`` from ``vmap`` would
        reach ``if pred:`` and raise ``TracerBoolConversionError``.
        """
        def f(pred, x):
            return brainstate.transform.cond(pred, lambda v: v + 1.0, lambda v: v - 1.0, x)

        preds = jnp.array([True, False])
        xs = jnp.array([10.0, 20.0])
        with jax.disable_jit():
            out = jax.vmap(f)(preds, xs)
        # True -> +1, False -> -1, selected per element.
        self.assertEqual(float(out[0]), 11.0)
        self.assertEqual(float(out[1]), 19.0)


class TestSwitchValidation(unittest.TestCase):
    """Argument validation and the disable-jit fast path for ``switch``."""

    def test_non_callable_branch_raises(self):
        """A non-callable branch raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.switch(0, [lambda x: x, 5], 1.0)

    def test_non_scalar_index_raises(self):
        """A non-scalar index raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.switch(jnp.array([0, 1]), [lambda x: x, lambda x: x], 1.0)

    def test_non_integer_index_raises(self):
        """A float index raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.switch(1.5, [lambda x: x, lambda x: x], 1.0)

    def test_index_with_bad_type_raises(self):
        """An index whose type cannot be resolved raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.switch(object(), [lambda x: x, lambda x: x], 1.0)

    def test_empty_branches_raise(self):
        """An empty branch sequence raises ``ValueError``."""
        with self.assertRaises(ValueError):
            brainstate.transform.switch(0, [], 1.0)

    def test_disable_jit_fast_path(self):
        """Under ``disable_jit`` a concrete index runs the branch directly."""
        with jax.disable_jit():
            out = brainstate.transform.switch(1, [lambda x: x, lambda x: x + 100], 1.0)
            self.assertEqual(int(out), 101)

    def test_vmap_traced_index_under_disable_jit(self):
        """``vmap`` over ``switch`` with a traced index works under ``disable_jit``.

        The eager fast-path must be gated on the *index* being concrete, not on
        ``disable_jit`` alone; otherwise a traced ``index`` from ``vmap`` would
        reach ``int(index)`` and raise a concretization error.
        """
        branches = [lambda x: x + 1.0, lambda x: x + 10.0, lambda x: x + 100.0]

        def f(index, x):
            return brainstate.transform.switch(index, branches, x)

        indices = jnp.array([0, 1, 2])
        xs = jnp.array([1.0, 1.0, 1.0])
        with jax.disable_jit():
            out = jax.vmap(f)(indices, xs)
        self.assertEqual(float(out[0]), 2.0)
        self.assertEqual(float(out[1]), 11.0)
        self.assertEqual(float(out[2]), 101.0)


class TestIfElseValidation(unittest.TestCase):
    """Validation and degenerate cases for ``ifelse``."""

    def test_non_callable_branch_raises(self):
        """A non-callable branch raises ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.ifelse([True, False], [lambda: 1, 2])

    def test_empty_branches_raise(self):
        """An empty branch sequence raises ``ValueError``."""
        with self.assertRaises(ValueError):
            brainstate.transform.ifelse([], [])

    def test_single_branch_short_circuits(self):
        """A single branch is invoked directly without indexing."""
        out = brainstate.transform.ifelse([True], [lambda x: x + 1], 5)
        self.assertEqual(int(out), 6)

    def test_mismatched_lengths_raise(self):
        """Mismatched condition/branch counts raise ``ValueError``."""
        with self.assertRaises(ValueError):
            brainstate.transform.ifelse([True, False, False], [lambda: 1, lambda: 2])

    def test_single_branch_mismatched_conditions_raise(self):
        """A single branch with >1 conditions must not silently short-circuit."""
        # Previously the single-branch early return bypassed the length check and
        # silently ignored the extra conditions, returning the lone branch's value.
        with self.assertRaises(ValueError):
            brainstate.transform.ifelse([True, False], [lambda: 42.0])

    def test_check_cond_false_skips_validation(self):
        """``check_cond=False`` skips the one-hot validation."""
        out = brainstate.transform.ifelse(
            [True, True], [lambda: 1, lambda: 2], check_cond=False
        )
        self.assertEqual(int(out), 1)


class TestCrossBranchStateAccess(unittest.TestCase):
    """States written in one branch but only read (or untouched) in another.

    Regression tests for the ``return_only_write=True`` interaction where a
    branch that does not write a state in the merged write set produced a
    ``None`` output slot, crashing ``lax.cond``/``lax.switch`` with a
    type-structure mismatch between branches.
    """

    def test_cond_write_in_true_read_in_false(self):
        a = brainstate.State(jnp.asarray(1.0))
        b = brainstate.State(jnp.asarray(2.0))
        r = brainstate.State(jnp.asarray(0.0))

        def true_fn(x):
            a.value = b.value + x

        def false_fn(x):
            r.value = a.value + 1.0

        brainstate.transform.cond(True, true_fn, false_fn, 3.0)
        self.assertEqual(float(a.value), 5.0)
        self.assertEqual(float(r.value), 0.0)
        self.assertFalse(isinstance(a.value, Tracer))
        self.assertFalse(isinstance(r.value, Tracer))

        brainstate.transform.cond(False, true_fn, false_fn, 3.0)
        self.assertEqual(float(a.value), 5.0)  # untouched by false branch
        self.assertEqual(float(r.value), 6.0)

    def test_cond_write_in_one_branch_untouched_in_other(self):
        a = brainstate.State(jnp.asarray(1.0))

        def true_fn():
            a.value = a.value * 10.0

        def false_fn():
            pass  # does not touch ``a`` at all

        brainstate.transform.cond(False, true_fn, false_fn)
        self.assertEqual(float(a.value), 1.0)

        brainstate.transform.cond(True, true_fn, false_fn)
        self.assertEqual(float(a.value), 10.0)

    def test_cond_cross_writes_under_jit(self):
        a = brainstate.State(jnp.asarray(1.0))
        b = brainstate.State(jnp.asarray(2.0))

        @brainstate.transform.jit
        def step(pred, x):
            def true_fn():
                a.value = b.value + x

            def false_fn():
                b.value = a.value * 10.0

            brainstate.transform.cond(pred, true_fn, false_fn)
            return a.value + b.value

        out = step(jnp.asarray(True), 3.0)
        self.assertEqual(float(out), 7.0)
        self.assertEqual(float(a.value), 5.0)
        self.assertEqual(float(b.value), 2.0)

        out = step(jnp.asarray(False), 3.0)
        self.assertEqual(float(out), 55.0)
        self.assertEqual(float(a.value), 5.0)
        self.assertEqual(float(b.value), 50.0)

    def test_switch_mixed_read_write_branches(self):
        a = brainstate.State(jnp.asarray(1.0))
        c = brainstate.State(jnp.asarray(10.0))

        def write_a(x):
            a.value = x * 2.0

        def write_c(x):
            c.value = a.value + x

        def read_only(x):
            _ = a.value + c.value

        branches = [write_a, write_c, read_only]

        brainstate.transform.switch(jnp.asarray(0), branches, 3.0)
        self.assertEqual(float(a.value), 6.0)
        self.assertEqual(float(c.value), 10.0)

        brainstate.transform.switch(jnp.asarray(1), branches, 3.0)
        self.assertEqual(float(a.value), 6.0)
        self.assertEqual(float(c.value), 9.0)

        brainstate.transform.switch(jnp.asarray(2), branches, 3.0)
        self.assertEqual(float(a.value), 6.0)
        self.assertEqual(float(c.value), 9.0)

    def test_ifelse_mixed_read_write_branches(self):
        a = brainstate.State(jnp.asarray(0.0))

        def write_branch():
            a.value = a.value + 1.0

        def read_branch():
            _ = a.value

        def f(x):
            return brainstate.transform.ifelse(
                [x >= 0, x < 0],
                [write_branch, read_branch],
            )

        f(jnp.asarray(1.0))
        self.assertEqual(float(a.value), 1.0)

        f(jnp.asarray(-1.0))
        self.assertEqual(float(a.value), 1.0)

    def test_cond_read_only_state_in_all_branches(self):
        """A state read everywhere and written nowhere stays untouched."""
        a = brainstate.State(jnp.asarray(4.0))

        out = brainstate.transform.cond(
            True, lambda: a.value + 1.0, lambda: a.value - 1.0
        )
        self.assertEqual(float(out), 5.0)
        self.assertEqual(float(a.value), 4.0)
        self.assertFalse(isinstance(a.value, Tracer))
