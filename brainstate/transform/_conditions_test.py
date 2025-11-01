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
        assert not isinstance(st1.value, jax.core.Tracer)
        assert not isinstance(st2.value, jax.core.Tracer)
        assert not isinstance(st3.value, jax.core.Tracer)
        assert not isinstance(st4.value, jax.core.Tracer)


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
