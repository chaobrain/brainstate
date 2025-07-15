# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

import unittest
from pprint import pprint

import brainunit as u
import jax
import jax.numpy as jnp
import pytest

import brainstate
from brainstate.augment._autograd import _jacfwd


class TestPureFuncGrad(unittest.TestCase):
    def test_grad_pure_func_1(self):
        def call(a, b, c): return jnp.sum(a + b + c)

        brainstate.random.seed(1)
        a = jnp.ones(10)
        b = brainstate.random.randn(10)
        c = brainstate.random.uniform(size=10)
        f_grad = brainstate.augment.grad(call, argnums=[0, 1, 2])
        grads = f_grad(a, b, c)

        for g in grads: assert (g == 1.).all()

    def test_grad_pure_func_2(self):
        def call(a, b, c): return jnp.sum(a + b + c)

        brainstate.random.seed(1)
        a = jnp.ones(10)
        b = brainstate.random.randn(10)
        c = brainstate.random.uniform(size=10)
        f_grad = brainstate.augment.grad(call)
        assert (f_grad(a, b, c) == 1.).all()

    def test_grad_pure_func_aux1(self):
        def call(a, b, c):
            return jnp.sum(a + b + c), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(1)
        f_grad = brainstate.augment.grad(call, argnums=[0, 1, 2])
        with pytest.raises(TypeError):
            f_grad(jnp.ones(10), brainstate.random.randn(10), brainstate.random.uniform(size=10))

    def test_grad_pure_func_aux2(self):
        def call(a, b, c):
            return jnp.sum(a + b + c), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(1)
        f_grad = brainstate.augment.grad(call, argnums=[0, 1, 2], has_aux=True)
        grads, aux = f_grad(jnp.ones(10), brainstate.random.randn(10), brainstate.random.uniform(size=10))
        for g in grads: assert (g == 1.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

    def test_grad_pure_func_return1(self):
        def call(a, b, c): return jnp.sum(a + b + c)

        brainstate.random.seed(1)
        a = jnp.ones(10)
        b = brainstate.random.randn(10)
        c = brainstate.random.uniform(size=10)
        f_grad = brainstate.augment.grad(call, return_value=True)
        grads, returns = f_grad(a, b, c)
        assert (grads == 1.).all()
        assert returns == jnp.sum(a + b + c)

    def test_grad_func_return_aux1(self):
        def call(a, b, c):
            return jnp.sum(a + b + c), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(1)
        a = jnp.ones(10)
        b = brainstate.random.randn(10)
        c = brainstate.random.uniform(size=10)
        f_grad = brainstate.augment.grad(call, return_value=True, has_aux=True)
        grads, returns, aux = f_grad(a, b, c)
        assert (grads == 1.).all()
        assert returns == jnp.sum(a + b + c)
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)


class TestObjectFuncGrad(unittest.TestCase):
    def test_grad_ob1(self):
        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()

                self.a = brainstate.ParamState(jnp.ones(10))
                self.b = brainstate.ParamState(brainstate.random.randn(10))
                self.c = brainstate.ParamState(brainstate.random.uniform(size=10))

            def __call__(self):
                return jnp.sum(self.a.value + self.b.value + self.c.value)

        brainstate.random.seed(0)

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states={'a': t.a, 'b': t.b, 'c': t.c})
        grads = f_grad()
        for g in grads.values():
            assert (g == 1.).all()

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=[t.a, t.b])
        grads = f_grad()
        for g in grads: assert (g == 1.).all()

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.a)
        grads = f_grad()
        assert (grads == 1.).all()

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.states())
        grads = f_grad()
        for g in grads.values():
            assert (g == 1.).all()

    def test_grad_ob_aux(self):
        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.a = brainstate.ParamState(jnp.ones(10))
                self.b = brainstate.ParamState(brainstate.random.randn(10))
                self.c = brainstate.ParamState(brainstate.random.uniform(size=10))

            def __call__(self):
                return jnp.sum(self.a.value + self.b.value + self.c.value), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(0)
        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=[t.a, t.b], has_aux=True)
        grads, aux = f_grad()
        for g in grads: assert (g == 1.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.a, has_aux=True)
        grads, aux = f_grad()
        assert (grads == 1.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.states(), has_aux=True)
        grads, aux = f_grad()
        self.assertTrue(len(grads) == len(t.states()))

    def test_grad_ob_return(self):
        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.a = brainstate.ParamState(jnp.ones(10))
                self.b = brainstate.ParamState(brainstate.random.randn(10))
                self.c = brainstate.ParamState(brainstate.random.uniform(size=10))

            def __call__(self):
                return jnp.sum(self.a.value + self.b.value + self.c.value)

        brainstate.random.seed(0)
        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=[t.a, t.b], return_value=True)
        grads, returns = f_grad()
        for g in grads: assert (g == 1.).all()
        assert returns == t()

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.a, return_value=True)
        grads, returns = f_grad()
        assert (grads == 1.).all()
        assert returns == t()

    def test_grad_ob_aux_return(self):
        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.a = brainstate.ParamState(jnp.ones(10))
                self.b = brainstate.ParamState(brainstate.random.randn(10))
                self.c = brainstate.ParamState(brainstate.random.uniform(size=10))

            def __call__(self):
                return jnp.sum(self.a.value + self.b.value + self.c.value), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(0)
        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=[t.a, t.b], has_aux=True, return_value=True)
        grads, returns, aux = f_grad()
        for g in grads: assert (g == 1.).all()
        assert returns == jnp.sum(t.a.value + t.b.value + t.c.value)
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.a, has_aux=True, return_value=True)
        grads, returns, aux = f_grad()
        assert (grads == 1.).all()
        assert returns == jnp.sum(t.a.value + t.b.value + t.c.value)
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

    def test_grad_ob_argnums(self):
        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                brainstate.random.seed()
                self.a = brainstate.ParamState(jnp.ones(10))
                self.b = brainstate.ParamState(brainstate.random.randn(10))
                self.c = brainstate.ParamState(brainstate.random.uniform(size=10))

            def __call__(self, d):
                return jnp.sum(self.a.value + self.b.value + self.c.value + 2 * d)

        brainstate.random.seed(0)

        t = Test()
        f_grad = brainstate.augment.grad(t, t.states(), argnums=0)
        var_grads, arg_grads = f_grad(brainstate.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()

        t = Test()
        f_grad = brainstate.augment.grad(t, t.states(), argnums=[0])
        var_grads, arg_grads = f_grad(brainstate.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()

        t = Test()
        f_grad = brainstate.augment.grad(t, argnums=0)
        arg_grads = f_grad(brainstate.random.random(10))
        assert (arg_grads == 2.).all()

        t = Test()
        f_grad = brainstate.augment.grad(t, argnums=[0])
        arg_grads = f_grad(brainstate.random.random(10))
        assert (arg_grads[0] == 2.).all()

    def test_grad_ob_argnums_aux(self):
        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.a = brainstate.ParamState(jnp.ones(10))
                self.b = brainstate.ParamState(brainstate.random.randn(10))
                self.c = brainstate.ParamState(brainstate.random.uniform(size=10))

            def __call__(self, d):
                return jnp.sum(self.a.value + self.b.value + self.c.value + 2 * d), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(0)

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.states(), argnums=0, has_aux=True)
        (var_grads, arg_grads), aux = f_grad(brainstate.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.states(), argnums=[0], has_aux=True)
        (var_grads, arg_grads), aux = f_grad(brainstate.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.augment.grad(t, argnums=0, has_aux=True)
        arg_grads, aux = f_grad(brainstate.random.random(10))
        assert (arg_grads == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.augment.grad(t, argnums=[0], has_aux=True)
        arg_grads, aux = f_grad(brainstate.random.random(10))
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

    def test_grad_ob_argnums_return(self):
        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()

                self.a = brainstate.ParamState(jnp.ones(10))
                self.b = brainstate.ParamState(brainstate.random.randn(10))
                self.c = brainstate.ParamState(brainstate.random.uniform(size=10))

            def __call__(self, d):
                return jnp.sum(self.a.value + self.b.value + self.c.value + 2 * d)

        brainstate.random.seed(0)

        t = Test()
        f_grad = brainstate.augment.grad(t, t.states(), argnums=0, return_value=True)
        d = brainstate.random.random(10)
        (var_grads, arg_grads), loss = f_grad(d)
        for g in var_grads.values():
            assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = brainstate.augment.grad(t, t.states(), argnums=[0], return_value=True)
        d = brainstate.random.random(10)
        (var_grads, arg_grads), loss = f_grad(d)
        for g in var_grads.values():
            assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = brainstate.augment.grad(t, argnums=0, return_value=True)
        d = brainstate.random.random(10)
        arg_grads, loss = f_grad(d)
        assert (arg_grads == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = brainstate.augment.grad(t, argnums=[0], return_value=True)
        d = brainstate.random.random(10)
        arg_grads, loss = f_grad(d)
        assert (arg_grads[0] == 2.).all()
        assert loss == t(d)

    def test_grad_ob_argnums_aux_return(self):
        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.a = brainstate.ParamState(jnp.ones(10))
                self.b = brainstate.ParamState(brainstate.random.randn(10))
                self.c = brainstate.ParamState(brainstate.random.uniform(size=10))

            def __call__(self, d):
                return jnp.sum(self.a.value + self.b.value + self.c.value + 2 * d), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(0)

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.states(), argnums=0, has_aux=True, return_value=True)
        d = brainstate.random.random(10)
        (var_grads, arg_grads), loss, aux = f_grad(d)
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = brainstate.augment.grad(t, grad_states=t.states(), argnums=[0], has_aux=True, return_value=True)
        d = brainstate.random.random(10)
        (var_grads, arg_grads), loss, aux = f_grad(d)
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = brainstate.augment.grad(t, argnums=0, has_aux=True, return_value=True)
        d = brainstate.random.random(10)
        arg_grads, loss, aux = f_grad(d)
        assert (arg_grads == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = brainstate.augment.grad(t, argnums=[0], has_aux=True, return_value=True)
        d = brainstate.random.random(10)
        arg_grads, loss, aux = f_grad(d)
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)
        assert loss == t(d)[0]


class TestPureFuncJacobian(unittest.TestCase):
    def test1(self):
        jac, aux = _jacfwd(lambda x: (x ** 3, [x ** 2]), has_aux=True)(3.)
        self.assertTrue(jax.numpy.allclose(jac, jax.jacfwd(lambda x: x ** 3)(3.)))
        self.assertTrue(aux[0] == 9.)

    def test_jacfwd_and_aux_nested(self):
        def f(x):
            jac, aux = _jacfwd(lambda x: (x ** 3, [x ** 3]), has_aux=True)(x)
            return aux[0]

        f2 = lambda x: x ** 3

        self.assertEqual(_jacfwd(f)(4.), _jacfwd(f2)(4.))
        self.assertEqual(jax.jit(_jacfwd(f))(4.), _jacfwd(f2)(4.))
        self.assertEqual(jax.jit(_jacfwd(jax.jit(f)))(4.), _jacfwd(f2)(4.))

        self.assertEqual(_jacfwd(f)(jnp.asarray(4.)), _jacfwd(f2)(jnp.asarray(4.)))
        self.assertEqual(jax.jit(_jacfwd(f))(jnp.asarray(4.)), _jacfwd(f2)(jnp.asarray(4.)))
        self.assertEqual(jax.jit(_jacfwd(jax.jit(f)))(jnp.asarray(4.)), _jacfwd(f2)(jnp.asarray(4.)))

        def f(x):
            jac, aux = _jacfwd(lambda x: (x ** 3, [x ** 3]), has_aux=True)(x)
            return aux[0] * jnp.sin(x)

        f2 = lambda x: x ** 3 * jnp.sin(x)

        self.assertEqual(_jacfwd(f)(4.), _jacfwd(f2)(4.))
        self.assertEqual(jax.jit(_jacfwd(f))(4.), _jacfwd(f2)(4.))
        self.assertEqual(jax.jit(_jacfwd(jax.jit(f)))(4.), _jacfwd(f2)(4.))

        self.assertEqual(_jacfwd(f)(jnp.asarray(4.)), _jacfwd(f2)(jnp.asarray(4.)))
        self.assertEqual(jax.jit(_jacfwd(f))(jnp.asarray(4.)), _jacfwd(f2)(jnp.asarray(4.)))
        self.assertEqual(jax.jit(_jacfwd(jax.jit(f)))(jnp.asarray(4.)), _jacfwd(f2)(jnp.asarray(4.)))

    def test_jacrev1(self):
        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        br = brainstate.augment.jacrev(f1)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        jr = jax.jacrev(f1)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        assert (br == jr).all()

        br = brainstate.augment.jacrev(f1, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        jr = jax.jacrev(f1, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        assert (br[0] == jr[0]).all()
        assert (br[1] == jr[1]).all()

    def test_jacrev2(self):
        print()

        def f2(x, y):
            r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
            r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r1, r2

        jr = jax.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(jr)

        br = brainstate.augment.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0], jr[0])
        assert jnp.array_equal(br[1], jr[1])

        br = brainstate.augment.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0], jr[0])
        assert jnp.array_equal(br[1], jr[1])

        def f2(x, y):
            r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
            r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r1, r2

        br = brainstate.augment.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0], jr[0])
        assert jnp.array_equal(br[1], jr[1])

        br = brainstate.augment.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0], jr[0])
        assert jnp.array_equal(br[1], jr[1])

    def test_jacrev3(self):
        print()

        def f3(x, y):
            r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
            r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r1, r2

        jr = jax.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(jr)

        br = brainstate.augment.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0][0], jr[0][0])
        assert jnp.array_equal(br[0][1], jr[0][1])
        assert jnp.array_equal(br[1][0], jr[1][0])
        assert jnp.array_equal(br[1][1], jr[1][1])

        br = brainstate.augment.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0][0], jr[0][0])
        assert jnp.array_equal(br[0][1], jr[0][1])
        assert jnp.array_equal(br[1][0], jr[1][0])
        assert jnp.array_equal(br[1][1], jr[1][1])

        def f3(x, y):
            r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
            r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r1, r2

        br = brainstate.augment.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0][0], jr[0][0])
        assert jnp.array_equal(br[0][1], jr[0][1])
        assert jnp.array_equal(br[1][0], jr[1][0])
        assert jnp.array_equal(br[1][1], jr[1][1])

        br = brainstate.augment.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0][0], jr[0][0])
        assert jnp.array_equal(br[0][1], jr[0][1])
        assert jnp.array_equal(br[1][0], jr[1][0])
        assert jnp.array_equal(br[1][1], jr[1][1])

    def test_jacrev_aux1(self):
        x = jnp.array([1., 2., 3.])
        y = jnp.array([10., 5.])

        def f1(x, y):
            a = 4 * x[1] ** 2 - 2 * x[2]
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], a, x[2] * jnp.sin(x[0])])
            return r, a

        f2 = lambda *args: f1(*args)[0]
        jr = jax.jacrev(f2)(x, y)  # jax jacobian
        pprint(jr)
        grads, aux = brainstate.augment.jacrev(f1, has_aux=True)(x, y)
        assert (grads == jr).all()
        assert aux == (4 * x[1] ** 2 - 2 * x[2])

        jr = jax.jacrev(f2, argnums=(0, 1))(x, y)  # jax jacobian
        pprint(jr)
        grads, aux = brainstate.augment.jacrev(f1, argnums=(0, 1), has_aux=True)(x, y)
        assert (grads[0] == jr[0]).all()
        assert (grads[1] == jr[1]).all()
        assert aux == (4 * x[1] ** 2 - 2 * x[2])

    def test_jacrev_return_aux1(self):
        with brainstate.environ.context(precision=64):
            def f1(x, y):
                a = 4 * x[1] ** 2 - 2 * x[2]
                r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], a, x[2] * jnp.sin(x[0])])
                return r, a

            _x = jnp.array([1., 2., 3.])
            _y = jnp.array([10., 5.])
            _r, _a = f1(_x, _y)
            f2 = lambda *args: f1(*args)[0]
            _g1 = jax.jacrev(f2)(_x, _y)  # jax jacobian
            pprint(_g1)
            _g2 = jax.jacrev(f2, argnums=(0, 1))(_x, _y)  # jax jacobian
            pprint(_g2)

            grads, vec, aux = brainstate.augment.jacrev(f1, return_value=True, has_aux=True)(_x, _y)
            assert (grads == _g1).all()
            assert aux == _a
            assert (vec == _r).all()

            grads, vec, aux = brainstate.augment.jacrev(f1, return_value=True, argnums=(0, 1), has_aux=True)(_x, _y)
            assert (grads[0] == _g2[0]).all()
            assert (grads[1] == _g2[1]).all()
            assert aux == _a
            assert (vec == _r).all()


class TestClassFuncJacobian(unittest.TestCase):
    def test_jacrev1(self):
        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = jnp.array([1., 2., 3.])
        _y = jnp.array([10., 5.])

        class Test(brainstate.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.x = brainstate.State(jnp.array([1., 2., 3.]))
                self.y = brainstate.State(jnp.array([10., 5.]))

            def __call__(self, ):
                a = self.x.value[0] * self.y.value[0]
                b = 5 * self.x.value[2] * self.y.value[1]
                c = 4 * self.x.value[1] ** 2 - 2 * self.x.value[2]
                d = self.x.value[2] * jnp.sin(self.x.value[0])
                r = jnp.asarray([a, b, c, d])
                return r

        _jr = jax.jacrev(f1)(_x, _y)
        t = Test()
        br = brainstate.augment.jacrev(t, grad_states=t.x)()
        self.assertTrue((br == _jr).all())

        _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
        t = Test()
        br = brainstate.augment.jacrev(t, grad_states=[t.x, t.y])()
        self.assertTrue((br[0] == _jr[0]).all())
        self.assertTrue((br[1] == _jr[1]).all())


#
#   def test_jacfwd1(self):
#     def f1(x, y):
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r
#
#     _x = jnp.array([1., 2., 3.])
#     _y = jnp.array([10., 5.])
#
#     class Test(brainstate.nn.Module):
#       def __init__(self):
#         super(Test, self).__init__()
#         self.x = jnp.Variable(jnp.array([1., 2., 3.]))
#         self.y = jnp.Variable(jnp.array([10., 5.]))
#
#       def __call__(self, ):
#         a = self.x[0] * self.y[0]
#         b = 5 * self.x[2] * self.y[1]
#         c = 4 * self.x[1] ** 2 - 2 * self.x[2]
#         d = self.x[2] * jnp.sin(self.x[0])
#         r = jnp.asarray([a, b, c, d])
#         return r
#
#     _jr = jax.jacfwd(f1)(_x, _y)
#     t = Test()
#     br = brainstate.augment.jacfwd(t, grad_states=t.x)()
#     self.assertTrue((br == _jr).all())
#
#     _jr = jax.jacfwd(f1, argnums=(0, 1))(_x, _y)
#     t = Test()
#     br = brainstate.augment.jacfwd(t, grad_states=[t.x, t.y])()
#     self.assertTrue((br[0] == _jr[0]).all())
#     self.assertTrue((br[1] == _jr[1]).all())
#
#   def test_jacrev2(self):
#     def f1(x, y):
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r
#
#     _x = jnp.array([1., 2., 3.])
#     _y = jnp.array([10., 5.])
#
#     class Test(brainstate.nn.Module):
#       def __init__(self):
#         super(Test, self).__init__()
#         self.x = jnp.Variable(jnp.array([1., 2., 3.]))
#
#       def __call__(self, y):
#         a = self.x[0] * y[0]
#         b = 5 * self.x[2] * y[1]
#         c = 4 * self.x[1] ** 2 - 2 * self.x[2]
#         d = self.x[2] * jnp.sin(self.x[0])
#         r = jnp.asarray([a, b, c, d])
#         return r
#
#     _jr = jax.jacrev(f1)(_x, _y)
#     t = Test()
#     br = brainstate.augment.jacrev(t, grad_states=t.x)(_y)
#     self.assertTrue((br == _jr).all())
#
#     _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
#     t = Test()
#     var_grads, arg_grads = brainstate.augment.jacrev(t, grad_states=t.x, argnums=0)(_y)
#     print(var_grads, )
#     print(arg_grads, )
#     self.assertTrue((var_grads == _jr[0]).all())
#     self.assertTrue((arg_grads == _jr[1]).all())
#
#   def test_jacfwd2(self):
#     def f1(x, y):
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r
#
#     _x = jnp.array([1., 2., 3.])
#     _y = jnp.array([10., 5.])
#
#     class Test(brainstate.nn.Module):
#       def __init__(self):
#         super(Test, self).__init__()
#         self.x = jnp.Variable(jnp.array([1., 2., 3.]))
#
#       def __call__(self, y):
#         a = self.x[0] * y[0]
#         b = 5 * self.x[2] * y[1]
#         c = 4 * self.x[1] ** 2 - 2 * self.x[2]
#         d = self.x[2] * jnp.sin(self.x[0])
#         r = jnp.asarray([a, b, c, d])
#         return r
#
#     _jr = jax.jacfwd(f1)(_x, _y)
#     t = Test()
#     br = brainstate.augment.jacfwd(t, grad_states=t.x)(_y)
#     self.assertTrue((br == _jr).all())
#
#     _jr = jax.jacfwd(f1, argnums=(0, 1))(_x, _y)
#     t = Test()
#     var_grads, arg_grads = brainstate.augment.jacfwd(t, grad_states=t.x, argnums=0)(_y)
#     print(var_grads, )
#     print(arg_grads, )
#     self.assertTrue((var_grads == _jr[0]).all())
#     self.assertTrue((arg_grads == _jr[1]).all())
#
#   def test_jacrev_aux1(self):
#     jnp.enable_x64()
#
#     def f1(x, y):
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r
#
#     _x = jnp.array([1., 2., 3.])
#     _y = jnp.array([10., 5.])
#
#     class Test(brainstate.nn.Module):
#       def __init__(self):
#         super(Test, self).__init__()
#         self.x = jnp.Variable(jnp.array([1., 2., 3.]))
#
#       def __call__(self, y):
#         a = self.x[0] * y[0]
#         b = 5 * self.x[2] * y[1]
#         c = 4 * self.x[1] ** 2 - 2 * self.x[2]
#         d = self.x[2] * jnp.sin(self.x[0])
#         r = jnp.asarray([a, b, c, d])
#         return r, (c, d)
#
#     _jr = jax.jacrev(f1)(_x, _y)
#     t = Test()
#     br, _ = brainstate.augment.jacrev(t, grad_states=t.x, has_aux=True)(_y)
#     self.assertTrue((br == _jr).all())
#
#     t = Test()
#     _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
#     _aux = t(_y)[1]
#     (var_grads, arg_grads), aux = brainstate.augment.jacrev(t, grad_states=t.x, argnums=0, has_aux=True)(_y)
#     print(var_grads, )
#     print(arg_grads, )
#     self.assertTrue((var_grads == _jr[0]).all())
#     self.assertTrue((arg_grads == _jr[1]).all())
#     self.assertTrue(jnp.array_equal(aux, _aux))
#
#     jnp.disable_x64()
#
#   def test_jacfwd_aux1(self):
#     jnp.enable_x64()
#
#     def f1(x, y):
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r
#
#     _x = jnp.array([1., 2., 3.])
#     _y = jnp.array([10., 5.])
#
#     class Test(brainstate.nn.Module):
#       def __init__(self):
#         super(Test, self).__init__()
#         self.x = jnp.Variable(jnp.array([1., 2., 3.]))
#
#       def __call__(self, y):
#         a = self.x[0] * y[0]
#         b = 5 * self.x[2] * y[1]
#         c = 4 * self.x[1] ** 2 - 2 * self.x[2]
#         d = self.x[2] * jnp.sin(self.x[0])
#         r = jnp.asarray([a, b, c, d])
#         return r, (c, d)
#
#     _jr = jax.jacfwd(f1)(_x, _y)
#     t = Test()
#     br, (c, d) = brainstate.augment.jacfwd(t, grad_states=t.x, has_aux=True)(_y)
#     # print(_jr)
#     # print(br)
#     a = (br == _jr)
#     self.assertTrue(a.all())
#
#     t = Test()
#     _jr = jax.jacfwd(f1, argnums=(0, 1))(_x, _y)
#     _aux = t(_y)[1]
#     (var_grads, arg_grads), aux = brainstate.augment.jacfwd(t, grad_states=t.x, argnums=0, has_aux=True)(_y)
#     print(var_grads, )
#     print(arg_grads, )
#     self.assertTrue((var_grads == _jr[0]).all())
#     self.assertTrue((arg_grads == _jr[1]).all())
#     self.assertTrue(jnp.array_equal(aux, _aux))
#
#     jnp.disable_x64()
#
#   def test_jacrev_return_aux1(self):
#     jnp.enable_x64()
#
#     def f1(x, y):
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r
#
#     _x = jnp.array([1., 2., 3.])
#     _y = jnp.array([10., 5.])
#
#     class Test(brainstate.nn.Module):
#       def __init__(self):
#         super(Test, self).__init__()
#         self.x = jnp.Variable(jnp.array([1., 2., 3.]))
#
#       def __call__(self, y):
#         a = self.x[0] * y[0]
#         b = 5 * self.x[2] * y[1]
#         c = 4 * self.x[1] ** 2 - 2 * self.x[2]
#         d = self.x[2] * jnp.sin(self.x[0])
#         r = jnp.asarray([a, b, c, d])
#         return r, (c, d)
#
#     _jr = jax.jacrev(f1)(_x, _y)
#     t = Test()
#     br, _ = brainstate.augment.jacrev(t, grad_states=t.x, has_aux=True)(_y)
#     self.assertTrue((br == _jr).all())
#
#     t = Test()
#     _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
#     _val, _aux = t(_y)
#     (var_grads, arg_grads), value, aux = brainstate.augment.jacrev(t, grad_states=t.x, argnums=0, has_aux=True, return_value=True)(_y)
#     print(var_grads, )
#     print(arg_grads, )
#     self.assertTrue((var_grads == _jr[0]).all())
#     self.assertTrue((arg_grads == _jr[1]).all())
#     self.assertTrue(jnp.array_equal(aux, _aux))
#     self.assertTrue(jnp.array_equal(value, _val))
#
#     jnp.disable_x64()
#
#   def test_jacfwd_return_aux1(self):
#     jnp.enable_x64()
#
#     def f1(x, y):
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r
#
#     _x = jnp.array([1., 2., 3.])
#     _y = jnp.array([10., 5.])
#
#     class Test(brainstate.nn.Module):
#       def __init__(self):
#         super(Test, self).__init__()
#         self.x = jnp.Variable(jnp.array([1., 2., 3.]))
#
#       def __call__(self, y):
#         a = self.x[0] * y[0]
#         b = 5 * self.x[2] * y[1]
#         c = 4 * self.x[1] ** 2 - 2 * self.x[2]
#         d = self.x[2] * jnp.sin(self.x[0])
#         r = jnp.asarray([a, b, c, d])
#         return r, (c, d)
#
#     _jr = jax.jacfwd(f1)(_x, _y)
#     t = Test()
#     br, _ = brainstate.augment.jacfwd(t, grad_states=t.x, has_aux=True)(_y)
#     self.assertTrue((br == _jr).all())
#
#     t = Test()
#     _jr = jax.jacfwd(f1, argnums=(0, 1))(_x, _y)
#     _val, _aux = t(_y)
#     (var_grads, arg_grads), value, aux = brainstate.augment.jacfwd(t, grad_states=t.x, argnums=0, has_aux=True, return_value=True)(_y)
#     print(_val, )
#     print('_aux: ', _aux, 'aux: ', aux)
#     print(var_grads, )
#     print(arg_grads, )
#     self.assertTrue((var_grads == _jr[0]).all())
#     self.assertTrue((arg_grads == _jr[1]).all())
#     self.assertTrue(jnp.array_equal(aux, _aux))
#     self.assertTrue(jnp.array_equal(value, _val))
#
#     jnp.disable_x64()
#
#
# class TestPureFuncVectorGrad(unittest.TestCase):
#   def test1(self):
#     f = lambda x: 3 * x ** 2
#     _x = jnp.ones(10)
#     pprint(brainstate.augment.vector_grad(f, argnums=0)(_x))
#
#   def test2(self):
#     def f(x, y):
#       dx = x ** 2 + y ** 2 + 10
#       return dx
#
#     _x = jnp.ones(5)
#     _y = jnp.ones(5)
#
#     g = brainstate.augment.vector_grad(f, argnums=0)(_x, _y)
#     pprint(g)
#     self.assertTrue(jnp.array_equal(g, 2 * _x))
#
#     g = brainstate.augment.vector_grad(f, argnums=(0,))(_x, _y)
#     self.assertTrue(jnp.array_equal(g[0], 2 * _x))
#
#     g = brainstate.augment.vector_grad(f, argnums=(0, 1))(_x, _y)
#     pprint(g)
#     self.assertTrue(jnp.array_equal(g[0], 2 * _x))
#     self.assertTrue(jnp.array_equal(g[1], 2 * _y))
#
#   def test3(self):
#     def f(x, y):
#       dx = x ** 2 + y ** 2 + 10
#       dy = x ** 3 + y ** 3 - 10
#       return dx, dy
#
#     _x = jnp.ones(5)
#     _y = jnp.ones(5)
#
#     g = brainstate.augment.vector_grad(f, argnums=0)(_x, _y)
#     # pprint(g)
#     self.assertTrue(jnp.array_equal(g, 2 * _x + 3 * _x ** 2))
#
#     g = brainstate.augment.vector_grad(f, argnums=(0,))(_x, _y)
#     self.assertTrue(jnp.array_equal(g[0], 2 * _x + 3 * _x ** 2))
#
#     g = brainstate.augment.vector_grad(f, argnums=(0, 1))(_x, _y)
#     # pprint(g)
#     self.assertTrue(jnp.array_equal(g[0], 2 * _x + 3 * _x ** 2))
#     self.assertTrue(jnp.array_equal(g[1], 2 * _y + 3 * _y ** 2))
#
#   def test4_2d(self):
#     def f(x, y):
#       dx = x ** 2 + y ** 2 + 10
#       return dx
#
#     _x = jnp.ones((5, 5))
#     _y = jnp.ones((5, 5))
#
#     g = brainstate.augment.vector_grad(f, argnums=0)(_x, _y)
#     pprint(g)
#     self.assertTrue(jnp.array_equal(g, 2 * _x))
#
#     g = brainstate.augment.vector_grad(f, argnums=(0,))(_x, _y)
#     self.assertTrue(jnp.array_equal(g[0], 2 * _x))
#
#     g = brainstate.augment.vector_grad(f, argnums=(0, 1))(_x, _y)
#     pprint(g)
#     self.assertTrue(jnp.array_equal(g[0], 2 * _x))
#     self.assertTrue(jnp.array_equal(g[1], 2 * _y))
#
#   def test_aux1(self):
#     def f(x, y):
#       dx = x ** 2 + y ** 2 + 10
#       dy = x ** 3 + y ** 3 - 10
#       return dx, dy
#
#     _x = jnp.ones(5)
#     _y = jnp.ones(5)
#
#     g, aux = brainstate.augment.vector_grad(f, has_aux=True)(_x, _y)
#     pprint(g, )
#     pprint(aux)
#     self.assertTrue(jnp.array_equal(g, 2 * _x))
#     self.assertTrue(jnp.array_equal(aux, _x ** 3 + _y ** 3 - 10))
#
#   def test_return1(self):
#     def f(x, y):
#       dx = x ** 2 + y ** 2 + 10
#       return dx
#
#     _x = jnp.ones(5)
#     _y = jnp.ones(5)
#
#     g, value = brainstate.augment.vector_grad(f, return_value=True)(_x, _y)
#     pprint(g, )
#     pprint(value)
#     self.assertTrue(jnp.array_equal(g, 2 * _x))
#     self.assertTrue(jnp.array_equal(value, _x ** 2 + _y ** 2 + 10))
#
#   def test_return_aux1(self):
#     def f(x, y):
#       dx = x ** 2 + y ** 2 + 10
#       dy = x ** 3 + y ** 3 - 10
#       return dx, dy
#
#     _x = jnp.ones(5)
#     _y = jnp.ones(5)
#
#     g, value, aux = brainstate.augment.vector_grad(f, has_aux=True, return_value=True)(_x, _y)
#     print('grad', g)
#     print('value', value)
#     print('aux', aux)
#     self.assertTrue(jnp.array_equal(g, 2 * _x))
#     self.assertTrue(jnp.array_equal(value, _x ** 2 + _y ** 2 + 10))
#     self.assertTrue(jnp.array_equal(aux, _x ** 3 + _y ** 3 - 10))
#
#
# class TestClassFuncVectorGrad(unittest.TestCase):
#   def test1(self):
#     class Test(brainstate.nn.Module):
#       def __init__(self):
#         super(Test, self).__init__()
#         self.x = jnp.Variable(jnp.ones(5))
#         self.y = jnp.Variable(jnp.ones(5))
#
#       def __call__(self, *args, **kwargs):
#         return self.x ** 2 + self.y ** 2 + 10
#
#     t = Test()
#
#     g = brainstate.augment.vector_grad(t, grad_states=t.x)()
#     self.assertTrue(jnp.array_equal(g, 2 * t.x))
#
#     g = brainstate.augment.vector_grad(t, grad_states=(t.x,))()
#     self.assertTrue(jnp.array_equal(g[0], 2 * t.x))
#
#     g = brainstate.augment.vector_grad(t, grad_states=(t.x, t.y))()
#     self.assertTrue(jnp.array_equal(g[0], 2 * t.x))
#     self.assertTrue(jnp.array_equal(g[1], 2 * t.y))
#
#
# def vgrad(f, *x):
#   y, vjp_fn = jax.vjp(f, *x)
#   return vjp_fn(jnp.ones(y.shape).value)[0]
#
#
# class TestDebug(parameterized.TestCase):
#   def test_debug1(self):
#     a = brainstate.random.RandomState()
#
#     def f(b):
#       print(a.value)
#       return a + b + a.random()
#
#     f = brainstate.augment.vector_grad(f, argnums=0)
#     f(1.)
#
#     with jax.disable_jit():
#       f(1.)
#
#   @parameterized.product(
#     grad_fun=[brainstate.augment.grad, brainstate.augment.vector_grad]
#   )
#   def test_print_info1(self, grad_fun):
#     file = tempfile.TemporaryFile(mode='w+')
#
#     @functools.partial(grad_fun, argnums=0)
#     def f2(a, b):
#       print('compiling f2 ...', file=file)
#       return a + b
#
#     @functools.partial(grad_fun, argnums=0)
#     def f1(a):
#       print('compiling f1 ...', file=file)
#       return f2(a, 1.)
#
#     expect_res = '''
# compiling f1 ...
# compiling f2 ...
# compiling f1 ...
# compiling f2 ...
#     '''
#
#     print(f1(1.))
#     file.seek(0)
#     self.assertTrue(file.read().strip() == expect_res.strip())
#
#     file = tempfile.TemporaryFile(mode='w+')
#     with jax.disable_jit():
#       expect_res = '''
# compiling f1 ...
# compiling f2 ...
#       '''
#       self.assertTrue(f1(1.) == 0.)
#       file.seek(0)
#       self.assertTrue(file.read().strip() == expect_res.strip())
#
#   @parameterized.product(
#     grad_fun=[brainstate.augment.grad, brainstate.augment.vector_grad]
#   )
#   def test_print_info2(self, grad_fun):
#     file = tempfile.TemporaryFile(mode='w+')
#
#     @functools.partial(grad_fun, argnums=0)
#     def f1(a):
#       @functools.partial(grad_fun, argnums=0)
#       def f2(a, b):
#         print('compiling f2 ...', file=file)
#         return a + b
#
#       print('compiling f1 ...', file=file)
#       return f2(a, 1.)
#
#     expect_res = '''
# compiling f1 ...
# compiling f2 ...
# compiling f1 ...
# compiling f2 ...
# compiling f2 ...
#     '''
#     self.assertTrue(f1(1.) == 0.)
#     file.seek(0)
#     self.assertTrue(file.read().strip() == expect_res.strip())
#
#     file = tempfile.TemporaryFile(mode='w+')
#     with jax.disable_jit():
#       expect_res = '''
# compiling f1 ...
# compiling f2 ...
#       '''
#       self.assertTrue(f1(1.) == 0.)
#       file.seek(0)
#       # print(file.read().strip())
#       self.assertTrue(file.read().strip() == expect_res.strip())
#
#   def test_debug_correctness1(self):
#     def test_f():
#       a = jnp.Variable(jnp.ones(2))
#       b = jnp.Variable(jnp.zeros(2))
#
#       @brainstate.augment.vector_grad(argnums=0)
#       def f1(c):
#         a.value += 1
#         b.value += 10
#         return a * b * c
#
#       return a, b, f1(1.)
#
#     r1 = test_f()
#     print(r1)
#
#     with jax.disable_jit():
#       r2 = test_f()
#       print(r2)
#       self.assertTrue(jnp.allclose(r1[0], r2[0]))
#       self.assertTrue(jnp.allclose(r1[1], r2[1]))
#       self.assertTrue(jnp.allclose(r1[2], r2[2]))
#
#     def f1(c, a, b):
#       a += 1
#       b += 10
#       return a * b * c
#
#     r3 = vgrad(f1, 1., jnp.ones(2).value, jnp.zeros(2).value)
#     self.assertTrue(jnp.allclose(r1[2], r3))
#
#   def _bench_f2(self, dd):
#     a = jnp.Variable(jnp.ones(2))
#     b = jnp.Variable(jnp.zeros(2))
#
#
#     def run_fun(d):
#       @brainstate.augment.vector_grad(argnums=0)
#       def f1(c):
#         a.value += d
#         b.value += 10
#         return a * b * c
#
#       return a, b, f1(1.)
#
#     return run_fun(dd)
#
#   def test_debug_correctness2(self):
#     r1 = self._bench_f2(1.)
#     print(r1)
#
#     with jax.disable_jit():
#       r2 = self._bench_f2(1.)
#       print(r2)
#
#     self.assertTrue(jnp.allclose(r1[0], r2[0]))
#     self.assertTrue(jnp.allclose(r1[1], r2[1]))
#     self.assertTrue(jnp.allclose(r1[2], r2[2]))
#
#   def test_cache1(self):
#       file = tempfile.TemporaryFile(mode='w+')
#
#       def f(a, b):
#         print('compiling f ...', file=file)
#         return a + b
#
#       grad1 = brainstate.augment.grad(f)(1., 2.)  # call "f" twice, one for Variable finding, one for compiling
#       grad2 = brainstate.augment.vector_grad(f)(1., 2.)  # call "f" once for compiling
#
#       file.seek(0)
#       print(file.read().strip())
#
#       expect_res = '''
# compiling f ...
# compiling f ...
# compiling f ...
#       '''
#       file.seek(0)
#       self.assertTrue(file.read().strip() == expect_res.strip())
#
#


class TestUnitAwareGrad(unittest.TestCase):
    def test_grad1(self):
        def f(x):
            return u.math.sum(x ** 2)

        x = jnp.array([1., 2., 3.]) * u.ms
        g = brainstate.augment.grad(f, unit_aware=True)(x)
        self.assertTrue(u.math.allclose(g, 2 * x))

    def test_vector_grad1(self):
        def f(x):
            return x ** 3

        x = jnp.array([1., 2., 3.]) * u.ms
        g = brainstate.augment.vector_grad(f, unit_aware=True)(x)
        self.assertTrue(u.math.allclose(g, 3 * x ** 2))

    def test_jacrev1(self):
        def f(x, y):
            return u.math.asarray([x[0] * y[0],
                                   5 * x[2] * y[1],
                                   4 * x[1] ** 2, ])

        _x = jnp.array([1., 2., 3.]) * u.ms
        _y = jnp.array([10., 5.]) * u.ms

        g = brainstate.augment.jacrev(f, unit_aware=True, argnums=(0, 1))(_x, _y)
        self.assertTrue(
            u.math.allclose(
                g[0],
                u.math.asarray([
                    [10., 0., 0.],
                    [0., 0., 25.],
                    [0., 16., 0.]
                ]) * u.ms
            )
        )

        self.assertTrue(
            u.math.allclose(
                g[1],
                u.math.asarray([
                    [1., 0.],
                    [0., 15.],
                    [0., 0.]
                ]) * u.ms
            )
        )

    def test_jacfwd1(self):
        def f(x, y):
            return u.math.asarray([x[0] * y[0],
                                   5 * x[2] * y[1],
                                   4 * x[1] ** 2, ])

        _x = jnp.array([1., 2., 3.]) * u.ms
        _y = jnp.array([10., 5.]) * u.ms

        g = brainstate.augment.jacfwd(f, unit_aware=True, argnums=(0, 1))(_x, _y)
        self.assertTrue(
            u.math.allclose(
                g[0],
                u.math.asarray([
                    [10., 0., 0.],
                    [0., 0., 25.],
                    [0., 16., 0.]
                ]) * u.ms
            )
        )

        self.assertTrue(
            u.math.allclose(
                g[1],
                u.math.asarray([
                    [1., 0.],
                    [0., 15.],
                    [0., 0.]
                ]) * u.ms
            )
        )

    def test_hessian(self):
        unit = u.ms

        def scalar_function(x):
            return x ** 3 + 3 * x * unit * unit + 2 * unit * unit * unit

        hess = brainstate.augment.hessian(scalar_function, unit_aware=True)
        x = jnp.array(1.0) * unit
        res = hess(x)
        expected_hessian = jnp.array([[6.0]]) * unit
        assert u.math.allclose(res, expected_hessian)
