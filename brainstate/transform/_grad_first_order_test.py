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

import brainunit as u
import jax.numpy as jnp
import pytest

import brainstate


class TestPureFuncGrad(unittest.TestCase):
    def test_grad_pure_func_1(self):
        def call(a, b, c): return jnp.sum(a + b + c)

        brainstate.random.seed(1)
        a = jnp.ones(10)
        b = brainstate.random.randn(10)
        c = brainstate.random.uniform(size=10)
        f_grad = brainstate.transform.grad(call, argnums=[0, 1, 2])
        grads = f_grad(a, b, c)

        for g in grads: assert (g == 1.).all()

    def test_grad_pure_func_2(self):
        def call(a, b, c): return jnp.sum(a + b + c)

        brainstate.random.seed(1)
        a = jnp.ones(10)
        b = brainstate.random.randn(10)
        c = brainstate.random.uniform(size=10)
        f_grad = brainstate.transform.grad(call)
        assert (f_grad(a, b, c) == 1.).all()

    def test_grad_pure_func_aux1(self):
        def call(a, b, c):
            return jnp.sum(a + b + c), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(1)
        f_grad = brainstate.transform.grad(call, argnums=[0, 1, 2])
        with pytest.raises(TypeError):
            f_grad(jnp.ones(10), brainstate.random.randn(10), brainstate.random.uniform(size=10))

    def test_grad_pure_func_aux2(self):
        def call(a, b, c):
            return jnp.sum(a + b + c), (jnp.sin(100), jnp.exp(0.1))

        brainstate.random.seed(1)
        f_grad = brainstate.transform.grad(call, argnums=[0, 1, 2], has_aux=True)
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
        f_grad = brainstate.transform.grad(call, return_value=True)
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
        f_grad = brainstate.transform.grad(call, return_value=True, has_aux=True)
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
        f_grad = brainstate.transform.grad(t, grad_states={'a': t.a, 'b': t.b, 'c': t.c})
        grads = f_grad()
        for g in grads.values():
            assert (g == 1.).all()

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=[t.a, t.b])
        grads = f_grad()
        for g in grads: assert (g == 1.).all()

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=t.a)
        grads = f_grad()
        assert (grads == 1.).all()

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=t.states())
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
        f_grad = brainstate.transform.grad(t, grad_states=[t.a, t.b], has_aux=True)
        grads, aux = f_grad()
        for g in grads: assert (g == 1.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=t.a, has_aux=True)
        grads, aux = f_grad()
        assert (grads == 1.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=t.states(), has_aux=True)
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
        f_grad = brainstate.transform.grad(t, grad_states=[t.a, t.b], return_value=True)
        grads, returns = f_grad()
        for g in grads: assert (g == 1.).all()
        assert returns == t()

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=t.a, return_value=True)
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
        f_grad = brainstate.transform.grad(t, grad_states=[t.a, t.b], has_aux=True, return_value=True)
        grads, returns, aux = f_grad()
        for g in grads: assert (g == 1.).all()
        assert returns == jnp.sum(t.a.value + t.b.value + t.c.value)
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=t.a, has_aux=True, return_value=True)
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
        f_grad = brainstate.transform.grad(t, t.states(), argnums=0)
        var_grads, arg_grads = f_grad(brainstate.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()

        t = Test()
        f_grad = brainstate.transform.grad(t, t.states(), argnums=[0])
        var_grads, arg_grads = f_grad(brainstate.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()

        t = Test()
        f_grad = brainstate.transform.grad(t, argnums=0)
        arg_grads = f_grad(brainstate.random.random(10))
        assert (arg_grads == 2.).all()

        t = Test()
        f_grad = brainstate.transform.grad(t, argnums=[0])
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
        f_grad = brainstate.transform.grad(t, grad_states=t.states(), argnums=0, has_aux=True)
        (var_grads, arg_grads), aux = f_grad(brainstate.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=t.states(), argnums=[0], has_aux=True)
        (var_grads, arg_grads), aux = f_grad(brainstate.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.transform.grad(t, argnums=0, has_aux=True)
        arg_grads, aux = f_grad(brainstate.random.random(10))
        assert (arg_grads == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)

        t = Test()
        f_grad = brainstate.transform.grad(t, argnums=[0], has_aux=True)
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
        f_grad = brainstate.transform.grad(t, t.states(), argnums=0, return_value=True)
        d = brainstate.random.random(10)
        (var_grads, arg_grads), loss = f_grad(d)
        for g in var_grads.values():
            assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = brainstate.transform.grad(t, t.states(), argnums=[0], return_value=True)
        d = brainstate.random.random(10)
        (var_grads, arg_grads), loss = f_grad(d)
        for g in var_grads.values():
            assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = brainstate.transform.grad(t, argnums=0, return_value=True)
        d = brainstate.random.random(10)
        arg_grads, loss = f_grad(d)
        assert (arg_grads == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = brainstate.transform.grad(t, argnums=[0], return_value=True)
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
        f_grad = brainstate.transform.grad(t, grad_states=t.states(), argnums=0, has_aux=True, return_value=True)
        d = brainstate.random.random(10)
        (var_grads, arg_grads), loss, aux = f_grad(d)
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = brainstate.transform.grad(t, grad_states=t.states(), argnums=[0], has_aux=True, return_value=True)
        d = brainstate.random.random(10)
        (var_grads, arg_grads), loss, aux = f_grad(d)
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = brainstate.transform.grad(t, argnums=0, has_aux=True, return_value=True)
        d = brainstate.random.random(10)
        arg_grads, loss, aux = f_grad(d)
        assert (arg_grads == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = brainstate.transform.grad(t, argnums=[0], has_aux=True, return_value=True)
        d = brainstate.random.random(10)
        arg_grads, loss, aux = f_grad(d)
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == jnp.sin(100)
        assert aux[1] == jnp.exp(0.1)
        assert loss == t(d)[0]


class TestUnitAwareGrad(unittest.TestCase):
    def test_grad1(self):
        def f(x):
            return u.math.sum(x ** 2)

        x = jnp.array([1., 2., 3.]) * u.ms
        g = brainstate.transform.grad(f, unit_aware=True)(x)
        self.assertTrue(u.math.allclose(g, 2 * x))

    def test_vector_grad1(self):
        def f(x):
            return x ** 3

        x = jnp.array([1., 2., 3.]) * u.ms
        g = brainstate.transform.vector_grad(f, unit_aware=True)(x)
        self.assertTrue(u.math.allclose(g, 3 * x ** 2))

    def test_jacrev1(self):
        def f(x, y):
            return u.math.asarray([x[0] * y[0],
                                   5 * x[2] * y[1],
                                   4 * x[1] ** 2, ])

        _x = jnp.array([1., 2., 3.]) * u.ms
        _y = jnp.array([10., 5.]) * u.ms

        g = brainstate.transform.jacrev(f, unit_aware=True, argnums=(0, 1))(_x, _y)
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

        g = brainstate.transform.jacfwd(f, unit_aware=True, argnums=(0, 1))(_x, _y)
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

        hess = brainstate.transform.hessian(scalar_function, unit_aware=True)
        x = jnp.array(1.0) * unit
        res = hess(x)
        expected_hessian = jnp.array([[6.0]]) * unit
        assert u.math.allclose(res, expected_hessian)
