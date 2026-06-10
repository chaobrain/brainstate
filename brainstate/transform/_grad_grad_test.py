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
import jax
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


class TestGradDecoratorForm(unittest.TestCase):
    """The decorator-factory form: ``grad(...)`` returns a decorator."""

    def test_grad_decorator_factory_over_state(self):
        """``grad(grad_states=...)`` used as a decorator differentiates the state."""
        p = brainstate.ParamState(jnp.array([3.0, 4.0]))

        @brainstate.transform.grad(grad_states=p)
        def loss():
            return jnp.sum(p.value ** 2)

        grads = loss()
        self.assertTrue(bool(jnp.allclose(grads, jnp.array([6.0, 8.0]))))

    def test_vector_grad_decorator_factory(self):
        """``vector_grad(grad_states=...)`` works as a decorator."""
        p = brainstate.ParamState(jnp.array([2.0, 3.0]))

        @brainstate.transform.vector_grad(grad_states=p)
        def model():
            return p.value ** 2

        grads = model()
        self.assertTrue(bool(jnp.allclose(grads, jnp.array([4.0, 6.0]))))

    def test_vector_grad_direct_elementwise(self):
        """``vector_grad`` gives per-element gradients for a vector output."""
        g = brainstate.transform.vector_grad(lambda x: x ** 2)(jnp.array([1.0, 2.0, 3.0]))
        self.assertTrue(bool(jnp.allclose(g, jnp.array([2.0, 4.0, 6.0]))))


class TestForwardGrad(unittest.TestCase):
    """Forward-mode gradient estimation via :func:`fwd_grad`.

    Forward-mode uses random tangents, so the estimate is stochastic; the tests
    fix the key for reproducibility and assert structure/finiteness rather than
    exact values, plus exact agreement of the single-direction estimator with
    its analytic construction for a quadratic.
    """

    def test_fwd_grad_single_direction_shape(self):
        """A single tangent yields an estimate with the input's shape."""
        key = brainstate.random.split_key()
        g = brainstate.transform.fwd_grad(lambda x: jnp.sum(x ** 2), key=key)(
            jnp.array([1.0, 2.0, 3.0])
        )
        self.assertEqual(g.shape, (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    def test_fwd_grad_averaged_directions(self):
        """``tangent_size`` averages several random directions."""
        key = brainstate.random.split_key()
        g = brainstate.transform.fwd_grad(
            lambda x: jnp.sum(x ** 2), tangent_size=16, key=key
        )(jnp.ones((4,)))
        self.assertEqual(g.shape, (4,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    def test_fwd_grad_with_clip(self):
        """``drct_der_clip`` bounds the directional derivative without error."""
        key = brainstate.random.split_key()
        g = brainstate.transform.fwd_grad(
            lambda x: jnp.sum(x ** 2), drct_der_clip=1.0, key=key
        )(jnp.ones((3,)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    def test_fwd_grad_has_aux(self):
        """``has_aux=True`` returns ``(grads, aux)``."""
        key = brainstate.random.split_key()

        def f(x):
            return jnp.sum(x ** 2), {"n": x.shape[0]}

        g, aux = brainstate.transform.fwd_grad(f, has_aux=True, key=key)(jnp.ones((2,)))
        self.assertEqual(g.shape, (2,))
        self.assertEqual(aux["n"], 2)

    def test_fwd_grad_averaged_with_clip(self):
        """Averaging directions together with ``drct_der_clip`` runs cleanly."""
        key = brainstate.random.split_key()
        g = brainstate.transform.fwd_grad(
            lambda x: jnp.sum(x ** 2), tangent_size=8, drct_der_clip=0.5, key=key
        )(jnp.ones((3,)))
        self.assertEqual(g.shape, (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    def test_fwd_grad_over_state(self):
        """``fwd_grad`` w.r.t. a state returns a gradient with the state shape."""
        key = brainstate.random.split_key()
        p = brainstate.ParamState(jnp.array([1.0, 2.0]))

        @brainstate.transform.fwd_grad(grad_states=p, key=key)
        def loss():
            return jnp.sum(p.value ** 2)

        grads = loss()
        leaf = jax.tree.leaves(grads)[0]
        self.assertEqual(leaf.shape, (2,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))


class TestArgnumsValidation(unittest.TestCase):
    """Negative or non-integer ``argnums`` must be rejected (audit H3).

    The internal calling convention shifts user argnums by +2 (two leading
    state-value slots), so a negative argnum silently differentiated an
    internal slot — e.g. ``argnums=-1`` returned the gradient w.r.t. the
    state-value dict keyed by raw ``id()`` — instead of raising.
    """

    def _apis(self):
        t = brainstate.transform
        return [t.grad, t.vector_grad, t.jacrev, t.jacfwd, t.hessian]

    def test_negative_int_argnums_raises(self):
        def loss(x, y):
            return (x * y) ** 2

        for api in self._apis():
            with self.assertRaises(ValueError, msg=f'api={api}'):
                api(loss, argnums=-1)
        key = brainstate.random.split_key()
        with self.assertRaises(ValueError):
            brainstate.transform.fwd_grad(loss, argnums=-1, key=key)

    def test_negative_argnums_in_sequence_raises(self):
        def loss(x, y):
            return (x * y) ** 2

        for api in self._apis():
            with self.assertRaises(ValueError, msg=f'api={api}'):
                api(loss, argnums=[0, -1])

    def test_negative_argnums_with_grad_states_raises(self):
        w = brainstate.State(jnp.asarray(1.0))

        def loss(x, y):
            return (w.value * x * y) ** 2

        with self.assertRaises(ValueError):
            brainstate.transform.grad(loss, grad_states=w, argnums=-1)

    def test_non_integer_argnums_raises(self):
        def loss(x, y):
            return (x * y) ** 2

        for bad in (0.5, True, '0'):
            with self.assertRaises(ValueError, msg=f'argnums={bad!r}'):
                brainstate.transform.grad(loss, argnums=bad)

    def test_positive_argnums_values_unchanged(self):
        w = brainstate.State(jnp.asarray(2.0))

        def loss(x, y):
            return w.value * x * y

        g = brainstate.transform.grad(loss, argnums=1)(3.0, 4.0)
        self.assertEqual(float(g), 6.0)  # d/dy = w * x

        state_grads, arg_grads = brainstate.transform.grad(
            loss, grad_states=w, argnums=1
        )(3.0, 4.0)
        self.assertEqual(float(state_grads), 12.0)  # d/dw = x * y
        self.assertEqual(float(arg_grads), 6.0)


class TestDebugNanPhaseName(unittest.TestCase):
    """``debug_nan=True`` must work when the underlying transform has no
    ``__name__`` (audit F5: ``vector_grad`` uses a ``functools.partial``)."""

    def test_vector_grad_debug_nan_clean_input(self):
        def f(x):
            return x ** 2

        g = brainstate.transform.vector_grad(f, debug_nan=True)(jnp.asarray([1.0, 2.0]))
        self.assertTrue(bool(jnp.allclose(g, jnp.asarray([2.0, 4.0]))))

    def test_vector_grad_debug_nan_still_raises_on_nan(self):
        def f(x):
            return jnp.sqrt(x)  # NaN value and NaN gradient at x < 0

        with self.assertRaises(Exception) as ctx:
            brainstate.transform.vector_grad(f, debug_nan=True)(jnp.asarray([-1.0]))
        self.assertNotIsInstance(ctx.exception, AttributeError)
