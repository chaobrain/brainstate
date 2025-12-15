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
from pprint import pprint

import jax
import jax.numpy as jnp

import brainstate
from brainstate.transform._grad_jacobian import _jacfwd


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

        br = brainstate.transform.jacrev(f1)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        jr = jax.jacrev(f1)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        assert (br == jr).all()

        br = brainstate.transform.jacrev(f1, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
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

        br = brainstate.transform.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0], jr[0])
        assert jnp.array_equal(br[1], jr[1])

        br = brainstate.transform.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0], jr[0])
        assert jnp.array_equal(br[1], jr[1])

        def f2(x, y):
            r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
            r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r1, r2

        br = brainstate.transform.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0], jr[0])
        assert jnp.array_equal(br[1], jr[1])

        br = brainstate.transform.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
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

        br = brainstate.transform.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0][0], jr[0][0])
        assert jnp.array_equal(br[0][1], jr[0][1])
        assert jnp.array_equal(br[1][0], jr[1][0])
        assert jnp.array_equal(br[1][1], jr[1][1])

        br = brainstate.transform.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0][0], jr[0][0])
        assert jnp.array_equal(br[0][1], jr[0][1])
        assert jnp.array_equal(br[1][0], jr[1][0])
        assert jnp.array_equal(br[1][1], jr[1][1])

        def f3(x, y):
            r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
            r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r1, r2

        br = brainstate.transform.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
        pprint(br)
        assert jnp.array_equal(br[0][0], jr[0][0])
        assert jnp.array_equal(br[0][1], jr[0][1])
        assert jnp.array_equal(br[1][0], jr[1][0])
        assert jnp.array_equal(br[1][1], jr[1][1])

        br = brainstate.transform.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
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
        grads, aux = brainstate.transform.jacrev(f1, has_aux=True)(x, y)
        assert (grads == jr).all()
        assert aux == (4 * x[1] ** 2 - 2 * x[2])

        jr = jax.jacrev(f2, argnums=(0, 1))(x, y)  # jax jacobian
        pprint(jr)
        grads, aux = brainstate.transform.jacrev(f1, argnums=(0, 1), has_aux=True)(x, y)
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

            grads, vec, aux = brainstate.transform.jacrev(f1, return_value=True, has_aux=True)(_x, _y)
            assert (grads == _g1).all()
            assert aux == _a
            assert (vec == _r).all()

            grads, vec, aux = brainstate.transform.jacrev(f1, return_value=True, argnums=(0, 1), has_aux=True)(_x, _y)
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
        br = brainstate.transform.jacrev(t, grad_states=t.x)()
        self.assertTrue((br == _jr).all())

        _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
        t = Test()
        br = brainstate.transform.jacrev(t, grad_states=[t.x, t.y])()
        self.assertTrue((br[0] == _jr[0]).all())
        self.assertTrue((br[1] == _jr[1]).all())
