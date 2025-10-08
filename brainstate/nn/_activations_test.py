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

"""Tests for nn module."""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from absl.testing import absltest, parameterized
from jax.test_util import check_grads

import brainstate


class NNFunctionsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.rng_key = jax.random.PRNGKey(0)

    def assertAllClose(self, a, b, check_dtypes=True, atol=None, rtol=None):
        """Helper method for backwards compatibility with JAX test utilities."""
        a = np.asarray(a)
        b = np.asarray(b)
        kw = {}
        if atol is not None:
            kw['atol'] = atol
        if rtol is not None:
            kw['rtol'] = rtol
        np.testing.assert_allclose(a, b, **kw)
        if check_dtypes:
            self.assertEqual(a.dtype, b.dtype)

    def assertArraysEqual(self, a, b):
        """Helper method for backwards compatibility with JAX test utilities."""
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def testSoftplusGrad(self):
        check_grads(brainstate.nn.softplus, (1e-8,), order=4, )

    def testSoftplusGradZero(self):
        check_grads(brainstate.nn.softplus, (0.,), order=1)

    def testSoftplusGradInf(self):
        self.assertAllClose(1., jax.grad(brainstate.nn.softplus)(float('inf')), check_dtypes=False)

    def testSoftplusGradNegInf(self):
        check_grads(brainstate.nn.softplus, (-float('inf'),), order=1)

    def testSoftplusGradNan(self):
        check_grads(brainstate.nn.softplus, (float('nan'),), order=1)

    @parameterized.parameters([int, float, jnp.float32, jnp.float64, jnp.int32, jnp.int64])
    def testSoftplusZero(self, dtype):
        self.assertEqual(jnp.log(dtype(2)), brainstate.nn.softplus(dtype(0)))

    def testSparseplusGradZero(self):
        check_grads(brainstate.nn.sparse_plus, (-2.,), order=1)

    def testSparseplusGrad(self):
        check_grads(brainstate.nn.sparse_plus, (0.,), order=1)

    def testSparseplusAndSparseSigmoid(self):
        self.assertAllClose(
            jax.grad(brainstate.nn.sparse_plus)(0.),
            brainstate.nn.sparse_sigmoid(0.),
            check_dtypes=False)
        self.assertAllClose(
            jax.grad(brainstate.nn.sparse_plus)(2.),
            brainstate.nn.sparse_sigmoid(2.),
            check_dtypes=False)
        self.assertAllClose(
            jax.grad(brainstate.nn.sparse_plus)(-2.),
            brainstate.nn.sparse_sigmoid(-2.),
            check_dtypes=False)

    #   def testSquareplusGrad(self):
    #     check_grads(brainstate.nn.squareplus, (1e-8,), order=4,
    #                 )

    #   def testSquareplusGradZero(self):
    #     check_grads(brainstate.nn.squareplus, (0.,), order=1,
    #                 )

    #   def testSquareplusGradNegInf(self):
    #     check_grads(brainstate.nn.squareplus, (-float('inf'),), order=1,
    #                 )

    #   def testSquareplusGradNan(self):
    #     check_grads(brainstate.nn.squareplus, (float('nan'),), order=1,
    #                 )

    #   @parameterized.parameters([float, jnp.float32, jnp.float64])
    #   def testSquareplusZero(self, dtype):
    #     self.assertEqual(dtype(1), brainstate.nn.squareplus(dtype(0), dtype(4)))
    #
    # def testMishGrad(self):
    #   check_grads(brainstate.nn.mish, (1e-8,), order=4,
    #               )
    #
    # def testMishGradZero(self):
    #   check_grads(brainstate.nn.mish, (0.,), order=1,
    #               )
    #
    # def testMishGradNegInf(self):
    #   check_grads(brainstate.nn.mish, (-float('inf'),), order=1,
    #               )
    #
    # def testMishGradNan(self):
    #   check_grads(brainstate.nn.mish, (float('nan'),), order=1,
    #               )

    @parameterized.parameters([float, jnp.float32, jnp.float64])
    def testMishZero(self, dtype):
        self.assertEqual(dtype(0), brainstate.nn.mish(dtype(0)))

    def testReluGrad(self):
        rtol = None
        check_grads(brainstate.nn.relu, (1.,), order=3, rtol=rtol)
        check_grads(brainstate.nn.relu, (-1.,), order=3, rtol=rtol)
        jaxpr = jax.make_jaxpr(jax.grad(brainstate.nn.relu))(0.)
        self.assertGreaterEqual(len(jaxpr.jaxpr.eqns), 2)

    def testRelu6Grad(self):
        rtol = None
        check_grads(brainstate.nn.relu6, (1.,), order=3, rtol=rtol)
        check_grads(brainstate.nn.relu6, (-1.,), order=3, rtol=rtol)
        self.assertAllClose(jax.grad(brainstate.nn.relu6)(0.), 0., check_dtypes=False)
        self.assertAllClose(jax.grad(brainstate.nn.relu6)(6.), 0., check_dtypes=False)

    def testSoftplusValue(self):
        val = brainstate.nn.softplus(89.)
        self.assertAllClose(val, 89., check_dtypes=False)

    def testSparseplusValue(self):
        val = brainstate.nn.sparse_plus(89.)
        self.assertAllClose(val, 89., check_dtypes=False)

    def testSparsesigmoidValue(self):
        self.assertAllClose(brainstate.nn.sparse_sigmoid(-2.), 0., check_dtypes=False)
        self.assertAllClose(brainstate.nn.sparse_sigmoid(2.), 1., check_dtypes=False)
        self.assertAllClose(brainstate.nn.sparse_sigmoid(0.), .5, check_dtypes=False)

    #   def testSquareplusValue(self):
    #     val = brainstate.nn.squareplus(1e3)
    #     self.assertAllClose(val, 1e3, check_dtypes=False, atol=1e-3)

    def testMishValue(self):
        val = brainstate.nn.mish(1e3)
        self.assertAllClose(val, 1e3, check_dtypes=False, atol=1e-3)

    def testEluValue(self):
        val = brainstate.nn.elu(1e4)
        self.assertAllClose(val, 1e4, check_dtypes=False)

    def testGluValue(self):
        val = brainstate.nn.glu(jnp.array([1.0, 0.0]), axis=0)
        self.assertAllClose(val, jnp.array([0.5]))

    @parameterized.parameters(False, True)
    def testGeluIntType(self, approximate):
        val_float = brainstate.nn.gelu(jnp.array(-1.0), approximate=approximate)
        val_int = brainstate.nn.gelu(jnp.array(-1), approximate=approximate)
        self.assertAllClose(val_float, val_int)

    @parameterized.parameters(False, True)
    def testGelu(self, approximate):
        def gelu_reference(x):
            return x * scipy.stats.norm.cdf(x)

        x = jax.random.normal(self.rng_key, (4, 5, 6), dtype=jnp.float32)
        expected = gelu_reference(x)
        actual = brainstate.nn.gelu(x, approximate=approximate)
        np.testing.assert_allclose(actual, expected, rtol=1e-2 if approximate else 1e-5, atol=1e-3 if approximate else 1e-5)

    @parameterized.parameters(*itertools.product(
        (jnp.float32, jnp.bfloat16, jnp.float16),
        (partial(brainstate.nn.gelu, approximate=False),
         partial(brainstate.nn.gelu, approximate=True),
         brainstate.nn.relu,
         brainstate.nn.softplus,
         brainstate.nn.sparse_plus,
         brainstate.nn.sigmoid,
         #  brainstate.nn.squareplus,
         brainstate.nn.mish)))
    def testDtypeMatchesInput(self, dtype, fn):
        x = jnp.zeros((), dtype=dtype)
        out = fn(x)
        self.assertEqual(out.dtype, dtype)

    def testEluMemory(self):
        # see https://github.com/google/jax/pull/1640
        with jax.enable_checks(False):  # With checks we materialize the array
            jax.make_jaxpr(lambda: brainstate.nn.elu(jnp.ones((10 ** 12,))))  # don't oom

    def testHardTanhMemory(self):
        # see https://github.com/google/jax/pull/1640
        with jax.enable_checks(False):  # With checks we materialize the array
            jax.make_jaxpr(lambda: brainstate.nn.hard_tanh(jnp.ones((10 ** 12,))))  # don't oom

    @parameterized.parameters([brainstate.nn.softmax, brainstate.nn.log_softmax])
    def testSoftmaxEmptyArray(self, fn):
        x = jnp.array([], dtype=float)
        self.assertArraysEqual(fn(x), x)

    @parameterized.parameters([brainstate.nn.softmax, brainstate.nn.log_softmax])
    def testSoftmaxEmptyMask(self, fn):
        x = jnp.array([5.5, 1.3, -4.2, 0.9])
        m = jnp.zeros_like(x, dtype=bool)
        expected = jnp.full_like(x, 0.0 if fn is brainstate.nn.softmax else -jnp.inf)
        self.assertArraysEqual(fn(x, where=m), expected)

    @parameterized.parameters([brainstate.nn.softmax, brainstate.nn.log_softmax])
    def testSoftmaxWhereMask(self, fn):
        x = jnp.array([5.5, 1.3, -4.2, 0.9])
        m = jnp.array([True, False, True, True])

        out = fn(x, where=m)
        self.assertAllClose(out[m], fn(x[m]))

        probs = out if fn is brainstate.nn.softmax else jnp.exp(out)
        self.assertAllClose(probs.sum(), 1.0, check_dtypes=False)

    @parameterized.parameters([brainstate.nn.softmax, brainstate.nn.log_softmax])
    def testSoftmaxWhereGrad(self, fn):
        # regression test for https://github.com/google/jax/issues/19490
        x = jnp.array([36., 10000.])
        mask = x < 1000

        f = lambda x, mask: fn(x, where=mask)[0]

        self.assertAllClose(jax.grad(f)(x, mask), jnp.zeros_like(x))

    def testSoftmaxGrad(self):
        x = jnp.array([5.5, 1.3, -4.2, 0.9])
        check_grads(brainstate.nn.softmax, (x,), order=2, atol=5e-3)

    def testStandardizeWhereMask(self):
        x = jnp.array([5.5, 1.3, -4.2, 0.9])
        m = jnp.array([True, False, True, True])
        x_filtered = jnp.take(x, jnp.array([0, 2, 3]))

        out_masked = jnp.take(brainstate.nn.standardize(x, where=m), jnp.array([0, 2, 3]))
        out_filtered = brainstate.nn.standardize(x_filtered)

        self.assertAllClose(out_masked, out_filtered, rtol=1e-6, atol=1e-6)

    def testOneHot(self):
        actual = brainstate.nn.one_hot(jnp.array([0, 1, 2]), 3)
        expected = jnp.array([[1., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.]])
        self.assertAllClose(actual, expected, check_dtypes=False)

        actual = brainstate.nn.one_hot(jnp.array([1, 2, 0]), 3)
        expected = jnp.array([[0., 1., 0.],
                              [0., 0., 1.],
                              [1., 0., 0.]])
        self.assertAllClose(actual, expected, check_dtypes=False)

    def testOneHotOutOfBound(self):
        actual = brainstate.nn.one_hot(jnp.array([-1, 3]), 3)
        expected = jnp.array([[0., 0., 0.],
                              [0., 0., 0.]])
        self.assertAllClose(actual, expected, check_dtypes=False)

    def testOneHotNonArrayInput(self):
        actual = brainstate.nn.one_hot([0, 1, 2], 3)
        expected = jnp.array([[1., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.]])
        self.assertAllClose(actual, expected, check_dtypes=False)

    def testOneHotCustomDtype(self):
        actual = brainstate.nn.one_hot(jnp.array([0, 1, 2]), 3, dtype=jnp.bool_)
        expected = jnp.array([[True, False, False],
                              [False, True, False],
                              [False, False, True]])
        self.assertAllClose(actual, expected)

    def testOneHotAxis(self):
        expected = jnp.array([[0., 1., 0.],
                              [0., 0., 1.],
                              [1., 0., 0.]]).T

        actual = brainstate.nn.one_hot(jnp.array([1, 2, 0]), 3, axis=0)
        self.assertAllClose(actual, expected, check_dtypes=False)

        actual = brainstate.nn.one_hot(jnp.array([1, 2, 0]), 3, axis=-2)
        self.assertAllClose(actual, expected, check_dtypes=False)

    def testTanhExists(self):
        print(brainstate.nn.tanh)  # doesn't crash

    def testCustomJVPLeak(self):
        # https://github.com/google/jax/issues/8171
        @jax.jit
        def fwd():
            a = jnp.array(1.)

            def f(hx, _):
                hx = brainstate.nn.sigmoid(hx + a)
                return hx, None

            hx = jnp.array(0.)
            jax.lax.scan(f, hx, None, length=2)

        with jax.checking_leaks():
            fwd()  # doesn't crash

    def testCustomJVPLeak2(self):
        # https://github.com/google/jax/issues/8171
        # The above test uses jax.brainstate.nn.sigmoid, as in the original #8171, but that
        # function no longer actually has a custom_jvp! So we inline the old def.

        @jax.custom_jvp
        def sigmoid(x):
            one = jnp.float32(1)
            return jax.lax.div(one, jax.lax.add(one, jax.lax.exp(jax.lax.neg(x))))

        sigmoid.defjvps(lambda g, ans, x: g * ans * (jnp.float32(1) - ans))

        @jax.jit
        def fwd():
            a = jnp.array(1., 'float32')

            def f(hx, _):
                hx = sigmoid(hx + a)
                return hx, None

            hx = jnp.array(0., 'float32')
            jax.lax.scan(f, hx, None, length=2)

        with jax.checking_leaks():
            fwd()  # doesn't crash


if __name__ == '__main__':
    absltest.main()
