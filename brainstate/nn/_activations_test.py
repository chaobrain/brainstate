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

"""Tests for nn module."""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import scipy.stats
from absl.testing import parameterized
from jax._src import test_util as jtu
from jax.test_util import check_grads

import brainstate


class NNFunctionsTest(jtu.JaxTestCase):
    @jtu.skip_on_flag("jax_skip_slow_tests", True)
    def testSoftplusGrad(self):
        check_grads(brainstate.functional.softplus, (1e-8,), order=4, )

    def testSoftplusGradZero(self):
        check_grads(brainstate.functional.softplus, (0.,), order=1)

    def testSoftplusGradInf(self):
        self.assertAllClose(1., jax.grad(brainstate.functional.softplus)(float('inf')))

    def testSoftplusGradNegInf(self):
        check_grads(brainstate.functional.softplus, (-float('inf'),), order=1)

    def testSoftplusGradNan(self):
        check_grads(brainstate.functional.softplus, (float('nan'),), order=1)

    @parameterized.parameters([int, float] + jtu.dtypes.floating + jtu.dtypes.integer)
    def testSoftplusZero(self, dtype):
        self.assertEqual(jnp.log(dtype(2)), brainstate.functional.softplus(dtype(0)))

    def testSparseplusGradZero(self):
        check_grads(brainstate.functional.sparse_plus, (-2.,), order=1)

    def testSparseplusGrad(self):
        check_grads(brainstate.functional.sparse_plus, (0.,), order=1)

    def testSparseplusAndSparseSigmoid(self):
        self.assertAllClose(
            jax.grad(brainstate.functional.sparse_plus)(0.),
            brainstate.functional.sparse_sigmoid(0.),
            check_dtypes=False)
        self.assertAllClose(
            jax.grad(brainstate.functional.sparse_plus)(2.),
            brainstate.functional.sparse_sigmoid(2.),
            check_dtypes=False)
        self.assertAllClose(
            jax.grad(brainstate.functional.sparse_plus)(-2.),
            brainstate.functional.sparse_sigmoid(-2.),
            check_dtypes=False)

    #   def testSquareplusGrad(self):
    #     check_grads(brainstate.functional.squareplus, (1e-8,), order=4,
    #                 )

    #   def testSquareplusGradZero(self):
    #     check_grads(brainstate.functional.squareplus, (0.,), order=1,
    #                 )

    #   def testSquareplusGradNegInf(self):
    #     check_grads(brainstate.functional.squareplus, (-float('inf'),), order=1,
    #                 )

    #   def testSquareplusGradNan(self):
    #     check_grads(brainstate.functional.squareplus, (float('nan'),), order=1,
    #                 )

    #   @parameterized.parameters([float] + jtu.dtypes.floating)
    #   def testSquareplusZero(self, dtype):
    #     self.assertEqual(dtype(1), brainstate.functional.squareplus(dtype(0), dtype(4)))
    #
    # def testMishGrad(self):
    #   check_grads(brainstate.functional.mish, (1e-8,), order=4,
    #               )
    #
    # def testMishGradZero(self):
    #   check_grads(brainstate.functional.mish, (0.,), order=1,
    #               )
    #
    # def testMishGradNegInf(self):
    #   check_grads(brainstate.functional.mish, (-float('inf'),), order=1,
    #               )
    #
    # def testMishGradNan(self):
    #   check_grads(brainstate.functional.mish, (float('nan'),), order=1,
    #               )

    @parameterized.parameters([float] + jtu.dtypes.floating)
    def testMishZero(self, dtype):
        self.assertEqual(dtype(0), brainstate.functional.mish(dtype(0)))

    def testReluGrad(self):
        rtol = None
        check_grads(brainstate.functional.relu, (1.,), order=3, rtol=rtol)
        check_grads(brainstate.functional.relu, (-1.,), order=3, rtol=rtol)
        jaxpr = jax.make_jaxpr(jax.grad(brainstate.functional.relu))(0.)
        self.assertGreaterEqual(len(jaxpr.jaxpr.eqns), 2)

    def testRelu6Grad(self):
        rtol = None
        check_grads(brainstate.functional.relu6, (1.,), order=3, rtol=rtol)
        check_grads(brainstate.functional.relu6, (-1.,), order=3, rtol=rtol)
        self.assertAllClose(jax.grad(brainstate.functional.relu6)(0.), 0., check_dtypes=False)
        self.assertAllClose(jax.grad(brainstate.functional.relu6)(6.), 0., check_dtypes=False)

    def testSoftplusValue(self):
        val = brainstate.functional.softplus(89.)
        self.assertAllClose(val, 89., check_dtypes=False)

    def testSparseplusValue(self):
        val = brainstate.functional.sparse_plus(89.)
        self.assertAllClose(val, 89., check_dtypes=False)

    def testSparsesigmoidValue(self):
        self.assertAllClose(brainstate.functional.sparse_sigmoid(-2.), 0., check_dtypes=False)
        self.assertAllClose(brainstate.functional.sparse_sigmoid(2.), 1., check_dtypes=False)
        self.assertAllClose(brainstate.functional.sparse_sigmoid(0.), .5, check_dtypes=False)

    #   def testSquareplusValue(self):
    #     val = brainstate.functional.squareplus(1e3)
    #     self.assertAllClose(val, 1e3, check_dtypes=False, atol=1e-3)

    def testMishValue(self):
        val = brainstate.functional.mish(1e3)
        self.assertAllClose(val, 1e3, check_dtypes=False, atol=1e-3)

    def testEluValue(self):
        val = brainstate.functional.elu(1e4)
        self.assertAllClose(val, 1e4, check_dtypes=False)

    def testGluValue(self):
        val = brainstate.functional.glu(jnp.array([1.0, 0.0]), axis=0)
        self.assertAllClose(val, jnp.array([0.5]))

    @parameterized.parameters(False, True)
    def testGeluIntType(self, approximate):
        val_float = brainstate.functional.gelu(jnp.array(-1.0), approximate=approximate)
        val_int = brainstate.functional.gelu(jnp.array(-1), approximate=approximate)
        self.assertAllClose(val_float, val_int)

    @parameterized.parameters(False, True)
    def testGelu(self, approximate):
        def gelu_reference(x):
            return x * scipy.stats.norm.cdf(x)

        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng((4, 5, 6), jnp.float32)]
        self._CheckAgainstNumpy(
            gelu_reference, partial(brainstate.functional.gelu, approximate=approximate), args_maker,
            check_dtypes=False, tol=1e-3 if approximate else None)

    @parameterized.parameters(*itertools.product(
        (jnp.float32, jnp.bfloat16, jnp.float16),
        (partial(brainstate.functional.gelu, approximate=False),
         partial(brainstate.functional.gelu, approximate=True),
         brainstate.functional.relu,
         brainstate.functional.softplus,
         brainstate.functional.sparse_plus,
         brainstate.functional.sigmoid,
         #  brainstate.functional.squareplus,
         brainstate.functional.mish)))
    def testDtypeMatchesInput(self, dtype, fn):
        x = jnp.zeros((), dtype=dtype)
        out = fn(x)
        self.assertEqual(out.dtype, dtype)

    def testEluMemory(self):
        # see https://github.com/google/jax/pull/1640
        with jax.enable_checks(False):  # With checks we materialize the array
            jax.make_jaxpr(lambda: brainstate.functional.elu(jnp.ones((10 ** 12,))))  # don't oom

    def testHardTanhMemory(self):
        # see https://github.com/google/jax/pull/1640
        with jax.enable_checks(False):  # With checks we materialize the array
            jax.make_jaxpr(lambda: brainstate.functional.hard_tanh(jnp.ones((10 ** 12,))))  # don't oom

    @parameterized.parameters([brainstate.functional.softmax, brainstate.functional.log_softmax])
    def testSoftmaxEmptyArray(self, fn):
        x = jnp.array([], dtype=float)
        self.assertArraysEqual(fn(x), x)

    @parameterized.parameters([brainstate.functional.softmax, brainstate.functional.log_softmax])
    def testSoftmaxEmptyMask(self, fn):
        x = jnp.array([5.5, 1.3, -4.2, 0.9])
        m = jnp.zeros_like(x, dtype=bool)
        expected = jnp.full_like(x, 0.0 if fn is brainstate.functional.softmax else -jnp.inf)
        self.assertArraysEqual(fn(x, where=m), expected)

    @parameterized.parameters([brainstate.functional.softmax, brainstate.functional.log_softmax])
    def testSoftmaxWhereMask(self, fn):
        x = jnp.array([5.5, 1.3, -4.2, 0.9])
        m = jnp.array([True, False, True, True])

        out = fn(x, where=m)
        self.assertAllClose(out[m], fn(x[m]))

        probs = out if fn is brainstate.functional.softmax else jnp.exp(out)
        self.assertAllClose(probs.sum(), 1.0)

    @parameterized.parameters([brainstate.functional.softmax, brainstate.functional.log_softmax])
    def testSoftmaxWhereGrad(self, fn):
        # regression test for https://github.com/google/jax/issues/19490
        x = jnp.array([36., 10000.])
        mask = x < 1000

        f = lambda x, mask: fn(x, where=mask)[0]

        self.assertAllClose(jax.grad(f)(x, mask), jnp.zeros_like(x))

    def testSoftmaxGrad(self):
        x = jnp.array([5.5, 1.3, -4.2, 0.9])
        jtu.check_grads(brainstate.functional.softmax, (x,), order=2, atol=5e-3)

    def testStandardizeWhereMask(self):
        x = jnp.array([5.5, 1.3, -4.2, 0.9])
        m = jnp.array([True, False, True, True])
        x_filtered = jnp.take(x, jnp.array([0, 2, 3]))

        out_masked = jnp.take(brainstate.functional.standardize(x, where=m), jnp.array([0, 2, 3]))
        out_filtered = brainstate.functional.standardize(x_filtered)

        self.assertAllClose(out_masked, out_filtered)

    def testOneHot(self):
        actual = brainstate.functional.one_hot(jnp.array([0, 1, 2]), 3)
        expected = jnp.array([[1., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.]])
        self.assertAllClose(actual, expected, check_dtypes=False)

        actual = brainstate.functional.one_hot(jnp.array([1, 2, 0]), 3)
        expected = jnp.array([[0., 1., 0.],
                              [0., 0., 1.],
                              [1., 0., 0.]])
        self.assertAllClose(actual, expected, check_dtypes=False)

    def testOneHotOutOfBound(self):
        actual = brainstate.functional.one_hot(jnp.array([-1, 3]), 3)
        expected = jnp.array([[0., 0., 0.],
                              [0., 0., 0.]])
        self.assertAllClose(actual, expected, check_dtypes=False)

    def testOneHotNonArrayInput(self):
        actual = brainstate.functional.one_hot([0, 1, 2], 3)
        expected = jnp.array([[1., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.]])
        self.assertAllClose(actual, expected, check_dtypes=False)

    def testOneHotCustomDtype(self):
        actual = brainstate.functional.one_hot(jnp.array([0, 1, 2]), 3, dtype=jnp.bool_)
        expected = jnp.array([[True, False, False],
                              [False, True, False],
                              [False, False, True]])
        self.assertAllClose(actual, expected)

    def testOneHotAxis(self):
        expected = jnp.array([[0., 1., 0.],
                              [0., 0., 1.],
                              [1., 0., 0.]]).T

        actual = brainstate.functional.one_hot(jnp.array([1, 2, 0]), 3, axis=0)
        self.assertAllClose(actual, expected, check_dtypes=False)

        actual = brainstate.functional.one_hot(jnp.array([1, 2, 0]), 3, axis=-2)
        self.assertAllClose(actual, expected, check_dtypes=False)

    def testTanhExists(self):
        print(brainstate.functional.tanh)  # doesn't crash

    def testCustomJVPLeak(self):
        # https://github.com/google/jax/issues/8171
        @jax.jit
        def fwd():
            a = jnp.array(1.)

            def f(hx, _):
                hx = brainstate.functional.sigmoid(hx + a)
                return hx, None

            hx = jnp.array(0.)
            jax.lax.scan(f, hx, None, length=2)

        with jax.checking_leaks():
            fwd()  # doesn't crash

    def testCustomJVPLeak2(self):
        # https://github.com/google/jax/issues/8171
        # The above test uses jax.brainstate.functional.sigmoid, as in the original #8171, but that
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
