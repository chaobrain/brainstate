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


import platform
import unittest

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

import brainstate


class TestRandomExamples(unittest.TestCase):
    """Test cases that demonstrate usage examples from docstrings."""

    def test_rand_examples(self):
        """Test examples from rand function docstring."""
        # Generate random values in a 3x2 array
        arr = brainstate.random.rand(3, 2)
        self.assertEqual(arr.shape, (3, 2))
        self.assertTrue((arr >= 0).all() and (arr < 1).all())

    def test_randint_examples(self):
        """Test examples from randint function docstring."""
        # Generate 10 random integers from 0 to 1 (exclusive)
        arr = brainstate.random.randint(2, size=10)
        self.assertEqual(arr.shape, (10,))
        self.assertTrue((arr >= 0).all() and (arr < 2).all())

        # Generate a 2x4 array of integers from 0 to 4 (exclusive)
        arr = brainstate.random.randint(5, size=(2, 4))
        self.assertEqual(arr.shape, (2, 4))
        self.assertTrue((arr >= 0).all() and (arr < 5).all())

        # Generate integers with different upper bounds using broadcasting
        arr = brainstate.random.randint(1, [3, 5, 10])
        self.assertEqual(arr.shape, (3,))

        # Generate integers with different lower bounds
        arr = brainstate.random.randint([1, 5, 7], 10)
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr >= jnp.array([1, 5, 7])).all())

    def test_randn_examples(self):
        """Test examples from randn function docstring."""
        # Generate standard normal distributed values
        arr = brainstate.random.randn(3, 2)
        self.assertEqual(arr.shape, (3, 2))

    def test_choice_examples(self):
        """Test examples from choice function docstring."""
        # Choose from range
        result = brainstate.random.choice(5)
        self.assertTrue(0 <= result < 5)

        # Choose multiple with probabilities
        arr = brainstate.random.choice(5, 3, p=[0.1, 0.4, 0.2, 0.0, 0.3])
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr >= 0).all() and (arr < 5).all())

    def test_normal_examples(self):
        """Test examples from normal function docstring."""
        # Standard normal
        result = brainstate.random.normal()
        self.assertEqual(result.shape, ())

        # With different parameters
        arr = brainstate.random.normal(loc=0.0, scale=1.0, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))

    def test_uniform_examples(self):
        """Test examples from uniform function docstring."""
        # Standard uniform
        result = brainstate.random.uniform()
        self.assertEqual(result.shape, ())
        self.assertTrue(0.0 <= result < 1.0)

        # With custom range
        arr = brainstate.random.uniform(low=2.0, high=5.0, size=(3, 2))
        self.assertEqual(arr.shape, (3, 2))
        self.assertTrue((arr >= 2.0).all() and (arr < 5.0).all())


class TestRandom(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(precision=32)

    def test_rand(self):
        brainstate.random.seed()
        a = brainstate.random.rand(3, 2)
        self.assertTupleEqual(a.shape, (3, 2))
        self.assertTrue((a >= 0).all() and (a < 1).all())

        # Drawing with the same integer-seed key is deterministic and the
        # samples stay within the half-open unit interval [0, 1).
        res1 = brainstate.random.rand(10, 100, key=123)
        res2 = brainstate.random.rand(10, 100, key=123)
        self.assertTrue(jnp.allclose(res1, res2))
        self.assertTrue((res1 >= 0).all() and (res1 < 1).all())
        # An explicit PRNG key drawn from brainstate also reproduces its draw.
        key = brainstate.random.split_key()
        self.assertTrue(jnp.allclose(brainstate.random.rand(10, 100, key=key),
                                     brainstate.random.rand(10, 100, key=key)))

    def test_randint1(self):
        brainstate.random.seed()
        a = brainstate.random.randint(5)
        self.assertTupleEqual(a.shape, ())
        self.assertTrue(0 <= a < 5)

    def test_randint2(self):
        brainstate.random.seed()
        a = brainstate.random.randint(2, 6, size=(4, 3))
        self.assertTupleEqual(a.shape, (4, 3))
        self.assertTrue((a >= 2).all() and (a < 6).all())

    def test_randint3(self):
        brainstate.random.seed()
        a = brainstate.random.randint([1, 2, 3], [10, 7, 8])
        self.assertTupleEqual(a.shape, (3,))
        self.assertTrue((a - jnp.array([1, 2, 3]) >= 0).all()
                        and (-a + jnp.array([10, 7, 8]) > 0).all())

    def test_randint4(self):
        brainstate.random.seed()
        a = brainstate.random.randint([1, 2, 3], [10, 7, 8], size=(2, 3))
        self.assertTupleEqual(a.shape, (2, 3))

    def test_randn(self):
        brainstate.random.seed()
        a = brainstate.random.randn(3, 2)
        self.assertTupleEqual(a.shape, (3, 2))

    def test_random1(self):
        brainstate.random.seed()
        a = brainstate.random.random()
        self.assertTrue(0. <= a < 1)

    def test_random2(self):
        brainstate.random.seed()
        a = brainstate.random.random(size=(3, 2))
        self.assertTupleEqual(a.shape, (3, 2))
        self.assertTrue((a >= 0).all() and (a < 1).all())

    def test_random_sample(self):
        brainstate.random.seed()
        a = brainstate.random.random_sample(size=(3, 2))
        self.assertTupleEqual(a.shape, (3, 2))
        self.assertTrue((a >= 0).all() and (a < 1).all())

    def test_choice1(self):
        brainstate.random.seed()
        a = brainstate.random.choice(5)
        self.assertTupleEqual(jnp.shape(a), ())
        self.assertTrue(0 <= a < 5)

    def test_choice2(self):
        brainstate.random.seed()
        a = brainstate.random.choice(5, 3, p=[0.1, 0.4, 0.2, 0., 0.3])
        self.assertTupleEqual(a.shape, (3,))
        self.assertTrue((a >= 0).all() and (a < 5).all())

    def test_choice3(self):
        brainstate.random.seed()
        a = brainstate.random.choice(jnp.arange(2, 20), size=(4, 3), replace=False)
        self.assertTupleEqual(a.shape, (4, 3))
        self.assertTrue((a >= 2).all() and (a < 20).all())
        self.assertEqual(len(jnp.unique(a)), 12)

    def test_permutation1(self):
        brainstate.random.seed()
        a = brainstate.random.permutation(10)
        self.assertTupleEqual(a.shape, (10,))
        self.assertEqual(len(jnp.unique(a)), 10)

    def test_permutation2(self):
        brainstate.random.seed()
        a = brainstate.random.permutation(jnp.arange(10))
        self.assertTupleEqual(a.shape, (10,))
        self.assertEqual(len(jnp.unique(a)), 10)

    def test_shuffle1(self):
        brainstate.random.seed()
        a = jnp.arange(10)
        brainstate.random.shuffle(a)
        self.assertTupleEqual(a.shape, (10,))
        self.assertEqual(len(jnp.unique(a)), 10)

    def test_shuffle2(self):
        brainstate.random.seed()
        a = jnp.arange(12).reshape(4, 3)
        brainstate.random.shuffle(a, axis=1)
        self.assertTupleEqual(a.shape, (4, 3))
        self.assertEqual(len(jnp.unique(a)), 12)

        # test that a is only shuffled along axis 1
        uni = jnp.unique(jnp.diff(a, axis=0))
        self.assertEqual(uni, jnp.asarray([3]))

    def test_beta1(self):
        brainstate.random.seed()
        a = brainstate.random.beta(2, 2)
        self.assertTupleEqual(a.shape, ())

    def test_beta2(self):
        brainstate.random.seed()
        a = brainstate.random.beta([2, 2, 3], 2, size=(3,))
        self.assertTupleEqual(a.shape, (3,))

    def test_exponential1(self):
        brainstate.random.seed()
        a = brainstate.random.exponential(10., size=[3, 2])
        self.assertTupleEqual(a.shape, (3, 2))

    def test_exponential2(self):
        brainstate.random.seed()
        a = brainstate.random.exponential([1., 2., 5.])
        self.assertTupleEqual(a.shape, (3,))

    def test_gamma(self):
        brainstate.random.seed()
        a = brainstate.random.gamma(2, 10., size=[3, 2])
        self.assertTupleEqual(a.shape, (3, 2))

    def test_gumbel(self):
        brainstate.random.seed()
        a = brainstate.random.gumbel(0., 2., size=[3, 2])
        self.assertTupleEqual(a.shape, (3, 2))

    def test_laplace(self):
        brainstate.random.seed()
        a = brainstate.random.laplace(0., 2., size=[3, 2])
        self.assertTupleEqual(a.shape, (3, 2))

    def test_logistic(self):
        brainstate.random.seed()
        a = brainstate.random.logistic(0., 2., size=[3, 2])
        self.assertTupleEqual(a.shape, (3, 2))

    def test_normal1(self):
        brainstate.random.seed()
        a = brainstate.random.normal()
        self.assertTupleEqual(a.shape, ())

    def test_normal2(self):
        brainstate.random.seed()
        a = brainstate.random.normal(loc=[0., 2., 4.], scale=[1., 2., 3.])
        self.assertTupleEqual(a.shape, (3,))

    def test_normal3(self):
        brainstate.random.seed()
        a = brainstate.random.normal(loc=[0., 2., 4.], scale=[[1., 2., 3.], [1., 1., 1.]])
        print(a)
        self.assertTupleEqual(a.shape, (2, 3))

    def test_pareto(self):
        brainstate.random.seed()
        a = brainstate.random.pareto([1, 2, 2])
        self.assertTupleEqual(a.shape, (3,))

    def test_poisson(self):
        brainstate.random.seed()
        a = brainstate.random.poisson([1., 2., 2.], size=3)
        self.assertTupleEqual(a.shape, (3,))

    def test_standard_cauchy(self):
        brainstate.random.seed()
        a = brainstate.random.standard_cauchy(size=(3, 2))
        self.assertTupleEqual(a.shape, (3, 2))

    def test_standard_exponential(self):
        brainstate.random.seed()
        a = brainstate.random.standard_exponential(size=(3, 2))
        self.assertTupleEqual(a.shape, (3, 2))

    def test_standard_gamma(self):
        brainstate.random.seed()
        a = brainstate.random.standard_gamma(shape=[1, 2, 4], size=3)
        self.assertTupleEqual(a.shape, (3,))

    def test_standard_normal(self):
        brainstate.random.seed()
        a = brainstate.random.standard_normal(size=(3, 2))
        self.assertTupleEqual(a.shape, (3, 2))

    def test_standard_t(self):
        brainstate.random.seed()
        a = brainstate.random.standard_t(df=[1, 2, 4], size=3)
        self.assertTupleEqual(a.shape, (3,))

    def test_standard_uniform1(self):
        brainstate.random.seed()
        a = brainstate.random.uniform()
        self.assertTupleEqual(a.shape, ())
        self.assertTrue(0 <= a < 1)

    def test_uniform2(self):
        brainstate.random.seed()
        a = brainstate.random.uniform(low=[-1., 5., 2.], high=[2., 6., 10.], size=3)
        self.assertTupleEqual(a.shape, (3,))
        self.assertTrue((a - jnp.array([-1., 5., 2.]) >= 0).all()
                        and (-a + jnp.array([2., 6., 10.]) > 0).all())

    def test_uniform3(self):
        brainstate.random.seed()
        a = brainstate.random.uniform(low=-1., high=[2., 6., 10.], size=(2, 3))
        self.assertTupleEqual(a.shape, (2, 3))

    def test_uniform4(self):
        brainstate.random.seed()
        a = brainstate.random.uniform(low=[-1., 5., 2.], high=[[2., 6., 10.], [10., 10., 10.]])
        self.assertTupleEqual(a.shape, (2, 3))

    def test_truncated_normal1(self):
        brainstate.random.seed()
        a = brainstate.random.truncated_normal(-1., 1.)
        self.assertTupleEqual(a.shape, ())
        self.assertTrue(-1. <= a <= 1.)

    def test_truncated_normal2(self):
        brainstate.random.seed()
        a = brainstate.random.truncated_normal(-1., [1., 2., 1.], size=(4, 3))
        self.assertTupleEqual(a.shape, (4, 3))

    def test_truncated_normal3(self):
        brainstate.random.seed()
        a = brainstate.random.truncated_normal([-1., 0., 1.], [[2., 2., 4.], [2., 2., 4.]])
        self.assertTupleEqual(a.shape, (2, 3))
        self.assertTrue((a - jnp.array([-1., 0., 1.]) >= 0.).all()
                        and (- a + jnp.array([2., 2., 4.]) >= 0.).all())

    def test_bernoulli1(self):
        brainstate.random.seed()
        a = brainstate.random.bernoulli()
        self.assertTupleEqual(a.shape, ())
        self.assertTrue(a == 0 or a == 1)

    def test_bernoulli2(self):
        brainstate.random.seed()
        a = brainstate.random.bernoulli([0.5, 0.6, 0.8])
        self.assertTupleEqual(a.shape, (3,))
        self.assertTrue(jnp.logical_xor(a == 1, a == 0).all())

    def test_bernoulli3(self):
        brainstate.random.seed()
        a = brainstate.random.bernoulli([0.5, 0.6], size=(3, 2))
        self.assertTupleEqual(a.shape, (3, 2))
        self.assertTrue(jnp.logical_xor(a == 1, a == 0).all())

    def test_lognormal1(self):
        brainstate.random.seed()
        a = brainstate.random.lognormal()
        self.assertTupleEqual(a.shape, ())

    def test_lognormal2(self):
        brainstate.random.seed()
        a = brainstate.random.lognormal(sigma=[2., 1.], size=[3, 2])
        self.assertTupleEqual(a.shape, (3, 2))

    def test_lognormal3(self):
        brainstate.random.seed()
        a = brainstate.random.lognormal([2., 0.], [[2., 1.], [3., 1.2]])
        self.assertTupleEqual(a.shape, (2, 2))

    def test_binomial1(self):
        brainstate.random.seed()
        a = brainstate.random.binomial(5, 0.5)
        b = np.random.binomial(5, 0.5)
        print(a)
        print(b)
        self.assertTupleEqual(a.shape, ())
        self.assertTrue(a.dtype, int)

    def test_binomial2(self):
        brainstate.random.seed()
        a = brainstate.random.binomial(5, 0.5, size=(3, 2))
        self.assertTupleEqual(a.shape, (3, 2))
        self.assertTrue((a >= 0).all() and (a <= 5).all())

    def test_binomial3(self):
        brainstate.random.seed()
        a = brainstate.random.binomial(n=jnp.asarray([2, 3, 4]), p=jnp.asarray([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]))
        self.assertTupleEqual(a.shape, (2, 3))

    def test_chisquare1(self):
        brainstate.random.seed()
        a = brainstate.random.chisquare(3)
        self.assertTupleEqual(a.shape, ())
        self.assertTrue(a.dtype, float)

    def test_chisquare2(self):
        # Array ``df`` with ``size=None`` infers the shape from ``df`` (gamma-based
        # implementation supports non-scalar and non-integer degrees of freedom).
        brainstate.random.seed()
        a = brainstate.random.chisquare(df=[2, 3, 4])
        self.assertTupleEqual(a.shape, (3,))

    def test_chisquare3(self):
        brainstate.random.seed()
        a = brainstate.random.chisquare(df=2, size=100)
        self.assertTupleEqual(a.shape, (100,))

    def test_chisquare4(self):
        brainstate.random.seed()
        a = brainstate.random.chisquare(df=2, size=(100, 10))
        self.assertTupleEqual(a.shape, (100, 10))

    def test_dirichlet1(self):
        brainstate.random.seed()
        a = brainstate.random.dirichlet((10, 5, 3))
        self.assertTupleEqual(a.shape, (3,))

    def test_dirichlet2(self):
        brainstate.random.seed()
        a = brainstate.random.dirichlet((10, 5, 3), 20)
        self.assertTupleEqual(a.shape, (20, 3))

    def test_f(self):
        brainstate.random.seed()
        a = brainstate.random.f(1., 48., 100)
        self.assertTupleEqual(a.shape, (100,))

    def test_geometric(self):
        brainstate.random.seed()
        a = brainstate.random.geometric([0.7, 0.5, 0.2])
        self.assertTupleEqual(a.shape, (3,))

    def test_hypergeometric1(self):
        brainstate.random.seed()
        a = brainstate.random.hypergeometric(10, 10, 10, 20)
        self.assertTupleEqual(a.shape, (20,))

    def test_hypergeometric2(self):
        brainstate.random.seed()
        a = brainstate.random.hypergeometric(8, [10, 4], [[5, 2], [5, 5]])
        self.assertTupleEqual(a.shape, (2, 2))

    def test_hypergeometric3(self):
        brainstate.random.seed()
        a = brainstate.random.hypergeometric(8, [10, 4], [[5, 2], [5, 5]], size=(3, 2, 2))
        self.assertTupleEqual(a.shape, (3, 2, 2))

    def test_logseries(self):
        brainstate.random.seed()
        a = brainstate.random.logseries([0.7, 0.5, 0.2], size=[4, 3])
        self.assertTupleEqual(a.shape, (4, 3))

    def test_multinominal1(self):
        brainstate.random.seed()
        a = np.random.multinomial(100, (0.5, 0.2, 0.3), size=[4, 2])
        print(a, a.shape)
        b = brainstate.random.multinomial(100, (0.5, 0.2, 0.3), size=[4, 2])
        print(b, b.shape)
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (4, 2, 3))

    def test_multinominal2(self):
        brainstate.random.seed()
        a = brainstate.random.multinomial(100, (0.5, 0.2, 0.3))
        self.assertTupleEqual(a.shape, (3,))
        self.assertTrue(a.sum() == 100)

    def test_multivariate_normal1(self):
        brainstate.random.seed()
        # self.skipTest('Windows jaxlib error')
        a = np.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], size=3)
        b = brainstate.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], size=3)
        print('test_multivariate_normal1')
        print(a)
        print(b)
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(a.shape, (3, 2))

    def test_multivariate_normal2(self):
        brainstate.random.seed()
        a = np.random.multivariate_normal([1, 2], [[1, 3], [3, 1]])
        b = brainstate.random.multivariate_normal([1, 2], [[1, 3], [3, 1]], method='svd')
        print(a)
        print(b)
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(a.shape, (2,))

    def test_negative_binomial(self):
        brainstate.random.seed()
        a = np.random.negative_binomial([3., 10.], 0.5)
        b = brainstate.random.negative_binomial([3., 10.], 0.5)
        print(a)
        print(b)
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (2,))

    def test_negative_binomial2(self):
        brainstate.random.seed()
        a = np.random.negative_binomial(3., 0.5, 10)
        b = brainstate.random.negative_binomial(3., 0.5, 10)
        print(a)
        print(b)
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (10,))

    def test_noncentral_chisquare(self):
        brainstate.random.seed()
        a = np.random.noncentral_chisquare(3, [3., 2.], (4, 2))
        b = brainstate.random.noncentral_chisquare(3, [3., 2.], (4, 2))
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (4, 2))

    def test_noncentral_chisquare2(self):
        brainstate.random.seed()
        a = brainstate.random.noncentral_chisquare(3, [3., 2.])
        self.assertTupleEqual(a.shape, (2,))

    def test_noncentral_f(self):
        brainstate.random.seed()
        a = brainstate.random.noncentral_f(3, 20, 3., 100)
        self.assertTupleEqual(a.shape, (100,))

    def test_power(self):
        brainstate.random.seed()
        a = np.random.power(2, (4, 2))
        b = brainstate.random.power(2, (4, 2))
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (4, 2))

    def test_rayleigh(self):
        brainstate.random.seed()
        a = brainstate.random.power(2., (4, 2))
        self.assertTupleEqual(a.shape, (4, 2))

    def test_triangular(self):
        brainstate.random.seed()
        a = brainstate.random.triangular(-3.0, 0.0, 8.0, (2, 2))
        self.assertTupleEqual(a.shape, (2, 2))
        self.assertTrue(((a >= -3.0) & (a <= 8.0)).all())

    def test_vonmises(self):
        brainstate.random.seed()
        a = np.random.vonmises(2., 2.)
        b = brainstate.random.vonmises(2., 2.)
        print(a, b)
        self.assertTupleEqual(np.shape(a), b.shape)
        self.assertTupleEqual(b.shape, ())

    def test_vonmises2(self):
        brainstate.random.seed()
        a = np.random.vonmises(2., 2., 10)
        b = brainstate.random.vonmises(2., 2., 10)
        print(a, b)
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (10,))

    def test_wald(self):
        brainstate.random.seed()
        a = np.random.wald([2., 0.5], 2.)
        b = brainstate.random.wald([2., 0.5], 2.)
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (2,))

    def test_wald2(self):
        brainstate.random.seed()
        a = np.random.wald(2., 2., 100)
        b = brainstate.random.wald(2., 2., 100)
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (100,))

    def test_weibull(self):
        brainstate.random.seed()
        a = brainstate.random.weibull(2., (4, 2))
        self.assertTupleEqual(a.shape, (4, 2))

    def test_weibull2(self):
        brainstate.random.seed()
        a = brainstate.random.weibull(2., )
        self.assertTupleEqual(a.shape, ())

    def test_weibull3(self):
        brainstate.random.seed()
        a = brainstate.random.weibull([2., 3.], )
        self.assertTupleEqual(a.shape, (2,))

    def test_weibull_min(self):
        brainstate.random.seed()
        a = brainstate.random.weibull_min(2., 2., (4, 2))
        self.assertTupleEqual(a.shape, (4, 2))

    def test_weibull_min2(self):
        brainstate.random.seed()
        a = brainstate.random.weibull_min(2., 2.)
        self.assertTupleEqual(a.shape, ())

    def test_weibull_min3(self):
        brainstate.random.seed()
        a = brainstate.random.weibull_min([2., 3.], 2.)
        self.assertTupleEqual(a.shape, (2,))

    def test_zipf(self):
        brainstate.random.seed()
        a = brainstate.random.zipf(2., (4, 2))
        self.assertTupleEqual(a.shape, (4, 2))

    def test_zipf2(self):
        brainstate.random.seed()
        a = np.random.zipf([1.1, 2.])
        b = brainstate.random.zipf([1.1, 2.])
        self.assertTupleEqual(a.shape, b.shape)
        self.assertTupleEqual(b.shape, (2,))

    def test_maxwell(self):
        brainstate.random.seed()
        a = brainstate.random.maxwell(10)
        self.assertTupleEqual(a.shape, (10,))

    def test_maxwell2(self):
        brainstate.random.seed()
        a = brainstate.random.maxwell()
        self.assertTupleEqual(a.shape, ())

    def test_t(self):
        brainstate.random.seed()
        a = brainstate.random.t(1., size=10)
        self.assertTupleEqual(a.shape, (10,))

    def test_t2(self):
        brainstate.random.seed()
        a = brainstate.random.t([1., 2.], size=None)
        self.assertTupleEqual(a.shape, (2,))

# class TestRandomKey(unittest.TestCase):
#   def test_clear_memory(self):
#     brainstate.random.split_key()
#     print(brainstate.random.DEFAULT.value)
#     self.assertTrue(isinstance(brainstate.random.DEFAULT.value, np.ndarray))


# Distributions that accept a ``size`` argument and return an array of exactly
# that shape. Each entry is a one-line lambda mapping a requested ``size`` to a
# draw, so a single contract can be exercised across the whole catalogue.
_SIZED_DISTRIBUTIONS = (
    ("normal", lambda size: brainstate.random.normal(0.0, 1.0, size)),
    ("uniform", lambda size: brainstate.random.uniform(0.0, 1.0, size)),
    ("rand", lambda size: brainstate.random.rand(*size)),
    ("randn", lambda size: brainstate.random.randn(*size)),
    ("random", lambda size: brainstate.random.random(size)),
    ("random_sample", lambda size: brainstate.random.random_sample(size)),
    ("ranf", lambda size: brainstate.random.ranf(size)),
    ("sample", lambda size: brainstate.random.sample(size)),
    ("random_integers", lambda size: brainstate.random.random_integers(1, 6, size)),
    ("randint", lambda size: brainstate.random.randint(0, 10, size=size)),
    ("standard_normal", lambda size: brainstate.random.standard_normal(size)),
    ("standard_exponential", lambda size: brainstate.random.standard_exponential(size)),
    ("standard_cauchy", lambda size: brainstate.random.standard_cauchy(size)),
    ("exponential", lambda size: brainstate.random.exponential(1.0, size)),
    ("poisson", lambda size: brainstate.random.poisson(3.0, size)),
    ("bernoulli", lambda size: brainstate.random.bernoulli(0.5, size)),
    ("beta", lambda size: brainstate.random.beta(2.0, 2.0, size)),
    ("gamma", lambda size: brainstate.random.gamma(2.0, 1.0, size)),
    ("standard_gamma", lambda size: brainstate.random.standard_gamma(2.0, size)),
    ("laplace", lambda size: brainstate.random.laplace(0.0, 1.0, size)),
    ("logistic", lambda size: brainstate.random.logistic(0.0, 1.0, size)),
    ("gumbel", lambda size: brainstate.random.gumbel(0.0, 1.0, size)),
    ("loggamma", lambda size: brainstate.random.loggamma(2.0, size)),
    ("weibull", lambda size: brainstate.random.weibull(2.0, size)),
    ("weibull_min", lambda size: brainstate.random.weibull_min(2.0, 1.0, size)),
    ("pareto", lambda size: brainstate.random.pareto(2.0, size)),
    ("rayleigh", lambda size: brainstate.random.rayleigh(1.0, size)),
    ("lognormal", lambda size: brainstate.random.lognormal(0.0, 1.0, size)),
    ("geometric", lambda size: brainstate.random.geometric(0.5, size)),
    ("chisquare", lambda size: brainstate.random.chisquare(3, size)),
    ("t", lambda size: brainstate.random.t(5.0, size)),
    ("standard_t", lambda size: brainstate.random.standard_t(5.0, size)),
    ("triangular", lambda size: brainstate.random.triangular(0.0, 0.5, 1.0, size)),
    ("vonmises", lambda size: brainstate.random.vonmises(0.0, 1.0, size)),
    ("maxwell", lambda size: brainstate.random.maxwell(size)),
    ("f", lambda size: brainstate.random.f(2.0, 5.0, size)),
    ("zipf", lambda size: brainstate.random.zipf(2.0, size)),
    ("power", lambda size: brainstate.random.power(2.0, size)),
    ("wald", lambda size: brainstate.random.wald(1.0, 1.0, size)),
    ("logseries", lambda size: brainstate.random.logseries(0.5, size)),
    ("binomial", lambda size: brainstate.random.binomial(10, 0.5, size)),
    ("negative_binomial", lambda size: brainstate.random.negative_binomial(5.0, 0.5, size)),
    ("noncentral_chisquare", lambda size: brainstate.random.noncentral_chisquare(3.0, 1.0, size)),
    ("noncentral_f", lambda size: brainstate.random.noncentral_f(3.0, 20.0, 1.0, size)),
)


class TestDistributionContract(parameterized.TestCase):
    """Shared contract for common distributions: shape, determinism, jit, vmap."""

    @parameterized.named_parameters(*_SIZED_DISTRIBUTIONS)
    def test_shape_and_determinism(self, draw):
        """Output matches the requested size and is reproducible under a fixed seed."""
        size = (4, 3)
        brainstate.random.seed(0)
        a = draw(size)
        brainstate.random.seed(0)
        b = draw(size)
        self.assertEqual(tuple(jnp.shape(a)), size)
        self.assertTrue(bool(jnp.allclose(jnp.asarray(a), jnp.asarray(b))))

    @parameterized.named_parameters(*_SIZED_DISTRIBUTIONS)
    def test_finite_values(self, draw):
        """Every drawn sample is finite (no NaN or inf leaks through)."""
        brainstate.random.seed(0)
        a = jnp.asarray(draw((6, 5)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(a))))

    @parameterized.named_parameters(
        ("normal", lambda size: brainstate.random.normal(0.0, 1.0, size, dtype=jnp.float16)),
        ("uniform", lambda size: brainstate.random.uniform(0.0, 1.0, size, dtype=jnp.float16)),
        ("exponential", lambda size: brainstate.random.exponential(1.0, size, dtype=jnp.float16)),
        ("gamma", lambda size: brainstate.random.gamma(2.0, 1.0, size, dtype=jnp.float16)),
        ("rayleigh", lambda size: brainstate.random.rayleigh(1.0, size, dtype=jnp.float16)),
        ("loggamma", lambda size: brainstate.random.loggamma(2.0, size, dtype=jnp.float16)),
    )
    def test_dtype_argument_honored(self, draw):
        """The requested ``dtype`` is reflected in the output array."""
        brainstate.random.seed(0)
        a = draw((4, 3))
        self.assertEqual(jnp.asarray(a).dtype, jnp.float16)

    def test_uniform_bounds(self):
        """``uniform`` samples lie within the requested half-open interval."""
        brainstate.random.seed(0)
        a = brainstate.random.uniform(2.0, 5.0, (200,))
        self.assertTrue(bool(jnp.all(a >= 2.0)))
        self.assertTrue(bool(jnp.all(a < 5.0)))

    def test_rand_bounds(self):
        """``rand`` samples lie within the standard unit interval [0, 1)."""
        brainstate.random.seed(0)
        a = brainstate.random.rand(200)
        self.assertTrue(bool(jnp.all(a >= 0.0)))
        self.assertTrue(bool(jnp.all(a < 1.0)))

    def test_random_integers_bounds(self):
        """``random_integers`` is inclusive of both endpoints of [low, high]."""
        brainstate.random.seed(0)
        a = brainstate.random.random_integers(1, 6, (500,))
        self.assertTrue(bool(jnp.all(a >= 1)))
        self.assertTrue(bool(jnp.all(a <= 6)))

    def test_bernoulli_is_binary(self):
        """``bernoulli`` only ever returns 0/1 valued samples."""
        brainstate.random.seed(0)
        a = brainstate.random.bernoulli(0.5, (200,))
        self.assertTrue(bool(jnp.all(jnp.logical_or(a == 0, a == 1))))

    def test_bernoulli_invalid_p_raises(self):
        """``bernoulli`` rejects a probability outside [0, 1] when validation is on."""
        brainstate.random.seed(0)
        with self.assertRaises(Exception):
            np.asarray(brainstate.random.bernoulli(1.5, (3,), check_valid=True))

    def test_aliases_match_random_sample(self):
        """``ranf`` and ``sample`` are exact aliases of ``random_sample`` for one seed."""
        brainstate.random.seed(0)
        ref = brainstate.random.random_sample((4, 3))
        brainstate.random.seed(0)
        self.assertTrue(bool(jnp.allclose(ref, brainstate.random.ranf((4, 3)))))
        brainstate.random.seed(0)
        self.assertTrue(bool(jnp.allclose(ref, brainstate.random.sample((4, 3)))))

    def test_orthogonal_is_orthonormal(self):
        """``orthogonal`` returns matrices whose columns are orthonormal."""
        brainstate.random.seed(0)
        q = brainstate.random.orthogonal(3, (2,))
        self.assertEqual(tuple(q.shape), (2, 3, 3))
        eye = jnp.einsum('...ij,...ik->...jk', q, q)
        self.assertTrue(bool(jnp.allclose(eye, jnp.eye(3)[None], atol=1e-4)))

    def test_categorical_shape_and_range(self):
        """``categorical`` draws integer class indices within the logits' range."""
        brainstate.random.seed(0)
        logits = jnp.zeros((4, 3, 5))
        a = brainstate.random.categorical(logits, axis=-1)
        self.assertEqual(tuple(a.shape), (4, 3))
        self.assertTrue(bool(jnp.all(a >= 0)))
        self.assertTrue(bool(jnp.all(a < 5)))

    @parameterized.named_parameters(
        ("rand_like", lambda x: brainstate.random.rand_like(x)),
        ("randn_like", lambda x: brainstate.random.randn_like(x)),
        ("randint_like", lambda x: brainstate.random.randint_like(x, 0, 5)),
    )
    def test_like_helpers_match_input_shape(self, draw):
        """``*_like`` helpers mirror the shape of their template tensor."""
        brainstate.random.seed(0)
        template = jnp.ones((4, 3))
        a = draw(template)
        self.assertEqual(tuple(a.shape), (4, 3))

    def test_normal_under_jit(self):
        """``normal`` produces the right shape when traced through ``transform.jit``."""

        @brainstate.transform.jit
        def draw():
            return brainstate.random.normal(0.0, 1.0, (4, 3))

        brainstate.random.seed(0)
        a = draw()
        self.assertEqual(tuple(a.shape), (4, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(a))))

    def test_independent_draws_under_vmap(self):
        """Mapping a per-key draw over split keys yields independent lanes."""
        brainstate.random.seed(0)
        keys = brainstate.random.split_keys(4)

        def draw(key):
            return brainstate.random.normal(0.0, 1.0, (3,), key=key)

        out = brainstate.transform.vmap(draw)(keys)
        self.assertEqual(tuple(out.shape), (4, 3))
        # Distinct keys must give distinct lanes.
        self.assertFalse(bool(jnp.allclose(out[0], out[1])))

    @pytest.mark.slow
    def test_normal_statistics(self):
        """A large normal sample has ~0 mean and ~1 std (loose tolerance)."""
        brainstate.random.seed(0)
        x = brainstate.random.normal(0.0, 1.0, (10000,))
        self.assertLess(abs(float(jnp.mean(x))), 0.1)
        self.assertLess(abs(float(jnp.std(x)) - 1.0), 0.1)

    @pytest.mark.slow
    def test_uniform_statistics(self):
        """A large uniform[0,1) sample has a mean near 0.5 and stays in bounds."""
        brainstate.random.seed(0)
        x = brainstate.random.uniform(0.0, 1.0, (10000,))
        self.assertLess(abs(float(jnp.mean(x)) - 0.5), 0.05)
        self.assertTrue(bool(jnp.all(x >= 0.0)))
        self.assertTrue(bool(jnp.all(x < 1.0)))

    def test_exponential_is_nonnegative(self):
        """``exponential`` only produces non-negative samples."""
        brainstate.random.seed(0)
        x = brainstate.random.exponential(2.0, (500,))
        self.assertTrue(bool(jnp.all(x >= 0.0)))

    @pytest.mark.slow
    def test_exponential_statistics(self):
        """A large exponential(scale=2) sample has a mean near its scale (numpy convention)."""
        brainstate.random.seed(0)
        x = brainstate.random.exponential(2.0, (200000,))
        self.assertLess(abs(float(jnp.mean(x)) - 2.0), 0.05)

    @pytest.mark.slow
    def test_poisson_statistics(self):
        """A large Poisson(lam=3) sample has a mean near its rate."""
        brainstate.random.seed(0)
        x = brainstate.random.poisson(3.0, (10000,))
        self.assertTrue(bool(jnp.all(x >= 0)))
        self.assertLess(abs(float(jnp.mean(x)) - 3.0), 0.2)


class TestAuditRegressions(parameterized.TestCase):
    """Regression tests for bugs found in the brainstate.random audit."""

    # --- A: standard_t array df with size=None ----------------------------------

    def test_standard_t_array_df_infers_shape(self):
        """standard_t infers output shape from array ``df`` when ``size`` is None."""
        brainstate.random.seed(0)
        a = brainstate.random.standard_t([1.0, 2.0, 4.0])
        self.assertTupleEqual(tuple(a.shape), (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(a))))

    def test_standard_t_scalar_df_unchanged(self):
        """standard_t with a scalar ``df`` still returns a scalar."""
        brainstate.random.seed(0)
        self.assertTupleEqual(tuple(brainstate.random.standard_t(3.0).shape), ())

    # --- B: weibull_min scale multiplies -----------------------------------------

    @pytest.mark.slow
    def test_weibull_min_scale_multiplies(self):
        """weibull_min(a, scale) scales the standard draw by ``scale`` (not 1/scale)."""
        brainstate.random.seed(0)
        base = np.asarray(brainstate.random.weibull(2.0, 200000))
        brainstate.random.seed(0)
        scaled = np.asarray(brainstate.random.weibull_min(2.0, 4.0, 200000))
        # Same key/uniforms, so the ratio is exactly the scale factor elementwise.
        np.testing.assert_allclose(scaled / base, 4.0, rtol=1e-4)

    # --- C: triangular is a real triangular distribution -------------------------

    def test_triangular_within_bounds(self):
        """triangular(left, mode, right) stays within [left, right]."""
        brainstate.random.seed(0)
        a = np.asarray(brainstate.random.triangular(-3.0, 0.0, 8.0, 1000))
        self.assertTupleEqual(a.shape, (1000,))
        self.assertTrue((a >= -3.0).all() and (a <= 8.0).all())

    def test_triangular_scalar(self):
        """triangular returns a scalar when all parameters are scalars and size is None."""
        brainstate.random.seed(0)
        self.assertTupleEqual(tuple(brainstate.random.triangular(0.0, 0.5, 1.0).shape), ())

    @pytest.mark.slow
    def test_triangular_mode_skews_mean(self):
        """The sample mean approaches the analytic mean (left+mode+right)/3."""
        brainstate.random.seed(0)
        a = np.asarray(brainstate.random.triangular(0.0, 2.0, 3.0, 200000))
        self.assertLess(abs(a.mean() - (0.0 + 2.0 + 3.0) / 3.0), 0.02)

    def test_triangular_docstring_example_runs(self):
        """The documented ``triangular(-3, 0, 8, N)`` call no longer raises."""
        brainstate.random.seed(0)
        a = brainstate.random.triangular(-3, 0, 8, 100)
        self.assertTupleEqual(tuple(a.shape), (100,))

    # --- D: geometric off-by-one and integer dtype -------------------------------

    def test_geometric_support_starts_at_one(self):
        """geometric is supported on the positive integers {1, 2, ...}."""
        brainstate.random.seed(0)
        a = np.asarray(brainstate.random.geometric(0.5, size=(5000,)))
        self.assertGreaterEqual(int(a.min()), 1)
        self.assertTrue(np.issubdtype(a.dtype, np.integer))

    @pytest.mark.slow
    def test_geometric_pmf_first_success(self):
        """P(k == 1) approaches ``p`` (NumPy convention)."""
        brainstate.random.seed(0)
        a = np.asarray(brainstate.random.geometric(0.35, size=(200000,)))
        self.assertLess(abs((a == 1).mean() - 0.35), 0.01)

    # --- E: randint_like default high with ndim>1 input --------------------------

    def test_randint_like_multidim_default_high(self):
        """randint_like infers ``high`` from a multi-dimensional template."""
        brainstate.random.seed(0)
        template = jnp.array([[3, 7], [2, 9]])
        a = brainstate.random.randint_like(template)
        self.assertTupleEqual(tuple(a.shape), (2, 2))
        self.assertTrue(bool(jnp.all(a >= 0)) and bool(jnp.all(a < 9)))

    # --- F: chisquare accepts float and array df ---------------------------------

    def test_chisquare_float_scalar_df(self):
        """chisquare accepts a non-integer scalar ``df``."""
        brainstate.random.seed(0)
        a = brainstate.random.chisquare(3.5)
        self.assertTupleEqual(tuple(a.shape), ())
        self.assertTrue(float(a) >= 0.0)

    def test_chisquare_array_df_infers_shape(self):
        """chisquare with array ``df`` and ``size=None`` infers the shape from ``df``."""
        brainstate.random.seed(0)
        a = brainstate.random.chisquare(jnp.array([2.0, 3.0, 4.0]))
        self.assertTupleEqual(tuple(a.shape), (3,))
        self.assertTrue(bool(jnp.all(a >= 0.0)))

    @pytest.mark.slow
    def test_chisquare_mean_matches_df(self):
        """A large chi-square sample has mean ~ df."""
        brainstate.random.seed(0)
        a = np.asarray(brainstate.random.chisquare(7.0, size=(100000,)))
        self.assertLess(abs(a.mean() - 7.0), 0.1)

    # --- G: power() enforces the documented ``Raises ValueError if a <= 0`` -------

    def test_power_nonpositive_a_raises(self):
        """power(a<=0) raises, matching the documented numpy contract."""
        brainstate.random.seed(0)
        for bad in (0.0, -1.0):
            with self.subTest(a=bad):
                with self.assertRaises(Exception):
                    np.asarray(brainstate.random.power(bad, 100))

    def test_power_check_valid_false_skips_guard(self):
        """power(check_valid=False) skips the a>0 guard."""
        brainstate.random.seed(0)
        out = brainstate.random.power(0.0, 100, check_valid=False)
        self.assertTupleEqual(tuple(jnp.shape(out)), (100,))

    def test_power_valid_a_unaffected(self):
        """A valid positive ``a`` still samples within [0, 1]."""
        brainstate.random.seed(0)
        a = np.asarray(brainstate.random.power(2.0, 1000))
        self.assertTupleEqual(a.shape, (1000,))
        self.assertTrue((a >= 0).all() and (a <= 1).all())

    # --- H: chisquare() enforces ``Raises ValueError when df <= 0`` (public API) --

    def test_chisquare_nonpositive_df_raises(self):
        """The public chisquare wrapper raises for df<=0."""
        brainstate.random.seed(0)
        for bad in (0.0, -2.0):
            with self.subTest(df=bad):
                with self.assertRaises(Exception):
                    np.asarray(brainstate.random.chisquare(bad, 5))
