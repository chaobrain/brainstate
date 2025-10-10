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
import jax.random as jr
import numpy as np
import pytest

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

        key = jr.PRNGKey(123)
        jres = jr.uniform(key, shape=(10, 100))
        self.assertTrue(jnp.allclose(jres, brainstate.random.rand(10, 100, key=key)))
        self.assertTrue(jnp.allclose(jres, brainstate.random.rand(10, 100, key=123)))

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
        brainstate.random.seed()
        with self.assertRaises(NotImplementedError):
            a = brainstate.random.chisquare(df=[2, 3, 4])

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
        a = brainstate.random.triangular((2, 2))
        self.assertTupleEqual(a.shape, (2, 2))

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
