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


import platform
import unittest

import jax.numpy as jnp
import jax.random
import jax.random as jr
import numpy as np
import pytest

import brainstate as bst


class TestRandom(unittest.TestCase):

  def test_rand(self):
    bst.random.seed()
    a = bst.random.rand(3, 2)
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())

    key = jr.PRNGKey(123)
    jres = jr.uniform(key, shape=(10, 100))
    self.assertTrue(jnp.allclose(jres, bst.random.rand(10, 100, key=key)))
    self.assertTrue(jnp.allclose(jres, bst.random.rand(10, 100, key=123)))

  def test_randint1(self):
    bst.random.seed()
    a = bst.random.randint(5)
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(0 <= a < 5)

  def test_randint2(self):
    bst.random.seed()
    a = bst.random.randint(2, 6, size=(4, 3))
    self.assertTupleEqual(a.shape, (4, 3))
    self.assertTrue((a >= 2).all() and (a < 6).all())

  def test_randint3(self):
    bst.random.seed()
    a = bst.random.randint([1, 2, 3], [10, 7, 8])
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue((a - jnp.array([1, 2, 3]) >= 0).all()
                    and (-a + jnp.array([10, 7, 8]) > 0).all())

  def test_randint4(self):
    bst.random.seed()
    a = bst.random.randint([1, 2, 3], [10, 7, 8], size=(2, 3))
    self.assertTupleEqual(a.shape, (2, 3))

  def test_randn(self):
    bst.random.seed()
    a = bst.random.randn(3, 2)
    self.assertTupleEqual(a.shape, (3, 2))

  def test_random1(self):
    bst.random.seed()
    a = bst.random.random()
    self.assertTrue(0. <= a < 1)

  def test_random2(self):
    bst.random.seed()
    a = bst.random.random(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())

  def test_random_sample(self):
    bst.random.seed()
    a = bst.random.random_sample(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())

  def test_choice1(self):
    bst.random.seed()
    a = bst.random.choice(5)
    self.assertTupleEqual(jnp.shape(a), ())
    self.assertTrue(0 <= a < 5)

  def test_choice2(self):
    bst.random.seed()
    a = bst.random.choice(5, 3, p=[0.1, 0.4, 0.2, 0., 0.3])
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue((a >= 0).all() and (a < 5).all())

  def test_choice3(self):
    bst.random.seed()
    a = bst.random.choice(jnp.arange(2, 20), size=(4, 3), replace=False)
    self.assertTupleEqual(a.shape, (4, 3))
    self.assertTrue((a >= 2).all() and (a < 20).all())
    self.assertEqual(len(jnp.unique(a)), 12)

  def test_permutation1(self):
    bst.random.seed()
    a = bst.random.permutation(10)
    self.assertTupleEqual(a.shape, (10,))
    self.assertEqual(len(jnp.unique(a)), 10)

  def test_permutation2(self):
    bst.random.seed()
    a = bst.random.permutation(jnp.arange(10))
    self.assertTupleEqual(a.shape, (10,))
    self.assertEqual(len(jnp.unique(a)), 10)

  def test_shuffle1(self):
    bst.random.seed()
    a = jnp.arange(10)
    bst.random.shuffle(a)
    self.assertTupleEqual(a.shape, (10,))
    self.assertEqual(len(jnp.unique(a)), 10)

  def test_shuffle2(self):
    bst.random.seed()
    a = jnp.arange(12).reshape(4, 3)
    bst.random.shuffle(a, axis=1)
    self.assertTupleEqual(a.shape, (4, 3))
    self.assertEqual(len(jnp.unique(a)), 12)

    # test that a is only shuffled along axis 1
    uni = jnp.unique(jnp.diff(a, axis=0))
    self.assertEqual(uni, jnp.asarray([3]))

  def test_beta1(self):
    bst.random.seed()
    a = bst.random.beta(2, 2)
    self.assertTupleEqual(a.shape, ())

  def test_beta2(self):
    bst.random.seed()
    a = bst.random.beta([2, 2, 3], 2, size=(3,))
    self.assertTupleEqual(a.shape, (3,))

  def test_exponential1(self):
    bst.random.seed()
    a = bst.random.exponential(10., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_exponential2(self):
    bst.random.seed()
    a = bst.random.exponential([1., 2., 5.])
    self.assertTupleEqual(a.shape, (3,))

  def test_gamma(self):
    bst.random.seed()
    a = bst.random.gamma(2, 10., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_gumbel(self):
    bst.random.seed()
    a = bst.random.gumbel(0., 2., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_laplace(self):
    bst.random.seed()
    a = bst.random.laplace(0., 2., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_logistic(self):
    bst.random.seed()
    a = bst.random.logistic(0., 2., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_normal1(self):
    bst.random.seed()
    a = bst.random.normal()
    self.assertTupleEqual(a.shape, ())

  def test_normal2(self):
    bst.random.seed()
    a = bst.random.normal(loc=[0., 2., 4.], scale=[1., 2., 3.])
    self.assertTupleEqual(a.shape, (3,))

  def test_normal3(self):
    bst.random.seed()
    a = bst.random.normal(loc=[0., 2., 4.], scale=[[1., 2., 3.], [1., 1., 1.]])
    print(a)
    self.assertTupleEqual(a.shape, (2, 3))

  def test_pareto(self):
    bst.random.seed()
    a = bst.random.pareto([1, 2, 2])
    self.assertTupleEqual(a.shape, (3,))

  def test_poisson(self):
    bst.random.seed()
    a = bst.random.poisson([1., 2., 2.], size=3)
    self.assertTupleEqual(a.shape, (3,))

  def test_standard_cauchy(self):
    bst.random.seed()
    a = bst.random.standard_cauchy(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))

  def test_standard_exponential(self):
    bst.random.seed()
    a = bst.random.standard_exponential(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))

  def test_standard_gamma(self):
    bst.random.seed()
    a = bst.random.standard_gamma(shape=[1, 2, 4], size=3)
    self.assertTupleEqual(a.shape, (3,))

  def test_standard_normal(self):
    bst.random.seed()
    a = bst.random.standard_normal(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))

  def test_standard_t(self):
    bst.random.seed()
    a = bst.random.standard_t(df=[1, 2, 4], size=3)
    self.assertTupleEqual(a.shape, (3,))

  def test_standard_uniform1(self):
    bst.random.seed()
    a = bst.random.uniform()
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(0 <= a < 1)

  def test_uniform2(self):
    bst.random.seed()
    a = bst.random.uniform(low=[-1., 5., 2.], high=[2., 6., 10.], size=3)
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue((a - jnp.array([-1., 5., 2.]) >= 0).all()
                    and (-a + jnp.array([2., 6., 10.]) > 0).all())

  def test_uniform3(self):
    bst.random.seed()
    a = bst.random.uniform(low=-1., high=[2., 6., 10.], size=(2, 3))
    self.assertTupleEqual(a.shape, (2, 3))

  def test_uniform4(self):
    bst.random.seed()
    a = bst.random.uniform(low=[-1., 5., 2.], high=[[2., 6., 10.], [10., 10., 10.]])
    self.assertTupleEqual(a.shape, (2, 3))

  def test_truncated_normal1(self):
    bst.random.seed()
    a = bst.random.truncated_normal(-1., 1.)
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(-1. <= a <= 1.)

  def test_truncated_normal2(self):
    bst.random.seed()
    a = bst.random.truncated_normal(-1., [1., 2., 1.], size=(4, 3))
    self.assertTupleEqual(a.shape, (4, 3))

  def test_truncated_normal3(self):
    bst.random.seed()
    a = bst.random.truncated_normal([-1., 0., 1.], [[2., 2., 4.], [2., 2., 4.]])
    self.assertTupleEqual(a.shape, (2, 3))
    self.assertTrue((a - jnp.array([-1., 0., 1.]) >= 0.).all()
                    and (- a + jnp.array([2., 2., 4.]) >= 0.).all())

  def test_bernoulli1(self):
    bst.random.seed()
    a = bst.random.bernoulli()
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(a == 0 or a == 1)

  def test_bernoulli2(self):
    bst.random.seed()
    a = bst.random.bernoulli([0.5, 0.6, 0.8])
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue(jnp.logical_xor(a == 1, a == 0).all())

  def test_bernoulli3(self):
    bst.random.seed()
    a = bst.random.bernoulli([0.5, 0.6], size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue(jnp.logical_xor(a == 1, a == 0).all())

  def test_lognormal1(self):
    bst.random.seed()
    a = bst.random.lognormal()
    self.assertTupleEqual(a.shape, ())

  def test_lognormal2(self):
    bst.random.seed()
    a = bst.random.lognormal(sigma=[2., 1.], size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_lognormal3(self):
    bst.random.seed()
    a = bst.random.lognormal([2., 0.], [[2., 1.], [3., 1.2]])
    self.assertTupleEqual(a.shape, (2, 2))

  def test_binomial1(self):
    bst.random.seed()
    a = bst.random.binomial(5, 0.5)
    b = np.random.binomial(5, 0.5)
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(a.dtype, int)

  def test_binomial2(self):
    bst.random.seed()
    a = bst.random.binomial(5, 0.5, size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a <= 5).all())

  def test_binomial3(self):
    bst.random.seed()
    a = bst.random.binomial(n=jnp.asarray([2, 3, 4]), p=jnp.asarray([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]))
    self.assertTupleEqual(a.shape, (2, 3))

  def test_chisquare1(self):
    bst.random.seed()
    a = bst.random.chisquare(3)
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(a.dtype, float)

  def test_chisquare2(self):
    bst.random.seed()
    with self.assertRaises(NotImplementedError):
      a = bst.random.chisquare(df=[2, 3, 4])

  def test_chisquare3(self):
    bst.random.seed()
    a = bst.random.chisquare(df=2, size=100)
    self.assertTupleEqual(a.shape, (100,))

  def test_chisquare4(self):
    bst.random.seed()
    a = bst.random.chisquare(df=2, size=(100, 10))
    self.assertTupleEqual(a.shape, (100, 10))

  def test_dirichlet1(self):
    bst.random.seed()
    a = bst.random.dirichlet((10, 5, 3))
    self.assertTupleEqual(a.shape, (3,))

  def test_dirichlet2(self):
    bst.random.seed()
    a = bst.random.dirichlet((10, 5, 3), 20)
    self.assertTupleEqual(a.shape, (20, 3))

  def test_f(self):
    bst.random.seed()
    a = bst.random.f(1., 48., 100)
    self.assertTupleEqual(a.shape, (100,))

  def test_geometric(self):
    bst.random.seed()
    a = bst.random.geometric([0.7, 0.5, 0.2])
    self.assertTupleEqual(a.shape, (3,))

  def test_hypergeometric1(self):
    bst.random.seed()
    a = bst.random.hypergeometric(10, 10, 10, 20)
    self.assertTupleEqual(a.shape, (20,))

  @pytest.mark.skipif(platform.system() == 'Windows', reason='Windows jaxlib error')
  def test_hypergeometric2(self):
    bst.random.seed()
    a = bst.random.hypergeometric(8, [10, 4], [[5, 2], [5, 5]])
    self.assertTupleEqual(a.shape, (2, 2))

  @pytest.mark.skipif(platform.system() == 'Windows', reason='Windows jaxlib error')
  def test_hypergeometric3(self):
    bst.random.seed()
    a = bst.random.hypergeometric(8, [10, 4], [[5, 2], [5, 5]], size=(3, 2, 2))
    self.assertTupleEqual(a.shape, (3, 2, 2))

  def test_logseries(self):
    bst.random.seed()
    a = bst.random.logseries([0.7, 0.5, 0.2], size=[4, 3])
    self.assertTupleEqual(a.shape, (4, 3))

  def test_multinominal1(self):
    bst.random.seed()
    a = np.random.multinomial(100, (0.5, 0.2, 0.3), size=[4, 2])
    print(a, a.shape)
    b = bst.random.multinomial(100, (0.5, 0.2, 0.3), size=[4, 2])
    print(b, b.shape)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (4, 2, 3))

  def test_multinominal2(self):
    bst.random.seed()
    a = bst.random.multinomial(100, (0.5, 0.2, 0.3))
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue(a.sum() == 100)

  def test_multivariate_normal1(self):
    bst.random.seed()
    # self.skipTest('Windows jaxlib error')
    a = np.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], size=3)
    b = bst.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], size=3)
    print('test_multivariate_normal1')
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(a.shape, (3, 2))

  def test_multivariate_normal2(self):
    bst.random.seed()
    a = np.random.multivariate_normal([1, 2], [[1, 3], [3, 1]])
    b = bst.random.multivariate_normal([1, 2], [[1, 3], [3, 1]], method='svd')
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(a.shape, (2,))

  def test_negative_binomial(self):
    bst.random.seed()
    a = np.random.negative_binomial([3., 10.], 0.5)
    b = bst.random.negative_binomial([3., 10.], 0.5)
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (2,))

  def test_negative_binomial2(self):
    bst.random.seed()
    a = np.random.negative_binomial(3., 0.5, 10)
    b = bst.random.negative_binomial(3., 0.5, 10)
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (10,))

  def test_noncentral_chisquare(self):
    bst.random.seed()
    a = np.random.noncentral_chisquare(3, [3., 2.], (4, 2))
    b = bst.random.noncentral_chisquare(3, [3., 2.], (4, 2))
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (4, 2))

  def test_noncentral_chisquare2(self):
    bst.random.seed()
    a = bst.random.noncentral_chisquare(3, [3., 2.])
    self.assertTupleEqual(a.shape, (2,))

  def test_noncentral_f(self):
    bst.random.seed()
    a = bst.random.noncentral_f(3, 20, 3., 100)
    self.assertTupleEqual(a.shape, (100,))

  def test_power(self):
    bst.random.seed()
    a = np.random.power(2, (4, 2))
    b = bst.random.power(2, (4, 2))
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (4, 2))

  def test_rayleigh(self):
    bst.random.seed()
    a = bst.random.power(2., (4, 2))
    self.assertTupleEqual(a.shape, (4, 2))

  def test_triangular(self):
    bst.random.seed()
    a = bst.random.triangular((2, 2))
    self.assertTupleEqual(a.shape, (2, 2))

  def test_vonmises(self):
    bst.random.seed()
    a = np.random.vonmises(2., 2.)
    b = bst.random.vonmises(2., 2.)
    print(a, b)
    self.assertTupleEqual(np.shape(a), b.shape)
    self.assertTupleEqual(b.shape, ())

  def test_vonmises2(self):
    bst.random.seed()
    a = np.random.vonmises(2., 2., 10)
    b = bst.random.vonmises(2., 2., 10)
    print(a, b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (10,))

  def test_wald(self):
    bst.random.seed()
    a = np.random.wald([2., 0.5], 2.)
    b = bst.random.wald([2., 0.5], 2.)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (2,))

  def test_wald2(self):
    bst.random.seed()
    a = np.random.wald(2., 2., 100)
    b = bst.random.wald(2., 2., 100)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (100,))

  def test_weibull(self):
    bst.random.seed()
    a = bst.random.weibull(2., (4, 2))
    self.assertTupleEqual(a.shape, (4, 2))

  def test_weibull2(self):
    bst.random.seed()
    a = bst.random.weibull(2., )
    self.assertTupleEqual(a.shape, ())

  def test_weibull3(self):
    bst.random.seed()
    a = bst.random.weibull([2., 3.], )
    self.assertTupleEqual(a.shape, (2,))

  def test_weibull_min(self):
    bst.random.seed()
    a = bst.random.weibull_min(2., 2., (4, 2))
    self.assertTupleEqual(a.shape, (4, 2))

  def test_weibull_min2(self):
    bst.random.seed()
    a = bst.random.weibull_min(2., 2.)
    self.assertTupleEqual(a.shape, ())

  def test_weibull_min3(self):
    bst.random.seed()
    a = bst.random.weibull_min([2., 3.], 2.)
    self.assertTupleEqual(a.shape, (2,))

  def test_zipf(self):
    bst.random.seed()
    a = bst.random.zipf(2., (4, 2))
    self.assertTupleEqual(a.shape, (4, 2))

  def test_zipf2(self):
    bst.random.seed()
    a = np.random.zipf([1.1, 2.])
    b = bst.random.zipf([1.1, 2.])
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (2,))

  def test_maxwell(self):
    bst.random.seed()
    a = bst.random.maxwell(10)
    self.assertTupleEqual(a.shape, (10,))

  def test_maxwell2(self):
    bst.random.seed()
    a = bst.random.maxwell()
    self.assertTupleEqual(a.shape, ())

  def test_t(self):
    bst.random.seed()
    a = bst.random.t(1., size=10)
    self.assertTupleEqual(a.shape, (10,))

  def test_t2(self):
    bst.random.seed()
    a = bst.random.t([1., 2.], size=None)
    self.assertTupleEqual(a.shape, (2,))


# class TestRandomKey(unittest.TestCase):
#   def test_clear_memory(self):
#     bst.random.split_key()
#     print(bst.random.DEFAULT.value)
#     self.assertTrue(isinstance(bst.random.DEFAULT.value, np.ndarray))
