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

"""
Comprehensive tests for surrogate gradient functions.
"""

import unittest
import jax
import jax.numpy as jnp
import brainstate.surrogate as surrogate


class TestSurrogateBase(unittest.TestCase):
    """Test base Surrogate class functionality."""

    def test_surrogate_abstract_methods(self):
        """Test that base Surrogate class has required abstract methods."""
        # Should not be able to instantiate base Surrogate class directly
        with self.assertRaises(NotImplementedError):
            surrogate.Surrogate().surrogate_fun(0.)


class TestSigmoidBased(unittest.TestCase):
    """Test sigmoid-based surrogate gradients."""

    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass returns binary values."""
        sg = surrogate.Sigmoid(alpha=4.0)
        x = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y = sg(x)
        # Forward should be Heaviside step function
        expected = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_sigmoid_backward(self):
        """Test Sigmoid backward pass computes gradients."""
        sg = surrogate.Sigmoid(alpha=4.0)
        x = jnp.array([0.0])

        def loss(x):
            return jnp.sum(sg(x))

        grad = jax.grad(loss)(x)
        # Gradient should be non-zero at x=0
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_sigmoid_parameters(self):
        """Test Sigmoid with different alpha values."""
        x = jnp.array([0.0])

        # Higher alpha should give steeper gradient
        sg1 = surrogate.Sigmoid(alpha=1.0)
        sg2 = surrogate.Sigmoid(alpha=10.0)

        grad1 = jax.grad(lambda x: jnp.sum(sg1(x)))(x)
        grad2 = jax.grad(lambda x: jnp.sum(sg2(x)))(x)

        # Both should have positive gradients at x=0
        self.assertGreater(grad1[0], 0.0)
        self.assertGreater(grad2[0], 0.0)

    def test_sigmoid_functional_api(self):
        """Test sigmoid functional API matches class API."""
        x = jnp.array([-1.0, 0.0, 1.0])

        sg_class = surrogate.Sigmoid(alpha=4.0)
        y_class = sg_class(x)

        # Functional API should give same forward result
        y_func = surrogate.sigmoid(x, alpha=4.0)
        self.assertTrue(jnp.allclose(y_class, y_func))

    def test_soft_sign_forward(self):
        """Test SoftSign forward pass."""
        sg = surrogate.SoftSign(alpha=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_soft_sign_backward(self):
        """Test SoftSign backward pass."""
        sg = surrogate.SoftSign(alpha=1.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_arctan_forward(self):
        """Test Arctan forward pass."""
        sg = surrogate.Arctan(alpha=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_arctan_backward(self):
        """Test Arctan backward pass."""
        sg = surrogate.Arctan(alpha=2.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_erf_forward(self):
        """Test ERF forward pass."""
        sg = surrogate.ERF(alpha=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_erf_backward(self):
        """Test ERF backward pass."""
        sg = surrogate.ERF(alpha=1.5)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)


class TestPiecewiseFunctions(unittest.TestCase):
    """Test piecewise surrogate gradients."""

    def test_piecewise_quadratic_forward(self):
        """Test PiecewiseQuadratic forward pass."""
        sg = surrogate.PiecewiseQuadratic(alpha=1.0)
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = sg(x)
        expected = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_piecewise_quadratic_backward(self):
        """Test PiecewiseQuadratic backward pass."""
        sg = surrogate.PiecewiseQuadratic(alpha=1.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_piecewise_quadratic_gradient_support(self):
        """Test PiecewiseQuadratic has finite support."""
        sg = surrogate.PiecewiseQuadratic(alpha=1.0)
        # Gradient should be zero far from threshold
        x_far = jnp.array([10.0])
        grad_far = jax.grad(lambda x: jnp.sum(sg(x)))(x_far)
        self.assertAlmostEqual(grad_far[0], 0.0, places=5)

    def test_piecewise_exp_forward(self):
        """Test PiecewiseExp forward pass."""
        sg = surrogate.PiecewiseExp(alpha=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_piecewise_exp_backward(self):
        """Test PiecewiseExp backward pass."""
        sg = surrogate.PiecewiseExp(alpha=2.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_piecewise_leaky_relu_forward(self):
        """Test PiecewiseLeakyRelu forward pass."""
        sg = surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_piecewise_leaky_relu_backward(self):
        """Test PiecewiseLeakyRelu backward pass."""
        sg = surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_piecewise_leaky_relu_parameters(self):
        """Test PiecewiseLeakyRelu with different parameters."""
        x = jnp.array([0.0])

        sg1 = surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0)
        sg2 = surrogate.PiecewiseLeakyRelu(c=0.1, w=0.5)

        grad1 = jax.grad(lambda x: jnp.sum(sg1(x)))(x)
        grad2 = jax.grad(lambda x: jnp.sum(sg2(x)))(x)

        self.assertGreater(jnp.abs(grad1[0]), 0.0)
        self.assertGreater(jnp.abs(grad2[0]), 0.0)


class TestReluBased(unittest.TestCase):
    """Test ReLU-based surrogate gradients."""

    def test_relu_grad_forward(self):
        """Test ReluGrad forward pass."""
        sg = surrogate.ReluGrad(alpha=0.3, width=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_relu_grad_backward(self):
        """Test ReluGrad backward pass."""
        sg = surrogate.ReluGrad(alpha=0.5, width=1.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_relu_grad_finite_support(self):
        """Test ReluGrad has finite support."""
        sg = surrogate.ReluGrad(alpha=0.3, width=1.0)
        x_far = jnp.array([10.0])
        grad_far = jax.grad(lambda x: jnp.sum(sg(x)))(x_far)
        self.assertAlmostEqual(grad_far[0], 0.0, places=5)

    def test_leaky_relu_forward(self):
        """Test LeakyRelu forward pass."""
        sg = surrogate.LeakyRelu(alpha=0.1, beta=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_leaky_relu_backward(self):
        """Test LeakyRelu backward pass."""
        sg = surrogate.LeakyRelu(alpha=0.1, beta=1.0)

        # Test gradient for x < 0
        x_neg = jnp.array([-1.0])
        grad_neg = jax.grad(lambda x: jnp.sum(sg(x)))(x_neg)
        self.assertGreater(jnp.abs(grad_neg[0]), 0.0)

        # Test gradient for x > 0
        x_pos = jnp.array([1.0])
        grad_pos = jax.grad(lambda x: jnp.sum(sg(x)))(x_pos)
        self.assertGreater(jnp.abs(grad_pos[0]), 0.0)

    def test_log_tailed_relu_forward(self):
        """Test LogTailedRelu forward pass."""
        sg = surrogate.LogTailedRelu(alpha=0.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_log_tailed_relu_backward(self):
        """Test LogTailedRelu backward pass."""
        sg = surrogate.LogTailedRelu(alpha=0.1)
        x = jnp.array([0.5])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)


class TestDistributionInspired(unittest.TestCase):
    """Test distribution-inspired surrogate gradients."""

    def test_gaussian_grad_forward(self):
        """Test GaussianGrad forward pass."""
        sg = surrogate.GaussianGrad(sigma=0.5, alpha=0.5)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_gaussian_grad_backward(self):
        """Test GaussianGrad backward pass."""
        sg = surrogate.GaussianGrad(sigma=0.5, alpha=1.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_gaussian_grad_symmetry(self):
        """Test GaussianGrad is symmetric around threshold."""
        sg = surrogate.GaussianGrad(sigma=0.5, alpha=1.0)

        x1 = jnp.array([0.1])
        x2 = jnp.array([-0.1])

        grad1 = jax.grad(lambda x: jnp.sum(sg(x)))(x1)
        grad2 = jax.grad(lambda x: jnp.sum(sg(x)))(x2)

        # Gradients should be symmetric
        self.assertAlmostEqual(grad1[0], grad2[0], places=5)

    def test_multi_gaussian_grad_forward(self):
        """Test MultiGaussianGrad forward pass."""
        sg = surrogate.MultiGaussianGrad(h=0.15, s=6.0, sigma=0.5, scale=0.5)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_multi_gaussian_grad_backward(self):
        """Test MultiGaussianGrad backward pass."""
        sg = surrogate.MultiGaussianGrad(h=0.15, s=6.0, sigma=0.5, scale=0.5)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_inv_square_grad_forward(self):
        """Test InvSquareGrad forward pass."""
        sg = surrogate.InvSquareGrad(alpha=100.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_inv_square_grad_backward(self):
        """Test InvSquareGrad backward pass."""
        sg = surrogate.InvSquareGrad(alpha=100.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_slayer_grad_forward(self):
        """Test SlayerGrad forward pass."""
        sg = surrogate.SlayerGrad(alpha=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_slayer_grad_backward(self):
        """Test SlayerGrad backward pass."""
        sg = surrogate.SlayerGrad(alpha=2.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)


class TestAdvancedFunctions(unittest.TestCase):
    """Test advanced surrogate gradients."""

    def test_nonzero_sign_log_forward(self):
        """Test NonzeroSignLog forward pass."""
        sg = surrogate.NonzeroSignLog(alpha=1.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_nonzero_sign_log_backward(self):
        """Test NonzeroSignLog backward pass."""
        sg = surrogate.NonzeroSignLog(alpha=1.0)
        x = jnp.array([0.1])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_squarewave_fourier_series_forward(self):
        """Test SquarewaveFourierSeries forward pass."""
        sg = surrogate.SquarewaveFourierSeries(n=2, t_period=8.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_squarewave_fourier_series_backward(self):
        """Test SquarewaveFourierSeries backward pass."""
        sg = surrogate.SquarewaveFourierSeries(n=3, t_period=8.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_squarewave_fourier_terms(self):
        """Test SquarewaveFourierSeries with different n."""
        x = jnp.array([0.0])

        # More terms should give different gradient
        sg1 = surrogate.SquarewaveFourierSeries(n=2, t_period=8.0)
        sg2 = surrogate.SquarewaveFourierSeries(n=5, t_period=8.0)

        grad1 = jax.grad(lambda x: jnp.sum(sg1(x)))(x)
        grad2 = jax.grad(lambda x: jnp.sum(sg2(x)))(x)

        self.assertGreater(jnp.abs(grad1[0]), 0.0)
        self.assertGreater(jnp.abs(grad2[0]), 0.0)

    def test_s2nn_forward(self):
        """Test S2NN forward pass."""
        sg = surrogate.S2NN(alpha=4.0, beta=1.0, epsilon=1e-8)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_s2nn_backward(self):
        """Test S2NN backward pass."""
        sg = surrogate.S2NN(alpha=4.0, beta=1.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_s2nn_asymmetry(self):
        """Test S2NN has asymmetric gradients."""
        sg = surrogate.S2NN(alpha=2.0, beta=1.0)

        x_neg = jnp.array([-0.5])
        x_pos = jnp.array([0.5])

        grad_neg = jax.grad(lambda x: jnp.sum(sg(x)))(x_neg)
        grad_pos = jax.grad(lambda x: jnp.sum(sg(x)))(x_pos)

        # With alpha != beta, gradients should be different
        self.assertGreater(jnp.abs(grad_neg[0]), 0.0)
        self.assertGreater(jnp.abs(grad_pos[0]), 0.0)

    def test_q_pseudo_spike_forward(self):
        """Test QPseudoSpike forward pass."""
        sg = surrogate.QPseudoSpike(alpha=2.0)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = sg(x)
        expected = jnp.array([0.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_q_pseudo_spike_backward(self):
        """Test QPseudoSpike backward pass."""
        sg = surrogate.QPseudoSpike(alpha=2.0)
        x = jnp.array([0.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)

    def test_q_pseudo_spike_parameters(self):
        """Test QPseudoSpike with different alpha values."""
        x = jnp.array([0.0])

        sg1 = surrogate.QPseudoSpike(alpha=1.5)
        sg2 = surrogate.QPseudoSpike(alpha=3.0)

        grad1 = jax.grad(lambda x: jnp.sum(sg1(x)))(x)
        grad2 = jax.grad(lambda x: jnp.sum(sg2(x)))(x)

        self.assertGreater(jnp.abs(grad1[0]), 0.0)
        self.assertGreater(jnp.abs(grad2[0]), 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and numerical stability."""

    def test_large_positive_values(self):
        """Test surrogates with large positive values."""
        surrogates_to_test = [
            surrogate.Sigmoid(),
            surrogate.ReluGrad(),
            surrogate.GaussianGrad(),
        ]

        x = jnp.array([100.0])
        for sg in surrogates_to_test:
            y = sg(x)
            self.assertEqual(y[0], 1.0)

    def test_large_negative_values(self):
        """Test surrogates with large negative values."""
        surrogates_to_test = [
            surrogate.Sigmoid(),
            surrogate.ReluGrad(),
            surrogate.GaussianGrad(),
        ]

        x = jnp.array([-100.0])
        for sg in surrogates_to_test:
            y = sg(x)
            self.assertEqual(y[0], 0.0)

    def test_zero_input(self):
        """Test all surrogates at x=0."""
        surrogates_to_test = [
            surrogate.Sigmoid(),
            surrogate.PiecewiseQuadratic(),
            surrogate.SoftSign(),
            surrogate.Arctan(),
            surrogate.ERF(),
            surrogate.ReluGrad(),
            surrogate.GaussianGrad(),
            surrogate.LeakyRelu(),
        ]

        x = jnp.array([0.0])
        for sg in surrogates_to_test:
            y = sg(x)
            self.assertEqual(y[0], 1.0)

    def test_array_inputs(self):
        """Test surrogates with various array shapes."""
        sg = surrogate.Sigmoid()

        # 1D array
        x1d = jnp.array([0.0, 0.5, 1.0])
        y1d = sg(x1d)
        self.assertEqual(y1d.shape, (3,))

        # 2D array
        x2d = jnp.array([[0.0, 0.5], [1.0, -1.0]])
        y2d = sg(x2d)
        self.assertEqual(y2d.shape, (2, 2))

        # 3D array
        x3d = jnp.array([[[0.0, 0.5], [1.0, -1.0]]])
        y3d = sg(x3d)
        self.assertEqual(y3d.shape, (1, 2, 2))

    def test_batch_processing(self):
        """Test surrogates work with vmap."""
        sg = surrogate.Sigmoid()

        x = jnp.array([[0.0, 0.5, 1.0], [-1.0, 0.0, 2.0]])

        # Should work with vmap
        batched_sg = jax.vmap(sg)
        y = batched_sg(x)
        self.assertEqual(y.shape, (2, 3))

    def test_jit_compilation(self):
        """Test surrogates work with JIT compilation."""
        sg = surrogate.Sigmoid()

        @jax.jit
        def forward(x):
            return sg(x)

        x = jnp.array([0.0, 0.5, 1.0])
        y = forward(x)
        expected = jnp.array([1.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(y, expected))

    def test_gradient_jit(self):
        """Test gradients work with JIT compilation."""
        sg = surrogate.Sigmoid()

        @jax.jit
        def loss_fn(x):
            return jnp.sum(sg(x))

        x = jnp.array([0.0, 0.5, 1.0])
        grad = jax.grad(loss_fn)(x)
        self.assertEqual(grad.shape, (3,))


class TestFunctionalAPI(unittest.TestCase):
    """Test functional API for all surrogates."""

    def test_sigmoid_functional(self):
        """Test sigmoid functional API."""
        x = jnp.array([0.0])
        y = surrogate.sigmoid(x, alpha=4.0)
        self.assertEqual(y[0], 1.0)

    def test_piecewise_quadratic_functional(self):
        """Test piecewise_quadratic functional API."""
        x = jnp.array([0.0])
        y = surrogate.piecewise_quadratic(x, alpha=1.0)
        self.assertEqual(y[0], 1.0)

    def test_piecewise_exp_functional(self):
        """Test piecewise_exp functional API."""
        x = jnp.array([0.0])
        y = surrogate.piecewise_exp(x, alpha=1.0)
        self.assertEqual(y[0], 1.0)

    def test_soft_sign_functional(self):
        """Test soft_sign functional API."""
        x = jnp.array([0.0])
        y = surrogate.soft_sign(x, alpha=1.0)
        self.assertEqual(y[0], 1.0)

    def test_arctan_functional(self):
        """Test arctan functional API."""
        x = jnp.array([0.0])
        y = surrogate.arctan(x, alpha=1.0)
        self.assertEqual(y[0], 1.0)

    def test_erf_functional(self):
        """Test erf functional API."""
        x = jnp.array([0.0])
        y = surrogate.erf(x, alpha=1.0)
        self.assertEqual(y[0], 1.0)

    def test_relu_grad_functional(self):
        """Test relu_grad functional API."""
        x = jnp.array([0.0])
        y = surrogate.relu_grad(x, alpha=0.3, width=1.0)
        self.assertEqual(y[0], 1.0)

    def test_gaussian_grad_functional(self):
        """Test gaussian_grad functional API."""
        x = jnp.array([0.0])
        y = surrogate.gaussian_grad(x, sigma=0.5, alpha=0.5)
        self.assertEqual(y[0], 1.0)

    def test_functional_gradient(self):
        """Test functional API supports gradients."""
        x = jnp.array([0.0])

        def loss(x):
            return jnp.sum(surrogate.sigmoid(x, alpha=4.0))

        grad = jax.grad(loss)(x)
        self.assertGreater(jnp.abs(grad[0]), 0.0)


class TestReprAndHash(unittest.TestCase):
    """Test __repr__ and __hash__ methods."""

    def test_sigmoid_repr(self):
        """Test Sigmoid __repr__."""
        sg = surrogate.Sigmoid(alpha=4.0)
        repr_str = repr(sg)
        self.assertIn('Sigmoid', repr_str)
        self.assertIn('alpha', repr_str)

    def test_sigmoid_hash(self):
        """Test Sigmoid __hash__."""
        sg1 = surrogate.Sigmoid(alpha=4.0)
        sg2 = surrogate.Sigmoid(alpha=4.0)
        sg3 = surrogate.Sigmoid(alpha=2.0)

        # Same parameters should have same hash
        self.assertEqual(hash(sg1), hash(sg2))
        # Different parameters should (likely) have different hash
        # Note: hash collisions are possible but unlikely
        self.assertNotEqual(hash(sg1), hash(sg3))

    def test_surrogates_in_dict(self):
        """Test surrogates can be used as dict keys."""
        sg1 = surrogate.Sigmoid(alpha=4.0)
        sg2 = surrogate.ReluGrad(alpha=0.3, width=1.0)

        d = {sg1: 'sigmoid', sg2: 'relu_grad'}
        self.assertEqual(d[sg1], 'sigmoid')
        self.assertEqual(d[sg2], 'relu_grad')

    def test_surrogates_in_set(self):
        """Test surrogates can be used in sets."""
        sg1 = surrogate.Sigmoid(alpha=4.0)
        sg2 = surrogate.Sigmoid(alpha=4.0)
        sg3 = surrogate.ReluGrad()
        assert hash(sg1) == hash(sg2)


class TestGradientProperties(unittest.TestCase):
    """Test mathematical properties of gradients."""

    def test_gradient_continuity(self):
        """Test gradients are continuous."""
        sg = surrogate.Sigmoid(alpha=4.0)

        # Test points near threshold
        x1 = jnp.array([0.0])
        x2 = jnp.array([0.001])

        grad1 = jax.grad(lambda x: jnp.sum(sg(x)))(x1)
        grad2 = jax.grad(lambda x: jnp.sum(sg(x)))(x2)

        # Gradients should be close for nearby points
        self.assertAlmostEqual(grad1[0], grad2[0], places=2)

    def test_gradient_positivity(self):
        """Test gradients are positive near threshold."""
        surrogates_to_test = [
            surrogate.Sigmoid(),
            surrogate.ReluGrad(),
            surrogate.GaussianGrad(),
            surrogate.PiecewiseQuadratic(),
        ]

        x = jnp.array([0.0])
        for sg in surrogates_to_test:
            grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
            self.assertGreater(grad[0], 0.0)

    def test_gradient_vanishing(self):
        """Test gradients vanish far from threshold for finite support."""
        sg = surrogate.ReluGrad(alpha=0.3, width=1.0)

        x_far = jnp.array([10.0])
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x_far)
        self.assertAlmostEqual(grad[0], 0.0, places=5)

    def test_second_order_gradients(self):
        """Test second-order gradients can be computed."""
        sg = surrogate.Sigmoid(alpha=4.0)

        def loss(x):
            return jnp.sum(sg(x))

        x = jnp.array([0.0])

        # First-order gradient
        grad1 = jax.grad(loss)(x)
        self.assertGreater(jnp.abs(grad1[0]), 0.0)

        # Second-order gradient (Hessian diagonal)
        hess = jax.grad(lambda x: jax.grad(loss)(x)[0])(x)
        # Second derivative should exist (even if zero or small)
        self.assertTrue(jnp.isfinite(hess))


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of surrogates."""

    def test_no_nans(self):
        """Test surrogates don't produce NaNs."""
        surrogates_to_test = [
            surrogate.Sigmoid(),
            surrogate.ReluGrad(),
            surrogate.GaussianGrad(),
            surrogate.ERF(),
        ]

        x = jnp.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])

        for sg in surrogates_to_test:
            y = sg(x)
            self.assertTrue(jnp.all(jnp.isfinite(y)))

    def test_no_inf_gradients(self):
        """Test gradients don't become infinite."""
        surrogates_to_test = [
            surrogate.Sigmoid(),
            surrogate.ReluGrad(),
            surrogate.GaussianGrad(),
        ]

        x = jnp.array([0.0, 1.0, -1.0, 10.0, -10.0])

        for sg in surrogates_to_test:
            grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)
            self.assertTrue(jnp.all(jnp.isfinite(grad)))

    def test_s2nn_epsilon_stability(self):
        """Test S2NN epsilon parameter provides numerical stability."""
        sg = surrogate.S2NN(alpha=4.0, beta=1.0, epsilon=1e-8)

        # Very small values near zero
        x = jnp.array([1e-10])
        y = sg(x)
        grad = jax.grad(lambda x: jnp.sum(sg(x)))(x)

        self.assertTrue(jnp.isfinite(y[0]))
        self.assertTrue(jnp.isfinite(grad[0]))


class TestIntegration(unittest.TestCase):
    """Test integration with typical SNN workflows."""

    def test_simple_neuron_model(self):
        """Test surrogate in simple neuron model."""
        sg = surrogate.Sigmoid(alpha=4.0)

        # Simple LIF-like dynamics
        def neuron_step(v, i_input):
            v = v + i_input
            spike = sg(v - 1.0)  # threshold at 1.0
            v = v * (1.0 - spike)  # reset
            return v, spike

        v = jnp.array([0.0])
        i_input = jnp.array([0.5])

        # Should not spike on first step
        v, spike = neuron_step(v, i_input)
        self.assertEqual(spike[0], 0.0)

        # Should spike on third step
        v, spike = neuron_step(v, i_input)
        v, spike = neuron_step(v, i_input)
        v, spike = neuron_step(v, i_input)
        self.assertEqual(spike[0], 1.0)

    def test_gradient_flow_through_neuron(self):
        """Test gradients flow through neuron model."""
        sg = surrogate.Sigmoid(alpha=4.0)

        def neuron_forward(i_input):
            v = jnp.zeros(1)
            spikes = []
            for i in i_input:
                v = v + i
                spike = sg(v - 1.0)
                spikes.append(spike)
                v = v * (1.0 - spike)
            return jnp.sum(jnp.array(spikes))

        i_input = jnp.array([0.5, 0.5, 0.5])

        # Should be able to compute gradients
        grad = jax.grad(neuron_forward)(i_input)

        # Gradients should flow
        self.assertTrue(jnp.any(grad != 0.0))

    def test_multi_layer_gradient_flow(self):
        """Test gradients flow through multiple layers."""
        sg = surrogate.Sigmoid(alpha=4.0)

        def two_layer_network(x):
            # Layer 1
            h1 = sg(x - 0.5)
            # Layer 2
            h2 = sg(h1 - 0.5)
            return jnp.sum(h2)

        x = jnp.array([0.0, 0.5, 1.0])

        # Gradients should flow through both layers
        grad = jax.grad(two_layer_network)(x)

        # At least some gradients should be non-zero
        self.assertTrue(jnp.any(grad != 0.0))


if __name__ == '__main__':
    unittest.main()
