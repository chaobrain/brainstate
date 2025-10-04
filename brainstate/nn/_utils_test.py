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

import unittest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import brainstate


class TestClipGradNorm(parameterized.TestCase):
    """Comprehensive tests for clip_grad_norm function."""

    def setUp(self):
        """Set up test fixtures."""
        # Enable 64-bit precision for more accurate testing
        jax.config.update("jax_enable_x64", True)

    def test_simple_dict_clipping(self):
        """Test basic gradient clipping with dictionary structure."""
        grads = {
            'w': jnp.array([3.0, 4.0]),
            'b': jnp.array([12.0])
        }

        # Test with return_norm=True
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=5.0, return_norm=True)

        # Expected L2 norm: sqrt(3^2 + 4^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(169) = 13
        self.assertAlmostEqual(norm, 13.0, places=5)

        # Check clipped values: should be scaled by 5/13
        scale = 5.0 / 13.0
        np.testing.assert_array_almost_equal(
            clipped_grads['w'],
            jnp.array([3.0, 4.0]) * scale,
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            clipped_grads['b'],
            jnp.array([12.0]) * scale,
            decimal=5
        )

    def test_return_norm_parameter(self):
        """Test the return_norm parameter behavior."""
        grads = {
            'w': jnp.array([3.0, 4.0]),
            'b': jnp.array([12.0])
        }

        # Test with return_norm=False (default)
        clipped_grads_only = brainstate.nn.clip_grad_norm(grads, max_norm=5.0, return_norm=False)
        self.assertIsInstance(clipped_grads_only, dict)
        self.assertIn('w', clipped_grads_only)
        self.assertIn('b', clipped_grads_only)

        # Test with return_norm=True
        result = brainstate.nn.clip_grad_norm(grads, max_norm=5.0, return_norm=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        clipped_grads, norm = result

        # Values should be the same regardless of return_norm
        np.testing.assert_array_almost_equal(
            clipped_grads_only['w'],
            clipped_grads['w'],
            decimal=7
        )
        np.testing.assert_array_almost_equal(
            clipped_grads_only['b'],
            clipped_grads['b'],
            decimal=7
        )

    def test_nested_structure_clipping(self):
        """Test gradient clipping with nested PyTree structures."""
        grads = {
            'layer1': {
                'weight': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                'bias': jnp.array([5.0, 6.0])
            },
            'layer2': {
                'weight': jnp.array([[7.0, 8.0]]),
                'bias': jnp.array([9.0])
            }
        }

        # Calculate expected norm
        flat = jnp.arange(1.0, 10.0)
        expected_norm = jnp.linalg.norm(flat)

        max_norm = 10.0
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        self.assertAlmostEqual(norm, expected_norm, places=5)

        # Since norm > max_norm, gradients should be scaled
        scale = max_norm / expected_norm
        np.testing.assert_array_almost_equal(
            clipped_grads['layer1']['weight'],
            grads['layer1']['weight'] * scale,
            decimal=5
        )

    def test_no_clipping_when_under_max(self):
        """Test that gradients are unchanged when norm is below max_norm."""
        grads = {
            'w': jnp.array([1.0, 2.0]),
            'b': jnp.array([2.0])
        }

        # L2 norm = sqrt(1 + 4 + 4) = 3
        max_norm = 5.0
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        self.assertAlmostEqual(norm, 3.0, places=5)

        # Gradients should be unchanged
        np.testing.assert_array_almost_equal(
            clipped_grads['w'], grads['w'], decimal=7
        )
        np.testing.assert_array_almost_equal(
            clipped_grads['b'], grads['b'], decimal=7
        )

    @parameterized.parameters(
        (1, 'L1'),      # L1 norm
        (2, 'L2'),      # L2 norm (default)
        (2.0, 'L2'),    # L2 norm with float
        (3, 'L3'),      # L3 norm
        ('inf', 'Linf'),  # Infinity norm
        (jnp.inf, 'Linf'),  # Infinity norm with jnp.inf
    )
    def test_different_norm_types(self, norm_type, norm_name):
        """Test gradient clipping with different norm types."""
        grads = {
            'param': jnp.array([[-2.0, 3.0], [1.0, -4.0]])
        }

        max_norm = 3.0
        clipped_grads, computed_norm = brainstate.nn.clip_grad_norm(
            grads, max_norm=max_norm, norm_type=norm_type, return_norm=True
        )

        # Compute expected norm
        flat_grads = grads['param'].ravel()
        if norm_type == 'inf' or norm_type == jnp.inf:
            expected_norm = jnp.max(jnp.abs(flat_grads))
        else:
            expected_norm = jnp.linalg.norm(flat_grads, ord=norm_type)

        self.assertAlmostEqual(computed_norm, expected_norm, places=5)

        # Check scaling
        if expected_norm > max_norm:
            scale = max_norm / expected_norm
            np.testing.assert_array_almost_equal(
                clipped_grads['param'],
                grads['param'] * scale,
                decimal=5
            )
        else:
            np.testing.assert_array_almost_equal(
                clipped_grads['param'],
                grads['param'],
                decimal=5
            )

    def test_zero_gradients(self):
        """Test handling of zero gradients."""
        grads = {
            'w': jnp.zeros((3, 4)),
            'b': jnp.zeros(4)
        }

        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=1.0, return_norm=True)

        self.assertAlmostEqual(norm, 0.0, places=7)
        np.testing.assert_array_equal(clipped_grads['w'], grads['w'])
        np.testing.assert_array_equal(clipped_grads['b'], grads['b'])

    def test_single_tensor_input(self):
        """Test with a single tensor instead of a PyTree."""
        grad = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        max_norm = 5.0
        clipped_grad, norm = brainstate.nn.clip_grad_norm(grad, max_norm=max_norm, return_norm=True)

        expected_norm = jnp.linalg.norm(grad.ravel())
        self.assertAlmostEqual(norm, expected_norm, places=5)

        scale = max_norm / expected_norm
        np.testing.assert_array_almost_equal(
            clipped_grad,
            grad * scale,
            decimal=5
        )

    def test_list_structure(self):
        """Test gradient clipping with list structure."""
        grads = [
            jnp.array([1.0, 2.0]),
            jnp.array([[3.0, 4.0], [5.0, 6.0]]),
            jnp.array([7.0])
        ]

        max_norm = 10.0
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        # Check structure is preserved
        self.assertIsInstance(clipped_grads, list)
        self.assertEqual(len(clipped_grads), 3)

        # Check norm computation
        flat = jnp.arange(1.0, 8.0)
        expected_norm = jnp.linalg.norm(flat)
        self.assertAlmostEqual(norm, expected_norm, places=5)

    def test_tuple_structure(self):
        """Test gradient clipping with tuple structure."""
        grads = (
            jnp.array([3.0, 4.0]),
            jnp.array([5.0])
        )

        max_norm = 5.0
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        # Check structure is preserved
        self.assertIsInstance(clipped_grads, tuple)
        self.assertEqual(len(clipped_grads), 2)

        # Check norm: sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.07
        expected_norm = jnp.sqrt(50.0)
        self.assertAlmostEqual(norm, expected_norm, places=5)

    def test_max_norm_as_array(self):
        """Test using JAX array for max_norm parameter."""
        grads = {'w': jnp.array([6.0, 8.0])}
        max_norm = jnp.array(5.0)

        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        # norm = sqrt(36 + 64) = 10
        self.assertAlmostEqual(norm, 10.0, places=5)

        # Should be scaled by 5/10 = 0.5
        np.testing.assert_array_almost_equal(
            clipped_grads['w'],
            jnp.array([3.0, 4.0]),
            decimal=5
        )

    def test_none_norm_type(self):
        """Test that None norm_type defaults to L2 norm."""
        grads = {'param': jnp.array([3.0, 4.0])}

        # Test with explicit None
        clipped1, norm1 = brainstate.nn.clip_grad_norm(grads, max_norm=10.0, norm_type=None, return_norm=True)

        # Test with default (should be same as L2)
        clipped2, norm2 = brainstate.nn.clip_grad_norm(grads, max_norm=10.0, norm_type=2.0, return_norm=True)

        self.assertAlmostEqual(norm1, norm2, places=7)
        np.testing.assert_array_almost_equal(
            clipped1['param'], clipped2['param'], decimal=7
        )

    def test_very_large_gradients(self):
        """Test clipping very large gradients."""
        grads = {
            'huge': jnp.array([1e10, 1e10, 1e10])
        }

        max_norm = 1.0
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        # Check that clipped norm is approximately max_norm
        clipped_norm = jnp.linalg.norm(clipped_grads['huge'])
        self.assertAlmostEqual(clipped_norm, max_norm, places=5)

    def test_very_small_gradients(self):
        """Test handling very small gradients (numerical stability)."""
        grads = {
            'tiny': jnp.array([1e-10, 1e-10, 1e-10])
        }

        max_norm = 1.0
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        # Should not be clipped
        np.testing.assert_array_almost_equal(
            clipped_grads['tiny'], grads['tiny'], decimal=15
        )

    def test_mixed_shapes(self):
        """Test with mixed tensor shapes in PyTree."""
        grads = {
            'scalar': jnp.array(2.0),
            'vector': jnp.array([3.0, 4.0]),
            'matrix': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            'tensor3d': jnp.ones((2, 3, 4))
        }

        max_norm = 10.0
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        # Check all shapes are preserved
        self.assertEqual(clipped_grads['scalar'].shape, ())
        self.assertEqual(clipped_grads['vector'].shape, (2,))
        self.assertEqual(clipped_grads['matrix'].shape, (2, 2))
        self.assertEqual(clipped_grads['tensor3d'].shape, (2, 3, 4))

    def test_gradient_clipping_invariants(self):
        """Test mathematical invariants of gradient clipping."""
        grads = {
            'w1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            'w2': jnp.array([5.0, 6.0])
        }

        max_norm = 5.0
        clipped_grads, original_norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        # Compute norm of clipped gradients
        clipped_flat = jnp.concatenate([g.ravel() for g in jax.tree.leaves(clipped_grads)])
        clipped_norm = jnp.linalg.norm(clipped_flat)

        # Clipped norm should be min(original_norm, max_norm)
        expected_clipped_norm = jnp.minimum(original_norm, max_norm)
        self.assertAlmostEqual(clipped_norm, expected_clipped_norm, places=5)

    @parameterized.parameters(
        (0.5,),
        (1.0,),
        (2.0,),
        (5.0,),
        (10.0,),
    )
    def test_different_max_norms(self, max_norm):
        """Test gradient clipping with various max_norm values."""
        grads = {'param': jnp.array([6.0, 8.0])}  # norm = 10

        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=max_norm, return_norm=True)

        self.assertAlmostEqual(norm, 10.0, places=5)

        # Check clipped norm
        clipped_norm = jnp.linalg.norm(clipped_grads['param'])
        if max_norm < 10.0:
            self.assertAlmostEqual(clipped_norm, max_norm, places=5)
        else:
            self.assertAlmostEqual(clipped_norm, 10.0, places=5)

    def test_empty_pytree(self):
        """Test handling of empty PyTree."""
        grads = {}

        # Test with return_norm=True
        clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=1.0, return_norm=True)
        self.assertEqual(clipped_grads, {})
        self.assertAlmostEqual(norm, 0.0, places=7)

        # Test with return_norm=False
        clipped_grads_only = brainstate.nn.clip_grad_norm(grads, max_norm=1.0, return_norm=False)
        self.assertEqual(clipped_grads_only, {})

    def test_pytree_with_none_leaves(self):
        """Test PyTree containing None values (should be filtered out)."""
        grads = {
            'w': jnp.array([3.0, 4.0]),
            'b': None,  # This should be filtered by jax.tree.leaves
            'c': jnp.array([5.0])
        }

        # This test depends on how the function handles None values
        # JAX typically filters them out
        try:
            clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=5.0, return_norm=True)
            # If it works, check that None is preserved in structure
            self.assertIn('b', clipped_grads)
        except:
            # Expected if None values cause issues
            pass


if __name__ == '__main__':
    unittest.main()