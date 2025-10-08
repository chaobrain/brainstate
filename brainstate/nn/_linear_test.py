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

import brainunit as u
import jax.numpy as jnp
from absl.testing import parameterized

import brainstate
import braintools


class TestLinear(parameterized.TestCase):
    """Test suite for Linear layer."""

    @parameterized.product(
        size=[(10,), (20, 10), (5, 8, 10)],
        num_out=[20, 5]
    )
    def test_linear_shapes(self, size, num_out):
        """Test output shapes with various input dimensions."""
        layer = brainstate.nn.Linear(10, num_out)
        x = brainstate.random.random(size)
        y = layer(x)
        self.assertEqual(y.shape, size[:-1] + (num_out,))

    def test_linear_with_bias(self):
        """Test linear layer with bias."""
        layer = brainstate.nn.Linear(10, 5)
        self.assertIn('bias', layer.weight.value)
        x = brainstate.random.random((3, 10))
        y = layer(x)
        self.assertEqual(y.shape, (3, 5))

    def test_linear_without_bias(self):
        """Test linear layer without bias."""
        layer = brainstate.nn.Linear(10, 5, b_init=None)
        self.assertNotIn('bias', layer.weight.value)
        x = brainstate.random.random((3, 10))
        y = layer(x)
        self.assertEqual(y.shape, (3, 5))

    def test_linear_with_mask(self):
        """Test linear layer with weight mask."""
        w_mask = jnp.ones((10, 5))
        w_mask = w_mask.at[:, 0].set(0)  # mask out first output column
        layer = brainstate.nn.Linear(10, 5, w_mask=w_mask)
        x = jnp.ones((3, 10))
        y = layer(x)
        self.assertEqual(y.shape, (3, 5))

    def test_linear_weight_initialization(self):
        """Test custom weight initialization."""
        layer = brainstate.nn.Linear(
            10, 5,
            w_init=braintools.init.ZeroInit(),
            b_init=braintools.init.Constant(1.0)
        )
        self.assertTrue(jnp.allclose(layer.weight.value['weight'], 0.0))
        self.assertTrue(jnp.allclose(layer.weight.value['bias'], 1.0))

    def test_linear_computation(self):
        """Test that computation is correct."""
        layer = brainstate.nn.Linear(3, 2, b_init=None)
        # Set known weights
        layer.weight.value = {'weight': jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}
        x = jnp.array([[1.0, 2.0, 3.0]])
        y = layer(x)
        expected = jnp.array([[4.0, 5.0]])  # [1*1+2*0+3*1, 1*0+2*1+3*1]
        self.assertTrue(jnp.allclose(y, expected))


class TestSignedWLinear(parameterized.TestCase):
    """Test suite for SignedWLinear layer."""

    @parameterized.product(
        in_size=[10, 20],
        out_size=[5, 10]
    )
    def test_signed_linear_shapes(self, in_size, out_size):
        """Test output shapes."""
        layer = brainstate.nn.SignedWLinear((in_size,), (out_size,))
        x = brainstate.random.random((3, in_size))
        y = layer(x)
        self.assertEqual(y.shape, (3, out_size))

    def test_signed_linear_positive_weights(self):
        """Test that weights are positive when w_sign is None."""
        layer = brainstate.nn.SignedWLinear((5,), (3,))
        # Set weights to negative values
        layer.weight.value = jnp.array([[-1.0, -2.0, -3.0]] * 5)
        x = jnp.ones((1, 5))
        y = layer(x)
        # Output should be positive since abs is applied
        self.assertTrue(jnp.all(y > 0))

    def test_signed_linear_with_sign_matrix(self):
        """Test signed linear with custom sign matrix."""
        w_sign = jnp.ones((5, 3)) * -1.0  # all negative
        layer = brainstate.nn.SignedWLinear((5,), (3,), w_sign=w_sign)
        layer.weight.value = jnp.ones((5, 3))
        x = jnp.ones((1, 5))
        y = layer(x)
        # All outputs should be negative
        self.assertTrue(jnp.all(y < 0))

    def test_signed_linear_mixed_signs(self):
        """Test with mixed positive and negative signs."""
        w_sign = jnp.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
        layer = brainstate.nn.SignedWLinear((3,), (2,), w_sign=w_sign)
        layer.weight.value = jnp.ones((3, 2))
        x = jnp.array([[1.0, 1.0, 1.0]])
        y = layer(x)
        expected = jnp.array([[1.0, -1.0]])  # [1-1, -1+1]
        self.assertTrue(jnp.allclose(y, expected))


class TestScaledWSLinear(parameterized.TestCase):
    """Test suite for ScaledWSLinear layer."""

    @parameterized.product(
        in_size=[10, 20],
        out_size=[5, 10],
        ws_gain=[True, False]
    )
    def test_scaled_ws_shapes(self, in_size, out_size, ws_gain):
        """Test output shapes with and without gain."""
        layer = brainstate.nn.ScaledWSLinear((in_size,), (out_size,), ws_gain=ws_gain)
        x = brainstate.random.random((3, in_size))
        y = layer(x)
        self.assertEqual(y.shape, (3, out_size))

    def test_scaled_ws_with_gain(self):
        """Test that gain parameter exists when ws_gain=True."""
        layer = brainstate.nn.ScaledWSLinear((10,), (5,), ws_gain=True)
        self.assertIn('gain', layer.weight.value)

    def test_scaled_ws_without_gain(self):
        """Test that gain parameter is absent when ws_gain=False."""
        layer = brainstate.nn.ScaledWSLinear((10,), (5,), ws_gain=False)
        self.assertNotIn('gain', layer.weight.value)

    def test_scaled_ws_with_mask(self):
        """Test scaled WS linear with weight mask."""
        w_mask = jnp.ones((10, 1))
        layer = brainstate.nn.ScaledWSLinear((10,), (5,), w_mask=w_mask)
        x = brainstate.random.random((3, 10))
        y = layer(x)
        self.assertEqual(y.shape, (3, 5))

    def test_scaled_ws_without_bias(self):
        """Test scaled WS linear without bias."""
        layer = brainstate.nn.ScaledWSLinear((10,), (5,), b_init=None)
        self.assertNotIn('bias', layer.weight.value)
        x = brainstate.random.random((3, 10))
        y = layer(x)
        self.assertEqual(y.shape, (3, 5))

    def test_scaled_ws_eps_parameter(self):
        """Test that eps parameter is stored correctly."""
        eps_value = 1e-5
        layer = brainstate.nn.ScaledWSLinear((10,), (5,), eps=eps_value)
        self.assertEqual(layer.eps, eps_value)


class TestSparseLinear(unittest.TestCase):
    """Test suite for SparseLinear layer."""

    def test_sparse_csr(self):
        """Test SparseLinear with CSR format."""
        data = brainstate.random.rand(10, 20)
        data = data * (data > 0.9)
        layer = brainstate.nn.SparseLinear(u.sparse.CSR.fromdense(data))

        x = brainstate.random.rand(10)
        y = layer(x)
        self.assertTrue(u.math.allclose(y, x @ data))

        x = brainstate.random.rand(5, 10)
        y = layer(x)
        self.assertTrue(u.math.allclose(y, x @ data))

    def test_sparse_csc(self):
        """Test SparseLinear with CSC format."""
        data = brainstate.random.rand(10, 20)
        data = data * (data > 0.9)
        layer = brainstate.nn.SparseLinear(u.sparse.CSC.fromdense(data))

        x = brainstate.random.rand(10)
        y = layer(x)
        self.assertTrue(u.math.allclose(y, x @ data))

        x = brainstate.random.rand(5, 10)
        y = layer(x)
        self.assertTrue(u.math.allclose(y, x @ data))

    def test_sparse_coo(self):
        """Test SparseLinear with COO format."""
        data = brainstate.random.rand(10, 20)
        data = data * (data > 0.9)
        layer = brainstate.nn.SparseLinear(u.sparse.COO.fromdense(data))

        x = brainstate.random.rand(10)
        y = layer(x)
        self.assertTrue(u.math.allclose(y, x @ data))

        x = brainstate.random.rand(5, 10)
        y = layer(x)
        self.assertTrue(u.math.allclose(y, x @ data))

    def test_sparse_with_bias(self):
        """Test SparseLinear with bias."""
        data = brainstate.random.rand(10, 20)
        data = data * (data > 0.9)
        spar_mat = u.sparse.CSR.fromdense(data)
        layer = brainstate.nn.SparseLinear(
            spar_mat,
            b_init=braintools.init.Constant(0.5),
            in_size=(10,)
        )
        self.assertIn('bias', layer.weight.value)
        x = brainstate.random.rand(5, 10)
        y = layer(x)
        expected = x @ data + 0.5
        self.assertTrue(u.math.allclose(y, expected))

    def test_sparse_without_bias(self):
        """Test SparseLinear without bias."""
        data = brainstate.random.rand(10, 20)
        data = data * (data > 0.9)
        spar_mat = u.sparse.CSR.fromdense(data)
        layer = brainstate.nn.SparseLinear(spar_mat, b_init=None)
        self.assertNotIn('bias', layer.weight.value)


class TestAllToAll(parameterized.TestCase):
    """Test suite for AllToAll connection layer."""

    @parameterized.product(
        in_size=[10, 20],
        out_size=[10, 15],
        include_self=[True, False]
    )
    def test_all_to_all_shapes(self, in_size, out_size, include_self):
        """Test output shapes with various configurations."""
        layer = brainstate.nn.AllToAll((in_size,), (out_size,), include_self=include_self)
        x = brainstate.random.random((3, in_size))
        y = layer(x)
        self.assertEqual(y.shape, (3, out_size))

    def test_all_to_all_with_self(self):
        """Test all-to-all with self-connections."""
        layer = brainstate.nn.AllToAll((5,), (5,), include_self=True)
        layer.weight.value = {'weight': jnp.eye(5)}
        x = jnp.ones((1, 5))
        y = layer(x)
        expected = jnp.ones((1, 5))
        self.assertTrue(jnp.allclose(y, expected))

    def test_all_to_all_without_self(self):
        """Test all-to-all without self-connections."""
        layer = brainstate.nn.AllToAll((5,), (5,), include_self=False)
        layer.weight.value = {'weight': jnp.eye(5)}
        x = jnp.ones((1, 5))
        y = layer(x)
        # Diagonal should be zeroed out
        expected = jnp.zeros((1, 5))
        self.assertTrue(jnp.allclose(y, expected))

    def test_all_to_all_scalar_weight(self):
        """Test all-to-all with scalar weight."""
        layer = brainstate.nn.AllToAll((5,), (5,), w_init=braintools.init.Constant(2.0))
        # Override with scalar
        layer.weight.value = {'weight': 2.0}
        x = jnp.ones((1, 5))
        y = layer(x)
        expected = jnp.ones((1, 5)) * 10.0  # sum of 5 ones * 2
        self.assertTrue(jnp.allclose(y, expected))

    def test_all_to_all_with_bias(self):
        """Test all-to-all with bias."""
        layer = brainstate.nn.AllToAll(
            (5,), (5,),
            b_init=braintools.init.Constant(1.0)
        )
        self.assertIn('bias', layer.weight.value)
        x = brainstate.random.random((3, 5))
        y = layer(x)
        self.assertEqual(y.shape, (3, 5))

    def test_all_to_all_with_units(self):
        """Test all-to-all with brainunit quantities."""
        layer = brainstate.nn.AllToAll((5,), (5,))
        layer.weight.value = {'weight': jnp.ones((5, 5)) * u.siemens}
        x = jnp.ones((1, 5)) * u.volt
        y = layer(x)
        # Should have units of siemens * volt
        self.assertTrue(hasattr(y, 'unit') or isinstance(y, u.Quantity))


class TestOneToOne(parameterized.TestCase):
    """Test suite for OneToOne connection layer."""

    @parameterized.parameters(5, 10, 20)
    def test_one_to_one_shapes(self, size):
        """Test output shapes."""
        layer = brainstate.nn.OneToOne((size,))
        x = brainstate.random.random((3, size))
        y = layer(x)
        self.assertEqual(y.shape, (3, size))

    def test_one_to_one_computation(self):
        """Test element-wise multiplication."""
        layer = brainstate.nn.OneToOne((5,), b_init=None)
        layer.weight.value = {'weight': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        x = jnp.ones((1, 5))
        y = layer(x)
        expected = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertTrue(jnp.allclose(y, expected))

    def test_one_to_one_with_bias(self):
        """Test one-to-one with bias."""
        layer = brainstate.nn.OneToOne((5,), b_init=braintools.init.Constant(1.0))
        self.assertIn('bias', layer.weight.value)
        layer.weight.value = {
            'weight': jnp.ones(5),
            'bias': jnp.ones(5)
        }
        x = jnp.ones((1, 5))
        y = layer(x)
        expected = jnp.ones((1, 5)) * 2.0  # 1*1 + 1
        self.assertTrue(jnp.allclose(y, expected))

    def test_one_to_one_without_bias(self):
        """Test one-to-one without bias."""
        layer = brainstate.nn.OneToOne((5,), b_init=None)
        self.assertNotIn('bias', layer.weight.value)

    def test_one_to_one_zero_weights(self):
        """Test one-to-one with zero weights."""
        layer = brainstate.nn.OneToOne((5,), w_init=braintools.init.ZeroInit(), b_init=None)
        x = jnp.ones((1, 5))
        y = layer(x)
        expected = jnp.zeros((1, 5))
        self.assertTrue(jnp.allclose(y, expected))


class TestLoRA(parameterized.TestCase):
    """Test suite for LoRA layer."""

    @parameterized.product(
        in_features=[10, 20],
        lora_rank=[2, 4],
        out_features=[5, 10]
    )
    def test_lora_shapes(self, in_features, lora_rank, out_features):
        """Test output shapes with various configurations."""
        layer = brainstate.nn.LoRA(in_features, lora_rank, out_features)
        x = brainstate.random.random((3, in_features))
        y = layer(x)
        self.assertEqual(y.shape, (3, out_features))

    def test_lora_parameter_count(self):
        """Test that LoRA has correct number of parameters."""
        in_features, lora_rank, out_features = 10, 2, 5
        layer = brainstate.nn.LoRA(in_features, lora_rank, out_features)
        # lora_a: 10 x 2, lora_b: 2 x 5
        self.assertEqual(layer.weight.value['lora_a'].shape, (10, 2))
        self.assertEqual(layer.weight.value['lora_b'].shape, (2, 5))

    def test_lora_standalone(self):
        """Test standalone LoRA without base module."""
        layer = brainstate.nn.LoRA(5, 2, 3)
        layer.weight.value = {
            'lora_a': jnp.ones((5, 2)),
            'lora_b': jnp.ones((2, 3))
        }
        x = jnp.ones((1, 5))
        y = layer(x)
        # Each output: sum(5 ones) * 2 = 10
        expected = jnp.ones((1, 3)) * 10.0
        self.assertTrue(jnp.allclose(y, expected))

    def test_lora_with_base_module(self):
        """Test LoRA wrapped around base module."""
        base = brainstate.nn.Linear(5, 3, b_init=None)
        base.weight.value = {'weight': jnp.ones((5, 3))}
        layer = brainstate.nn.LoRA(5, 2, 3, base_module=base)
        layer.weight.value = {
            'lora_a': jnp.ones((5, 2)),
            'lora_b': jnp.ones((2, 3))
        }
        x = jnp.ones((1, 5))
        y = layer(x)
        # LoRA output: 10, Base output: 5, Total: 15
        expected = jnp.ones((1, 3)) * 15.0
        self.assertTrue(jnp.allclose(y, expected))

    def test_lora_base_module_attribute(self):
        """Test that base_module attribute is set correctly."""
        base = brainstate.nn.Linear(5, 3)
        layer = brainstate.nn.LoRA(5, 2, 3, base_module=base)
        self.assertEqual(layer.base_module, base)

    def test_lora_without_base_module(self):
        """Test that base_module is None when not provided."""
        layer = brainstate.nn.LoRA(5, 2, 3)
        self.assertIsNone(layer.base_module)

    def test_lora_size_attributes(self):
        """Test that size attributes are set correctly."""
        layer = brainstate.nn.LoRA(10, 3, 5, in_size=(10,))
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.in_size[0], 10)
        self.assertEqual(layer.out_size[0], 5)

    def test_lora_custom_initialization(self):
        """Test LoRA with custom initialization."""
        layer = brainstate.nn.LoRA(
            5, 2, 3,
            kernel_init=braintools.init.ZeroInit()
        )
        self.assertTrue(jnp.allclose(layer.weight.value['lora_a'], 0.0))
        self.assertTrue(jnp.allclose(layer.weight.value['lora_b'], 0.0))


class TestLinearEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for linear layers."""

    def test_linear_size_mismatch(self):
        """Test that size mismatch raises error."""
        with self.assertRaises(AssertionError):
            # Mismatched first dimensions
            brainstate.nn.Linear((5, 10), (3, 5))

    def test_linear_1d_sizes(self):
        """Test with 1D size specifications."""
        layer = brainstate.nn.Linear(10, 5)
        x = brainstate.random.random((3, 10))
        y = layer(x)
        self.assertEqual(y.shape, (3, 5))

    def test_signed_linear_size_mismatch(self):
        """Test SignedWLinear with size mismatch."""
        with self.assertRaises(AssertionError):
            brainstate.nn.SignedWLinear((5, 10), (3, 5))

    def test_all_to_all_size_mismatch(self):
        """Test AllToAll with size mismatch."""
        with self.assertRaises(AssertionError):
            brainstate.nn.AllToAll((5, 10), (3, 5))

    def test_sparse_linear_invalid_input(self):
        """Test SparseLinear with invalid sparse matrix."""
        with self.assertRaises(AssertionError):
            # Not a SparseMatrix
            brainstate.nn.SparseLinear(jnp.ones((5, 5)))


if __name__ == '__main__':
    unittest.main()
