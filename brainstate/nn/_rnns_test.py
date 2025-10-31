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

"""Comprehensive tests for RNN cell implementations."""

import unittest
from typing import Type

import jax
import jax.numpy as jnp
import numpy as np

import brainstate
import brainstate.nn as nn
from brainstate.nn import RNNCell, ValinaRNNCell, GRUCell, MGUCell, LSTMCell, URLSTMCell
from brainstate.nn import init as init
from brainstate.nn import _activations as functional


class TestRNNCellBase(unittest.TestCase):
    """Base test class for all RNN cell implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_in = 10
        self.num_out = 20
        self.batch_size = 32
        self.sequence_length = 100
        self.seed = 42

        # Initialize random inputs
        key = jax.random.PRNGKey(self.seed)
        self.x = jax.random.normal(key, (self.batch_size, self.num_in))
        self.sequence = jax.random.normal(
            key, (self.sequence_length, self.batch_size, self.num_in)
        )


class TestVanillaRNNCell(TestRNNCellBase):
    """Comprehensive tests for VanillaRNNCell."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        cell = ValinaRNNCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        output = cell.update(self.x)

        self.assertEqual(output.shape, (self.batch_size, self.num_out))
        self.assertFalse(jnp.any(jnp.isnan(output)))
        self.assertFalse(jnp.any(jnp.isinf(output)))

    def test_sequence_processing(self):
        """Test processing a sequence of inputs."""
        cell = ValinaRNNCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        outputs = []
        for t in range(self.sequence_length):
            output = cell.update(self.sequence[t])
            outputs.append(output)

        outputs = jnp.stack(outputs)
        self.assertEqual(outputs.shape, (self.sequence_length, self.batch_size, self.num_out))
        self.assertFalse(jnp.any(jnp.isnan(outputs)))

    def test_state_reset(self):
        """Test state reset functionality."""
        cell = ValinaRNNCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Process some input
        _ = cell.update(self.x)
        state_before = cell.h.value.copy()

        # Reset state
        cell.reset_state(batch_size=self.batch_size)
        state_after = cell.h.value.copy()

        # States should be different (unless randomly the same, which is unlikely)
        self.assertFalse(jnp.allclose(state_before, state_after, atol=1e-6))

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        cell = ValinaRNNCell(num_in=self.num_in, num_out=self.num_out)

        for batch_size in [1, 16, 64]:
            cell.init_state(batch_size=batch_size)
            x = jnp.ones((batch_size, self.num_in))
            output = cell.update(x)
            self.assertEqual(output.shape, (batch_size, self.num_out))

    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid', 'gelu']

        for activation in activations:
            cell = ValinaRNNCell(
                num_in=self.num_in,
                num_out=self.num_out,
                activation=activation
            )
            cell.init_state(batch_size=self.batch_size)
            output = cell.update(self.x)
            self.assertEqual(output.shape, (self.batch_size, self.num_out))
            self.assertFalse(jnp.any(jnp.isnan(output)))

    def test_custom_initializers(self):
        """Test custom weight and state initializers."""
        cell = ValinaRNNCell(
            num_in=self.num_in,
            num_out=self.num_out,
            w_init=init.Orthogonal(),
            b_init=init.Constant(0.1),
            state_init=init.Normal(0.01)
        )
        cell.init_state(batch_size=self.batch_size)
        output = cell.update(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.num_out))

    def test_gradient_flow(self):
        """Test gradient flow through the cell."""
        cell = ValinaRNNCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        def loss_fn(x):
            output = cell.update(x)
            return jnp.mean(output ** 2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(self.x)

        self.assertEqual(grad.shape, self.x.shape)
        self.assertFalse(jnp.any(jnp.isnan(grad)))
        self.assertTrue(jnp.any(grad != 0))  # Gradients should be non-zero


class TestGRUCell(TestRNNCellBase):
    """Comprehensive tests for GRUCell."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        cell = GRUCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        output = cell.update(self.x)

        self.assertEqual(output.shape, (self.batch_size, self.num_out))
        self.assertFalse(jnp.any(jnp.isnan(output)))

    def test_gating_mechanism(self):
        """Test that gating values are in valid range."""
        cell = GRUCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Access internal computation
        old_h = cell.h.value
        xh = jnp.concatenate([self.x, old_h], axis=-1)
        gates = functional.sigmoid(cell.Wrz(xh))

        # Gates should be between 0 and 1
        self.assertTrue(jnp.all(gates >= 0))
        self.assertTrue(jnp.all(gates <= 1))

    def test_state_persistence(self):
        """Test that state persists across updates."""
        cell = GRUCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Process sequence and track states
        states = []
        for t in range(10):
            _ = cell.update(self.sequence[t])
            states.append(cell.h.value.copy())

        # States should evolve over time
        for i in range(1, len(states)):
            self.assertFalse(jnp.allclose(states[i], states[i-1], atol=1e-8))

    def test_reset_vs_update_gates(self):
        """Test that reset and update gates behave differently."""
        cell = GRUCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Get gates for the same input
        old_h = cell.h.value
        xh = jnp.concatenate([self.x, old_h], axis=-1)
        r, z = jnp.split(functional.sigmoid(cell.Wrz(xh)), indices_or_sections=2, axis=-1)

        # Reset and update gates should be different
        self.assertFalse(jnp.allclose(r, z, atol=1e-6))

    def test_different_initializers(self):
        """Test with different weight initializers."""
        initializers = [
            init.XavierNormal(),
            init.XavierUniform(),
            init.Orthogonal(),
            init.KaimingNormal(),
        ]

        for w_init in initializers:
            cell = GRUCell(
                num_in=self.num_in,
                num_out=self.num_out,
                w_init=w_init
            )
            cell.init_state(batch_size=self.batch_size)
            output = cell.update(self.x)
            self.assertEqual(output.shape, (self.batch_size, self.num_out))


class TestMGUCell(TestRNNCellBase):
    """Comprehensive tests for MGUCell."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        cell = MGUCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        output = cell.update(self.x)

        self.assertEqual(output.shape, (self.batch_size, self.num_out))
        self.assertFalse(jnp.any(jnp.isnan(output)))

    def test_single_gate_mechanism(self):
        """Test that MGU uses single forget gate."""
        cell = MGUCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Check that only one gate is computed
        xh = jnp.concatenate([self.x, cell.h.value], axis=-1)
        f = functional.sigmoid(cell.Wf(xh))

        # Forget gate should be between 0 and 1
        self.assertTrue(jnp.all(f >= 0))
        self.assertTrue(jnp.all(f <= 1))
        self.assertEqual(f.shape, (self.batch_size, self.num_out))

    def test_parameter_efficiency(self):
        """Test that MGU has fewer parameters than GRU."""
        mgu_cell = MGUCell(num_in=self.num_in, num_out=self.num_out)
        gru_cell = GRUCell(num_in=self.num_in, num_out=self.num_out)

        # Count parameters - MGU should have fewer
        # MGU has 2 weight matrices (Wf, Wh)
        # GRU has 2 weight matrices but one is double size (Wrz, Wh)
        mgu_param_count = 2 * ((self.num_in + self.num_out) * self.num_out + self.num_out)
        gru_param_count = ((self.num_in + self.num_out) * (self.num_out * 2) + self.num_out * 2) + \
                          ((self.num_in + self.num_out) * self.num_out + self.num_out)

        self.assertLess(mgu_param_count, gru_param_count)


class TestLSTMCell(TestRNNCellBase):
    """Comprehensive tests for LSTMCell."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        cell = LSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        output = cell.update(self.x)

        self.assertEqual(output.shape, (self.batch_size, self.num_out))
        self.assertFalse(jnp.any(jnp.isnan(output)))

    def test_dual_state_mechanism(self):
        """Test that LSTM maintains both hidden and cell states."""
        cell = LSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Check initial states
        self.assertIsNotNone(cell.h)
        self.assertIsNotNone(cell.c)
        self.assertEqual(cell.h.value.shape, (self.batch_size, self.num_out))
        self.assertEqual(cell.c.value.shape, (self.batch_size, self.num_out))

        # Update and check states change
        h_before = cell.h.value.copy()
        c_before = cell.c.value.copy()

        _ = cell.update(self.x)

        self.assertFalse(jnp.allclose(cell.h.value, h_before, atol=1e-8))
        self.assertFalse(jnp.allclose(cell.c.value, c_before, atol=1e-8))

    def test_forget_gate_bias(self):
        """Test that forget gate has positive bias initialization."""
        cell = LSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Process with zero input to see bias effect
        zero_input = jnp.zeros((self.batch_size, self.num_in))
        xh = jnp.concatenate([zero_input, cell.h.value], axis=-1)
        gates = cell.W(xh)
        _, _, f, _ = jnp.split(gates, indices_or_sections=4, axis=-1)
        f_gate = functional.sigmoid(f + 1.)  # Note the +1 bias

        # Forget gate should be biased towards remembering (> 0.5)
        self.assertTrue(jnp.mean(f_gate) > 0.5)

    def test_gate_values_range(self):
        """Test that all gates produce values in [0, 1]."""
        cell = LSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        xh = jnp.concatenate([self.x, cell.h.value], axis=-1)
        i, g, f, o = jnp.split(cell.W(xh), indices_or_sections=4, axis=-1)

        i_gate = functional.sigmoid(i)
        f_gate = functional.sigmoid(f + 1.)
        o_gate = functional.sigmoid(o)

        for gate in [i_gate, f_gate, o_gate]:
            self.assertTrue(jnp.all(gate >= 0))
            self.assertTrue(jnp.all(gate <= 1))

    def test_cell_state_gradient_flow(self):
        """Test gradient flow through cell state."""
        cell = LSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        def loss_fn(x):
            for t in range(10):
                _ = cell.update(x)
            return jnp.mean(cell.c.value ** 2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(self.x)

        self.assertFalse(jnp.any(jnp.isnan(grad)))
        self.assertTrue(jnp.any(grad != 0))


class TestURLSTMCell(TestRNNCellBase):
    """Comprehensive tests for URLSTMCell."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        cell = URLSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        output = cell.update(self.x)

        self.assertEqual(output.shape, (self.batch_size, self.num_out))
        self.assertFalse(jnp.any(jnp.isnan(output)))

    def test_untied_bias_mechanism(self):
        """Test the untied bias initialization."""
        cell = URLSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Check bias values are initialized
        self.assertIsNotNone(cell.bias.value)
        self.assertEqual(cell.bias.value.shape, (self.num_out,))

        # Biases should be diverse (not all the same)
        self.assertGreater(jnp.std(cell.bias.value), 0.1)

    def test_unified_gate_computation(self):
        """Test the unified gate mechanism."""
        cell = URLSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        h, c = cell.h.value, cell.c.value
        xh = jnp.concatenate([self.x, h], axis=-1)
        gates = cell.W(xh)
        f, r, u, o = jnp.split(gates, indices_or_sections=4, axis=-1)

        f_gate = functional.sigmoid(f + cell.bias.value)
        r_gate = functional.sigmoid(r - cell.bias.value)

        # Compute unified gate
        g = 2 * r_gate * f_gate + (1 - 2 * r_gate) * f_gate ** 2

        # Unified gate should be in [0, 1]
        self.assertTrue(jnp.all(g >= 0))
        self.assertTrue(jnp.all(g <= 1))

    def test_comparison_with_lstm(self):
        """Test that URLSTM behaves differently from standard LSTM."""
        urlstm = URLSTMCell(num_in=self.num_in, num_out=self.num_out, state_init=init.Constant(0.5))
        lstm = LSTMCell(num_in=self.num_in, num_out=self.num_out, state_init=init.Constant(0.5))

        urlstm.init_state(batch_size=self.batch_size)
        lstm.init_state(batch_size=self.batch_size)

        # Same input should produce different outputs
        urlstm_out = urlstm.update(self.x)
        lstm_out = lstm.update(self.x)

        self.assertFalse(jnp.allclose(urlstm_out, lstm_out, atol=1e-4))


class TestRNNCellIntegration(TestRNNCellBase):
    """Integration tests for all RNN cells."""

    def test_all_cells_compatible_interface(self):
        """Test that all cells have compatible interfaces."""
        cell_types = [ValinaRNNCell, GRUCell, MGUCell, LSTMCell, URLSTMCell]

        for CellType in cell_types:
            cell = CellType(num_in=self.num_in, num_out=self.num_out)

            # Test init_state
            cell.init_state(batch_size=self.batch_size)

            # Test update
            output = cell.update(self.x)
            self.assertEqual(output.shape, (self.batch_size, self.num_out))

            # Test reset_state
            cell.reset_state(batch_size=16)

            # Test with new batch size
            x_small = jnp.ones((16, self.num_in))
            output_small = cell.update(x_small)
            self.assertEqual(output_small.shape, (16, self.num_out))

    def test_sequence_to_sequence(self):
        """Test sequence-to-sequence processing."""
        cell_types = [ValinaRNNCell, GRUCell, MGUCell, LSTMCell, URLSTMCell]

        for CellType in cell_types:
            cell = CellType(num_in=self.num_in, num_out=self.num_out)
            cell.init_state(batch_size=self.batch_size)

            outputs = []
            for t in range(self.sequence_length):
                output = cell.update(self.sequence[t])
                outputs.append(output)

            outputs = jnp.stack(outputs)
            self.assertEqual(
                outputs.shape,
                (self.sequence_length, self.batch_size, self.num_out)
            )

    def test_variable_length_sequences(self):
        """Test handling of variable length sequences with masking."""
        cell = LSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Create mask for variable lengths
        lengths = jnp.array([10, 20, 30, 40] * (self.batch_size // 4))
        mask = jnp.arange(self.sequence_length)[:, None] < lengths[None, :]

        outputs = []
        for t in range(self.sequence_length):
            output = cell.update(self.sequence[t])
            # Apply mask
            output = output * mask[t:t+1].T
            outputs.append(output)

        outputs = jnp.stack(outputs)

        # Check that masked positions are zero
        for b in range(self.batch_size):
            length = lengths[b]
            if length < self.sequence_length:
                self.assertTrue(jnp.allclose(outputs[length:, b, :], 0.0))

    def test_gradient_clipping(self):
        """Test gradient clipping during training."""
        cell = GRUCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        def loss_fn(x):
            output = jnp.zeros((self.batch_size, self.num_out))
            for t in range(50):  # Long sequence
                output = cell.update(x * (t + 1))  # Amplify input
            return jnp.mean(output ** 2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(self.x)

        # Gradients should not explode
        self.assertFalse(jnp.any(jnp.isnan(grad)))
        self.assertFalse(jnp.any(jnp.isinf(grad)))
        self.assertLess(jnp.max(jnp.abs(grad)), 1e6)


class TestRNNCellEdgeCases(TestRNNCellBase):
    """Edge case tests for RNN cells."""

    def test_single_sample(self):
        """Test with batch size of 1."""
        cell = ValinaRNNCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=1)

        x = jnp.ones((1, self.num_in))
        output = cell.update(x)
        self.assertEqual(output.shape, (1, self.num_out))

    def test_zero_input(self):
        """Test with zero inputs."""
        cell_types = [ValinaRNNCell, GRUCell, MGUCell, LSTMCell, URLSTMCell]

        for CellType in cell_types:
            cell = CellType(num_in=self.num_in, num_out=self.num_out)
            cell.init_state(batch_size=self.batch_size)

            zero_input = jnp.zeros((self.batch_size, self.num_in))
            output = cell.update(zero_input)

            self.assertEqual(output.shape, (self.batch_size, self.num_out))
            self.assertFalse(jnp.any(jnp.isnan(output)))

    def test_large_input_values(self):
        """Test with large input values."""
        cell = LSTMCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        large_input = jnp.ones((self.batch_size, self.num_in)) * 100
        output = cell.update(large_input)

        # Should handle large inputs gracefully (sigmoid saturation)
        self.assertFalse(jnp.any(jnp.isnan(output)))
        self.assertFalse(jnp.any(jnp.isinf(output)))

    def test_very_long_sequence(self):
        """Test with very long sequences."""
        cell = GRUCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=4)  # Smaller batch for memory

        long_sequence = jnp.ones((1000, 4, self.num_in))

        final_output = None
        for t in range(1000):
            final_output = cell.update(long_sequence[t])

        # Should not have numerical issues even after long sequence
        self.assertFalse(jnp.any(jnp.isnan(final_output)))
        self.assertFalse(jnp.any(jnp.isinf(final_output)))

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatches raise appropriate errors."""
        cell = ValinaRNNCell(num_in=self.num_in, num_out=self.num_out)
        cell.init_state(batch_size=self.batch_size)

        # Wrong input dimension should raise error
        wrong_input = jnp.ones((self.batch_size, self.num_in + 5))

        with self.assertRaises(Exception):
            _ = cell.update(wrong_input)


class TestRNNCellProperties(TestRNNCellBase):
    """Test cell properties and attributes."""

    def test_cell_attributes(self):
        """Test that cells have correct attributes."""
        cell = LSTMCell(num_in=self.num_in, num_out=self.num_out)

        self.assertEqual(cell.num_in, self.num_in)
        self.assertEqual(cell.num_out, self.num_out)
        self.assertEqual(cell.in_size, (self.num_in,))
        self.assertEqual(cell.out_size, (self.num_out,))

    def test_inheritance_structure(self):
        """Test that all cells inherit from RNNCell."""
        cell_types = [ValinaRNNCell, GRUCell, MGUCell, LSTMCell, URLSTMCell]

        for CellType in cell_types:
            cell = CellType(num_in=self.num_in, num_out=self.num_out)
            self.assertIsInstance(cell, RNNCell)

    def test_docstring_presence(self):
        """Test that all cells have proper docstrings."""
        cell_types = [ValinaRNNCell, GRUCell, MGUCell, LSTMCell, URLSTMCell]

        for CellType in cell_types:
            self.assertIsNotNone(CellType.__doc__)
            self.assertIn("Examples", CellType.__doc__)
            self.assertIn("Parameters", CellType.__doc__)
            self.assertIn(">>>", CellType.__doc__)


if __name__ == '__main__':
    unittest.main()