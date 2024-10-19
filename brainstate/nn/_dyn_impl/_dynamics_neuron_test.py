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

# -*- coding: utf-8 -*-


import unittest

import jax
import jax.numpy as jnp

import brainstate as bst
from brainstate.nn import IF, LIF, ALIF


class TestNeuron(unittest.TestCase):
  def setUp(self):
    self.in_size = 10
    self.batch_size = 5
    self.time_steps = 100

  def test_neuron_base_class(self):
    with self.assertRaises(NotImplementedError):
      bst.nn.Neuron(self.in_size).get_spike()  # Neuron is an abstract base class

  def generate_input(self):
    return bst.random.randn(self.time_steps, self.batch_size, self.in_size)

  def test_if_neuron(self):
    neuron = IF(self.in_size)
    inputs = self.generate_input()

    # Test initialization
    self.assertEqual(neuron.in_size, (self.in_size,))
    self.assertEqual(neuron.out_size, (self.in_size,))

    # Test forward pass
    state = neuron.init_state(self.batch_size)
    for t in range(self.time_steps):
      out = neuron(inputs[t])
      self.assertEqual(out.shape, (self.batch_size, self.in_size))

    # Test spike generation
    v = jnp.linspace(-1, 1, 100)
    spikes = neuron.get_spike(v)
    self.assertTrue(jnp.all((spikes >= 0) & (spikes <= 1)))

  def test_lif_neuron(self):
    tau = 20.0
    neuron = LIF(self.in_size, tau=tau)
    inputs = self.generate_input()

    # Test initialization
    self.assertEqual(neuron.in_size, (self.in_size,))
    self.assertEqual(neuron.out_size, (self.in_size,))
    self.assertEqual(neuron.tau, tau)

    # Test forward pass
    state = neuron.init_state(self.batch_size)
    for t in range(self.time_steps):
      out = neuron(inputs[t])
      self.assertEqual(out.shape, (self.batch_size, self.in_size))

  def test_alif_neuron(self):
    tau = 20.0
    tau_ada = 100.0
    neuron = ALIF(self.in_size, tau=tau, tau_a=tau_ada)
    inputs = self.generate_input()

    # Test initialization
    self.assertEqual(neuron.in_size, (self.in_size,))
    self.assertEqual(neuron.out_size, (self.in_size,))
    self.assertEqual(neuron.tau, tau)
    self.assertEqual(neuron.tau_a, tau_ada)

    # Test forward pass
    neuron.init_state(self.batch_size)
    for t in range(self.time_steps):
      out = neuron(inputs[t])
      self.assertEqual(out.shape, (self.batch_size, self.in_size))

  def test_spike_function(self):
    for NeuronClass in [IF, LIF, ALIF]:
      neuron = NeuronClass(self.in_size)
      neuron.init_state()
      v = jnp.linspace(-1, 1, self.in_size)
      spikes = neuron.get_spike(v)
      self.assertTrue(jnp.all((spikes >= 0) & (spikes <= 1)))

  def test_soft_reset(self):
    for NeuronClass in [IF, LIF, ALIF]:
      neuron = NeuronClass(self.in_size, spk_reset='soft')
      inputs = self.generate_input()
      state = neuron.init_state(self.batch_size)
      for t in range(self.time_steps):
        out = neuron(inputs[t])
        self.assertTrue(jnp.all(neuron.V.value <= neuron.V_th))

  def test_hard_reset(self):
    for NeuronClass in [IF, LIF, ALIF]:
      neuron = NeuronClass(self.in_size, spk_reset='hard')
      inputs = self.generate_input()
      state = neuron.init_state(self.batch_size)
      for t in range(self.time_steps):
        out = neuron(inputs[t])
        self.assertTrue(jnp.all((neuron.V.value < neuron.V_th) | (neuron.V.value == 0)))

  def test_detach_spike(self):
    for NeuronClass in [IF, LIF, ALIF]:
      neuron = NeuronClass(self.in_size)
      inputs = self.generate_input()
      state = neuron.init_state(self.batch_size)
      for t in range(self.time_steps):
        out = neuron(inputs[t])
        self.assertFalse(jax.tree_util.tree_leaves(out)[0].aval.weak_type)

  def test_keep_size(self):
    in_size = (2, 3)
    for NeuronClass in [IF, LIF, ALIF]:
      neuron = NeuronClass(in_size, keep_size=True)
      self.assertEqual(neuron.in_size, in_size)
      self.assertEqual(neuron.out_size, in_size)

      inputs = bst.random.randn(self.time_steps, self.batch_size, *in_size)
      state = neuron.init_state(self.batch_size)
      for t in range(self.time_steps):
        out = neuron(inputs[t])
        self.assertEqual(out.shape, (self.batch_size, *in_size))


if __name__ == '__main__':
  unittest.main()
