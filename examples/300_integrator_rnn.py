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

from typing import Callable

import braintools as bt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import brainstate as bst

dt = 0.04
num_step = int(1.0 / dt)
num_batch = 512


@bst.compile.jit(static_argnums=2)
def build_inputs_and_targets(mean=0.025, scale=0.01, batch_size=10):
  # Create the white noise input
  sample = bst.random.normal(size=(1, batch_size, 1))
  bias = mean * 2.0 * (sample - 0.5)
  samples = bst.random.normal(size=(num_step, batch_size, 1))
  noise_t = scale / dt ** 0.5 * samples
  inputs = bias + noise_t
  targets = jnp.cumsum(inputs, axis=0)
  return inputs, targets


def train_data():
  for _ in range(500):
    yield build_inputs_and_targets(0.025, 0.01, num_batch)


class RNNCell(bst.nn.Module):
  def __init__(
      self,
      num_in: int,
      num_out: int,
      state_initializer: Callable = bst.init.ZeroInit(),
      w_initializer: Callable = bst.init.XavierNormal(),
      b_initializer: Callable = bst.init.ZeroInit(),
      activation: Callable = bst.functional.relu,
      train_state: bool = False,
  ):
    super().__init__()

    # parameters
    self.num_out = num_out
    self.train_state = train_state

    # parameters
    self.num_in = num_in

    # initializers
    self._state_initializer = state_initializer
    self._w_initializer = w_initializer
    self._b_initializer = b_initializer

    # activation function
    self.activation = activation

    # weights
    W = bst.init.param(self._w_initializer, (num_in + num_out, self.num_out))
    b = bst.init.param(self._b_initializer, (self.num_out,))
    self.W = bst.ParamState(W)
    self.b = None if (b is None) else bst.ParamState(b)

    # state
    if train_state:
      self.state2train = bst.ParamState(bst.init.param(bst.init.ZeroInit(), (self.num_out,), allow_none=False))

  def init_state(self, batch_size=None, **kwargs):
    self.state = bst.HiddenState(bst.init.param(self._state_initializer, (self.num_out,), batch_size))
    if self.train_state:
      self.state.value = jnp.repeat(jnp.expand_dims(self.state2train.value, axis=0), batch_size, axis=0)

  def update(self, x):
    x = jnp.concat([x, self.state.value], axis=-1)
    h = x @ self.W.value
    if self.b is not None:
      h += self.b.value
    h = self.activation(h)
    self.state.value = h
    return h


class RNN(bst.nn.Module):
  def __init__(self, num_in, num_hidden):
    super().__init__()
    self.rnn = RNNCell(num_in, num_hidden, train_state=True)
    self.out = bst.nn.Linear(num_hidden, 1)

  def update(self, x):
    return x >> self.rnn >> self.out


model = RNN(1, 100)
weights = model.states(bst.ParamState)


@bst.compile.jit
def f_predict(inputs):
  bst.nn.init_all_states(model, batch_size=inputs.shape[1])
  return bst.compile.for_loop(model.update, inputs)


def f_loss(inputs, targets, l2_reg=2e-4):
  predictions = f_predict(inputs)
  mse = bt.metric.squared_error(predictions, targets).mean()
  l2 = 0.0
  for weight in weights.values():
    for leaf in jax.tree.leaves(weight.value):
      l2 += jnp.sum(leaf ** 2)
  return mse + l2_reg * l2


# define optimizer
lr = bst.optim.ExponentialDecayLR(lr=0.025, decay_steps=1, decay_rate=0.99975)
opt = bst.optim.Adam(lr=lr, eps=1e-1)
opt.register_trainable_weights(weights)


@bst.compile.jit
def f_train(inputs, targets):
  grads, l = bst.augment.grad(f_loss, weights, return_value=True)(inputs, targets)
  opt.update(grads)
  return l


for i_epoch in range(5):
  for i_batch, (inps, tars) in enumerate(train_data()):
    loss = f_train(inps, tars)
    if (i_batch + 1) % 100 == 0:
      print(f'Epoch {i_epoch}, Batch {i_batch + 1:3d}, Loss {loss:.5f}')

bst.nn.init_all_states(model, 1)
x, y = build_inputs_and_targets(0.025, 0.01, 1)
predicts = f_predict(x)

plt.figure(figsize=(8, 2))
plt.plot(np.asarray(y[:, 0]).flatten(), label='Ground Truth')
plt.plot(np.asarray(predicts[:, 0]).flatten(), label='Prediction')
plt.legend()
plt.show()
