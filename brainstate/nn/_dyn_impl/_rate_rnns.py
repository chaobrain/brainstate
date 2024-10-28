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

from __future__ import annotations

from typing import Callable, Union

import jax.numpy as jnp

from brainstate import random, init, functional
from brainstate._state import ShortTermState, ParamState
from brainstate.mixin import ParamDesc
from brainstate.nn._interaction._connections import Linear
from brainstate.nn._module import Module
from brainstate.typing import ArrayLike

__all__ = [
  'RNNCell', 'ValinaRNNCell', 'GRUCell', 'MGUCell', 'LSTMCell', 'URLSTMCell',
]


class RNNCell(Module, ParamDesc):
  """
  Base class for RNN cells.
  """
  pass


class ValinaRNNCell(RNNCell):
  """
  Vanilla RNN cell.

  Args:
    num_in: int. The number of input units.
    num_out: int. The number of hidden units.
    state_init: callable, ArrayLike. The state initializer.
    w_init: callable, ArrayLike. The input weight initializer.
    b_init: optional, callable, ArrayLike. The bias weight initializer.
    activation: str, callable. The activation function. It can be a string or a callable function.
    name: optional, str. The name of the module.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      num_in: int,
      num_out: int,
      state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
      b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      activation: str | Callable = 'relu',
      name: str = None,
  ):
    super().__init__(name=name)

    # parameters
    self.num_out = num_out
    self.num_in = num_in
    self.in_size = (num_in,)
    self.out_size = (num_out,)
    self._state_initializer = state_init

    # activation function
    if isinstance(activation, str):
      self.activation = getattr(functional, activation)
    else:
      assert callable(activation), "The activation function should be a string or a callable function. "
      self.activation = activation

    # weights
    self.W = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)

  def init_state(self, batch_size: int = None, **kwargs):
    self.h = ShortTermState(init.param(self._state_initializer, self.num_out, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.h.value = init.param(self._state_initializer, self.num_out, batch_size)

  def update(self, x):
    xh = jnp.concatenate([x, self.h.value], axis=-1)
    h = self.W(xh)
    self.h.value = self.activation(h)
    return self.h.value


class GRUCell(RNNCell):
  """
  Gated Recurrent Unit (GRU) cell.

  Args:
    num_in: int. The number of input units.
    num_out: int. The number of hidden units.
    state_init: callable, ArrayLike. The state initializer.
    w_init: callable, ArrayLike. The input weight initializer.
    b_init: optional, callable, ArrayLike. The bias weight initializer.
    activation: str, callable. The activation function. It can be a string or a callable function.
    name: optional, str. The name of the module.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      num_in: int,
      num_out: int,
      w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
      b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      activation: str | Callable = 'tanh',
      name: str = None,
  ):
    super().__init__(name=name)

    # parameters
    self._state_initializer = state_init
    self.num_out = num_out
    self.num_in = num_in
    self.in_size = (num_in,)
    self.out_size = (num_out,)

    # activation function
    if isinstance(activation, str):
      self.activation = getattr(functional, activation)
    else:
      assert callable(activation), "The activation function should be a string or a callable function. "
      self.activation = activation

    # weights
    self.Wrz = Linear(num_in + num_out, num_out * 2, w_init=w_init, b_init=b_init)
    self.Wh = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)

  def init_state(self, batch_size: int = None, **kwargs):
    self.h = ShortTermState(init.param(self._state_initializer, [self.num_out], batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

  def update(self, x):
    old_h = self.h.value
    xh = jnp.concatenate([x, old_h], axis=-1)
    r, z = jnp.split(functional.sigmoid(self.Wrz(xh)), indices_or_sections=2, axis=-1)
    rh = r * old_h
    h = self.activation(self.Wh(jnp.concatenate([x, rh], axis=-1)))
    h = (1 - z) * old_h + z * h
    self.h.value = h
    return h


class MGUCell(RNNCell):
  r"""
  Minimal Gated Recurrent Unit (MGU) cell.

  .. math::

     \begin{aligned}
     f_{t}&=\sigma (W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\
     {\hat {h}}_{t}&=\phi (W_{h}x_{t}+U_{h}(f_{t}\odot h_{t-1})+b_{h})\\
     h_{t}&=(1-f_{t})\odot h_{t-1}+f_{t}\odot {\hat {h}}_{t}
     \end{aligned}

  where:

  - :math:`x_{t}`: input vector
  - :math:`h_{t}`: output vector
  - :math:`{\hat {h}}_{t}`: candidate activation vector
  - :math:`f_{t}`: forget vector
  - :math:`W, U, b`: parameter matrices and vector

  Args:
    num_in: int. The number of input units.
    num_out: int. The number of hidden units.
    state_init: callable, ArrayLike. The state initializer.
    w_init: callable, ArrayLike. The input weight initializer.
    b_init: optional, callable, ArrayLike. The bias weight initializer.
    activation: str, callable. The activation function. It can be a string or a callable function.
    name: optional, str. The name of the module.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      num_in: int,
      num_out: int,
      w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
      b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      activation: str | Callable = 'tanh',
      name: str = None,
  ):
    super().__init__(name=name)

    # parameters
    self._state_initializer = state_init
    self.num_out = num_out
    self.num_in = num_in
    self.in_size = (num_in,)
    self.out_size = (num_out,)

    # activation function
    if isinstance(activation, str):
      self.activation = getattr(functional, activation)
    else:
      assert callable(activation), "The activation function should be a string or a callable function. "
      self.activation = activation

    # weights
    self.Wf = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
    self.Wh = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)

  def init_state(self, batch_size: int = None, **kwargs):
    self.h = ShortTermState(init.param(self._state_initializer, [self.num_out], batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

  def update(self, x):
    old_h = self.h.value
    xh = jnp.concatenate([x, old_h], axis=-1)
    f = functional.sigmoid(self.Wf(xh))
    fh = f * old_h
    h = self.activation(self.Wh(jnp.concatenate([x, fh], axis=-1)))
    self.h.value = (1 - f) * self.h.value + f * h
    return self.h.value


class LSTMCell(RNNCell):
  r"""Long short-term memory (LSTM) RNN core.

  The implementation is based on (zaremba, et al., 2014) [1]_. Given
  :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})` the core
  computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  where :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and
  output gate activations, and :math:`g_t` is a vector of cell updates.

  The output is equal to the new hidden, :math:`h_t`.

  Notes
  -----

  Forget gate initialization: Following (Jozefowicz, et al., 2015) [2]_ we add 1.0
  to :math:`b_f` after initialization in order to reduce the scale of forgetting in
  the beginning of the training.


  Parameters
  ----------
  num_in: int
    The dimension of the input vector
  num_out: int
    The number of hidden unit in the node.
  state_init: callable, ArrayLike
    The state initializer.
  w_init: callable, ArrayLike
    The input weight initializer.
  b_init: optional, callable, ArrayLike
    The bias weight initializer.
  activation: str, callable
    The activation function. It can be a string or a callable function.

  References
  ----------

  .. [1] Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural
         network regularization." arXiv preprint arXiv:1409.2329 (2014).
  .. [2] Jozefowicz, Rafal, Wojciech Zaremba, and Ilya Sutskever. "An empirical
         exploration of recurrent network architectures." In International conference
         on machine learning, pp. 2342-2350. PMLR, 2015.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      num_in: int,
      num_out: int,
      w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
      b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      activation: str | Callable = 'tanh',
      name: str = None,
  ):
    super().__init__(name=name)

    # parameters
    self.num_out = num_out
    self.num_in = num_in
    self.in_size = (num_in,)
    self.out_size = (num_out,)

    # initializers
    self._state_initializer = state_init

    # activation function
    if isinstance(activation, str):
      self.activation = getattr(functional, activation)
    else:
      assert callable(activation), "The activation function should be a string or a callable function. "
      self.activation = activation

    # weights
    self.W = Linear(num_in + num_out, num_out * 4, w_init=w_init, b_init=b_init)

  def init_state(self, batch_size: int = None, **kwargs):
    self.c = ShortTermState(init.param(self._state_initializer, [self.num_out], batch_size))
    self.h = ShortTermState(init.param(self._state_initializer, [self.num_out], batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.c.value = init.param(self._state_initializer, [self.num_out], batch_size)
    self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

  def update(self, x):
    h, c = self.h.value, self.c.value
    xh = jnp.concat([x, h], axis=-1)
    i, g, f, o = jnp.split(self.W(xh), indices_or_sections=4, axis=-1)
    c = functional.sigmoid(f + 1.) * c + functional.sigmoid(i) * self.activation(g)
    h = functional.sigmoid(o) * self.activation(c)
    self.h.value = h
    self.c.value = c
    return h


class URLSTMCell(RNNCell):
  def __init__(
      self,
      num_in: int,
      num_out: int,
      w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
      state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
      activation: str | Callable = 'tanh',
      name: str = None,
  ):
    super().__init__(name=name)

    # parameters
    self.num_out = num_out
    self.num_in = num_in
    self.in_size = (num_in,)
    self.out_size = (num_out,)

    # initializers
    self._state_initializer = state_init

    # activation function
    if isinstance(activation, str):
      self.activation = getattr(functional, activation)
    else:
      assert callable(activation), "The activation function should be a string or a callable function. "
      self.activation = activation

    # weights
    self.W = Linear(num_in + num_out, num_out * 4, w_init=w_init, b_init=None)
    self.bias = ParamState(self._forget_bias())

  def _forget_bias(self):
    u = random.uniform(1 / self.num_out, 1 - 1 / self.num_out, (self.num_out,))
    return -jnp.log(1 / u - 1)

  def init_state(self, batch_size: int = None, **kwargs):
    self.c = ShortTermState(init.param(self._state_initializer, [self.num_out], batch_size))
    self.h = ShortTermState(init.param(self._state_initializer, [self.num_out], batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.c.value = init.param(self._state_initializer, [self.num_out], batch_size)
    self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

  def update(self, x: ArrayLike) -> ArrayLike:
    h, c = self.h.value, self.c.value
    xh = jnp.concat([x, h], axis=-1)
    f, r, u, o = jnp.split(self.W(xh), indices_or_sections=4, axis=-1)
    f_ = functional.sigmoid(f + self.bias.value)
    r_ = functional.sigmoid(r - self.bias.value)
    g = 2 * r_ * f_ + (1 - 2 * r_) * f_ ** 2
    next_cell = g * c + (1 - g) * self.activation(u)
    next_hidden = functional.sigmoid(o) * self.activation(next_cell)
    self.h.value = next_hidden
    self.c.value = next_cell
    return next_hidden
