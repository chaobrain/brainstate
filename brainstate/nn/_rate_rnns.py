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


from typing import Callable, Union

import jax.numpy as jnp

from brainstate import random, init, functional
from brainstate._state import HiddenState, ParamState
from brainstate.typing import ArrayLike
from ._linear import Linear
from ._module import Module

__all__ = [
    'RNNCell', 'ValinaRNNCell', 'GRUCell', 'MGUCell', 'LSTMCell', 'URLSTMCell',
]


class RNNCell(Module):
    """
    Base class for all recurrent neural network (RNN) cell implementations.

    This abstract class serves as the foundation for implementing various RNN cell types
    such as vanilla RNN, GRU, LSTM, and other recurrent architectures. It extends the
    Module class and provides common functionality and interface for recurrent units.

    All RNN cell implementations should inherit from this class and implement the required
    methods, particularly the `init_state()`, `reset_state()`, and `update()` methods that
    define the state initialization and recurrent dynamics.

    The RNNCell typically maintains hidden state(s) that are updated at each time step
    based on the current input and previous state values.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the cell state variables with appropriate dimensions.
    reset_state(batch_size=None, **kwargs)
        Reset the cell state variables to their initial values.
    update(x)
        Update the cell state for one time step based on input x and return output.
    """
    pass


class ValinaRNNCell(RNNCell):
    r"""
    Vanilla Recurrent Neural Network (RNN) cell implementation.

    This class implements the basic RNN model that updates a hidden state based on
    the current input and previous hidden state. The standard RNN cell follows the
    mathematical formulation:

    .. math::

        h_t = \phi(W [x_t, h_{t-1}] + b)

    where:

    - :math:`x_t` is the input vector at time t
    - :math:`h_t` is the hidden state at time t
    - :math:`h_{t-1}` is the hidden state at previous time step
    - :math:`W` is the weight matrix for the combined input-hidden linear transformation
    - :math:`b` is the bias vector
    - :math:`\phi` is the activation function

    Parameters
    ----------
    num_in : int
        The number of input units.
    num_out : int
        The number of hidden units.
    state_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the hidden state.
    w_init : Union[ArrayLike, Callable], default=init.XavierNormal()
        Initializer for the weight matrix.
    b_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the bias vector.
    activation : str or Callable, default='relu'
        Activation function to use. Can be a string (e.g., 'relu', 'tanh')
        or a callable function.
    name : str, optional
        Name of the module.

    State Variables
    --------------
    h : HiddenState
        Hidden state of the RNN cell.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the cell hidden state.
    reset_state(batch_size=None, **kwargs)
        Reset the cell hidden state to its initial value.
    update(x)
        Update the hidden state for one time step and return the new state.
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
        self.h = HiddenState(init.param(self._state_initializer, self.num_out, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.num_out, batch_size)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        h = self.W(xh)
        self.h.value = self.activation(h)
        return self.h.value


class GRUCell(RNNCell):
    r"""
    Gated Recurrent Unit (GRU) cell implementation.

    This class implements the GRU model that uses gating mechanisms to control
    information flow. The GRU has fewer parameters than LSTM as it combines
    the forget and input gates into a single update gate. The GRU follows the
    mathematical formulation:

    .. math::

        r_t &= \sigma(W_r [x_t, h_{t-1}] + b_r) \\
        z_t &= \sigma(W_z [x_t, h_{t-1}] + b_z) \\
        \tilde{h}_t &= \tanh(W_h [x_t, (r_t \odot h_{t-1})] + b_h) \\
        h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t

    where:

    - :math:`x_t` is the input vector at time t
    - :math:`h_t` is the hidden state at time t
    - :math:`r_t` is the reset gate vector
    - :math:`z_t` is the update gate vector
    - :math:`\tilde{h}_t` is the candidate hidden state
    - :math:`\odot` represents element-wise multiplication
    - :math:`\sigma` is the sigmoid activation function

    Parameters
    ----------
    num_in : int
        The number of input units.
    num_out : int
        The number of hidden units.
    w_init : Union[ArrayLike, Callable], default=init.Orthogonal()
        Initializer for the weight matrices.
    b_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the bias vectors.
    state_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the hidden state.
    activation : str or Callable, default='tanh'
        Activation function to use. Can be a string (e.g., 'tanh')
        or a callable function.
    name : str, optional
        Name of the module.

    State Variables
    --------------
    h : HiddenState
        Hidden state of the GRU cell.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the cell hidden state.
    reset_state(batch_size=None, **kwargs)
        Reset the cell hidden state to its initial value.
    update(x)
        Update the hidden state for one time step and return the new state.
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
        self.h = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))

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
    Minimal Gated Recurrent Unit (MGU) cell implementation.

    MGU is a simplified version of GRU that uses a single forget gate instead of
    separate reset and update gates. This results in fewer parameters while
    maintaining much of the gating capability. The MGU follows the mathematical
    formulation:

    .. math::

        f_t &= \\sigma(W_f [x_t, h_{t-1}] + b_f) \\\\
        \\tilde{h}_t &= \\phi(W_h [x_t, (f_t \\odot h_{t-1})] + b_h) \\\\
        h_t &= (1 - f_t) \\odot h_{t-1} + f_t \\odot \\tilde{h}_t

    where:

    - :math:`x_t` is the input vector at time t
    - :math:`h_t` is the hidden state at time t
    - :math:`f_t` is the forget gate vector
    - :math:`\\tilde{h}_t` is the candidate hidden state
    - :math:`\\odot` represents element-wise multiplication
    - :math:`\\sigma` is the sigmoid activation function
    - :math:`\\phi` is the activation function (typically tanh)

    Parameters
    ----------
    num_in : int
        The number of input units.
    num_out : int
        The number of hidden units.
    w_init : Union[ArrayLike, Callable], default=init.Orthogonal()
        Initializer for the weight matrices.
    b_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the bias vectors.
    state_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the hidden state.
    activation : str or Callable, default='tanh'
        Activation function to use. Can be a string (e.g., 'tanh')
        or a callable function.
    name : str, optional
        Name of the module.

    State Variables
    --------------
    h : HiddenState
        Hidden state of the MGU cell.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the cell hidden state.
    reset_state(batch_size=None, **kwargs)
        Reset the cell hidden state to its initial value.
    update(x)
        Update the hidden state for one time step and return the new state.
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
        self.h = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))

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
    r"""
    Long Short-Term Memory (LSTM) cell implementation.

    This class implements the LSTM architecture which uses multiple gating mechanisms
    to regulate information flow and address the vanishing gradient problem in RNNs.
    The LSTM follows the mathematical formulation:

    .. math::

        i_t &= \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
        f_t &= \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
        g_t &= \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
        o_t &= \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
        h_t &= o_t \odot \tanh(c_t)

    where:

    - :math:`x_t` is the input vector at time t
    - :math:`h_t` is the hidden state at time t
    - :math:`c_t` is the cell state at time t
    - :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and output gate activations
    - :math:`g_t` is the cell update vector
    - :math:`\odot` represents element-wise multiplication
    - :math:`\sigma` is the sigmoid activation function

    Notes
    -----
    Forget gate initialization: Following Jozefowicz et al. (2015), we add 1.0
    to the forget gate bias after initialization to reduce forgetting at the
    beginning of training.

    Parameters
    ----------
    num_in : int
        The number of input units.
    num_out : int
        The number of hidden/cell units.
    w_init : Union[ArrayLike, Callable], default=init.XavierNormal()
        Initializer for the weight matrices.
    b_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the bias vectors.
    state_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the hidden and cell states.
    activation : str or Callable, default='tanh'
        Activation function to use. Can be a string (e.g., 'tanh')
        or a callable function.
    name : str, optional
        Name of the module.

    State Variables
    --------------
    h : HiddenState
        Hidden state of the LSTM cell.
    c : HiddenState
        Cell state of the LSTM cell.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the cell and hidden states.
    reset_state(batch_size=None, **kwargs)
        Reset the cell and hidden states to their initial values.
    update(x)
        Update the states for one time step and return the new hidden state.

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
           Neural computation, 9(8), 1735-1780.
    .. [2] Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural
           network regularization. arXiv preprint arXiv:1409.2329.
    .. [3] Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An empirical
           exploration of recurrent network architectures. In International
           conference on machine learning, pp. 2342-2350.
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
        self.c = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))
        self.h = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))

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
        self.c = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))
        self.h = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))

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
