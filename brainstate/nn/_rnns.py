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

# -*- coding: utf-8 -*-


from typing import Callable, Union

import jax.numpy as jnp

from brainstate import random
from brainstate._state import HiddenState, ParamState
from brainstate.typing import ArrayLike
from . import _activations as functional
from . import init as init
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

    See Also
    --------
    ValinaRNNCell : Vanilla RNN cell implementation
    GRUCell : Gated Recurrent Unit cell implementation
    LSTMCell : Long Short-Term Memory cell implementation
    URLSTMCell : LSTM with UR gating mechanism
    MGUCell : Minimal Gated Unit cell implementation
    """
    __module__ = 'brainstate.nn'
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

    Attributes
    ----------
    num_in : int
        Number of input features.
    num_out : int
        Number of hidden units.
    in_size : tuple
        Shape of input (num_in,).
    out_size : tuple
        Shape of output (num_out,).

    State Variables
    ---------------
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

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bs
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a vanilla RNN cell
        >>> cell = bs.nn.ValinaRNNCell(num_in=10, num_out=20)
        >>>
        >>> # Initialize state for batch size 32
        >>> cell.init_state(batch_size=32)
        >>>
        >>> # Process a single time step
        >>> x = jnp.ones((32, 10))  # batch_size x num_in
        >>> output = cell.update(x)
        >>> print(output.shape)  # (32, 20)
        >>>
        >>> # Process a sequence of inputs
        >>> sequence = jnp.ones((100, 32, 10))  # time_steps x batch_size x num_in
        >>> outputs = []
        >>> for t in range(100):
        ...     output = cell.update(sequence[t])
        ...     outputs.append(output)
        >>> outputs = jnp.stack(outputs)
        >>> print(outputs.shape)  # (100, 32, 20)

    Notes
    -----
    Vanilla RNNs can suffer from vanishing or exploding gradient problems
    when processing long sequences. For better performance on long sequences,
    consider using gated architectures like GRU or LSTM.
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
        """
        Initialize the hidden state.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for state initialization.
        **kwargs
            Additional keyword arguments.
        """
        self.h = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the hidden state to initial value.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for state reset.
        **kwargs
            Additional keyword arguments.
        """
        self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        h = self.W(xh)
        self.h.value = self.activation(h)
        return self.h.value


class GRUCell(RNNCell):
    r"""
    Gated Recurrent Unit (GRU) cell implementation.

    The GRU is a gating mechanism in recurrent neural networks that aims to solve
    the vanishing gradient problem. It uses gating mechanisms to control information
    flow and has fewer parameters than LSTM as it combines the forget and input gates
    into a single update gate.

    The GRU cell follows the mathematical formulation:

    .. math::

        r_t &= \sigma(W_r [x_t, h_{t-1}] + b_r) \\
        z_t &= \sigma(W_z [x_t, h_{t-1}] + b_z) \\
        \tilde{h}_t &= \phi(W_h [x_t, (r_t \odot h_{t-1})] + b_h) \\
        h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t

    where:

    - :math:`x_t` is the input vector at time t
    - :math:`h_t` is the hidden state at time t
    - :math:`r_t` is the reset gate vector
    - :math:`z_t` is the update gate vector
    - :math:`\tilde{h}_t` is the candidate hidden state
    - :math:`\odot` represents element-wise multiplication
    - :math:`\sigma` is the sigmoid activation function
    - :math:`\phi` is the activation function (typically tanh)

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
        Activation function to use. Can be a string (e.g., 'tanh', 'relu')
        or a callable function.
    name : str, optional
        Name of the module.

    Attributes
    ----------
    num_in : int
        Number of input features.
    num_out : int
        Number of hidden units.
    in_size : tuple
        Shape of input (num_in,).
    out_size : tuple
        Shape of output (num_out,).

    State Variables
    ---------------
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

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bs
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a GRU cell
        >>> cell = bs.nn.GRUCell(num_in=10, num_out=20)
        >>>
        >>> # Initialize state for batch size 32
        >>> cell.init_state(batch_size=32)
        >>>
        >>> # Process a single time step
        >>> x = jnp.ones((32, 10))  # batch_size x num_in
        >>> output = cell.update(x)
        >>> print(output.shape)  # (32, 20)
        >>>
        >>> # Process a sequence
        >>> sequence = jnp.ones((100, 32, 10))  # time_steps x batch_size x num_in
        >>> outputs = []
        >>> for t in range(100):
        ...     output = cell.update(sequence[t])
        ...     outputs.append(output)
        >>> outputs = jnp.stack(outputs)
        >>> print(outputs.shape)  # (100, 32, 20)
        >>>
        >>> # Reset state with different batch size
        >>> cell.reset_state(batch_size=16)
        >>> x_new = jnp.ones((16, 10))
        >>> output_new = cell.update(x_new)
        >>> print(output_new.shape)  # (16, 20)

    Notes
    -----
    GRU cells are computationally more efficient than LSTM cells due to having
    fewer parameters, while often achieving comparable performance on many tasks.

    References
    ----------
    .. [1] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
           Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using
           RNN encoder-decoder for statistical machine translation.
           arXiv preprint arXiv:1406.1078.
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
    Minimal Gated Unit (MGU) cell implementation.

    MGU is a simplified version of GRU that uses a single forget gate instead of
    separate reset and update gates. This design significantly reduces the number
    of parameters while maintaining much of the gating capability. MGU provides
    a good trade-off between model complexity and performance.

    The MGU cell follows the mathematical formulation:

    .. math::

        f_t &= \sigma(W_f [x_t, h_{t-1}] + b_f) \\
        \tilde{h}_t &= \phi(W_h [x_t, (f_t \odot h_{t-1})] + b_h) \\
        h_t &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t

    where:

    - :math:`x_t` is the input vector at time t
    - :math:`h_t` is the hidden state at time t
    - :math:`f_t` is the forget gate vector
    - :math:`\tilde{h}_t` is the candidate hidden state
    - :math:`\odot` represents element-wise multiplication
    - :math:`\sigma` is the sigmoid activation function
    - :math:`\phi` is the activation function (typically tanh)

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
        Activation function to use. Can be a string (e.g., 'tanh', 'relu')
        or a callable function.
    name : str, optional
        Name of the module.

    Attributes
    ----------
    num_in : int
        Number of input features.
    num_out : int
        Number of hidden units.
    in_size : tuple
        Shape of input (num_in,).
    out_size : tuple
        Shape of output (num_out,).

    State Variables
    ---------------
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

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bs
        >>> import jax.numpy as jnp
        >>>
        >>> # Create an MGU cell
        >>> cell = bs.nn.MGUCell(num_in=10, num_out=20)
        >>>
        >>> # Initialize state for batch size 32
        >>> cell.init_state(batch_size=32)
        >>>
        >>> # Process a single time step
        >>> x = jnp.ones((32, 10))  # batch_size x num_in
        >>> output = cell.update(x)
        >>> print(output.shape)  # (32, 20)
        >>>
        >>> # Process a sequence
        >>> sequence = jnp.ones((100, 32, 10))  # time_steps x batch_size x num_in
        >>> outputs = []
        >>> for t in range(100):
        ...     output = cell.update(sequence[t])
        ...     outputs.append(output)
        >>> outputs = jnp.stack(outputs)
        >>> print(outputs.shape)  # (100, 32, 20)

    Notes
    -----
    MGU provides a lightweight alternative to GRU and LSTM, making it suitable
    for resource-constrained applications or when model simplicity is preferred.

    References
    ----------
    .. [1] Zhou, G. B., Wu, J., Zhang, C. L., & Zhou, Z. H. (2016). Minimal gated unit
           for recurrent neural networks. International Journal of Automation and Computing,
           13(3), 226-234.
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

    LSTM is a type of RNN architecture designed to address the vanishing gradient
    problem and learn long-term dependencies. It uses a cell state to carry
    information across time steps and three gates (input, forget, output) to
    control information flow.

    The LSTM cell follows the mathematical formulation:

    .. math::

        i_t &= \sigma(W_i [x_t, h_{t-1}] + b_i) \\
        f_t &= \sigma(W_f [x_t, h_{t-1}] + b_f) \\
        g_t &= \phi(W_g [x_t, h_{t-1}] + b_g) \\
        o_t &= \sigma(W_o [x_t, h_{t-1}] + b_o) \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
        h_t &= o_t \odot \phi(c_t)

    where:

    - :math:`x_t` is the input vector at time t
    - :math:`h_t` is the hidden state at time t
    - :math:`c_t` is the cell state at time t
    - :math:`i_t` is the input gate activation
    - :math:`f_t` is the forget gate activation
    - :math:`o_t` is the output gate activation
    - :math:`g_t` is the cell update (candidate) vector
    - :math:`\odot` represents element-wise multiplication
    - :math:`\sigma` is the sigmoid activation function
    - :math:`\phi` is the activation function (typically tanh)

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
        Activation function to use. Can be a string (e.g., 'tanh', 'relu')
        or a callable function.
    name : str, optional
        Name of the module.

    Attributes
    ----------
    num_in : int
        Number of input features.
    num_out : int
        Number of hidden/cell units.
    in_size : tuple
        Shape of input (num_in,).
    out_size : tuple
        Shape of output (num_out,).

    State Variables
    ---------------
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

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bs
        >>> import jax.numpy as jnp
        >>>
        >>> # Create an LSTM cell
        >>> cell = bs.nn.LSTMCell(num_in=10, num_out=20)
        >>>
        >>> # Initialize states for batch size 32
        >>> cell.init_state(batch_size=32)
        >>>
        >>> # Process a single time step
        >>> x = jnp.ones((32, 10))  # batch_size x num_in
        >>> output = cell.update(x)
        >>> print(output.shape)  # (32, 20)
        >>>
        >>> # Process a sequence
        >>> sequence = jnp.ones((100, 32, 10))  # time_steps x batch_size x num_in
        >>> outputs = []
        >>> for t in range(100):
        ...     output = cell.update(sequence[t])
        ...     outputs.append(output)
        >>> outputs = jnp.stack(outputs)
        >>> print(outputs.shape)  # (100, 32, 20)
        >>>
        >>> # Access cell state
        >>> print(cell.c.value.shape)  # (32, 20)
        >>> print(cell.h.value.shape)  # (32, 20)

    Notes
    -----
    - The forget gate bias is initialized with +1.0 following Jozefowicz et al. (2015)
      to reduce forgetting at the beginning of training.
    - LSTM cells are effective for learning long-term dependencies but require
      more parameters and computation than simpler RNN variants.

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
           Neural computation, 9(8), 1735-1780.
    .. [2] Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget:
           Continual prediction with LSTM. Neural computation, 12(10), 2451-2471.
    .. [3] Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An empirical
           exploration of recurrent network architectures. In International
           conference on machine learning (pp. 2342-2350).
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
    r"""LSTM with UR gating mechanism.

    URLSTM is a modification of the standard LSTM that uses untied (separate) biases
    for the forget and retention mechanisms, allowing for more flexible gating control.
    This implementation is based on the paper "Improving the Gating Mechanism of
    Recurrent Neural Networks" by Gers et al.

    The URLSTM cell follows the mathematical formulation:

    .. math::

        f_t &= \sigma(W_f [x_t, h_{t-1}] + b_f) \\
        r_t &= \sigma(W_r [x_t, h_{t-1}] - b_f) \\
        g_t &= 2 r_t \odot f_t + (1 - 2 r_t) \odot f_t^2 \\
        \tilde{c}_t &= \phi(W_c [x_t, h_{t-1}]) \\
        c_t &= g_t \odot c_{t-1} + (1 - g_t) \odot \tilde{c}_t \\
        o_t &= \sigma(W_o [x_t, h_{t-1}]) \\
        h_t &= o_t \odot \phi(c_t)

    where:

    - :math:`x_t` is the input vector at time t
    - :math:`h_t` is the hidden state at time t
    - :math:`c_t` is the cell state at time t
    - :math:`f_t` is the forget gate with positive bias
    - :math:`r_t` is the retention gate with negative bias
    - :math:`g_t` is the unified gate combining forget and retention
    - :math:`\tilde{c}_t` is the candidate cell state
    - :math:`o_t` is the output gate
    - :math:`\odot` represents element-wise multiplication
    - :math:`\sigma` is the sigmoid activation function
    - :math:`\phi` is the activation function (typically tanh)

    The key innovation is the untied bias mechanism where the forget and retention
    gates use opposite biases, initialized using a uniform distribution to encourage
    diverse gating behavior across units.

    Parameters
    ----------
    num_in : int
        The number of input units.
    num_out : int
        The number of hidden/output units.
    w_init : Union[ArrayLike, Callable], default=init.XavierNormal()
        Initializer for the weight matrix.
    state_init : Union[ArrayLike, Callable], default=init.ZeroInit()
        Initializer for the hidden and cell states.
    activation : str or Callable, default='tanh'
        Activation function to use. Can be a string (e.g., 'relu', 'tanh')
        or a callable function.
    name : str, optional
        Name of the module.

    State Variables
    ---------------
    h : HiddenState
        Hidden state of the URLSTM cell.
    c : HiddenState
        Cell state of the URLSTM cell.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the cell and hidden states.
    reset_state(batch_size=None, **kwargs)
        Reset the cell and hidden states to their initial values.
    update(x)
        Update the cell and hidden states for one time step and return the hidden state.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bs
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a URLSTM cell
        >>> cell = bs.nn.URLSTMCell(num_in=10, num_out=20)
        >>>
        >>> # Initialize the state for batch size 32
        >>> cell.init_state(batch_size=32)
        >>>
        >>> # Process a sequence
        >>> x = jnp.ones((32, 10))  # batch_size x num_in
        >>> output = cell.update(x)
        >>> print(output.shape)  # (32, 20)
        >>>
        >>> # Process multiple time steps
        >>> sequence = jnp.ones((100, 32, 10))  # time_steps x batch_size x num_in
        >>> outputs = []
        >>> for t in range(100):
        ...     output = cell.update(sequence[t])
        ...     outputs.append(output)
        >>> outputs = jnp.stack(outputs)
        >>> print(outputs.shape)  # (100, 32, 20)

    References
    ----------
    .. [1] Gu, Albert, et al. "Improving the gating mechanism of recurrent neural networks."
           International conference on machine learning. PMLR, 2020.
    """
    __module__ = 'brainstate.nn'

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
            assert callable(activation), "The activation function should be a string or a callable function."
            self.activation = activation

        # weights - 4 gates: forget, retention, candidate, output
        self.W = Linear(num_in + num_out, num_out * 4, w_init=w_init, b_init=None)

        # Initialize untied bias using uniform distribution
        self.bias = ParamState(self._forget_bias())

    def _forget_bias(self):
        """Initialize the forget gate bias using uniform distribution."""
        # Sample from uniform distribution to encourage diverse gating
        u = random.uniform(1 / self.num_out, 1 - 1 / self.num_out, (self.num_out,))
        # Transform to logit space for initialization
        return -jnp.log(1 / u - 1)

    def init_state(self, batch_size: int = None, **kwargs):
        """
        Initialize the cell and hidden states.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for state initialization.
        **kwargs
            Additional keyword arguments.
        """
        self.c = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))
        self.h = HiddenState(init.param(self._state_initializer, [self.num_out], batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the cell and hidden states to their initial values.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for state reset.
        **kwargs
            Additional keyword arguments.
        """
        self.c.value = init.param(self._state_initializer, [self.num_out], batch_size)
        self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

    def update(self, x: ArrayLike) -> ArrayLike:
        """
        Update the URLSTM cell for one time step.

        Parameters
        ----------
        x : ArrayLike
            Input tensor with shape (batch_size, num_in).

        Returns
        -------
        ArrayLike
            Hidden state tensor with shape (batch_size, num_out).
        """
        h, c = self.h.value, self.c.value

        # Concatenate input and hidden state
        xh = jnp.concatenate([x, h], axis=-1)

        # Compute all gates in one pass
        gates = self.W(xh)
        f, r, u, o = jnp.split(gates, indices_or_sections=4, axis=-1)

        # Apply untied biases to forget and retention gates
        f_gate = functional.sigmoid(f + self.bias.value)
        r_gate = functional.sigmoid(r - self.bias.value)

        # Compute unified gate
        g = 2 * r_gate * f_gate + (1 - 2 * r_gate) * f_gate ** 2

        # Update cell state
        next_cell = g * c + (1 - g) * self.activation(u)

        # Compute output gate and hidden state
        o_gate = functional.sigmoid(o)
        next_hidden = o_gate * self.activation(next_cell)

        # Update states
        self.h.value = next_hidden
        self.c.value = next_cell

        return next_hidden
