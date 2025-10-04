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


from functools import partial
from typing import Optional, Sequence

import brainunit as u
import jax.numpy as jnp

from brainstate import random, environ
from brainstate._state import ShortTermState
from brainstate.typing import Size
from . import init as init
from ._module import ElementWiseBlock

__all__ = [
    'Dropout',
    'Dropout1d',
    'Dropout2d',
    'Dropout3d',
    'AlphaDropout',
    'FeatureAlphaDropout',
    'DropoutFixed',
]


class Dropout(ElementWiseBlock):
    """A layer that stochastically ignores a subset of inputs each training step.

    In training, to compensate for the fraction of input values dropped (`rate`),
    all surviving values are multiplied by `1 / (1 - rate)`.

    This layer is active only during training (``mode=brainstate.mixin.Training``). In other
    circumstances it is a no-op.

    Parameters
    ----------
    prob : float
        Probability to keep element of the tensor. Default is 0.5.
    broadcast_dims : Sequence[int]
        Dimensions that will share the same dropout mask. Default is ().
    name : str, optional
        The name of the dynamic system.

    References
    ----------
    .. [1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent
           neural networks from overfitting." The journal of machine learning
           research 15.1 (2014): 1929-1958.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> layer = brainstate.nn.Dropout(prob=0.8)
        >>> x = brainstate.random.randn(10, 20)
        >>> with brainstate.environ.context(fit=True):
        ...     output = layer(x)
        >>> output.shape
        (10, 20)

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        prob: float = 0.5,
        broadcast_dims: Sequence[int] = (),
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        assert 0. <= prob <= 1., f"Dropout probability must be in the range [0, 1]. But got {prob}."
        self.prob = prob
        self.broadcast_dims = broadcast_dims

    def __call__(self, x):
        dtype = u.math.get_dtype(x)
        fit_phase = environ.get('fit', desc='Whether this is a fitting process. Bool.')
        if fit_phase and self.prob < 1.:
            broadcast_shape = list(x.shape)
            for dim in self.broadcast_dims:
                broadcast_shape[dim] = 1
            keep_mask = random.bernoulli(self.prob, broadcast_shape)
            keep_mask = u.math.broadcast_to(keep_mask, x.shape)
            return u.math.where(
                keep_mask,
                u.math.asarray(x / self.prob, dtype=dtype),
                u.math.asarray(0., dtype=dtype)
            )
        else:
            return x


class _DropoutNd(ElementWiseBlock):
    __module__ = 'brainstate.nn'
    prob: float
    channel_axis: int
    minimal_dim: int

    def __init__(
        self,
        prob: float = 0.5,
        channel_axis: int = -1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        assert 0. <= prob <= 1., f"Dropout probability must be in the range [0, 1]. But got {prob}."
        self.prob = prob
        self.channel_axis = channel_axis

    def __call__(self, x):
        # check input shape
        inp_dim = u.math.ndim(x)
        if inp_dim not in (self.minimal_dim, self.minimal_dim + 1):
            raise RuntimeError(f"dropout1d: Expected {self.minimal_dim}D or {self.minimal_dim + 1}D input, "
                               f"but received a {inp_dim}D input. {self._get_msg(x)}")
        is_not_batched = self.minimal_dim
        if is_not_batched:
            channel_axis = self.channel_axis if self.channel_axis >= 0 else (x.ndim + self.channel_axis)
            mask_shape = [(dim if i == channel_axis else 1) for i, dim in enumerate(x.shape)]
        else:
            channel_axis = (self.channel_axis + 1) if self.channel_axis >= 0 else (x.ndim + self.channel_axis)
            assert channel_axis != 0, f"Channel axis must not be 0. But got {self.channel_axis}."
            mask_shape = [(dim if i in (channel_axis, 0) else 1) for i, dim in enumerate(x.shape)]

        # get fit phase
        fit_phase = environ.get('fit', desc='Whether this is a fitting process. Bool.')

        # generate mask
        if fit_phase and self.prob < 1.:
            dtype = u.math.get_dtype(x)
            keep_mask = random.bernoulli(self.prob, mask_shape)
            keep_mask = jnp.broadcast_to(keep_mask, x.shape)
            return jnp.where(
                keep_mask,
                jnp.asarray(x / self.prob, dtype=dtype),
                jnp.asarray(0., dtype=dtype)
            )
        else:
            return x

    def _get_msg(self, x):
        return ''


class Dropout1d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 1D feature map).

    Each channel will be zeroed out independently on every forward call with
    probability using samples from a Bernoulli distribution. The channel is
    a 1D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th sample
    in the batched input is a 1D tensor :math:`\text{input}[i, j]`.

    Usually the input comes from :class:`Conv1d` modules.

    As described in the paper [1]_, if adjacent pixels within feature maps are
    strongly correlated (as is normally the case in early convolution layers)
    then i.i.d. dropout will not regularize the activations and will otherwise
    just result in an effective learning rate decrease.

    In this case, :class:`Dropout1d` will help promote independence between
    feature maps and should be used instead.

    Parameters
    ----------
    prob : float
        Probability of an element to be kept. Default is 0.5.
    channel_axis : int
        The axis representing the channel dimension. Default is -1.
    name : str, optional
        The name of the dynamic system.

    Notes
    -----
    Input shape: :math:`(N, C, L)` or :math:`(C, L)`.

    Output shape: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input).

    References
    ----------
    .. [1] Springenberg et al., "Striving for Simplicity: The All Convolutional Net"
           https://arxiv.org/abs/1411.4280

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> m = brainstate.nn.Dropout1d(prob=0.8)
        >>> x = brainstate.random.randn(20, 32, 16)
        >>> with brainstate.environ.context(fit=True):
        ...     output = m(x)
        >>> output.shape
        (20, 32, 16)

    """
    __module__ = 'brainstate.nn'
    minimal_dim: int = 2

    def _get_msg(self, x):
        return ("Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
                "spatial dimension, a channel dimension, and an optional batch dimension "
                "(i.e. 2D or 3D inputs).")


class Dropout2d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 2D feature map).

    Each channel will be zeroed out independently on every forward call with
    probability using samples from a Bernoulli distribution. The channel is
    a 2D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th sample
    in the batched input is a 2D tensor :math:`\text{input}[i, j]`.

    Usually the input comes from :class:`Conv2d` modules.

    As described in the paper [1]_, if adjacent pixels within feature maps are
    strongly correlated (as is normally the case in early convolution layers)
    then i.i.d. dropout will not regularize the activations and will otherwise
    just result in an effective learning rate decrease.

    In this case, :class:`Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Parameters
    ----------
    prob : float
        Probability of an element to be kept. Default is 0.5.
    channel_axis : int
        The axis representing the channel dimension. Default is -1.
    name : str, optional
        The name of the dynamic system.

    Notes
    -----
    Input shape: :math:`(N, C, H, W)` or :math:`(C, H, W)`.

    Output shape: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input).

    References
    ----------
    .. [1] Springenberg et al., "Striving for Simplicity: The All Convolutional Net"
           https://arxiv.org/abs/1411.4280

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> m = brainstate.nn.Dropout2d(prob=0.8)
        >>> x = brainstate.random.randn(20, 32, 32, 16)
        >>> with brainstate.environ.context(fit=True):
        ...     output = m(x)
        >>> output.shape
        (20, 32, 32, 16)

    """
    __module__ = 'brainstate.nn'
    minimal_dim: int = 3

    def _get_msg(self, x):
        return ("Note that dropout2d exists to provide channel-wise dropout on inputs with 2 "
                "spatial dimensions, a channel dimension, and an optional batch dimension "
                "(i.e. 3D or 4D inputs).")


class Dropout3d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map).

    Each channel will be zeroed out independently on every forward call with
    probability using samples from a Bernoulli distribution. The channel is
    a 3D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th sample
    in the batched input is a 3D tensor :math:`\text{input}[i, j]`.

    Usually the input comes from :class:`Conv3d` modules.

    As described in the paper [1]_, if adjacent pixels within feature maps are
    strongly correlated (as is normally the case in early convolution layers)
    then i.i.d. dropout will not regularize the activations and will otherwise
    just result in an effective learning rate decrease.

    In this case, :class:`Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Parameters
    ----------
    prob : float
        Probability of an element to be kept. Default is 0.5.
    channel_axis : int
        The axis representing the channel dimension. Default is -1.
    name : str, optional
        The name of the dynamic system.

    Notes
    -----
    Input shape: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.

    Output shape: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input).

    References
    ----------
    .. [1] Springenberg et al., "Striving for Simplicity: The All Convolutional Net"
           https://arxiv.org/abs/1411.4280

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> m = brainstate.nn.Dropout3d(prob=0.8)
        >>> x = brainstate.random.randn(20, 16, 4, 32, 32)
        >>> with brainstate.environ.context(fit=True):
        ...     output = m(x)
        >>> output.shape
        (20, 16, 4, 32, 32)

    """
    __module__ = 'brainstate.nn'
    minimal_dim: int = 4

    def _get_msg(self, x):
        return ("Note that dropout3d exists to provide channel-wise dropout on inputs with 3 "
                "spatial dimensions, a channel dimension, and an optional batch dimension "
                "(i.e. 4D or 5D inputs).")


class AlphaDropout(_DropoutNd):
    r"""Applies Alpha Dropout over the input.

    Alpha Dropout is a type of Dropout that maintains the self-normalizing
    property. For an input with zero mean and unit standard deviation, the output of
    Alpha Dropout maintains the original mean and standard deviation of the
    input.

    Alpha Dropout goes hand-in-hand with SELU activation function, which ensures
    that the outputs have zero mean and unit standard deviation.

    During training, it randomly masks some of the elements of the input
    tensor with probability using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit standard deviation.

    During evaluation the module simply computes an identity function.

    Parameters
    ----------
    prob : float
        Probability of an element to be kept. Default is 0.5.
    name : str, optional
        The name of the dynamic system.

    Notes
    -----
    Input shape: :math:`(*)`. Input can be of any shape.

    Output shape: :math:`(*)`. Output is of the same shape as input.

    References
    ----------
    .. [1] Klambauer et al., "Self-Normalizing Neural Networks"
           https://arxiv.org/abs/1706.02515

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> m = brainstate.nn.AlphaDropout(prob=0.8)
        >>> x = brainstate.random.randn(20, 16)
        >>> with brainstate.environ.context(fit=True):
        ...     output = m(x)
        >>> output.shape
        (20, 16)

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        prob: float = 0.5,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        assert 0. <= prob <= 1., f"Dropout probability must be in the range [0, 1]. But got {prob}."
        self.prob = prob

        # SELU parameters
        alpha = -1.7580993408473766
        self.alpha = alpha

        # Affine transformation parameters to maintain mean and variance
        self.a = ((1 - prob) * (1 + prob * alpha ** 2)) ** -0.5
        self.b = -self.a * alpha * prob

    def __call__(self, x):
        dtype = u.math.get_dtype(x)
        fit_phase = environ.get('fit', desc='Whether this is a fitting process. Bool.')
        if fit_phase and self.prob < 1.:
            keep_mask = random.bernoulli(self.prob, x.shape)
            return u.math.where(
                keep_mask,
                u.math.asarray(x, dtype=dtype),
                u.math.asarray(self.alpha, dtype=dtype)
            ) * self.a + self.b
        else:
            return x


class FeatureAlphaDropout(ElementWiseBlock):
    r"""Randomly masks out entire channels with Alpha Dropout properties.

    Instead of setting activations to zero as in regular Dropout, the activations
    are set to the negative saturation value of the SELU activation function to
    maintain self-normalizing properties.

    Each channel (e.g., the :math:`j`-th channel of the :math:`i`-th sample in
    the batch input is a tensor :math:`\text{input}[i, j]`) will be masked
    independently for each sample on every forward call with probability using
    samples from a Bernoulli distribution. The elements to be masked are randomized
    on every forward call, and scaled and shifted to maintain zero mean and unit
    variance.

    Usually the input comes from convolutional layers with SELU activation.

    As described in the paper [2]_, if adjacent pixels within feature maps are
    strongly correlated (as is normally the case in early convolution layers)
    then i.i.d. dropout will not regularize the activations and will otherwise
    just result in an effective learning rate decrease.

    In this case, :class:`FeatureAlphaDropout` will help promote independence between
    feature maps and should be used instead.

    Parameters
    ----------
    prob : float
        Probability of an element to be kept. Default is 0.5.
    channel_axis : int
        The axis representing the channel dimension. Default is -1.
    name : str, optional
        The name of the dynamic system.

    Notes
    -----
    Input shape: :math:`(N, C, *)` where C is the channel dimension.

    Output shape: Same shape as input.

    References
    ----------
    .. [1] Klambauer et al., "Self-Normalizing Neural Networks"
           https://arxiv.org/abs/1706.02515
    .. [2] Springenberg et al., "Striving for Simplicity: The All Convolutional Net"
           https://arxiv.org/abs/1411.4280

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> m = brainstate.nn.FeatureAlphaDropout(prob=0.8)
        >>> x = brainstate.random.randn(20, 16, 4, 32, 32)
        >>> with brainstate.environ.context(fit=True):
        ...     output = m(x)
        >>> output.shape
        (20, 16, 4, 32, 32)

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        prob: float = 0.5,
        channel_axis: int = -1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        assert 0. <= prob <= 1., f"Dropout probability must be in the range [0, 1]. But got {prob}."
        self.prob = prob
        self.channel_axis = channel_axis

        # SELU parameters
        alpha = -1.7580993408473766
        self.alpha = alpha

        # Affine transformation parameters to maintain mean and variance
        self.a = ((1 - prob) * (1 + prob * alpha ** 2)) ** -0.5
        self.b = -self.a * alpha * prob

    def __call__(self, x):
        dtype = u.math.get_dtype(x)
        fit_phase = environ.get('fit', desc='Whether this is a fitting process. Bool.')
        if fit_phase and self.prob < 1.:
            # Create mask shape with 1s except for batch and channel dimensions
            channel_axis = self.channel_axis if self.channel_axis >= 0 else (x.ndim + self.channel_axis)
            mask_shape = [1] * x.ndim
            mask_shape[0] = x.shape[0]  # batch dimension
            mask_shape[channel_axis] = x.shape[channel_axis]  # channel dimension

            keep_mask = random.bernoulli(self.prob, mask_shape)
            keep_mask = u.math.broadcast_to(keep_mask, x.shape)
            return u.math.where(
                keep_mask,
                u.math.asarray(x, dtype=dtype),
                u.math.asarray(self.alpha, dtype=dtype)
            ) * self.a + self.b
        else:
            return x


class DropoutFixed(ElementWiseBlock):
    """A dropout layer with a fixed dropout mask along the time axis.

    In training, to compensate for the fraction of input values dropped,
    all surviving values are multiplied by `1 / (1 - prob)`.

    This layer is active only during training (``mode=brainstate.mixin.Training``). In other
    circumstances it is a no-op.

    This kind of Dropout is particularly useful for spiking neural networks (SNNs) where
    the same dropout mask needs to be applied across multiple time steps within a single
    mini-batch iteration.

    Parameters
    ----------
    in_size : tuple or int
        The size of the input tensor.
    prob : float
        Probability to keep element of the tensor. Default is 0.5.
    name : str, optional
        The name of the dynamic system.

    Notes
    -----
    As described in [2]_, there is a subtle difference in the way dropout is applied in
    SNNs compared to ANNs. In ANNs, each epoch of training has several iterations of
    mini-batches. In each iteration, randomly selected units (with dropout ratio of
    :math:`p`) are disconnected from the network while weighting by its posterior
    probability (:math:`1-p`).

    However, in SNNs, each iteration has more than one forward propagation depending on
    the time length of the spike train. We back-propagate the output error and modify
    the network parameters only at the last time step. For dropout to be effective in
    our training method, it has to be ensured that the set of connected units within an
    iteration of mini-batch data is not changed, such that the neural network is
    constituted by the same random subset of units during each forward propagation within
    a single iteration.

    On the other hand, if the units are randomly connected at each time-step, the effect
    of dropout will be averaged out over the entire forward propagation time within an
    iteration. Then, the dropout effect would fade-out once the output error is propagated
    backward and the parameters are updated at the last time step. Therefore, we need to
    keep the set of randomly connected units for the entire time window within an iteration.

    References
    ----------
    .. [1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent
           neural networks from overfitting." The journal of machine learning
           research 15.1 (2014): 1929-1958.
    .. [2] Lee et al., "Enabling Spike-based Backpropagation for Training Deep Neural
           Network Architectures" https://arxiv.org/abs/1903.06379

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> layer = brainstate.nn.DropoutFixed(in_size=(20,), prob=0.8)
        >>> layer.init_state(batch_size=10)
        >>> x = brainstate.random.randn(10, 20)
        >>> with brainstate.environ.context(fit=True):
        ...     output = layer.update(x)
        >>> output.shape
        (10, 20)

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        prob: float = 0.5,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        assert 0. <= prob <= 1., f"Dropout probability must be in the range [0, 1]. But got {prob}."
        self.prob = prob
        self.in_size = in_size
        self.out_size = in_size

    def init_state(self, batch_size=None, **kwargs):
        if self.prob < 1.:
            self.mask = ShortTermState(init.param(partial(random.bernoulli, self.prob), self.in_size, batch_size))

    def update(self, x):
        dtype = u.math.get_dtype(x)
        fit_phase = environ.get('fit', desc='Whether this is a fitting process. Bool.')
        if fit_phase and self.prob < 1.:
            if self.mask.value.shape != x.shape:
                raise ValueError(f"Input shape {x.shape} does not match the mask shape {self.mask.value.shape}. "
                                 f"Please call `init_state()` method first.")
            return u.math.where(self.mask.value,
                                u.math.asarray(x / self.prob, dtype=dtype),
                                u.math.asarray(0., dtype=dtype) * u.get_unit(x))
        else:
            return x
