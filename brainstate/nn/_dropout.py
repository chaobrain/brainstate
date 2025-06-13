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


from functools import partial
from typing import Optional, Sequence

import brainunit as u
import jax.numpy as jnp

from brainstate import random, environ, init
from brainstate._state import ShortTermState
from brainstate.typing import Size
from ._module import ElementWiseBlock

__all__ = [
    'DropoutFixed', 'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d',
]


class Dropout(ElementWiseBlock):
    """A layer that stochastically ignores a subset of inputs each training step.

    In training, to compensate for the fraction of input values dropped (`rate`),
    all surviving values are multiplied by `1 / (1 - rate)`.

    This layer is active only during training (``mode=brainstate.mixin.Training``). In other
    circumstances it is a no-op.

    .. [1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent
           neural networks from overfitting." The journal of machine learning
           research 15.1 (2014): 1929-1958.

    Args:
        prob: Probability to keep element of the tensor.
        broadcast_dims: dimensions that will share the same dropout mask.
        name: str. The name of the dynamic system.

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
    r"""Randomly zero out entire channels (a channel is a 1D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv1d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout1d` will help promote independence between
    feature maps and should be used instead.

    Args:
        prob: float. probability of an element to be zero-ed.

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`.
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input).

    Examples::

        >>> m = Dropout1d(p=0.2)
        >>> x = random.randn(20, 32, 16)
        >>> output = m(x)
        >>> output.shape
        (20, 32, 16)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """
    __module__ = 'brainstate.nn'
    minimal_dim: int = 2

    def _get_msg(self, x):
        return ("Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
                "spatial dimension, a channel dimension, and an optional batch dimension "
                "(i.e. 2D or 3D inputs).")


class Dropout2d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        prob: float. probability of an element to be kept.

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, L)`.
        - Output: :math:`(N, C, H, W)` or :math:`(N, C, L)` (same shape as input).

    Examples::

        >>> m = Dropout2d(p=0.2)
        >>> x = random.randn(20, 32, 32, 16)
        >>> output = m(x)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """
    __module__ = 'brainstate.nn'
    minimal_dim: int = 3

    def _get_msg(self, x):
        return ("Note that dropout2d exists to provide channel-wise dropout on inputs with 2 "
                "spatial dimensions, a channel dimension, and an optional batch dimension "
                "(i.e. 3D or 4D inputs).")


class Dropout3d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv3d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        prob: float. probability of an element to be kept.

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input).

    Examples::

        >>> m = Dropout3d(p=0.2)
        >>> x = random.randn(20, 16, 4, 32, 32)
        >>> output = m(x)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
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
    property.
    For an input with zero mean and unit standard deviation, the output of
    Alpha Dropout maintains the original mean and standard deviation of the
    input.
    Alpha Dropout goes hand-in-hand with SELU activation function, which ensures
    that the outputs have zero mean and unit standard deviation.

    During training, it randomly masks some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit standard deviation.

    During evaluation the module simply computes an identity function.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        prob: float. probability of an element to be kept.

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = AlphaDropout(p=0.2)
        >>> x = random.randn(20, 16)
        >>> output = m(x)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """
    __module__ = 'brainstate.nn'

    def update(self, *args, **kwargs):
        raise NotImplementedError("AlphaDropout is not supported in the current version.")


class FeatureAlphaDropout(_DropoutNd):
    r"""Randomly masks out entire channels (a channel is a feature map,
    e.g. the :math:`j`-th channel of the :math:`i`-th sample in the batch input
    is a tensor :math:`\text{input}[i, j]`) of the input tensor). Instead of
    setting activations to zero, as in regular Dropout, the activations are set
    to the negative saturation value of the SELU activation function. More details
    can be found in the paper `Self-Normalizing Neural Networks`_ .

    Each element will be masked independently for each sample on every forward
    call with probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit variance.

    Usually the input comes from :class:`nn.AlphaDropout` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.AlphaDropout` will help promote independence between
    feature maps and should be used instead.

    Args:
        prob: float. probability of an element to be kept.

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input).

    Examples::

        >>> m = FeatureAlphaDropout(p=0.2)
        >>> x = random.randn(20, 16, 4, 32, 32)
        >>> output = m(x)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """
    __module__ = 'brainstate.nn'

    def update(self, *args, **kwargs):
        raise NotImplementedError("FeatureAlphaDropout is not supported in the current version.")


class DropoutFixed(ElementWiseBlock):
    """
    A dropout layer with the fixed dropout mask along the time axis once after initialized.

    In training, to compensate for the fraction of input values dropped (`rate`),
    all surviving values are multiplied by `1 / (1 - rate)`.

    This layer is active only during training (``mode=brainstate.mixin.Training``). In other
    circumstances it is a no-op.

    .. [1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent
           neural networks from overfitting." The journal of machine learning
           research 15.1 (2014): 1929-1958.

    .. admonition:: Tip
        :class: tip

        This kind of Dropout is firstly described in `Enabling Spike-based Backpropagation for Training Deep Neural
        Network Architectures <https://arxiv.org/abs/1903.06379>`_:

        There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
        training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
        are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
        iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
        the output error and modify the network parameters only at the last time step. For dropout to be effective in
        our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
        data is not changed, such that the neural network is constituted by the same random subset of units during
        each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
        each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
        iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
        are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
        time window within an iteration.

    Args:
      in_size: The size of the input tensor.
      prob: Probability to keep element of the tensor.
      mode: Mode. The computation mode of the object.
      name: str. The name of the dynamic system.
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
