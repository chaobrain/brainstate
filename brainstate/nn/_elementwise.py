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

from typing import Optional

import brainunit as u
import jax.numpy as jnp

from brainstate._state import ParamState
from brainstate.typing import ArrayLike
from . import _activations as F
from ._module import ElementWiseBlock

__all__ = [
    # activation functions
    'Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
    'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
    'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'PReLU',
    'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax',

    # others
    'Identity', 'SpikeBitwise',
]


class Threshold(ElementWiseBlock):
    r"""Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate
        >>> m = nn.Threshold(0.1, 20)
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    threshold: float
    value: float

    def __init__(self, threshold: float, value: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value

    def __call__(self, x: ArrayLike) -> ArrayLike:
        dtype = u.math.get_dtype(x)
        return jnp.where(x > jnp.asarray(self.threshold, dtype=dtype),
                         x,
                         jnp.asarray(self.value, dtype=dtype))

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold}, value={self.value})'


class ReLU(ElementWiseBlock):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.ReLU()
        >>> x = random.randn(2)
        >>> output = m(x)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.ReLU()
        >>> x = random.randn(2).unsqueeze(0)
        >>> output = jax.numpy.concat((m(x), m(-x)))
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.relu(x)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class RReLU(ElementWiseBlock):
    r"""Applies the randomized leaky rectified liner unit function, element-wise,
    as described in the paper:

    `Empirical Evaluation of Rectified Activations in Convolutional Network`_.

    The function is defined as:

    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})`.

     See: https://arxiv.org/pdf/1505.00853.pdf

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.RReLU(0.1, 0.3)
        >>> x = random.randn(2)
        >>> output = m(x)

    .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
        https://arxiv.org/abs/1505.00853
    """
    __module__ = 'brainstate.nn'
    lower: float
    upper: float

    def __init__(
        self,
        lower: float = 1. / 8,
        upper: float = 1. / 3,
    ):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.rrelu(x, self.lower, self.upper)

    def extra_repr(self):
        return f'{self.__class__.__name__}(lower={self.lower}, upper={self.upper})'


class Hardtanh(ElementWiseBlock):
    r"""Applies the HardTanh function element-wise.

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Hardtanh(-2, 2)
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    min_val: float
    max_val: float

    def __init__(
        self,
        min_val: float = -1.,
        max_val: float = 1.,
    ) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        assert self.max_val > self.min_val

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.hard_tanh(x, self.min_val, self.max_val)

    def extra_repr(self) -> str:
        return f'{self.__class__.__name__}(min_val={self.min_val}, max_val={self.max_val})'


class ReLU6(Hardtanh, ElementWiseBlock):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.ReLU6()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __init__(self):
        super().__init__(0., 6.)


class Sigmoid(ElementWiseBlock):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Sigmoid()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.sigmoid(x)


class Hardsigmoid(ElementWiseBlock):
    r"""Applies the Hardsigmoid function element-wise.

    Hardsigmoid is defined as:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Hardsigmoid()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.hard_sigmoid(x)


class Tanh(ElementWiseBlock):
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Tanh()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.tanh(x)


class SiLU(ElementWiseBlock):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> m = nn.SiLU()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.silu(x)


class Mish(ElementWiseBlock):
    r"""Applies the Mish function, element-wise.
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Mish()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.mish(x)


class Hardswish(ElementWiseBlock):
    r"""Applies the Hardswish function, element-wise, as described in the paper:
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    Hardswish is defined as:

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Hardswish()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.hard_swish(x)


class ELU(ElementWiseBlock):
    r"""Applies the Exponential Linear Unit (ELU) function, element-wise, as described
    in the paper: `Fast and Accurate Deep Network Learning by Exponential Linear
    Units (ELUs) <https://arxiv.org/abs/1511.07289>`__.

    ELU is defined as:

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.ELU()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    alpha: float

    def __init__(self, alpha: float = 1.) -> None:
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.elu(x, self.alpha)

    def extra_repr(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'


class CELU(ElementWiseBlock):
    r"""Applies the element-wise function:

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    More details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .

    Args:
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.CELU()
        >>> x = random.randn(2)
        >>> output = m(x)

    .. _`Continuously Differentiable Exponential Linear Units`:
        https://arxiv.org/abs/1704.07483
    """
    __module__ = 'brainstate.nn'
    alpha: float

    def __init__(self, alpha: float = 1.) -> None:
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.celu(x, self.alpha)

    def extra_repr(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'


class SELU(ElementWiseBlock):
    r"""Applied element-wise, as:

    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.SELU()
        >>> x = random.randn(2)
        >>> output = m(x)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.selu(x)


class GLU(ElementWiseBlock):
    r"""Applies the gated linear unit function
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.GLU()
        >>> x = random.randn(4, 2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.glu(x, self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class GELU(ElementWiseBlock):
    r"""Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.GELU()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    approximate: bool

    def __init__(self, approximate: bool = False) -> None:
        super().__init__()
        self.approximate = approximate

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.gelu(x, approximate=self.approximate)

    def __repr__(self):
        return f'{self.__class__.__name__}(approximate={self.approximate})'


class Hardshrink(ElementWiseBlock):
    r"""Applies the Hard Shrinkage (Hardshrink) function element-wise.

    Hardshrink is defined as:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Hardshrink()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.hard_shrink(x, self.lambd)

    def __repr__(self):
        return f'{self.__class__.__name__}(lambd={self.lambd})'


class LeakyReLU(ElementWiseBlock):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)


    or

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope (which is used for
          negative input values). Default: 1e-2

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.LeakyReLU(0.1)
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.leaky_relu(x, self.negative_slope)

    def __repr__(self):
        return f'{self.__class__.__name__}(negative_slope={self.negative_slope})'


class LogSigmoid(ElementWiseBlock):
    r"""Applies the element-wise function:

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.LogSigmoid()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.log_sigmoid(x)


class Softplus(ElementWiseBlock):
    r"""Applies the Softplus function :math:`\text{Softplus}(x) = \frac{1}{\beta} *
    \log(1 + \exp(\beta * x))` element-wise.

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Softplus()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.softplus(x)


class Softshrink(ElementWiseBlock):
    r"""Applies the soft shrinkage function elementwise:

    .. math::
        \text{SoftShrinkage}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` (must be no less than zero) value for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Softshrink()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.soft_shrink(x, self.lambd)

    def __repr__(self):
        return f'{self.__class__.__name__}(lambd={self.lambd})'


class PReLU(ElementWiseBlock):
    r"""Applies the element-wise function:

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    or

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.


    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`( *)` where `*` means, any number of additional
          dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Attributes:
        weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    Examples::

        >>> import brainstate as brainstate
        >>> m = brainstate.nn.PReLU()
        >>> x = brainstate.random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25, dtype=None) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = ParamState(jnp.ones(num_parameters, dtype=dtype) * init)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.prelu(x, self.weight.value)

    def __repr__(self):
        return f'{self.__class__.__name__}(num_parameters={self.num_parameters})'


class Softsign(ElementWiseBlock):
    r"""Applies the element-wise function:

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Softsign()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.soft_sign(x)


class Tanhshrink(ElementWiseBlock):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Tanhshrink()
        >>> x = random.randn(2)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.tanh_shrink(x)


class Softmin(ElementWiseBlock):
    r"""Applies the Softmin function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range `[0, 1]` and sum to 1.

    Softmin is defined as:

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which Softmin will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Softmin(dim=1)
        >>> x = random.randn(2, 3)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.softmin(x, self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class Softmax(ElementWiseBlock):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    When the input Tensor is a sparse tensor then the unspecified
    values are treated as ``-inf``.

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Softmax(dim=1)
        >>> x = random.randn(2, 3)
        >>> output = m(x)

    """
    __module__ = 'brainstate.nn'
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.softmax(x, self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class Softmax2d(ElementWiseBlock):
    r"""Applies SoftMax over features to each spatial location.

    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> x = random.randn(2, 3, 12, 13)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x: ArrayLike) -> ArrayLike:
        assert x.ndim == 4 or x.ndim == 3, 'Softmax2d requires a 3D or 4D tensor as input'
        return F.softmax(x, -3)


class LogSoftmax(ElementWiseBlock):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> import brainstate.nn as nn
        >>> import brainstate as brainstate
        >>> m = nn.LogSoftmax(dim=1)
        >>> x = random.randn(2, 3)
        >>> output = m(x)
    """
    __module__ = 'brainstate.nn'
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return F.log_softmax(x, self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class Identity(ElementWiseBlock):
    r"""A placeholder identity operator that is argument-insensitive.
    """
    __module__ = 'brainstate.nn'

    def __call__(self, x):
        return x


class SpikeBitwise(ElementWiseBlock):
    r"""Bitwise addition for the spiking inputs.

    .. math::

       \begin{array}{ccc}
        \hline \text { Mode } & \text { Expression for } \mathrm{g}(\mathrm{x}, \mathrm{y}) & \text { Code for } \mathrm{g}(\mathrm{x}, \mathrm{y}) \\
        \hline \text { ADD } & x+y & x+y \\
        \text { AND } & x \cap y & x \cdot y \\
        \text { IAND } & (\neg x) \cap y & (1-x) \cdot y \\
        \text { OR } & x \cup y & (x+y)-(x \cdot y) \\
        \hline
        \end{array}

    Args:
      op: str. The bitwise operation.
      name: str. The name of the dynamic system.
    """
    __module__ = 'brainstate.nn'

    def __init__(self,
                 op: str = 'add',
                 name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.op = op

    def __call__(self, x, y):
        return F.spike_bitwise(x, y, self.op)
