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


"""
Shared neural network activations and other functions.
"""

from typing import Any, Union, Sequence

import brainunit as u
import jax
from jax.scipy.special import logsumexp

from brainstate import random
from brainstate.typing import ArrayLike

__all__ = [
    "tanh",
    "relu",
    "squareplus",
    "softplus",
    "soft_sign",
    "sigmoid",
    "silu",
    "swish",
    "log_sigmoid",
    "elu",
    "leaky_relu",
    "hard_tanh",
    "celu",
    "selu",
    "gelu",
    "glu",
    "logsumexp",
    "log_softmax",
    "softmax",
    "standardize",
    "one_hot",
    "relu6",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    'hard_shrink',
    'rrelu',
    'mish',
    'soft_shrink',
    'prelu',
    'tanh_shrink',
    'softmin',
    'sparse_plus',
    'sparse_sigmoid',
]


def tanh(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Hyperbolic tangent activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.
    """
    return u.math.tanh(x)


def softmin(x, axis=-1):
    r"""
    Softmin activation function.

    Applies the Softmin function to an n-dimensional input tensor, rescaling elements
    so that they lie in the range [0, 1] and sum to 1 along the specified axis.

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

    Parameters
    ----------
    x : ArrayLike
        Input array of any shape.
    axis : int, optional
        The axis along which Softmin will be computed. Every slice along this
        dimension will sum to 1. Default is -1.

    Returns
    -------
    jax.Array or Quantity
        Output array with the same shape as the input.
    """
    unnormalized = u.math.exp(-x)
    return unnormalized / unnormalized.sum(axis, keepdims=True)


def tanh_shrink(x):
    r"""
    Tanh shrink activation function.

    Applies the element-wise function:

    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        Output array with the same shape as the input.
    """
    return x - u.math.tanh(x)


def prelu(x, a=0.25):
    r"""
    Parametric Rectified Linear Unit activation function.

    Applies the element-wise function:

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    or equivalently:

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Parameters
    ----------
    x : ArrayLike
        Input array.
    a : float or ArrayLike, optional
        The negative slope coefficient. Can be a learnable parameter.
        Default is 0.25.

    Returns
    -------
    jax.Array or Quantity
        Output array with the same shape as the input.

    Notes
    -----
    When used in neural network layers, :math:`a` can be a learnable parameter
    that is optimized during training.
    """
    return u.math.where(x >= 0., x, a * x)


def soft_shrink(x, lambd=0.5):
    r"""
    Soft shrinkage activation function.

    Applies the soft shrinkage function element-wise:

    .. math::
        \text{SoftShrinkage}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Parameters
    ----------
    x : ArrayLike
        Input array of any shape.
    lambd : float, optional
        The :math:`\lambda` value for the soft shrinkage formulation.
        Must be non-negative. Default is 0.5.

    Returns
    -------
    jax.Array or Quantity
        Output array with the same shape as the input.
    """
    return u.math.where(
        x > lambd,
        x - lambd,
        u.math.where(
            x < -lambd,
            x + lambd,
            u.Quantity(0., unit=u.get_unit(lambd))
        )
    )


def mish(x):
    r"""
    Mish activation function.

    Mish is a self-regularized non-monotonic activation function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    Parameters
    ----------
    x : ArrayLike
        Input array of any shape.

    Returns
    -------
    jax.Array or Quantity
        Output array with the same shape as the input.

    References
    ----------
    .. [1] Misra, D. (2019). "Mish: A Self Regularized Non-Monotonic Activation Function."
           arXiv:1908.08681
    """
    return x * u.math.tanh(softplus(x))


def rrelu(x, lower=0.125, upper=0.3333333333333333):
    r"""
    Randomized Leaky Rectified Linear Unit activation function.

    The function is defined as:

    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})`.

    Parameters
    ----------
    x : ArrayLike
        Input array of any shape.
    lower : float, optional
        Lower bound of the uniform distribution for sampling the negative slope.
        Default is 1/8.
    upper : float, optional
        Upper bound of the uniform distribution for sampling the negative slope.
        Default is 1/3.

    Returns
    -------
    jax.Array or Quantity
        Output array with the same shape as the input.

    References
    ----------
    .. [1] Xu, B., et al. (2015). "Empirical Evaluation of Rectified Activations
           in Convolutional Network." arXiv:1505.00853
    """
    a = random.uniform(lower, upper, size=u.math.shape(x), dtype=x.dtype)
    return u.math.where(u.get_mantissa(x) >= 0., x, a * x)


def hard_shrink(x, lambd=0.5):
    r"""
    Hard shrinkage activation function.

    Applies the hard shrinkage function element-wise:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Parameters
    ----------
    x : ArrayLike
        Input array of any shape.
    lambd : float, optional
        The :math:`\lambda` threshold value for the hard shrinkage formulation.
        Default is 0.5.

    Returns
    -------
    jax.Array or Quantity
        Output array with the same shape as the input.
    """
    return u.math.where(
        x > lambd,
        x,
        u.math.where(
            x < -lambd,
            x,
            u.Quantity(0., unit=u.get_unit(x))
        )
    )


def relu(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Rectified Linear Unit activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{relu}(x) = \max(x, 0)

    Under differentiation, we take:

    .. math::
      \nabla \mathrm{relu}(0) = 0

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate 
        >>> brainstate.nn.relu(jnp.array([-2., -1., -0.5, 0, 0.5, 1., 2.]))
        Array([0. , 0. , 0. , 0. , 0.5, 1. , 2. ], dtype=float32)

    See Also
    --------
    relu6 : ReLU6 activation function.
    leaky_relu : Leaky ReLU activation function.

    References
    ----------
    .. [1] For more information see "Numerical influence of ReLU'(0) on backpropagation"
           https://openreview.net/forum?id=urrcVI-_jRm
    """
    return u.math.relu(x)


def squareplus(x: ArrayLike, b: ArrayLike = 4) -> Union[jax.Array, u.Quantity]:
    r"""
    Squareplus activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{squareplus}(x) = \frac{x + \sqrt{x^2 + b}}{2}

    Parameters
    ----------
    x : ArrayLike
        Input array.
    b : ArrayLike, optional
        Smoothness parameter. Default is 4.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    References
    ----------
    .. [1] So, D., et al. (2021). "Primer: Searching for Efficient Transformers
           for Language Modeling." arXiv:2112.11687
    """
    return u.math.squareplus(x, b=b)


def softplus(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Softplus activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{softplus}(x) = \log(1 + e^x)

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.
    """
    return u.math.softplus(x)


def soft_sign(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Soft-sign activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.
    """
    return u.math.soft_sign(x)


def sigmoid(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Sigmoid activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    log_sigmoid : Logarithm of the sigmoid function.
    """
    return u.math.sigmoid(x)


def silu(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    SiLU (Sigmoid Linear Unit) activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    sigmoid : The sigmoid function.
    swish : Alias for silu.

    Notes
    -----
    `swish` and `silu` are both aliases for the same function.
    """
    return u.math.silu(x)


swish = silu


def log_sigmoid(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Log-sigmoid activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    sigmoid : The sigmoid function.
    """
    return u.math.log_sigmoid(x)


def elu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Union[jax.Array, u.Quantity]:
    r"""
    Exponential Linear Unit activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{elu}(x) = \begin{cases}
        x, & x > 0\\
        \alpha \left(\exp(x) - 1\right), & x \le 0
      \end{cases}

    Parameters
    ----------
    x : ArrayLike
        Input array.
    alpha : ArrayLike, optional
        Scalar or array of alpha values. Default is 1.0.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    selu : Scaled ELU activation function.
    celu : Continuously-differentiable ELU activation function.
    """
    return u.math.elu(x, alpha=alpha)


def leaky_relu(x: ArrayLike, negative_slope: ArrayLike = 1e-2) -> Union[jax.Array, u.Quantity]:
    r"""
    Leaky Rectified Linear Unit activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{leaky\_relu}(x) = \begin{cases}
        x, & x \ge 0\\
        \alpha x, & x < 0
      \end{cases}

    where :math:`\alpha` = :code:`negative_slope`.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    negative_slope : ArrayLike, optional
        Array or scalar specifying the negative slope. Default is 0.01.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    relu : Standard ReLU activation function.
    prelu : Parametric ReLU with learnable slope.
    """
    return u.math.leaky_relu(x, negative_slope=negative_slope)


def _hard_tanh(x, min_val=- 1.0, max_val=1.0):
    return jax.numpy.where(x > max_val, max_val, jax.numpy.where(x < min_val, min_val, x))


def hard_tanh(
    x: ArrayLike,
    min_val: float = - 1.0,
    max_val: float = 1.0
) -> Union[jax.Array, u.Quantity]:
    r"""
    Hard hyperbolic tangent activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{hard\_tanh}(x) = \begin{cases}
        -1, & x < -1\\
        x, & -1 \le x \le 1\\
        1, & 1 < x
      \end{cases}

    Parameters
    ----------
    x : ArrayLike
        Input array.
    min_val : float, optional
        Minimum value of the linear region range. Default is -1.
    max_val : float, optional
        Maximum value of the linear region range. Default is 1.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.
    """
    x = u.Quantity(x)
    min_val = u.Quantity(min_val).to(x.unit).mantissa
    max_val = u.Quantity(max_val).to(x.unit).mantissa
    return u.maybe_decimal(_hard_tanh(x.mantissa, min_val=min_val, max_val=max_val) * x.unit)


def celu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Union[jax.Array, u.Quantity]:
    r"""
    Continuously-differentiable Exponential Linear Unit activation.

    Computes the element-wise function:

    .. math::
      \mathrm{celu}(x) = \begin{cases}
        x, & x > 0\\
        \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
      \end{cases}

    Parameters
    ----------
    x : ArrayLike
        Input array.
    alpha : ArrayLike, optional
        Scalar or array value controlling the smoothness. Default is 1.0.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    References
    ----------
    .. [1] Barron, J. T. (2017). "Continuously Differentiable Exponential Linear Units."
           arXiv:1704.07483
    """
    return u.math.celu(x, alpha=alpha)


def selu(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Scaled Exponential Linear Unit activation.

    Computes the element-wise function:

    .. math::
      \mathrm{selu}(x) = \lambda \begin{cases}
        x, & x > 0\\
        \alpha e^x - \alpha, & x \le 0
      \end{cases}

    where :math:`\lambda = 1.0507009873554804934193349852946` and
    :math:`\alpha = 1.6732632423543772848170429916717`.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    elu : Exponential Linear Unit activation function.

    References
    ----------
    .. [1] Klambauer, G., et al. (2017). "Self-Normalizing Neural Networks."
           NeurIPS 2017.
    """
    return u.math.selu(x)


def gelu(x: ArrayLike, approximate: bool = True) -> Union[jax.Array, u.Quantity]:
    r"""
    Gaussian Error Linear Unit activation function.

    If ``approximate=False``, computes the element-wise function:

    .. math::
      \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{erf} \left(
        \frac{x}{\sqrt{2}} \right) \right)

    If ``approximate=True``, uses the approximate formulation of GELU:

    .. math::
      \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
        \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

    Parameters
    ----------
    x : ArrayLike
        Input array.
    approximate : bool, optional
        Whether to use the approximate (True) or exact (False) formulation.
        Default is True.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    References
    ----------
    .. [1] Hendrycks, D., & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)."
           arXiv:1606.08415
    """
    return u.math.gelu(x, approximate=approximate)


def glu(x: ArrayLike, axis: int = -1) -> Union[jax.Array, u.Quantity]:
    r"""
    Gated Linear Unit activation function.

    Computes the function:

    .. math::
      \mathrm{glu}(x) =  x\left[\ldots, 0:\frac{n}{2}, \ldots\right] \cdot
        \mathrm{sigmoid} \left( x\left[\ldots, \frac{n}{2}:n, \ldots\right]
          \right)

    where the array is split into two along ``axis``. The size of the ``axis``
    dimension must be divisible by two.

    Parameters
    ----------
    x : ArrayLike
        Input array. The dimension specified by ``axis`` must be divisible by 2.
    axis : int, optional
        The axis along which the split should be computed. Default is -1.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as input except the ``axis`` dimension
        is halved.

    See Also
    --------
    sigmoid : The sigmoid activation function.
    """
    return u.math.glu(x, axis=axis)


def log_softmax(x: ArrayLike,
                axis: int | tuple[int, ...] | None = -1,
                where: ArrayLike | None = None) -> Union[jax.Array, u.Quantity]:
    r"""
    Log-Softmax function.

    Computes the logarithm of the softmax function, which rescales
    elements to the range :math:`[-\infty, 0)`.

    .. math ::
      \mathrm{log\_softmax}(x)_i = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
      \right)

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axis : int or tuple of int, optional
        The axis or axes along which the log-softmax should be computed.
        Either an integer or a tuple of integers. Default is -1.
    where : ArrayLike, optional
        Elements to include in the log-softmax computation.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    softmax : The softmax function.
    """
    return jax.nn.log_softmax(x, axis=axis, where=where)


def softmax(x: ArrayLike,
            axis: int | tuple[int, ...] | None = -1,
            where: ArrayLike | None = None) -> Union[jax.Array, u.Quantity]:
    r"""
    Softmax activation function.

    Computes the function which rescales elements to the range :math:`[0, 1]`
    such that the elements along :code:`axis` sum to :math:`1`.

    .. math ::
      \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axis : int or tuple of int, optional
        The axis or axes along which the softmax should be computed. The
        softmax output summed across these dimensions should sum to :math:`1`.
        Either an integer or a tuple of integers. Default is -1.
    where : ArrayLike, optional
        Elements to include in the softmax computation.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    log_softmax : Logarithm of the softmax function.
    softmin : Softmin activation function.
    """
    return jax.nn.softmax(x, axis=axis, where=where)


def standardize(x: ArrayLike,
                axis: int | tuple[int, ...] | None = -1,
                variance: ArrayLike | None = None,
                epsilon: ArrayLike = 1e-5,
                where: ArrayLike | None = None) -> Union[jax.Array, u.Quantity]:
    r"""
    Standardize (normalize) an array.

    Normalizes an array by subtracting the mean and dividing by the standard
    deviation :math:`\sqrt{\mathrm{variance}}`.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axis : int or tuple of int, optional
        The axis or axes along which to compute the mean and variance.
        Default is -1.
    variance : ArrayLike, optional
        Pre-computed variance. If None, variance is computed from ``x``.
    epsilon : ArrayLike, optional
        A small constant added to the variance to avoid division by zero.
        Default is 1e-5.
    where : ArrayLike, optional
        Elements to include in the computation.

    Returns
    -------
    jax.Array or Quantity
        Standardized array with the same shape as the input.
    """
    return jax.nn.standardize(x, axis=axis, where=where, variance=variance, epsilon=epsilon)


def one_hot(x: Any,
            num_classes: int, *,
            dtype: Any = jax.numpy.float_,
            axis: Union[int, Sequence[int]] = -1) -> Union[jax.Array, u.Quantity]:
    """
    One-hot encode the given indices.

    Each index in the input ``x`` is encoded as a vector of zeros of length
    ``num_classes`` with the element at ``index`` set to one.

    Indices outside the range [0, num_classes) will be encoded as zeros.

    Parameters
    ----------
    x : ArrayLike
        A tensor of indices.
    num_classes : int
        Number of classes in the one-hot dimension.
    dtype : dtype, optional
        The dtype for the returned values. Default is ``jnp.float_``.
    axis : int or Sequence of int, optional
        The axis or axes along which the function should be computed.
        Default is -1.

    Returns
    -------
    jax.Array or Quantity
        One-hot encoded array.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate 
        >>> brainstate.nn.one_hot(jnp.array([0, 1, 2]), 3)
        Array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)

        >>> # Indices outside the range are encoded as zeros
        >>> brainstate.nn.one_hot(jnp.array([-1, 3]), 3)
        Array([[0., 0., 0.],
               [0., 0., 0.]], dtype=float32)
    """
    return jax.nn.one_hot(x, axis=axis, num_classes=num_classes, dtype=dtype)


def relu6(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Rectified Linear Unit 6 activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{relu6}(x) = \min(\max(x, 0), 6)

    Under differentiation, we take:

    .. math::
      \nabla \mathrm{relu}(0) = 0

    and

    .. math::
      \nabla \mathrm{relu}(6) = 0

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    relu : Standard ReLU activation function.
    """
    return u.math.relu6(x)


def hard_sigmoid(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Hard Sigmoid activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    relu6 : ReLU6 activation function.
    sigmoid : Standard sigmoid function.
    """
    return u.math.hard_sigmoid(x)


def hard_silu(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Hard SiLU (Swish) activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    hard_sigmoid : Hard sigmoid activation function.
    silu : Standard SiLU activation function.
    hard_swish : Alias for hard_silu.

    Notes
    -----
    Both `hard_silu` and `hard_swish` are aliases for the same function.
    """
    return u.math.hard_silu(x)


hard_swish = hard_silu


def sparse_plus(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Sparse plus activation function.

    Computes the function:

    .. math::

      \mathrm{sparse\_plus}(x) = \begin{cases}
        0, & x \leq -1\\
        \frac{1}{4}(x+1)^2, & -1 < x < 1 \\
        x, & 1 \leq x
      \end{cases}

    This is the twin function of the softplus activation, ensuring a zero output
    for inputs less than -1 and a linear output for inputs greater than 1,
    while remaining smooth, convex, and monotonic between -1 and 1.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    sparse_sigmoid : Derivative of sparse_plus.
    softplus : Standard softplus activation function.
    """
    return u.math.sparse_plus(x)


def sparse_sigmoid(x: ArrayLike) -> Union[jax.Array, u.Quantity]:
    r"""
    Sparse sigmoid activation function.

    Computes the function:

    .. math::

      \mathrm{sparse\_sigmoid}(x) = \begin{cases}
        0, & x \leq -1\\
        \frac{1}{2}(x+1), & -1 < x < 1 \\
        1, & 1 \leq x
      \end{cases}

    This is the twin function of the standard sigmoid activation, ensuring a zero
    output for inputs less than -1, a 1 output for inputs greater than 1, and a
    linear output for inputs between -1 and 1. It is the derivative of `sparse_plus`.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    jax.Array or Quantity
        An array with the same shape as the input.

    See Also
    --------
    sigmoid : Standard sigmoid activation function.
    sparse_plus : Sparse plus activation function.

    References
    ----------
    .. [1] Martins, A. F. T., & Astudillo, R. F. (2016). "From Softmax to Sparsemax:
           A Sparse Model of Attention and Multi-Label Classification."
           In ICML. See also "Learning with Fenchel-Young Losses", arXiv:1901.02324
    """
    return u.math.sparse_sigmoid(x)
