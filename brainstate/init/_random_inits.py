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

import math

import brainunit as u
import jax.numpy as jnp
import numpy as np

from brainstate import environ, random
from brainstate.typing import ArrayLike, SeedOrKey, DTypeLike
from ._base import Initializer, to_size

__all__ = [
    'Normal',
    'TruncatedNormal',
    'Uniform',
    'VarianceScaling',
    'KaimingUniform',
    'KaimingNormal',
    'XavierUniform',
    'XavierNormal',
    'LecunUniform',
    'LecunNormal',
    'Orthogonal',
    'DeltaOrthogonal',
]


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalisation
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _format_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    if len(shape) == 0:
        raise ValueError('Please provide shape.')
    if len(shape) == 1:
        if isinstance(shape[0], (tuple, list)):
            return shape[0]
        else:
            return shape
    else:
        return shape


def _compute_fans(shape, in_axis=-2, out_axis=-1):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out


class Normal(Initializer):
    """Initialize weights with normal distribution.

    Parameters
    ----------
    scale : float
      The gain of the derivation of the normal distribution.

    """
    __module__ = 'brainstate.init'

    def __init__(
        self,
        mean: ArrayLike = 0.,
        scale: ArrayLike = 1.,
        unit: u.Unit = u.UNITLESS,
        seed: SeedOrKey = None
    ):
        super().__init__()
        self.scale = scale
        self.mean = mean
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None):
        shape = to_size(shape)
        dtype = dtype or environ.dftype()
        weights = self.rng.normal(size=shape, loc=self.mean, scale=self.scale, dtype=dtype)
        return u.maybe_decimal(u.Quantity(weights, unit=self.unit))


class TruncatedNormal(Initializer):
    """Initialize weights with truncated normal distribution.

    Parameters
    ----------
    loc : float, ndarray
      Mean ("centre") of the distribution before truncating. Note that
      the mean of the truncated distribution will not be exactly equal
      to ``loc``.
    scale : float
      The standard deviation of the normal distribution before truncating.
    lower : float, ndarray
      A float or array of floats representing the lower bound for
        truncation. Must be broadcast-compatible with ``upper``.
    upper : float, ndarray
      A float or array of floats representing the  upper bound for
      truncation. Must be broadcast-compatible with ``lower``.

    """
    __module__ = 'brainstate.init'

    def __init__(
        self,
        loc: ArrayLike = 0.,
        scale: ArrayLike = 1.,
        unit: u.Unit = u.UNITLESS,
        lower: ArrayLike = None,
        upper: ArrayLike = None,
        seed: SeedOrKey = None,
    ):
        super().__init__()
        assert scale > 0, '`scale` must be positive.'
        self.scale = scale
        self.loc = loc
        self.lower = lower
        self.upper = upper
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        dtype = dtype or environ.dftype()
        weights = self.rng.truncated_normal(
            size=shape,
            scale=self.scale,
            lower=self.lower,
            upper=self.upper,
            loc=self.loc,
            dtype=dtype
        )
        return u.maybe_decimal(u.Quantity(weights, unit=self.unit))


class Gamma(Initializer):
    """Initialize weights with Gamma distribution.

    Parameters
    ----------
    shape: float, Array
      Shape parameter.
    scale: float, Array
      The gain of the derivation of the Gamma distribution.

    """
    __module__ = 'brainstate.init'

    def __init__(
        self,
        shape: ArrayLike,
        unit: u.Unit = u.UNITLESS,
        scale: ArrayLike = None,
        seed: SeedOrKey = None
    ):
        self.shape = shape
        self.scale = scale
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        shape = to_size(shape)
        dtype = dtype or environ.dftype()
        weights = self.rng.gamma(self.shape, scale=self.scale, size=shape, dtype=dtype)
        return u.maybe_decimal(u.Quantity(weights, unit=self.unit))


class Exponential(Initializer):
    """Initialize weights with Gamma distribution.

    Parameters
    ----------
    scale: float, Array
      The gain of the derivation of the Exponential distribution.

    """
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: ArrayLike = None,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        self.scale = scale
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        shape = to_size(shape)
        dtype = dtype or environ.dftype()
        weights = self.rng.exponential(scale=self.scale, size=shape, dtype=dtype)
        return u.maybe_decimal(u.Quantity(weights, unit=self.unit))


class Uniform(Initializer):
    """Initialize weights with uniform distribution.

    Parameters
    ----------
    min_val : float
      The lower limit of the uniform distribution.
    max_val : float
      The upper limit of the uniform distribution.
    """
    __module__ = 'brainstate.init'

    def __init__(
        self,
        min_val: ArrayLike = 0.,
        max_val: ArrayLike = 1.,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super(Uniform, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        shape = to_size(shape)
        dtype = dtype or environ.dftype()
        weights = self.rng.uniform(low=self.min_val, high=self.max_val, size=shape, dtype=dtype)
        return u.maybe_decimal(u.Quantity(weights, unit=self.unit))


class VarianceScaling(Initializer):
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: ArrayLike,
        mode: str,
        distribution: str,
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        assert mode in ['fan_in', 'fan_out', 'fan_avg']
        assert distribution in ['truncated_normal', 'normal', 'uniform']
        self.scale = scale
        self.mode = mode
        self.in_axis = in_axis
        self.out_axis = out_axis
        self.distribution = distribution
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        shape = to_size(shape)
        dtype = dtype or environ.dftype()
        fan_in, fan_out = _compute_fans(shape, in_axis=self.in_axis, out_axis=self.out_axis)
        if self.mode == "fan_in":
            denominator = fan_in
        elif self.mode == "fan_out":
            denominator = fan_out
        elif self.mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError("invalid mode for variance scaling initializer: {}".format(self.mode))
        variance = (self.scale / denominator).astype(dtype)
        if self.distribution == "truncated_normal":
            stddev = (jnp.sqrt(variance) / .87962566103423978).astype(dtype)
            res = self.rng.truncated_normal(-2, 2, shape, dtype=dtype) * stddev
        elif self.distribution == "normal":
            res = self.rng.randn(*shape, dtype=dtype) * jnp.sqrt(variance).astype(dtype)
        elif self.distribution == "uniform":
            res = (self.rng.uniform(low=-1, high=1, size=shape, dtype=dtype) *
                   jnp.sqrt(3 * variance).astype(dtype))
        else:
            raise ValueError("invalid distribution for variance scaling initializer")
        return u.maybe_decimal(u.Quantity(res, unit=self.unit))


class KaimingUniform(VarianceScaling):
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: float = 2.0,
        mode: str = "fan_in",
        distribution: str = "uniform",
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__(scale,
                         mode,
                         distribution,
                         in_axis=in_axis,
                         out_axis=out_axis,
                         seed=seed,
                         unit=unit)


class KaimingNormal(VarianceScaling):
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: float = 2.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__(scale,
                         mode,
                         distribution,
                         in_axis=in_axis,
                         out_axis=out_axis,
                         seed=seed,
                         unit=unit)


class XavierUniform(VarianceScaling):
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_avg",
        distribution: str = "uniform",
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__(scale,
                         mode,
                         distribution,
                         in_axis=in_axis,
                         out_axis=out_axis,
                         seed=seed,
                         unit=unit)


class XavierNormal(VarianceScaling):
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_avg",
        distribution: str = "truncated_normal",
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__(scale,
                         mode,
                         distribution,
                         in_axis=in_axis,
                         out_axis=out_axis,
                         seed=seed,
                         unit=unit)


class LecunUniform(VarianceScaling):
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "uniform",
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__(scale,
                         mode,
                         distribution,
                         in_axis=in_axis,
                         out_axis=out_axis,
                         seed=seed,
                         unit=unit)


class LecunNormal(VarianceScaling):
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__(scale,
                         mode,
                         distribution,
                         in_axis=in_axis,
                         out_axis=out_axis,
                         seed=seed,
                         unit=unit)


class Orthogonal(Initializer):
    """
    Construct an initializer for uniformly distributed orthogonal matrices.

    If the shape is not square, the matrix will have orthonormal rows or columns
    depending on which side is smaller.
    """
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: ArrayLike = 1.,
        axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__()
        self.scale = scale
        self.axis = axis
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        dtype = dtype or environ.dftype()
        shape = to_size(shape)
        n_rows = shape[self.axis]
        n_cols = np.prod(shape) // n_rows
        matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
        norm_dst = self.rng.normal(size=matrix_shape, dtype=dtype)

        q_mat, r_mat = jnp.linalg.qr(norm_dst)
        # Enforce Q is uniformly distributed
        q_mat *= jnp.sign(jnp.diag(r_mat))
        if n_rows < n_cols:
            q_mat = q_mat.T
        q_mat = jnp.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, self.axis)))
        q_mat = jnp.moveaxis(q_mat, 0, self.axis)
        r = jnp.asarray(self.scale, dtype=dtype) * q_mat
        return u.maybe_decimal(u.Quantity(r, unit=self.unit))


class DeltaOrthogonal(Initializer):
    """
    Construct an initializer for delta orthogonal kernels; see arXiv:1806.05393.

    The shape must be 3D, 4D or 5D.
    """
    __module__ = 'brainstate.init'

    def __init__(
        self,
        scale: ArrayLike = 1.0,
        axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__()
        self.scale = scale
        self.axis = axis
        self.orghogonal = Orthogonal(scale=scale, axis=axis, seed=seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        shape = to_size(shape)
        dtype = dtype or environ.dftype()
        if len(shape) not in [3, 4, 5]:
            raise ValueError("Delta orthogonal initializer requires a 3D, 4D or 5D shape.")
        if shape[-1] < shape[-2]:
            raise ValueError("`fan_in` must be less or equal than `fan_out`. ")
        ortho_matrix = u.Quantity(self.orghogonal(shape[-2:]))
        W = u.Quantity(u.math.zeros(shape, dtype=dtype), unit=u.get_unit(ortho_matrix))
        if len(shape) == 3:
            k = shape[0]
            W = W.at[(k - 1) // 2].set(ortho_matrix)
        elif len(shape) == 4:
            k1, k2 = shape[:2]
            W = W.at[(k1 - 1) // 2, (k2 - 1) // 2].set(ortho_matrix)
        else:
            k1, k2, k3 = shape[:3]
            W = W.at[(k1 - 1) // 2, (k2 - 1) // 2, (k3 - 1) // 2].set(ortho_matrix)
        return u.maybe_decimal(u.Quantity(W.mantissa, unit=self.unit))
