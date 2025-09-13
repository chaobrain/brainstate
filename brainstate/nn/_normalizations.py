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

from typing import Callable, Union, Sequence, Optional, Any

import jax
import jax.numpy as jnp

import brainunit as u
from brainstate import environ, init
from brainstate._state import ParamState, BatchState
from brainstate.typing import DTypeLike, ArrayLike, Size, Axes
from ._module import Module

__all__ = [
    'BatchNorm0d',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'LayerNorm',
    'RMSNorm',
    'GroupNorm',
]


def weight_standardization(
    w: ArrayLike,
    eps: float = 1e-4,
    gain: Optional[jax.Array] = None,
    out_axis: int = -1,
) -> Union[jax.Array, u.Quantity]:
    """
    Scaled Weight Standardization,
    see `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization <https://paperswithcode.com/paper/weight-standardization>`_.

    Parameters
    ----------
    w : ArrayLike
        The weight tensor.
    eps : float
        A small value to avoid division by zero.
    gain : Array
        The gain function, by default None.
    out_axis : int
        The output axis, by default -1.

    Returns
    -------
    ArrayLike
        The scaled weight tensor.
    """
    w = u.maybe_custom_array(w)
    if out_axis < 0:
        out_axis = w.ndim + out_axis
    fan_in = 1  # get the fan-in of the weight tensor
    axes = []  # get the axes of the weight tensor
    for i in range(w.ndim):
        if i != out_axis:
            fan_in *= w.shape[i]
            axes.append(i)
    # normalize the weight
    mean = u.math.mean(w, axis=axes, keepdims=True)
    var = u.math.var(w, axis=axes, keepdims=True)

    temp = u.math.maximum(var * fan_in, eps)
    if isinstance(temp, u.Quantity):
        unit = temp.unit
        temp = temp.mantissa
        if unit.is_unitless:
            scale = jax.lax.rsqrt(temp)
        else:
            scale = u.Quantity(jax.lax.rsqrt(temp), unit=1 / unit ** 0.5)
    else:
        scale = jax.lax.rsqrt(temp)
    if gain is not None:
        scale = gain * scale
    shift = mean * scale
    return w * scale - shift



def canonicalize_dtype(
    *args,
    dtype: jax.typing.DTypeLike | None = None,
    inexact: bool = True
) -> jax.typing.DTypeLike:
    """Canonicalize an optional dtype to the definitive dtype.

    If the ``dtype`` is None this function will infer the dtype. If it is not
    None it will be returned unmodified or an exceptions is raised if the dtype
    is invalid.
    from the input arguments using ``jnp.result_type``.

    Args:
      *args: JAX array compatible values. None values
        are ignored.
      dtype: Optional dtype override. If specified the arguments are cast to
        the specified dtype instead and dtype inference is disabled.
      inexact: When True, the output dtype must be a subdtype
      of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
      is useful when you want to apply operations that don't work directly on
      integers like taking a mean for example.
    Returns:
      The dtype that *args should be cast to.
    """
    if dtype is None:
        args_filtered = [jnp.asarray(x) for x in args if x is not None]
        dtype = jnp.result_type(*args_filtered)
        if inexact and not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.promote_types(jnp.float32, dtype)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
        raise ValueError(f'Dtype must be inexact: {dtype}')
    return dtype


def _canonicalize_axes(ndim: int, feature_axes: Sequence[int]):
    axes = []
    for axis in feature_axes:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f'Invalid axis {axis} for {ndim}D input')
        axes.append(axis)
    return tuple(axes)


def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
        return jax.lax.square(jax.lax.real(x)) + jax.lax.square(jax.lax.imag(x))
    else:
        return jax.lax.square(x)


class NormalizationParamState(ParamState):
    # This is a dummy class to be used as a compatibility
    # usage of `ETraceParam` for the layers in "brainetrace"
    def execute(self, x):
        param = self.value
        if 'scale' in param:
            x = x * param['scale']
        if 'bias' in param:
            x = x + param['bias']
        return x


def _compute_stats(
    x: ArrayLike,
    axes: Sequence[int],
    dtype: DTypeLike,
    axis_name: Optional[str] = None,
    axis_index_groups: Optional[Sequence[int]] = None,
    use_mean: bool = True,
    use_fast_variance: bool = True,
    mask: Optional[jax.Array] = None,
):
    """
    Computes mean and variance statistics.

    This implementation takes care of a few important details:
    - Computes in float32 precision for stability in half precision training.
    - If ``use_fast_variance`` is ``True``, mean and variance are computed using
      Var = E[|x|^2] - |E[x]|^2, instead of Var = E[|x - E[x]|^2]), in a single XLA fusion.
    - Clips negative variances to zero which can happen due to
      roundoff errors. This avoids downstream NaNs.
    - Supports averaging across a parallel axis and subgroups of a parallel axis
      with a single `lax.pmean` call to avoid latency.

    Arguments:
        x: Input array.
        axes: The axes in ``x`` to compute mean and variance statistics for.
        dtype: tp.Optional dtype specifying the minimal precision. Statistics
            are always at least float32 for stability (default: dtype of x).
        axis_name: Optional name for the pmapped axis to compute mean over. Note,
            this is only used for pmap and shard map. For SPMD jit, you do not need to
            manually synchronize. Just make sure that the axes are correctly annotated
            and XLA:SPMD will insert the necessary collectives.
        axis_index_groups: Optional axis indices.
        use_mean: If true, calculate the mean from the input and use it when
            computing the variance. If false, set the mean to zero and compute
            the variance without subtracting the mean.
        use_fast_variance: If true, use a faster, but less numerically stable,
            calculation for the variance.
        mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.

    Returns:
      A pair ``(mean, val)``.
    """
    if dtype is None:
        dtype = jax.numpy.result_type(x)
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jax.numpy.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)
    axes = _canonicalize_axes(x.ndim, axes)

    def maybe_distributed_mean(*xs, mask=None):
        mus = tuple(x.mean(axes, where=mask) for x in xs)
        if axis_name is None:
            return mus if len(xs) > 1 else mus[0]
        else:
            # In the distributed case we stack multiple arrays to speed comms.
            if len(xs) > 1:
                reduced_mus = jax.lax.pmean(
                    jnp.stack(mus, axis=0),
                    axis_name,
                    axis_index_groups=axis_index_groups,
                )
                return tuple(reduced_mus[i] for i in range(len(xs)))
            else:
                return jax.lax.pmean(
                    mus[0],
                    axis_name,
                    axis_index_groups=axis_index_groups
                )

    if use_mean:
        if use_fast_variance:
            mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
            # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
            # to floating point round-off errors.
            var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
        else:
            mu = maybe_distributed_mean(x, mask=mask)
            var = maybe_distributed_mean(_abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask)
    else:
        var = maybe_distributed_mean(_abs_sq(x), mask=mask)
        mu = jnp.zeros_like(var)
    return mu, var


def _normalize(
    x: ArrayLike,
    mean: Optional[ArrayLike],
    var: Optional[ArrayLike],
    weights: Optional[NormalizationParamState],
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: DTypeLike,
    epsilon: jax.typing.ArrayLike,
):
    """Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

    Arguments:
      x: The input.
      mean: Mean to use for normalization.
      var: Variance to use for normalization.
      weights: The scale and bias parameters.
      reduction_axes: The axes in ``x`` to reduce.
      feature_axes: The feature axes to apply the scale and bias.
      dtype: The dtype of the result (default: infer from input and params).
      epsilon: Normalization epsilon.

    Returns:
      The normalized input.
    """
    if mean is not None:
        assert var is not None, 'mean and val must be both None or not None.'
        reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
        feature_axes = _canonicalize_axes(x.ndim, feature_axes)
        stats_shape = list(x.shape)
        for axis in reduction_axes:
            stats_shape[axis] = 1
        mean = mean.reshape(stats_shape)
        var = var.reshape(stats_shape)
        feature_shape = [1] * x.ndim
        for ax in feature_axes:
            feature_shape[ax] = x.shape[ax]
        y = x - mean
        mul = jax.lax.rsqrt(var + epsilon)
        y = y * mul
        if weights is not None:
            y = weights.execute(y)
            dtype = canonicalize_dtype(x, *jax.tree.leaves(weights.value), dtype=dtype)
    else:
        assert var is None, 'mean and val must be both None or not None.'
        assert weights is None, 'scale and bias are not supported without mean and val'
        y = x
    return jnp.asarray(y, dtype)


class _BatchNorm(Module):
    __module__ = 'brainstate.nn'
    num_spatial_dims: int

    def __init__(
        self,
        in_size: Size,
        feature_axis: Axes = -1,
        *,
        track_running_stats: bool = True,
        epsilon: float = 1e-5,
        momentum: float = 0.99,
        affine: bool = True,
        bias_initializer: Union[ArrayLike, Callable] = init.Constant(0.),
        scale_initializer: Union[ArrayLike, Callable] = init.Constant(1.),
        axis_name: Optional[Union[str, Sequence[str]]] = None,
        axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
        use_fast_variance: bool = True,
        name: Optional[str] = None,
        dtype: Any = None,
        param_type: type = NormalizationParamState,
        mean_type: type = BatchState,
    ):
        super().__init__(name=name)

        # parameters
        self.in_size = in_size
        self.out_size = in_size
        self.affine = affine
        self.bias_initializer = bias_initializer
        self.scale_initializer = scale_initializer
        self.dtype = dtype or environ.dftype()
        self.track_running_stats = track_running_stats
        self.momentum = jnp.asarray(momentum, dtype=self.dtype)
        self.epsilon = jnp.asarray(epsilon, dtype=self.dtype)
        self.use_fast_variance = use_fast_variance

        # parameters about axis
        feature_axis = (feature_axis,) if isinstance(feature_axis, int) else feature_axis
        self.feature_axes = _canonicalize_axes(len(self.in_size), feature_axis)
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups

        # variables
        feature_shape = tuple([(ax if i in self.feature_axes else 1)
                               for i, ax in enumerate(self.in_size)])
        if self.track_running_stats:
            self.running_mean = mean_type(jnp.zeros(feature_shape, dtype=self.dtype))
            self.running_var = mean_type(jnp.ones(feature_shape, dtype=self.dtype))
        else:
            self.running_mean = None
            self.running_var = None

        # parameters
        if self.affine:
            assert track_running_stats, "Affine parameters are not needed when track_running_stats is False."
            bias = init.param(self.bias_initializer, feature_shape)
            scale = init.param(self.scale_initializer, feature_shape)
            self.weight = param_type(dict(bias=bias, scale=scale))
        else:
            self.weight = None

    def update(self, x, mask: Optional[jax.Array] = None):
        # input shape and batch mode or not
        if x.ndim == self.num_spatial_dims + 2:
            x_shape = x.shape[1:]
            batch = True
        elif x.ndim == self.num_spatial_dims + 1:
            x_shape = x.shape
            batch = False
        else:
            raise ValueError(f"expected {self.num_spatial_dims + 2}D (with batch) or "
                             f"{self.num_spatial_dims + 1}D (without batch) input (got {x.ndim}D input, {x.shape})")
        if self.in_size != x_shape:
            raise ValueError(f"The expected input shape is {self.in_size}, while we got {x_shape}.")

        # reduce the feature axis
        if batch:
            reduction_axes = tuple(i for i in range(x.ndim) if (i - 1) not in self.feature_axes)
        else:
            reduction_axes = tuple(i for i in range(x.ndim) if i not in self.feature_axes)

        # fitting phase
        fit_phase = environ.get('fit', desc='Whether this is a fitting process. Bool.')

        # compute the running mean and variance
        if self.track_running_stats:
            if fit_phase:
                mean, var = _compute_stats(
                    x,
                    reduction_axes,
                    dtype=self.dtype,
                    axis_name=self.axis_name,
                    axis_index_groups=self.axis_index_groups,
                    use_fast_variance=self.use_fast_variance,
                    mask=mask,
                )
                self.running_mean.value = self.momentum * self.running_mean.value + (1 - self.momentum) * mean
                self.running_var.value = self.momentum * self.running_var.value + (1 - self.momentum) * var
            else:
                mean = self.running_mean.value
                var = self.running_var.value
        else:
            mean, var = None, None

        # normalize
        return _normalize(
            x,
            mean=mean,
            var=var,
            weights=self.weight,
            reduction_axes=reduction_axes,
            feature_axes=self.feature_axes,
            dtype=self.dtype,
            epsilon=self.epsilon
        )


class BatchNorm0d(_BatchNorm):
    r"""0-D batch normalization [1]_.

    The data should be of `(b, c)`, where `b` is the batch dimension, and `c` is the channel dimension.

    %s
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 0


class BatchNorm1d(_BatchNorm):
    r"""1-D batch normalization [1]_.

    The data should be of `(b, l, c)`, where `b` is the batch dimension,
    `l` is the layer dimension, and `c` is the channel dimension.

    %s
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 1


class BatchNorm2d(_BatchNorm):
    r"""2-D batch normalization [1]_.

    The data should be of `(b, h, w, c)`, where `b` is the batch dimension,
    `h` is the height dimension, `w` is the width dimension, and `c` is the
    channel dimension.

    %s
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 2


class BatchNorm3d(_BatchNorm):
    r"""3-D batch normalization [1]_.

    The data should be of `(b, h, w, d, c)`, where `b` is the batch dimension,
    `h` is the height dimension, `w` is the width dimension, `d` is the depth
    dimension, and `c` is the channel dimension.

    %s
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 3


_bn_doc = r'''

  This layer aims to reduce the internal covariant shift of data. It
  normalizes a batch of data by fixing the mean and variance of inputs
  on each feature (channel). Most commonly, the first axis of the data
  is the batch, and the last is the channel. However, users can specify
  the axes to be normalized.

  .. math::
     y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta

  .. note::
      This :attr:`momentum` argument is different from one used in optimizer
      classes and the conventional notion of momentum. Mathematically, the
      update rule for running statistics here is
      :math:`\hat{x}_\text{new} = \text{momentum} \times \hat{x} + (1-\text{momentum}) \times x_t`,
      where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
      new observed value.

  Parameters
  ----------
  in_size: sequence of int
    The input shape, without batch size.
  feature_axis: int, tuple, list
    The feature or non-batch axis of the input.
  track_running_stats: bool
    A boolean value that when set to ``True``, this module tracks the running mean and variance, 
    and when set to ``False``, this module does not track such statistics, and initializes 
    statistics buffers ``running_mean`` and ``running_var`` as ``None``. When these buffers are ``None``, 
    this module always uses batch statistics. in both training and eval modes. Default: ``True``.
  momentum: float
    The value used for the ``running_mean`` and ``running_var`` computation. Default: 0.99
  epsilon: float
    A value added to the denominator for numerical stability. Default: 1e-5
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_initializer: ArrayLike, Callable
    An initializer generating the original translation matrix. If not ``None``, bias (beta) is added. 
    Default: ``init.Constant(0.)``
  scale_initializer: ArrayLike, Callable
    An initializer generating the original scaling matrix. If not ``None``, multiply by scale (gamma).
    Default: ``init.Constant(1.)``
  axis_name: optional, str, sequence of str
    If not ``None``, it should be a string (or sequence of
    strings) representing the axis name(s) over which this module is being
    run within a jax map (e.g. ``jax.pmap`` or ``jax.vmap``). Supplying this
    argument means that batch statistics are calculated across all replicas
    on the named axes.
  axis_index_groups: optional, sequence
    Specifies how devices are grouped. Valid
    only within ``jax.pmap`` collectives.
    Groups of axis indices within that named axis
    representing subsets of devices to reduce over (default: None). For
    example, `[[0, 1], [2, 3]]` would independently batch-normalize over
    the examples on the first two and last two devices. See `jax.lax.psum`
    for more details.
  use_fast_variance: If true, use a faster, but less numerically stable,
    calculation for the variance.
    
    
  References
  ----------
  .. [1] Ioffe, Sergey and Christian Szegedy. “Batch Normalization: Accelerating Deep Network Training
         by Reducing Internal Covariate Shift.” ArXiv abs/1502.03167 (2015): n. pag.

'''

BatchNorm1d.__doc__ = BatchNorm1d.__doc__ % _bn_doc
BatchNorm2d.__doc__ = BatchNorm2d.__doc__ % _bn_doc
BatchNorm3d.__doc__ = BatchNorm3d.__doc__ % _bn_doc


class LayerNorm(Module):
    """
    Layer normalization (https://arxiv.org/abs/1607.06450).

    LayerNorm normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Example usage::

      >>> import brainstate as brainstate
      >>> x = brainstate.random.normal(size=(3, 4, 5, 6))
      >>> layer = brainstate.nn.LayerNorm(x.shape)
      >>> layer.states()
      >>> y = layer(x)

    Attributes:
      in_size: The input shape, without batch size.
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
          (also e.g. nnx.relu), this can be disabled since the scaling will be done
          by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
      reduction_axes: Axes for computing normalization statistics. It is recommended
            to use the negative integer, since when the batch dimension is used,
            the reduction_axes may be wrong when using the positive integer.
      feature_axes: Feature axes for learned bias and scaling.
      axis_name: the axis name used to combine batch statistics from multiple
          devices. See ``jax.pmap`` for a description of axis names (default: None).
          This is only needed if the model is subdivided across devices, i.e. the
          array being normalized is sharded across devices within a pmap.
      axis_index_groups: groups of axis indices within that named axis
          representing subsets of devices to reduce over (default: None). For
          example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over
          the examples on the first two and last two devices. See ``jax.lax.psum``
          for more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
          calculation for the variance.
    """

    def __init__(
        self,
        in_size: Size,
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        *,
        epsilon: float = 1e-6,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Callable = init.ZeroInit(),
        scale_init: Callable = init.Constant(1.0),
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        dtype: Optional[jax.typing.DTypeLike] = None,
        param_type: type = NormalizationParamState,
    ):
        super().__init__()

        self.in_size = in_size
        self.out_size = in_size

        # parameters about axis
        feature_axes = (feature_axes,) if isinstance(feature_axes, int) else feature_axes
        self.feature_axes = _canonicalize_axes(len(self.in_size), feature_axes)
        self.reduction_axes = (reduction_axes,) if isinstance(reduction_axes, int) else reduction_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups

        # variables
        feature_shape = tuple([(ax if i in self.feature_axes else 1)
                               for i, ax in enumerate(self.in_size)])

        weights = dict()
        if use_scale:
            weights['scale'] = init.param(scale_init, feature_shape)
        if use_bias:
            weights['bias'] = init.param(bias_init, feature_shape)
        if len(weights):
            self.weight = param_type(weights)
        else:
            self.weight = None

        # parameters
        self.epsilon = epsilon
        self.dtype = dtype or environ.dftype()
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.use_fast_variance = use_fast_variance

    def update(self, x, *, mask: Optional[jax.Array] = None):
        """Applies layer normalization on the input.

        Args:
          x: the inputs

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        mean, var = _compute_stats(
            x,
            self.reduction_axes,
            dtype=self.dtype,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        return _normalize(
            x,
            mean=mean,
            var=var,
            weights=self.weight,
            reduction_axes=self.reduction_axes,
            feature_axes=self.feature_axes,
            dtype=self.dtype,
            epsilon=self.epsilon,
        )


class RMSNorm(Module):
    """
    RMS Layer normalization (https://arxiv.org/abs/1910.07467).

    RMSNorm normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    Unlike LayerNorm which re-centers the mean to be 0 and normalizes by the
    standard deviation of the activations, RMSNorm does not re-center at all
    and instead normalizes by the root mean square of the activations.

    Example usage::

      >>> import brainstate as brainstate
      >>> x = brainstate.random.normal(size=(5, 6))
      >>> layer = brainstate.nn.RMSNorm(num_features=6)
      >>> layer.states()
      >>> y = layer(x)

    Attributes:
        in_size: The input shape, without batch size.
        epsilon: A small float added to variance to avoid dividing by zero.
        dtype: the dtype of the result (default: infer from input and params).
        use_scale: If True, multiply by scale (gamma). When the next layer is linear
            (also e.g. nn.relu), this can be disabled since the scaling will be done
            by the next layer.
        scale_init: Initializer for scale, by default, one.
        reduction_axes: Axes for computing normalization statistics. It is recommended
            to use the negative integer, since when the batch dimension is used,
            the reduction_axes may be wrong when using the positive integer.
        feature_axes: Feature axes for learned bias and scaling.
        axis_name: the axis name used to combine batch statistics from multiple
            devices. See ``jax.pmap`` for a description of axis names (default: None).
            This is only needed if the model is subdivided across devices, i.e. the
            array being normalized is sharded across devices within a pmap.
        axis_index_groups: groups of axis indices within that named axis
            representing subsets of devices to reduce over (default: None). For
            example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over
            the examples on the first two and last two devices. See ``jax.lax.psum``
            for more details.
        use_fast_variance: If true, use a faster, but less numerically stable,
            calculation for the variance.
    """

    def __init__(
        self,
        in_size: Size,
        *,
        epsilon: float = 1e-6,
        dtype: Optional[jax.typing.DTypeLike] = None,
        use_scale: bool = True,
        scale_init: Callable = init.Constant(1.0),
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        param_type: type = NormalizationParamState,
    ):
        super().__init__()

        self.in_size = in_size
        self.out_size = in_size

        # parameters about axis
        feature_axes = (feature_axes,) if isinstance(feature_axes, int) else feature_axes
        self.feature_axes = _canonicalize_axes(len(self.in_size), feature_axes)
        self.reduction_axes = (reduction_axes,) if isinstance(reduction_axes, int) else reduction_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups

        # variables
        feature_shape = tuple([(ax if i in self.feature_axes else 1)
                               for i, ax in enumerate(self.in_size)])
        if use_scale:
            self.scale = param_type({'scale': init.param(scale_init, feature_shape)})
        else:
            self.scale = None

        # parameters
        self.epsilon = epsilon
        self.dtype = dtype or environ.dftype()
        self.use_scale = use_scale
        self.scale_init = scale_init
        self.use_fast_variance = use_fast_variance

    def update(self, x, *, mask: Optional[jax.Array] = None):
        """Applies layer normalization on the input.

        Args:
          x: the inputs
          mask: the mask

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        mean, var = _compute_stats(
            x,
            self.reduction_axes,
            dtype=self.dtype,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
            use_mean=False,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        return _normalize(
            x,
            mean=mean,
            var=var,
            weights=self.scale,
            reduction_axes=self.reduction_axes,
            feature_axes=self.feature_axes,
            dtype=self.dtype,
            epsilon=self.epsilon,
        )


class GroupNorm(Module):
    """
    Group normalization (arxiv.org/abs/1803.08494).

    This op is similar to batch normalization, but statistics are shared across
    equally-sized groups of channels and not shared across batch dimension.
    Thus, group normalization does not depend on the batch composition and does
    not require maintaining internal state for storing statistics.
    The user should either specify the total number of channel groups or the
    number of channels per group.

    .. note::
      LayerNorm is a special case of GroupNorm where ``num_groups=1``.

    Example usage::

      >>> import numpy as np
      >>> import brainstate as brainstate
      ...
      >>> x = brainstate.random.normal(size=(3, 4, 5, 6))
      >>> layer = brainstate.nn.GroupNorm(x.shape, num_groups=3)
      >>> layer.states()
      >>> y = layer(x)
      >>> y = brainstate.nn.GroupNorm(x.shape, num_groups=1)(x)
      >>> y2 = brainstate.nn.LayerNorm(x.shape, reduction_axes=(1, 2, 3))(x)
      >>> np.testing.assert_allclose(y, y2)

    Attributes:
      in_size: The input shape, without batch size.
      num_groups: the total number of channel groups. The default value of 32 is
        proposed by the original group normalization paper.
      group_size: the number of channels in a group.
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
      reduction_axes: List of axes used for computing normalization statistics.
        This list must include the final dimension, which is assumed to be the
        feature axis. Furthermore, if the input used at call time has additional
        leading axes compared to the data used for initialisation, for example due
        to batching, then the reduction axes need to be defined explicitly.
        It is recommended to use the negative integer, since when the batch dimension is used,
        the reduction_axes may be wrong when using the positive integer.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See ``jax.pmap`` for a description of axis names (default: None).
        This is only needed if the model is subdivided across devices, i.e. the
        array being normalized is sharded across devices within a pmap or shard
        map. For SPMD jit, you do not need to manually synchronize. Just make sure
        that the axes are correctly annotated and XLA:SPMD will insert the
        necessary collectives.
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
        examples on the first two and last two devices. See ``jax.lax.psum`` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    def __init__(
        self,
        in_size: Size,
        feature_axis: Axes = -1,
        num_groups: Optional[int] = 32,
        group_size: Optional[int] = None,
        *,
        epsilon: float = 1e-6,
        dtype: Optional[jax.typing.DTypeLike] = None,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Callable = init.ZeroInit(),
        scale_init: Callable = init.Constant(1.),
        reduction_axes: Optional[Axes] = None,
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        param_type: type = NormalizationParamState,
    ):
        super().__init__()

        self.in_size = in_size
        self.out_size = in_size

        # parameters about axis
        feature_axis = (feature_axis,) if isinstance(feature_axis, int) else feature_axis
        self.feature_axes = _canonicalize_axes(len(self.in_size), feature_axis)
        self.reduction_axes = (reduction_axes,) if isinstance(reduction_axes, int) else reduction_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups

        if (num_groups is None and group_size is None) or (
            num_groups is not None and group_size is not None
        ):
            raise ValueError(
                'Either `num_groups` or `group_size` should be '
                'specified. If `group_size` is to be specified, '
                'pass `num_groups=None` as argument to override '
                'the default `num_groups` value of 32.'
            )

        feature_shape = tuple([(ax if i in self.feature_axes else 1)
                               for i, ax in enumerate(self.in_size)])
        assert len(feature_shape) == 1, 'GroupNorm only supports 1D feature axis.'
        num_features = feature_shape[0]
        if group_size is not None:
            if num_features % group_size != 0:
                raise ValueError(
                    'Number of features ({}) is not multiple of the '
                    'group size ({}).'.format(num_features, group_size)
                )
            self.num_groups = num_features // group_size
            self.group_size = group_size
        else:
            if not isinstance(num_groups, int) or num_groups <= 0 or (
                num_features % num_groups != 0
            ):
                raise ValueError(
                    'Number of groups ({}) does not divide the number'
                    ' of channels ({}).'.format(num_groups, num_features)
                )
            self.num_groups = num_groups
            self.group_size = num_features // num_groups

        # variables
        weights = dict()
        if use_scale:
            weights['scale'] = init.param(scale_init, feature_shape)
        if use_bias:
            weights['bias'] = init.param(bias_init, feature_shape)
        if len(weights):
            self.weight = param_type(weights)
        else:
            self.weight = None

        # parameters
        self.epsilon = epsilon
        self.dtype = dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.use_fast_variance = use_fast_variance

    def update(self, x, *, mask: Optional[jax.Array] = None):
        """Applies group normalization to the input (arxiv.org/abs/1803.08494).

        Args:
          x: the input of shape ``...self.num_features`` where ``self.num_features``
            is a channels dimension and ``...`` represents an arbitrary number of
            extra dimensions that can be used to accumulate statistics over. If no
            reduction axes have been specified then all additional dimensions ``...``
            will be used to accumulate statistics apart from the leading dimension
            which is assumed to represent the batch.
          mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        if self.reduction_axes is not None:
            reduction_axes = self.reduction_axes
        else:
            reduction_axes = list(range(1, x.ndim - 1)) + [-1]
        reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)

        group_shape = x.shape[:-1] + (self.num_groups, self.group_size)
        if mask is not None:
            mask = mask.reshape(mask.shape[:-1] + (self.num_groups, self.group_size))

        mean, var = _compute_stats(
            x.reshape(group_shape),
            list(reduction_axes[:-1]) + [-1],
            dtype=self.dtype,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )
        mean = jnp.repeat(mean, self.group_size, axis=1)
        var = jnp.repeat(var, self.group_size, axis=1)
        return _normalize(
            x,
            mean=mean,
            var=var,
            weights=self.weight,
            reduction_axes=reduction_axes[:-1],
            feature_axes=self.feature_axes,
            dtype=self.dtype,
            epsilon=self.epsilon,
        )
