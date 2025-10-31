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

from typing import Callable, Union, Sequence, Optional, Any

import brainunit as u
import jax
import jax.numpy as jnp

from brainstate import environ
from brainstate._state import ParamState, BatchState
from brainstate.typing import DTypeLike, ArrayLike, Size, Axes
from . import init as init
from ._module import Module

__all__ = [
    'weight_standardization',
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
    Scaled Weight Standardization.

    Applies weight standardization to improve training stability, as described in
    "Micro-Batch Training with Batch-Channel Normalization and Weight Standardization" [1]_.

    Parameters
    ----------
    w : ArrayLike
        The weight tensor to be standardized.
    eps : float, optional
        A small value added to variance to avoid division by zero. Default is 1e-4.
    gain : jax.Array, optional
        Optional gain parameter to scale the standardized weights. Default is None.
    out_axis : int, optional
        The output axis of the weight tensor. Default is -1.

    Returns
    -------
    jax.Array or u.Quantity
        The standardized weight tensor with the same shape as input.

    References
    ----------
    .. [1] Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019).
       Micro-Batch Training with Batch-Channel Normalization and Weight Standardization.
       arXiv preprint arXiv:1903.10520.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Standardize a weight matrix
        >>> w = jnp.ones((3, 4))
        >>> w_std = brainstate.nn.weight_standardization(w)
        >>>
        >>> # With custom gain
        >>> gain = jnp.ones((4,))
        >>> w_std = brainstate.nn.weight_standardization(w, gain=gain)
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
    """
    Canonicalize an optional dtype to the definitive dtype.

    If the ``dtype`` is None, this function will infer the dtype from the input
    arguments using ``jnp.result_type``. If it is not None, it will be returned
    unmodified or an exception is raised if the dtype is invalid.

    Parameters
    ----------
    *args : ArrayLike
        JAX array compatible values. None values are ignored.
    dtype : jax.typing.DTypeLike, optional
        Optional dtype override. If specified, the arguments are cast to the
        specified dtype and dtype inference is disabled. Default is None.
    inexact : bool, optional
        When True, the output dtype must be a subtype of ``jnp.inexact``.
        Inexact dtypes are real or complex floating points. This is useful
        when applying operations that don't work directly on integers like
        taking a mean. Default is True.

    Returns
    -------
    jax.typing.DTypeLike
        The dtype that ``*args`` should be cast to.

    Raises
    ------
    ValueError
        If ``inexact=True`` and the resulting dtype is not an inexact type.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Infer dtype from arguments
        >>> x = jnp.array([1, 2, 3])
        >>> dtype = canonicalize_dtype(x)
        >>>
        >>> # Specify explicit dtype
        >>> dtype = canonicalize_dtype(x, dtype=jnp.float64)
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
    Compute mean and variance statistics for normalization.

    This implementation includes several optimizations:

    - Computes in float32 precision for stability in half precision training.
    - If ``use_fast_variance`` is True, uses the formula Var = E[|x|^2] - |E[x]|^2
      instead of Var = E[|x - E[x]|^2] in a single XLA fusion.
    - Clips negative variances to zero to avoid downstream NaNs from roundoff errors.
    - Supports averaging across parallel axes and subgroups with a single
      ``lax.pmean`` call to reduce latency.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axes : Sequence[int]
        The axes in ``x`` to compute mean and variance statistics for.
    dtype : DTypeLike
        Optional dtype specifying the minimal precision. Statistics are always
        at least float32 for stability. If None, uses the dtype of x.
    axis_name : str, optional
        Optional name for the pmapped axis to compute mean over. Only used for
        pmap and shard map. For SPMD jit, axes should be correctly annotated
        and XLA:SPMD will insert necessary collectives. Default is None.
    axis_index_groups : Sequence[int], optional
        Optional axis indices for grouped reductions. Default is None.
    use_mean : bool, optional
        If True, calculate the mean from the input and use it when computing
        the variance. If False, set the mean to zero and compute the variance
        without subtracting the mean. Default is True.
    use_fast_variance : bool, optional
        If True, use a faster but less numerically stable calculation for the
        variance. Default is True.
    mask : jax.Array, optional
        Binary array of shape broadcastable to ``x``, indicating the positions
        for which the mean and variance should be computed. Default is None.

    Returns
    -------
    tuple of jax.Array
        A pair ``(mean, var)`` containing the computed mean and variance.
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
    """
    Normalize the input and optionally apply learned scale and bias.

    Parameters
    ----------
    x : ArrayLike
        The input array.
    mean : ArrayLike, optional
        Mean to use for normalization. If None, normalization is skipped.
    var : ArrayLike, optional
        Variance to use for normalization. If None, normalization is skipped.
    weights : NormalizationParamState, optional
        The scale and bias parameters. If None, no affine transformation is applied.
    reduction_axes : Axes
        The axes in ``x`` to reduce.
    feature_axes : Axes
        The feature axes to apply the scale and bias.
    dtype : DTypeLike
        The dtype of the result. If None, inferred from input and parameters.
    epsilon : jax.typing.ArrayLike
        A small value added to variance to avoid division by zero.

    Returns
    -------
    jax.Array
        The normalized input array.
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
    """
    0-D batch normalization.

    Normalizes a batch of 0-D data (vectors) by fixing the mean and variance
    of inputs on each feature (channel). This layer aims to reduce the internal
    covariate shift of data.

    The input data should have shape ``(b, c)``, where ``b`` is the batch dimension
    and ``c`` is the channel dimension.

    The normalization is performed as:

    .. math::
        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\operatorname{Var}[x] + \\epsilon}} \\cdot \\gamma + \\beta

    where :math:`\\gamma` and :math:`\\beta` are learnable affine parameters (if ``affine=True``).

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without batch dimension.
    feature_axis : int or tuple of int, optional
        The feature or non-batch axis of the input. Default is -1.
    track_running_stats : bool, optional
        If True, tracks the running mean and variance. If False, uses batch
        statistics in both training and eval modes. Default is True.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        The momentum value used for the ``running_mean`` and ``running_var``
        computation. The update rule is:
        :math:`\\hat{x}_{\\text{new}} = \\text{momentum} \\times \\hat{x} + (1 - \\text{momentum}) \\times x_t`.
        Default is 0.99.
    affine : bool, optional
        If True, this module has learnable affine parameters (scale and bias).
        Default is True.
    bias_initializer : ArrayLike or Callable, optional
        Initializer for the bias (beta) parameter. Default is ``init.Constant(0.)``.
    scale_initializer : ArrayLike or Callable, optional
        Initializer for the scale (gamma) parameter. Default is ``init.Constant(1.)``.
    axis_name : str or sequence of str, optional
        The axis name(s) for parallel reduction using ``jax.pmap`` or ``jax.vmap``.
        If specified, batch statistics are calculated across all replicas on the
        named axes. Default is None.
    axis_index_groups : sequence of sequence of int, optional
        Groups of axis indices within the named axis representing subsets of
        devices to reduce over. For example, ``[[0, 1], [2, 3]]`` would
        independently batch-normalize over the first two and last two devices.
        See ``jax.lax.psum`` for more details. Default is None.
    use_fast_variance : bool, optional
        If True, use a faster but less numerically stable calculation for
        the variance. Default is True.

    Notes
    -----
    The ``momentum`` parameter is different from the conventional notion of
    momentum used in optimizers.

    References
    ----------
    .. [1] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating
       Deep Network Training by Reducing Internal Covariate Shift.
       In International Conference on Machine Learning (pp. 448-456).

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a BatchNorm0d layer
        >>> layer = brainstate.nn.BatchNorm0d(in_size=(10,))
        >>>
        >>> # Apply normalization to a batch of data
        >>> x = jnp.ones((32, 10))  # batch_size=32, features=10
        >>> y = layer(x)
        >>>
        >>> # Check output shape
        >>> print(y.shape)
        (32, 10)
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 0


class BatchNorm1d(_BatchNorm):
    """
    1-D batch normalization.

    Normalizes a batch of 1-D data by fixing the mean and variance of inputs
    on each feature (channel). This layer aims to reduce the internal covariate
    shift of data.

    The input data should have shape ``(b, l, c)``, where ``b`` is the batch
    dimension, ``l`` is the spatial/sequence dimension, and ``c`` is the channel
    dimension.

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without batch dimension. For 1-D data, typically ``(l, c)``.
    feature_axis : int or tuple of int, optional
        The feature or non-batch axis of the input. Default is -1.
    track_running_stats : bool, optional
        If True, tracks the running mean and variance. If False, uses batch
        statistics in both training and eval modes. Default is True.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        The momentum value for running statistics computation. Default is 0.99.
    affine : bool, optional
        If True, has learnable affine parameters (scale and bias). Default is True.
    bias_initializer : ArrayLike or Callable, optional
        Initializer for the bias parameter. Default is ``init.Constant(0.)``.
    scale_initializer : ArrayLike or Callable, optional
        Initializer for the scale parameter. Default is ``init.Constant(1.)``.
    axis_name : str or sequence of str, optional
        Axis name(s) for parallel reduction. Default is None.
    axis_index_groups : sequence of sequence of int, optional
        Groups of axis indices for device-grouped reduction. Default is None.
    use_fast_variance : bool, optional
        If True, use faster but less stable variance calculation. Default is True.

    References
    ----------
    .. [1] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating
       Deep Network Training by Reducing Internal Covariate Shift.
       In International Conference on Machine Learning (pp. 448-456).

    See Also
    --------
    BatchNorm0d : 0-D batch normalization
    BatchNorm2d : 2-D batch normalization
    BatchNorm3d : 3-D batch normalization

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a BatchNorm1d layer for sequence data
        >>> layer = brainstate.nn.BatchNorm1d(in_size=(100, 64))  # length=100, channels=64
        >>>
        >>> # Apply normalization
        >>> x = jnp.ones((8, 100, 64))  # batch_size=8
        >>> y = layer(x)
        >>> print(y.shape)
        (8, 100, 64)
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 1


class BatchNorm2d(_BatchNorm):
    """
    2-D batch normalization.

    Normalizes a batch of 2-D data (e.g., images) by fixing the mean and variance
    of inputs on each feature (channel). This layer aims to reduce the internal
    covariate shift of data.

    The input data should have shape ``(b, h, w, c)``, where ``b`` is the batch
    dimension, ``h`` is the height dimension, ``w`` is the width dimension, and
    ``c`` is the channel dimension.

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without batch dimension. For 2-D data, typically ``(h, w, c)``.
    feature_axis : int or tuple of int, optional
        The feature or non-batch axis of the input. Default is -1.
    track_running_stats : bool, optional
        If True, tracks the running mean and variance. If False, uses batch
        statistics in both training and eval modes. Default is True.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        The momentum value for running statistics computation. Default is 0.99.
    affine : bool, optional
        If True, has learnable affine parameters (scale and bias). Default is True.
    bias_initializer : ArrayLike or Callable, optional
        Initializer for the bias parameter. Default is ``init.Constant(0.)``.
    scale_initializer : ArrayLike or Callable, optional
        Initializer for the scale parameter. Default is ``init.Constant(1.)``.
    axis_name : str or sequence of str, optional
        Axis name(s) for parallel reduction. Default is None.
    axis_index_groups : sequence of sequence of int, optional
        Groups of axis indices for device-grouped reduction. Default is None.
    use_fast_variance : bool, optional
        If True, use faster but less stable variance calculation. Default is True.

    References
    ----------
    .. [1] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating
       Deep Network Training by Reducing Internal Covariate Shift.
       In International Conference on Machine Learning (pp. 448-456).

    See Also
    --------
    BatchNorm0d : 0-D batch normalization
    BatchNorm1d : 1-D batch normalization
    BatchNorm3d : 3-D batch normalization

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a BatchNorm2d layer for image data
        >>> layer = brainstate.nn.BatchNorm2d(in_size=(28, 28, 3))  # 28x28 RGB images
        >>>
        >>> # Apply normalization
        >>> x = jnp.ones((16, 28, 28, 3))  # batch_size=16
        >>> y = layer(x)
        >>> print(y.shape)
        (16, 28, 28, 3)
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 2


class BatchNorm3d(_BatchNorm):
    """
    3-D batch normalization.

    Normalizes a batch of 3-D data (e.g., video or volumetric data) by fixing
    the mean and variance of inputs on each feature (channel). This layer aims
    to reduce the internal covariate shift of data.

    The input data should have shape ``(b, h, w, d, c)``, where ``b`` is the
    batch dimension, ``h`` is the height dimension, ``w`` is the width dimension,
    ``d`` is the depth dimension, and ``c`` is the channel dimension.

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without batch dimension. For 3-D data, typically ``(h, w, d, c)``.
    feature_axis : int or tuple of int, optional
        The feature or non-batch axis of the input. Default is -1.
    track_running_stats : bool, optional
        If True, tracks the running mean and variance. If False, uses batch
        statistics in both training and eval modes. Default is True.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        The momentum value for running statistics computation. Default is 0.99.
    affine : bool, optional
        If True, has learnable affine parameters (scale and bias). Default is True.
    bias_initializer : ArrayLike or Callable, optional
        Initializer for the bias parameter. Default is ``init.Constant(0.)``.
    scale_initializer : ArrayLike or Callable, optional
        Initializer for the scale parameter. Default is ``init.Constant(1.)``.
    axis_name : str or sequence of str, optional
        Axis name(s) for parallel reduction. Default is None.
    axis_index_groups : sequence of sequence of int, optional
        Groups of axis indices for device-grouped reduction. Default is None.
    use_fast_variance : bool, optional
        If True, use faster but less stable variance calculation. Default is True.

    References
    ----------
    .. [1] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating
       Deep Network Training by Reducing Internal Covariate Shift.
       In International Conference on Machine Learning (pp. 448-456).

    See Also
    --------
    BatchNorm0d : 0-D batch normalization
    BatchNorm1d : 1-D batch normalization
    BatchNorm2d : 2-D batch normalization

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a BatchNorm3d layer for volumetric data
        >>> layer = brainstate.nn.BatchNorm3d(in_size=(32, 32, 32, 1))  # 32x32x32 volumes
        >>>
        >>> # Apply normalization
        >>> x = jnp.ones((4, 32, 32, 32, 1))  # batch_size=4
        >>> y = layer(x)
        >>> print(y.shape)
        (4, 32, 32, 32, 1)
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 3


class LayerNorm(Module):
    """
    Layer normalization layer [1]_.

    LayerNorm normalizes the activations of the layer for each given example in
    a batch independently, rather than across a batch like Batch Normalization.
    It applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without batch dimension.
    reduction_axes : int or tuple of int, optional
        Axes for computing normalization statistics. It is recommended to use
        negative integers, as positive integers may cause issues when batch
        dimensions are present. Default is -1.
    feature_axes : int or tuple of int, optional
        Feature axes for learned bias and scaling. Default is -1.
    epsilon : float, optional
        A small value added to variance to avoid division by zero. Default is 1e-6.
    use_bias : bool, optional
        If True, bias (beta) is added. Default is True.
    use_scale : bool, optional
        If True, multiply by scale (gamma). When the next layer is linear
        (e.g., nn.relu), this can be disabled since scaling will be done by
        the next layer. Default is True.
    bias_init : Callable, optional
        Initializer for bias parameter. Default is ``init.ZeroInit()``.
    scale_init : Callable, optional
        Initializer for scale parameter. Default is ``init.Constant(1.0)``.
    axis_name : str, optional
        The axis name used to combine batch statistics from multiple devices.
        See ``jax.pmap`` for axis name description. Only needed if the model
        is subdivided across devices. Default is None.
    axis_index_groups : sequence, optional
        Groups of axis indices within the named axis representing subsets of
        devices to reduce over. For example, ``[[0, 1], [2, 3]]`` would
        independently normalize over the first two and last two devices.
        See ``jax.lax.psum`` for details. Default is None.
    use_fast_variance : bool, optional
        If True, use a faster but less numerically stable calculation for
        the variance. Default is True.
    dtype : jax.typing.DTypeLike, optional
        The dtype of the result. If None, inferred from input and parameters.
        Default is None.

    References
    ----------
    .. [1] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization.
       arXiv preprint arXiv:1607.06450.

    See Also
    --------
    RMSNorm : Root Mean Square Layer Normalization
    GroupNorm : Group Normalization
    BatchNorm1d : 1-D Batch Normalization

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>>
        >>> # Create a LayerNorm layer
        >>> x = brainstate.random.normal(size=(3, 4, 5, 6))
        >>> layer = brainstate.nn.LayerNorm(x.shape)
        >>>
        >>> # Apply normalization
        >>> y = layer(x)
        >>> print(y.shape)
        (3, 4, 5, 6)
        >>>
        >>> # Normalize only the last dimension
        >>> layer = brainstate.nn.LayerNorm((10, 20), reduction_axes=-1, feature_axes=-1)
        >>> x = brainstate.random.normal((5, 10, 20))
        >>> y = layer(x)
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
        """
        Apply layer normalization on the input.

        Parameters
        ----------
        x : jax.Array
            The input array.
        mask : jax.Array, optional
            Binary array of shape broadcastable to ``x``, indicating the
            positions for which normalization should be computed. Default is None.

        Returns
        -------
        jax.Array
            Normalized inputs with the same shape as the input.
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
    Root Mean Square Layer Normalization [1]_.

    RMSNorm normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    Unlike LayerNorm which re-centers the mean to 0 and normalizes by the standard
    deviation, RMSNorm does not re-center at all and instead normalizes by the
    root mean square of the activations.

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without batch dimension.
    epsilon : float, optional
        A small value added to variance to avoid division by zero. Default is 1e-6.
    dtype : jax.typing.DTypeLike, optional
        The dtype of the result. If None, inferred from input and parameters.
        Default is None.
    use_scale : bool, optional
        If True, multiply by scale (gamma). When the next layer is linear
        (e.g., nn.relu), this can be disabled since scaling will be done by
        the next layer. Default is True.
    scale_init : Callable, optional
        Initializer for scale parameter. Default is ``init.Constant(1.0)``.
    reduction_axes : int or tuple of int, optional
        Axes for computing normalization statistics. It is recommended to use
        negative integers. Default is -1.
    feature_axes : int or tuple of int, optional
        Feature axes for learned scaling. Default is -1.
    axis_name : str, optional
        The axis name used to combine batch statistics from multiple devices.
        See ``jax.pmap`` for details. Default is None.
    axis_index_groups : sequence, optional
        Groups of axis indices within the named axis representing subsets of
        devices to reduce over. For example, ``[[0, 1], [2, 3]]`` would
        independently normalize over the first two and last two devices.
        Default is None.
    use_fast_variance : bool, optional
        If True, use a faster but less numerically stable calculation for
        the variance. Default is True.

    References
    ----------
    .. [1] Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
       Advances in Neural Information Processing Systems, 32.

    See Also
    --------
    LayerNorm : Layer Normalization
    GroupNorm : Group Normalization

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>>
        >>> # Create an RMSNorm layer
        >>> x = brainstate.random.normal(size=(5, 6))
        >>> layer = brainstate.nn.RMSNorm(in_size=(6,))
        >>>
        >>> # Apply normalization
        >>> y = layer(x)
        >>> print(y.shape)
        (5, 6)
        >>>
        >>> # Without scaling
        >>> layer = brainstate.nn.RMSNorm(in_size=(10,), use_scale=False)
        >>> x = brainstate.random.normal((3, 10))
        >>> y = layer(x)
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
        """
        Apply RMS normalization on the input.

        Parameters
        ----------
        x : jax.Array
            The input array.
        mask : jax.Array, optional
            Binary array of shape broadcastable to ``x``, indicating the
            positions for which normalization should be computed. Default is None.

        Returns
        -------
        jax.Array
            Normalized inputs with the same shape as the input.
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
    Group Normalization layer [1]_.

    Group normalization is similar to batch normalization, but statistics are
    shared across equally-sized groups of channels and not shared across the
    batch dimension. Thus, group normalization does not depend on the batch
    composition and does not require maintaining internal state for storing statistics.

    The user should specify either the total number of channel groups (``num_groups``)
    or the number of channels per group (``group_size``).

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without batch dimension.
    feature_axis : int or tuple of int, optional
        The feature axis of the input. Default is -1.
    num_groups : int, optional
        The total number of channel groups. The default value of 32 is proposed
        by the original group normalization paper. Either ``num_groups`` or
        ``group_size`` must be specified, but not both. Default is 32.
    group_size : int, optional
        The number of channels in each group. Either ``num_groups`` or
        ``group_size`` must be specified, but not both. Default is None.
    epsilon : float, optional
        A small value added to variance to avoid division by zero. Default is 1e-6.
    dtype : jax.typing.DTypeLike, optional
        The dtype of the result. If None, inferred from input and parameters.
        Default is None.
    use_bias : bool, optional
        If True, bias (beta) is added. Default is True.
    use_scale : bool, optional
        If True, multiply by scale (gamma). When the next layer is linear
        (e.g., nn.relu), this can be disabled. Default is True.
    bias_init : Callable, optional
        Initializer for bias parameter. Default is ``init.ZeroInit()``.
    scale_init : Callable, optional
        Initializer for scale parameter. Default is ``init.Constant(1.)``.
    reduction_axes : int or tuple of int, optional
        List of axes used for computing normalization statistics. Must include
        the final dimension (feature axis). It is recommended to use negative
        integers. Default is None.
    axis_name : str, optional
        The axis name used to combine batch statistics from multiple devices.
        See ``jax.pmap`` for details. Default is None.
    axis_index_groups : sequence, optional
        Groups of axis indices within the named axis representing subsets of
        devices to reduce over. For example, ``[[0, 1], [2, 3]]`` would
        independently normalize over the first two and last two devices.
        Default is None.
    use_fast_variance : bool, optional
        If True, use a faster but less numerically stable calculation for
        the variance. Default is True.

    Notes
    -----
    LayerNorm is a special case of GroupNorm where ``num_groups=1``.

    References
    ----------
    .. [1] Wu, Y., & He, K. (2018). Group Normalization.
       In Proceedings of the European Conference on Computer Vision (ECCV)
       (pp. 3-19).

    See Also
    --------
    LayerNorm : Layer Normalization
    BatchNorm2d : 2-D Batch Normalization

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainstate as brainstate
        >>>
        >>> # Create a GroupNorm layer with 3 groups
        >>> x = brainstate.random.normal(size=(3, 4, 5, 6))
        >>> layer = brainstate.nn.GroupNorm(x.shape, num_groups=3)
        >>> y = layer(x)
        >>>
        >>> # GroupNorm with num_groups=1 is equivalent to LayerNorm
        >>> y1 = brainstate.nn.GroupNorm(x.shape, num_groups=1)(x)
        >>> y2 = brainstate.nn.LayerNorm(x.shape, reduction_axes=(1, 2, 3))(x)
        >>> np.testing.assert_allclose(y1, y2, rtol=1e-5)
        >>>
        >>> # Specify group_size instead of num_groups
        >>> layer = brainstate.nn.GroupNorm((12,), num_groups=None, group_size=4)
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
        """
        Apply group normalization to the input.

        Parameters
        ----------
        x : jax.Array
            The input of shape ``...C`` where ``C`` is the channels dimension
            and ``...`` represents an arbitrary number of extra dimensions. If no
            reduction axes have been specified, all additional dimensions will be
            used to accumulate statistics apart from the leading dimension which
            is assumed to represent the batch.
        mask : jax.Array, optional
            Binary array of shape broadcastable to ``x``, indicating the
            positions for which the mean and variance should be computed.
            Default is None.

        Returns
        -------
        jax.Array
            Normalized inputs with the same shape as the input.
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
