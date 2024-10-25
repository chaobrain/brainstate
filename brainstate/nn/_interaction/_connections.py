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

import collections.abc
import numbers
from typing import Callable, Tuple, Union, Sequence, Optional, TypeVar

import brainunit as u
import jax
import jax.numpy as jnp

from brainstate import init, functional
from brainstate._state import ParamState
from brainstate.nn._module import Module
from brainstate.typing import ArrayLike

T = TypeVar('T')

__all__ = [
  'Linear', 'ScaledWSLinear', 'SignedWLinear', 'CSRLinear',
  'Conv1d', 'Conv2d', 'Conv3d',
  'ScaledWSConv1d', 'ScaledWSConv2d', 'ScaledWSConv3d',
  'AllToAll',
]


def to_dimension_numbers(
    num_spatial_dims: int,
    channels_last: bool,
    transpose: bool
) -> jax.lax.ConvDimensionNumbers:
  """Create a `lax.ConvDimensionNumbers` for the given inputs."""
  num_dims = num_spatial_dims + 2
  if channels_last:
    spatial_dims = tuple(range(1, num_dims - 1))
    image_dn = (0, num_dims - 1) + spatial_dims
  else:
    spatial_dims = tuple(range(2, num_dims))
    image_dn = (0, 1) + spatial_dims
  if transpose:
    kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
  else:
    kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))
  return jax.lax.ConvDimensionNumbers(lhs_spec=image_dn,
                                      rhs_spec=kernel_dn,
                                      out_spec=image_dn)


def replicate(
    element: Union[T, Sequence[T]],
    num_replicate: int,
    name: str,
) -> Tuple[T, ...]:
  """Replicates entry in `element` `num_replicate` if needed."""
  if isinstance(element, (str, bytes)) or not isinstance(element, collections.abc.Sequence):
    return (element,) * num_replicate
  elif len(element) == 1:
    return tuple(list(element) * num_replicate)
  elif len(element) == num_replicate:
    return tuple(element)
  else:
    raise TypeError(f"{name} must be a scalar or sequence of length 1 or "
                    f"sequence of length {num_replicate}.")


class Linear(Module):
  """
  Linear layer.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Union[int, Sequence[int]],
      out_size: Union[int, Sequence[int]],
      w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
      b_init: Optional[Union[Callable, ArrayLike]] = init.ZeroInit(),
      w_mask: Optional[Union[ArrayLike, Callable]] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    # input and output shape
    self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
    self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)

    # w_mask
    self.w_mask = init.param(w_mask, self.in_size + self.out_size)

    # weights
    params = dict(weight=init.param(w_init, (self.in_size[-1], self.out_size[-1]), allow_none=False))
    if b_init is not None:
      params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)

    # weight + op
    self.weight = ParamState(params)

  def update(self, x):
    params = self.weight.value
    weight = params['weight']
    if self.w_mask is not None:
      weight = weight * self.w_mask
    y = jnp.dot(x, weight)
    if 'bias' in params:
      y = y + params['bias']
    return y


class SignedWLinear(Module):
  """
  Linear layer with signed weights.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Union[int, Sequence[int]],
      out_size: Union[int, Sequence[int]],
      w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
      w_sign: Optional[ArrayLike] = None,
      name: Optional[str] = None,

  ):
    super().__init__(name=name)

    # input and output shape
    self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
    self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)

    # w_mask
    self.w_sign = w_sign

    # weights
    weight = init.param(w_init, self.in_size + self.out_size, allow_none=False)
    self.weight = ParamState(weight)

  def _operation(self, x, w):
    if self.w_sign is None:
      return jnp.matmul(x, jnp.abs(w))
    else:
      return jnp.matmul(x, jnp.abs(w) * self.w_sign)

  def update(self, x):
    return self._operation(x, self.weight.value)


class ScaledWSLinear(Module):
  """
  Linear Layer with Weight Standardization.

  Applies weight standardization to the weights of the linear layer.

  Parameters
  ----------
  in_size: int, sequence of int
    The input size.
  out_size: int, sequence of int
    The output size.
  w_init: Callable, ArrayLike
    The initializer for the weights.
  b_init: Callable, ArrayLike
    The initializer for the bias.
  w_mask: ArrayLike, Callable
    The optional mask of the weights.
  ws_gain: bool
    Whether to use gain for the weights. The default is True.
  eps: float
    The epsilon value for the weight standardization.
  name: str
    The name of the object.

  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Union[int, Sequence[int]],
      out_size: Union[int, Sequence[int]],
      w_init: Callable = init.KaimingNormal(),
      b_init: Callable = init.ZeroInit(),
      w_mask: Optional[Union[ArrayLike, Callable]] = None,
      ws_gain: bool = True,
      eps: float = 1e-4,
      name: str = None,
  ):
    super().__init__(name=name)

    # input and output shape
    self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
    self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)

    # w_mask
    self.w_mask = init.param(w_mask, (self.in_size[0], 1))

    # parameters
    self.eps = eps

    # weights
    params = dict(weight=init.param(w_init, self.in_size + self.out_size, allow_none=False))
    if b_init is not None:
      params['bias'] = init.param(b_init, self.out_size, allow_none=False)
    # gain
    if ws_gain:
      s = params['weight'].shape
      params['gain'] = jnp.ones((1,) * (len(s) - 1) + (s[-1],), dtype=params['weight'].dtype)

    # weight operation
    self.weight = ParamState(params)

  def update(self, x):
    return self._operation(x, self.weight.value)

  def _operation(self, x, params):
    w = params['weight']
    w = functional.weight_standardization(w, self.eps, params.get('gain', None))
    if self.w_mask is not None:
      w = w * self.w_mask
    y = jnp.dot(x, w)
    if 'bias' in params:
      y = y + params['bias']
    return y


class CSRLinear(Module):
  __module__ = 'brainstate.nn'


class _BaseConv(Module):
  # the number of spatial dimensions
  num_spatial_dims: int

  # the weight and its operations
  weight: ParamState

  def __init__(
      self,
      in_size: Sequence[int],
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_mask: Optional[Union[ArrayLike, Callable]] = None,
      name: str = None,
  ):
    super().__init__(name=name)

    # general parameters
    assert self.num_spatial_dims + 1 == len(in_size)
    self.in_size = tuple(in_size)
    self.in_channels = in_size[-1]
    self.out_channels = out_channels
    self.stride = replicate(stride, self.num_spatial_dims, 'stride')
    self.kernel_size = replicate(kernel_size, self.num_spatial_dims, 'kernel_size')
    self.lhs_dilation = replicate(lhs_dilation, self.num_spatial_dims, 'lhs_dilation')
    self.rhs_dilation = replicate(rhs_dilation, self.num_spatial_dims, 'rhs_dilation')
    self.groups = groups
    self.dimension_numbers = to_dimension_numbers(self.num_spatial_dims, channels_last=True, transpose=False)

    # the padding parameter
    if isinstance(padding, str):
      assert padding in ['SAME', 'VALID']
    elif isinstance(padding, int):
      padding = tuple((padding, padding) for _ in range(self.num_spatial_dims))
    elif isinstance(padding, (tuple, list)):
      if isinstance(padding[0], int):
        padding = (padding,) * self.num_spatial_dims
      elif isinstance(padding[0], (tuple, list)):
        if len(padding) == 1:
          padding = tuple(padding) * self.num_spatial_dims
        else:
          if len(padding) != self.num_spatial_dims:
            raise ValueError(
              f"Padding {padding} must be a Tuple[int, int], "
              f"or sequence of Tuple[int, int] with length 1, "
              f"or sequence of Tuple[int, int] with length {self.num_spatial_dims}."
            )
          padding = tuple(padding)
    else:
      raise ValueError
    self.padding = padding

    # the number of in-/out-channels
    assert self.out_channels % self.groups == 0, '"out_channels" should be divisible by groups'
    assert self.in_channels % self.groups == 0, '"in_channels" should be divisible by groups'

    # kernel shape and w_mask
    kernel_shape = tuple(self.kernel_size) + (self.in_channels // self.groups, self.out_channels)
    self.kernel_shape = kernel_shape
    self.w_mask = init.param(w_mask, kernel_shape, allow_none=True)

  def _check_input_dim(self, x):
    if x.ndim == self.num_spatial_dims + 2:
      x_shape = x.shape[1:]
    elif x.ndim == self.num_spatial_dims + 1:
      x_shape = x.shape
    else:
      raise ValueError(f"expected {self.num_spatial_dims + 2}D (with batch) or "
                       f"{self.num_spatial_dims + 1}D (without batch) input (got {x.ndim}D input, {x.shape})")
    if self.in_size != x_shape:
      raise ValueError(f"The expected input shape is {self.in_size}, while we got {x_shape}.")

  def update(self, x):
    self._check_input_dim(x)
    non_batching = False
    if x.ndim == self.num_spatial_dims + 1:
      x = jnp.expand_dims(x, 0)
      non_batching = True
    y = self._conv_op(x, self.weight.value)
    return y[0] if non_batching else y

  def _conv_op(self, x, params):
    raise NotImplementedError

  def __repr__(self):
    return (f'{self.__class__.__name__}('
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}, '
            f'groups={self.groups})')


class _Conv(_BaseConv):
  num_spatial_dims: int = None

  def __init__(
      self,
      in_size: Sequence[int],
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_init: Union[Callable, ArrayLike] = init.XavierNormal(),
      b_init: Optional[Union[Callable, ArrayLike]] = None,
      w_mask: Optional[Union[ArrayLike, Callable]] = None,
      name: str = None,
  ):
    super().__init__(in_size=in_size,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     lhs_dilation=lhs_dilation,
                     rhs_dilation=rhs_dilation,
                     groups=groups,
                     w_mask=w_mask,
                     name=name)

    self.w_initializer = w_init
    self.b_initializer = b_init

    # --- weights --- #
    weight = init.param(self.w_initializer, self.kernel_shape, allow_none=False)
    params = dict(weight=weight)
    if self.b_initializer is not None:
      bias_shape = (1,) * len(self.kernel_size) + (self.out_channels,)
      bias = init.param(self.b_initializer, bias_shape, allow_none=True)
      params['bias'] = bias

    # The weight operation
    self.weight = ParamState(params)

    # Evaluate the output shape
    abstract_y = jax.eval_shape(
      self._conv_op,
      jax.ShapeDtypeStruct((128,) + self.in_size, weight.dtype),
      params
    )
    y_shape = abstract_y.shape[1:]
    self.out_size = y_shape

  def _conv_op(self, x, params):
    w = params['weight']
    if self.w_mask is not None:
      w = w * self.w_mask
    y = jax.lax.conv_general_dilated(
      lhs=x,
      rhs=w,
      window_strides=self.stride,
      padding=self.padding,
      lhs_dilation=self.lhs_dilation,
      rhs_dilation=self.rhs_dilation,
      feature_group_count=self.groups,
      dimension_numbers=self.dimension_numbers
    )
    if 'bias' in params:
      y = y + params['bias']
    return y


class Conv1d(_Conv):
  """One-dimensional convolution.

  The input should be a 3d array with the shape of ``[B, H, C]``.

  Parameters
  ----------
  %s
  """
  __module__ = 'brainstate.nn'
  num_spatial_dims: int = 1


class Conv2d(_Conv):
  """Two-dimensional convolution.

  The input should be a 4d array with the shape of ``[B, H, W, C]``.

  Parameters
  ----------
  %s
  """
  __module__ = 'brainstate.nn'
  num_spatial_dims: int = 2


class Conv3d(_Conv):
  """Three-dimensional convolution.

  The input should be a 5d array with the shape of ``[B, H, W, D, C]``.

  Parameters
  ----------
  %s
  """
  __module__ = 'brainstate.nn'
  num_spatial_dims: int = 3


_conv_doc = '''
  in_size: tuple of int
    The input shape, without the batch size. This argument is important, since it is
    used to evaluate the shape of the output.
  out_channels: int
    The number of output channels.
  kernel_size: int, sequence of int
    The shape of the convolutional kernel.
    For 1D convolution, the kernel size can be passed as an integer.
    For all other cases, it must be a sequence of integers.
  stride: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, int, sequence of int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
    high)` integer pairs that give the padding to apply before and after each
    spatial dimension.
  lhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of `inputs`
    (default: 1). Convolution with input dilation `d` is equivalent to
    transposed convolution with stride `d`.
  rhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of the convolution
    kernel (default: 1). Convolution with kernel dilation
    is also known as 'atrous convolution'.
  groups: int
    If specified, divides the input features into groups. default 1.
  w_init: Callable, ArrayLike, Initializer
    The initializer for the convolutional kernel.
  b_init: Optional, Callable, ArrayLike, Initializer
    The initializer for the bias.
  w_mask: ArrayLike, Callable, Optional
    The optional mask of the weights.
  mode: Mode
    The computation mode of the current object. Default it is `training`.
  name: str, Optional
    The name of the object.
'''

Conv1d.__doc__ = Conv1d.__doc__ % _conv_doc
Conv2d.__doc__ = Conv2d.__doc__ % _conv_doc
Conv3d.__doc__ = Conv3d.__doc__ % _conv_doc


class _ScaledWSConv(_BaseConv):
  def __init__(
      self,
      in_size: Sequence[int],
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      ws_gain: bool = True,
      eps: float = 1e-4,
      w_init: Union[Callable, ArrayLike] = init.XavierNormal(),
      b_init: Optional[Union[Callable, ArrayLike]] = None,
      w_mask: Optional[Union[ArrayLike, Callable]] = None,
      name: str = None,
  ):
    super().__init__(in_size=in_size,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     lhs_dilation=lhs_dilation,
                     rhs_dilation=rhs_dilation,
                     groups=groups,
                     w_mask=w_mask,
                     name=name, )

    self.w_initializer = w_init
    self.b_initializer = b_init

    # --- weights --- #
    weight = init.param(self.w_initializer, self.kernel_shape, allow_none=False)
    params = dict(weight=weight)
    if self.b_initializer is not None:
      bias_shape = (1,) * len(self.kernel_size) + (self.out_channels,)
      bias = init.param(self.b_initializer, bias_shape, allow_none=True)
      params['bias'] = bias

    # gain
    if ws_gain:
      gain_size = (1,) * len(self.kernel_size) + (1, self.out_channels)
      ws_gain = jnp.ones(gain_size, dtype=params['weight'].dtype)
      params['gain'] = ws_gain

    # Epsilon, a small constant to avoid dividing by zero.
    self.eps = eps

    # The weight operation
    self.weight = ParamState(params)

    # Evaluate the output shape
    abstract_y = jax.eval_shape(
      self._conv_op,
      jax.ShapeDtypeStruct((128,) + self.in_size, weight.dtype),
      params
    )
    y_shape = abstract_y.shape[1:]
    self.out_size = y_shape

  def _conv_op(self, x, params):
    w = params['weight']
    w = functional.weight_standardization(w, self.eps, params.get('gain', None))
    if self.w_mask is not None:
      w = w * self.w_mask
    y = jax.lax.conv_general_dilated(
      lhs=x,
      rhs=w,
      window_strides=self.stride,
      padding=self.padding,
      lhs_dilation=self.lhs_dilation,
      rhs_dilation=self.rhs_dilation,
      feature_group_count=self.groups,
      dimension_numbers=self.dimension_numbers
    )
    if 'bias' in params:
      y = y + params['bias']
    return y


class ScaledWSConv1d(_ScaledWSConv):
  """One-dimensional convolution with weight standardization.

  The input should be a 3d array with the shape of ``[B, H, C]``.

  Parameters
  ----------
  %s
  """
  __module__ = 'brainstate.nn'
  num_spatial_dims: int = 1


class ScaledWSConv2d(_ScaledWSConv):
  """Two-dimensional convolution with weight standardization.

  The input should be a 4d array with the shape of ``[B, H, W, C]``.

  Parameters
  ----------
  %s
  """
  __module__ = 'brainstate.nn'
  num_spatial_dims: int = 2


class ScaledWSConv3d(_ScaledWSConv):
  """Three-dimensional convolution with weight standardization.

  The input should be a 5d array with the shape of ``[B, H, W, D, C]``.

  Parameters
  ----------
  %s
  """
  __module__ = 'brainstate.nn'
  num_spatial_dims: int = 3


_ws_conv_doc = '''
  in_size: tuple of int
    The input shape, without the batch size. This argument is important, since it is
    used to evaluate the shape of the output.
  out_channels: int
    The number of output channels.
  kernel_size: int, sequence of int
    The shape of the convolutional kernel.
    For 1D convolution, the kernel size can be passed as an integer.
    For all other cases, it must be a sequence of integers.
  stride: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, int, sequence of int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
    high)` integer pairs that give the padding to apply before and after each
    spatial dimension.
  lhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of `inputs`
    (default: 1). Convolution with input dilation `d` is equivalent to
    transposed convolution with stride `d`.
  rhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of the convolution
    kernel (default: 1). Convolution with kernel dilation
    is also known as 'atrous convolution'.
  groups: int
    If specified, divides the input features into groups. default 1.
  w_init: Callable, ArrayLike, Initializer
    The initializer for the convolutional kernel.
  b_init: Optional, Callable, ArrayLike, Initializer
    The initializer for the bias.
  ws_gain: bool
    Whether to add a gain term for the weight standarization. The default is `True`.
  eps: float
    The epsilon value for numerical stability.
  w_mask: ArrayLike, Callable, Optional
    The optional mask of the weights.
  mode: Mode
    The computation mode of the current object. Default it is `training`.
  name: str, Optional
    The name of the object.

'''

ScaledWSConv1d.__doc__ = ScaledWSConv1d.__doc__ % _ws_conv_doc
ScaledWSConv2d.__doc__ = ScaledWSConv2d.__doc__ % _ws_conv_doc
ScaledWSConv3d.__doc__ = ScaledWSConv3d.__doc__ % _ws_conv_doc


class AllToAll(Module):
  """Synaptic matrix multiplication with All2All connections.

  Args:
    in_size: int. The number of neurons in the presynaptic neuron group.
    out_size: int. The number of neurons in the postsynaptic neuron group.
    w_init: The synaptic weights.
    include_self: bool. Whether connect the neuron with at the same position.
    name: str. The object name.
  """

  def __init__(
      self,
      in_size: Union[int, Sequence[int]],
      out_size: Union[int, Sequence[int]],
      w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
      include_self: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    # input and output shape
    self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
    self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
    assert self.in_size[:-1] == self.out_size[:-1], ('The first n-1 dimensions of "in_size" '
                                                     'and "out_size" must be the same.')

    # weights
    self.weight = ParamState(init.param(w_init, (self.in_size[-1], self.out_size[-1]), allow_none=False))

    # others
    self.include_self = include_self

  def update(self, pre_val):
    if u.math.ndim(self.weight.value) == 0:  # weight is a scalar
      if pre_val.ndim == 1:
        post_val = u.math.sum(pre_val)
      else:
        post_val = u.math.sum(pre_val, keepdims=True, axis=-1)
      if not self.include_self:
        if self.in_size == self.out_size:
          post_val = post_val - pre_val
        elif self.in_size[-1] > self.out_size[-1]:
          val = pre_val[..., :self.out_size[-1]]
          post_val = post_val - val
        else:
          size = list(self.out_size)
          size[-1] = self.out_size[-1] - self.in_size[-1]
          val = u.math.concatenate([pre_val, u.math.zeros(size, dtype=pre_val.dtype)])
          post_val = post_val - val
      post_val = self.weight.value * post_val

    else:  # weight is a matrix
      assert u.math.ndim(self.weight.value) == 2, '"weight" must be a 2D matrix.'
      if not self.include_self:
        post_val = pre_val @ u.math.fill_diagonal(self.weight.value, 0.)
      else:
        post_val = pre_val @ self.weight.value
    return post_val
