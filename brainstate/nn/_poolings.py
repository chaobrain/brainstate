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

import functools
from typing import Sequence, Optional
from typing import Union, Tuple, Callable, List

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate import environ
from brainstate.typing import Size
from ._module import Module

__all__ = [
    'Flatten', 'Unflatten',

    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',

    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
    'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
]


class Flatten(Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.

    Shape:
        - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.

    Args:
        in_size: Sequence of int. The shape of the input tensor.
        start_axis: first dim to flatten (default = 1).
        end_axis: last dim to flatten (default = -1).

    Examples::
        >>> import brainstate as brainstate
        >>> inp = brainstate.random.randn(32, 1, 5, 5)
        >>> # With default parameters
        >>> m = Flatten()
        >>> output = m(inp)
        >>> output.shape
        (32, 25)
        >>> # With non-default parameters
        >>> m = Flatten(0, 2)
        >>> output = m(inp)
        >>> output.shape
        (160, 5)
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        start_axis: int = 0,
        end_axis: int = -1,
        in_size: Optional[Size] = None
    ) -> None:
        super().__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis

        if in_size is not None:
            self.in_size = tuple(in_size)
            y = jax.eval_shape(functools.partial(u.math.flatten, start_axis=start_axis, end_axis=end_axis),
                               jax.ShapeDtypeStruct(self.in_size, environ.dftype()))
            self.out_size = y.shape

    def update(self, x):
        if self._in_size is None:
            start_axis = self.start_axis if self.start_axis >= 0 else x.ndim + self.start_axis
        else:
            assert x.ndim >= len(self.in_size), 'Input tensor has fewer dimensions than the expected shape.'
            dim_diff = x.ndim - len(self.in_size)
            if self.in_size != x.shape[dim_diff:]:
                raise ValueError(f'Input tensor has shape {x.shape}, but expected shape {self.in_size}.')
            if self.start_axis >= 0:
                start_axis = self.start_axis + dim_diff
            else:
                start_axis = x.ndim + self.start_axis
        return u.math.flatten(x, start_axis, self.end_axis)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(start_axis={self.start_axis}, end_axis={self.end_axis})'


class Unflatten(Module):
    r"""
    Unflatten a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it can
      be either `int` or `str` when `Tensor` or `NamedTensor` is used, respectively.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `torch.Size` for `Tensor` input;  a `NamedShape`
      (tuple of `(name, size)` tuples) for `NamedTensor` input.

    Shape:
        - Input: :math:`(*, S_{\text{dim}}, *)`, where :math:`S_{\text{dim}}` is the size at
          dimension :attr:`dim` and :math:`*` means any number of dimensions including none.
        - Output: :math:`(*, U_1, ..., U_n, *)`, where :math:`U` = :attr:`unflattened_size` and
          :math:`\prod_{i=1}^n U_i = S_{\text{dim}}`.

    Args:
        axis: int, Dimension to be unflattened.
        sizes: Sequence of int. New shape of the unflattened dimension.
        in_size: Sequence of int. The shape of the input tensor.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        axis: int,
        sizes: Size,
        name: str = None,
        in_size: Optional[Size] = None
    ) -> None:
        super().__init__(name=name)

        self.axis = axis
        self.sizes = sizes
        if isinstance(sizes, (tuple, list)):
            for idx, elem in enumerate(sizes):
                if not isinstance(elem, int):
                    raise TypeError("unflattened sizes must be tuple of ints, " +
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
        else:
            raise TypeError("unflattened sizes must be tuple or list, but found type {}".format(type(sizes).__name__))

        if in_size is not None:
            self.in_size = tuple(in_size)
            y = jax.eval_shape(functools.partial(u.math.unflatten, axis=axis, sizes=sizes),
                               jax.ShapeDtypeStruct(self.in_size, environ.dftype()))
            self.out_size = y.shape

    def update(self, x):
        return u.math.unflatten(x, self.axis, self.sizes)

    def __repr__(self):
        return f'{self.__class__.__name__}(axis={self.axis}, sizes={self.sizes})'


class _MaxPool(Module):
    def __init__(
        self,
        init_value: float,
        computation: Callable,
        pool_dim: int,
        kernel_size: Size,
        stride: Union[int, Sequence[int]] = None,
        padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
        channel_axis: Optional[int] = -1,
        name: Optional[str] = None,
        in_size: Optional[Size] = None,
    ):
        super().__init__(name=name)

        self.init_value = init_value
        self.computation = computation
        self.pool_dim = pool_dim

        # kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * pool_dim
        elif isinstance(kernel_size, Sequence):
            assert isinstance(kernel_size, (tuple, list)), f'kernel_size should be a tuple, but got {type(kernel_size)}'
            assert all(
                [isinstance(x, int) for x in kernel_size]), f'kernel_size should be a tuple of ints. {kernel_size}'
            if len(kernel_size) != pool_dim:
                raise ValueError(f'kernel_size should a tuple with {pool_dim} ints, but got {len(kernel_size)}')
        else:
            raise TypeError(f'kernel_size should be a int or a tuple with {pool_dim} ints.')
        self.kernel_size = kernel_size

        # stride
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = (stride,) * pool_dim
        elif isinstance(stride, Sequence):
            assert isinstance(stride, (tuple, list)), f'stride should be a tuple, but got {type(stride)}'
            assert all([isinstance(x, int) for x in stride]), f'stride should be a tuple of ints. {stride}'
            if len(stride) != pool_dim:
                raise ValueError(f'stride should a tuple with {pool_dim} ints, but got {len(kernel_size)}')
        else:
            raise TypeError(f'stride should be a int or a tuple with {pool_dim} ints.')
        self.stride = stride

        # padding
        if isinstance(padding, str):
            if padding not in ("SAME", "VALID"):
                raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")
        elif isinstance(padding, int):
            padding = [(padding, padding) for _ in range(pool_dim)]
        elif isinstance(padding, (list, tuple)):
            if isinstance(padding[0], int):
                if len(padding) == pool_dim:
                    padding = [(x, x) for x in padding]
                else:
                    raise ValueError(f'If padding is a sequence of ints, it '
                                     f'should has the length of {pool_dim}.')
            else:
                if not all([isinstance(x, (tuple, list)) for x in padding]):
                    raise ValueError(f'padding should be sequence of Tuple[int, int]. {padding}')
                if not all([len(x) == 2 for x in padding]):
                    raise ValueError(f"Each entry in padding must be tuple of 2 ints. {padding} ")
                if len(padding) == 1:
                    padding = tuple(padding) * pool_dim
                assert len(padding) == pool_dim, f'padding should has the length of {pool_dim}. {padding}'
        else:
            raise ValueError
        self.padding = padding

        # channel_axis
        assert channel_axis is None or isinstance(channel_axis, int), \
            f'channel_axis should be an int, but got {channel_axis}'
        self.channel_axis = channel_axis

        # in & out shapes
        if in_size is not None:
            in_size = tuple(in_size)
            self.in_size = in_size
            y = jax.eval_shape(self.update, jax.ShapeDtypeStruct((128,) + in_size, environ.dftype()))
            self.out_size = y.shape[1:]

    def update(self, x):
        x_dim = self.pool_dim + (0 if self.channel_axis is None else 1)
        if x.ndim < x_dim:
            raise ValueError(f'Excepted input with >= {x_dim} dimensions, but got {x.ndim}.')
        window_shape = self._infer_shape(x.ndim, self.kernel_size, 1)
        stride = self._infer_shape(x.ndim, self.stride, 1)
        padding = (self.padding if isinstance(self.padding, str) else
                   self._infer_shape(x.ndim, self.padding, element=(0, 0)))
        r = jax.lax.reduce_window(
            x,
            init_value=self.init_value,
            computation=self.computation,
            window_dimensions=window_shape,
            window_strides=stride,
            padding=padding
        )
        return r

    def _infer_shape(self, x_dim, inputs, element):
        channel_axis = self.channel_axis
        if channel_axis and not 0 <= abs(channel_axis) < x_dim:
            raise ValueError(f"Invalid channel axis {channel_axis} for input with {x_dim} dimensions")
        if channel_axis and channel_axis < 0:
            channel_axis = x_dim + channel_axis
        all_dims = list(range(x_dim))
        if channel_axis is not None:
            all_dims.pop(channel_axis)
        pool_dims = all_dims[-self.pool_dim:]
        results = [element] * x_dim
        for i, dim in enumerate(pool_dims):
            results[dim] = inputs[i]
        return results


class _AvgPool(_MaxPool):
    def update(self, x):
        x_dim = self.pool_dim + (0 if self.channel_axis is None else 1)
        if x.ndim < x_dim:
            raise ValueError(f'Excepted input with >= {x_dim} dimensions, but got {x.ndim}.')
        dims = self._infer_shape(x.ndim, self.kernel_size, 1)
        stride = self._infer_shape(x.ndim, self.stride, 1)
        padding = (self.padding if isinstance(self.padding, str) else
                   self._infer_shape(x.ndim, self.padding, element=(0, 0)))
        pooled = jax.lax.reduce_window(x,
                                       init_value=self.init_value,
                                       computation=self.computation,
                                       window_dimensions=dims,
                                       window_strides=stride,
                                       padding=padding)
        if padding == "VALID":
            # Avoid the extra reduce_window.
            return pooled / np.prod(dims)
        else:
            # Count the number of valid entries at each input point, then use that for
            # computing average. Assumes that any two arrays of same shape will be
            # padded the same.
            window_counts = jax.lax.reduce_window(jnp.ones_like(x),
                                                  init_value=self.init_value,
                                                  computation=self.computation,
                                                  window_dimensions=dims,
                                                  window_strides=stride,
                                                  padding=padding)
            assert pooled.shape == window_counts.shape
            return pooled / window_counts


class MaxPool1d(_MaxPool):
    r"""Applies a 1D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, L, C)`
    and output :math:`(N, L_{out}, C)` can be precisely described as:

    .. math::
        out(N_i, k, C_j) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, stride \times k + m, C_j)

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the
    sliding window. This `link`_ has a nice visualization of the pooling parameters.

    Shape:
        - Input: :math:`(N, L_{in}, C)` or :math:`(L_{in}, C)`.
        - Output: :math:`(N, L_{out}, C)` or :math:`(L_{out}, C)`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor


    Examples::

        >>> import brainstate as brainstate
        >>> # pool of size=3, stride=2
        >>> m = MaxPool1d(3, stride=2, channel_axis=-1)
        >>> input = brainstate.random.randn(20, 50, 16)
        >>> output = m(input)
        >>> output.shape
        (20, 24, 16)

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    kernel_size: int, sequence of int
      An integer, or a sequence of integers defining the window to reduce over.
    stride: int, sequence of int
      An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
    padding: str, int, sequence of tuple
      Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of n `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: optional, str
      The object name.

    .. _link:
          https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        kernel_size: Size,
        stride: Union[int, Sequence[int]] = None,
        padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
        channel_axis: Optional[int] = -1,
        name: Optional[str] = None,
        in_size: Optional[Size] = None,
    ):
        super().__init__(in_size=in_size,
                         init_value=-jax.numpy.inf,
                         computation=jax.lax.max,
                         pool_dim=1,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         channel_axis=channel_axis,
                         name=name)


class MaxPool2d(_MaxPool):
    r"""Applies a 2D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, H, W, C)`,
    output :math:`(N, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, h, w, C_j) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n, C_j)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.


    Shape:
        - Input: :math:`(N, H_{in}, W_{in}, C)` or :math:`(H_{in}, W_{in}, C)`
        - Output: :math:`(N, H_{out}, W_{out}, C)` or :math:`(H_{out}, W_{out}, C)`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> import brainstate as brainstate
        >>> # pool of square window of size=3, stride=2
        >>> m = MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = MaxPool2d((3, 2), stride=(2, 1), channel_axis=-1)
        >>> input = brainstate.random.randn(20, 50, 32, 16)
        >>> output = m(input)
        >>> output.shape
        (20, 24, 31, 16)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    kernel_size: int, sequence of int
      An integer, or a sequence of integers defining the window to reduce over.
    stride: int, sequence of int
      An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
    padding: str, int, sequence of tuple
      Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of n `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: optional, str
      The object name.

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        kernel_size: Size,
        stride: Union[int, Sequence[int]] = None,
        padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
        channel_axis: Optional[int] = -1,
        name: Optional[str] = None,
        in_size: Optional[Size] = None,
    ):
        super().__init__(in_size=in_size,
                         init_value=-jax.numpy.inf,
                         computation=jax.lax.max,
                         pool_dim=2,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         channel_axis=channel_axis,
                         name=name)


class MaxPool3d(_MaxPool):
    r"""Applies a 3D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, D, H, W, C)`,
    output :math:`(N, D_{out}, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
          \text{out}(N_i, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                      & \text{input}(N_i, \text{stride[0]} \times d + k,
                                      \text{stride[1]} \times h + m, \text{stride[2]} \times w + n, C_j)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.


    Shape:
        - Input: :math:`(N, D_{in}, H_{in}, W_{in}, C)` or :math:`(D_{in}, H_{in}, W_{in}, C)`.
        - Output: :math:`(N, D_{out}, H_{out}, W_{out}, C)` or :math:`(D_{out}, H_{out}, W_{out}, C)`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> import brainstate as brainstate
        >>> # pool of square window of size=3, stride=2
        >>> m = MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = MaxPool3d((3, 2, 2), stride=(2, 1, 2), channel_axis=-1)
        >>> input = brainstate.random.randn(20, 50, 44, 31, 16)
        >>> output = m(input)
        >>> output.shape
        (20, 24, 43, 15, 16)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    kernel_size: int, sequence of int
      An integer, or a sequence of integers defining the window to reduce over.
    stride: int, sequence of int
      An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
    padding: str, int, sequence of tuple
      Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of n `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: optional, str
      The object name.

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        kernel_size: Size,
        stride: Union[int, Sequence[int]] = None,
        padding: Union[str, int, Tuple[int], Sequence[Tuple[int, int]]] = "VALID",
        channel_axis: Optional[int] = -1,
        name: Optional[str] = None,
        in_size: Optional[Size] = None,
    ):
        super().__init__(in_size=in_size,
                         init_value=-jax.numpy.inf,
                         computation=jax.lax.max,
                         pool_dim=3,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         channel_axis=channel_axis,
                         name=name)


class AvgPool1d(_AvgPool):
    r"""Applies a 1D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, L, C)`,
    output :math:`(N, L_{out}, C)` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \text{out}(N_i, l, C_j) = \frac{1}{k} \sum_{m=0}^{k-1}
                               \text{input}(N_i, \text{stride} \times l + m, C_j)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Shape:
        - Input: :math:`(N, L_{in}, C)` or :math:`(L_{in}, C)`.
        - Output: :math:`(N, L_{out}, C)` or :math:`(L_{out}, C)`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} +
              2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

    Examples::

        >>> import brainstate as brainstate
        >>> # pool with window of size=3, stride=2
        >>> m = AvgPool1d(3, stride=2)
        >>> input = brainstate.random.randn(20, 50, 16)
        >>> m(input).shape
        (20, 24, 16)

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    kernel_size: int, sequence of int
      An integer, or a sequence of integers defining the window to reduce over.
    stride: int, sequence of int
      An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
    padding: str, int, sequence of tuple
      Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of n `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: optional, str
      The object name.

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        kernel_size: Size,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
        channel_axis: Optional[int] = -1,
        name: Optional[str] = None,
        in_size: Optional[Size] = None,
    ):
        super().__init__(in_size=in_size,
                         init_value=0.,
                         computation=jax.lax.add,
                         pool_dim=1,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         channel_axis=channel_axis,
                         name=name)


class AvgPool2d(_AvgPool):
    r"""Applies a 2D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, H, W, C)`,
    output :math:`(N, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, h, w, C_j)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, stride[0] \times h + m, stride[1] \times w + n, C_j)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Shape:
        - Input: :math:`(N, H_{in}, W_{in}, C)` or :math:`(H_{in}, W_{in}, C)`.
        - Output: :math:`(N, H_{out}, W_{out}, C)` or :math:`(H_{out}, W_{out}, C)`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> import brainstate as brainstate
        >>> # pool of square window of size=3, stride=2
        >>> m = AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = AvgPool2d((3, 2), stride=(2, 1))
        >>> input = brainstate.random.randn(20, 50, 32, , 16)
        >>> output = m(input)
        >>> output.shape
        (20, 24, 31, 16)

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    kernel_size: int, sequence of int
      An integer, or a sequence of integers defining the window to reduce over.
    stride: int, sequence of int
      An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
    padding: str, int, sequence of tuple
      Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of n `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: optional, str
      The object name.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        kernel_size: Size,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
        channel_axis: Optional[int] = -1,
        name: Optional[str] = None,
        in_size: Optional[Size] = None,
    ):
        super().__init__(in_size=in_size,
                         init_value=0.,
                         computation=jax.lax.add,
                         pool_dim=2,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         channel_axis=channel_axis,
                         name=name)


class AvgPool3d(_AvgPool):
    r"""Applies a 3D average pooling over an input signal composed of several input planes.


    In the simplest case, the output value of the layer with input size :math:`(N, D, H, W, C)`,
    output :math:`(N, D_{out}, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, d, h, w, C_j) ={} & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
                                          & \frac{\text{input}(N_i, \text{stride}[0] \times d + k,
                                                  \text{stride}[1] \times h + m, \text{stride}[2] \times w + n, C_j)}
                                                 {kD \times kH \times kW}
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides
    for :attr:`padding` number of points.

    Shape:
        - Input: :math:`(N, D_{in}, H_{in}, W_{in}, C)` or :math:`(D_{in}, H_{in}, W_{in}, C)`.
        - Output: :math:`(N, D_{out}, H_{out}, W_{out}, C)` or
          :math:`(D_{out}, H_{out}, W_{out}, C)`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] -
                    \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -
                    \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -
                    \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> import brainstate as brainstate
        >>> # pool of square window of size=3, stride=2
        >>> m = AvgPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = AvgPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = brainstate.random.randn(20, 50, 44, 31, 16)
        >>> output = m(input)
        >>> output.shape
        (20, 24, 43, 15, 16)

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    kernel_size: int, sequence of int
      An integer, or a sequence of integers defining the window to reduce over.
    stride: int, sequence of int
      An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
    padding: str, int, sequence of tuple
      Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of n `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: optional, str
      The object name.

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        kernel_size: Size,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
        channel_axis: Optional[int] = -1,
        name: Optional[str] = None,
        in_size: Optional[Size] = None,
    ):
        super().__init__(in_size=in_size,
                         init_value=0.,
                         computation=jax.lax.add,
                         pool_dim=3,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         channel_axis=channel_axis,
                         name=name)


def _adaptive_pool1d(x, target_size: int, operation: Callable):
    """Adaptive pool 1D.

    Args:
      x: The input. Should be a JAX array of shape `(dim,)`.
      target_size: The shape of the output after the pooling operation `(target_size,)`.
      operation: The pooling operation to be performed on the input array.

    Returns:
      A JAX array of shape `(target_size, )`.
    """
    size = jnp.size(x)
    num_head_arrays = size % target_size
    num_block = size // target_size
    if num_head_arrays != 0:
        head_end_index = num_head_arrays * (num_block + 1)
        heads = jax.vmap(operation)(x[:head_end_index].reshape(num_head_arrays, -1))
        tails = jax.vmap(operation)(x[head_end_index:].reshape(-1, num_block))
        outs = jnp.concatenate([heads, tails])
    else:
        outs = jax.vmap(operation)(x.reshape(-1, num_block))
    return outs


def _generate_vmap(fun: Callable, map_axes: List[int]):
    map_axes = sorted(map_axes)
    for axis in map_axes:
        fun = jax.vmap(fun, in_axes=(axis, None, None), out_axes=axis)
    return fun


class _AdaptivePool(Module):
    """General N dimensional adaptive down-sampling to a target shape.

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    target_size: int, sequence of int
      The target output shape.
    num_spatial_dims: int
      The number of spatial dimensions.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    operation: Callable
      The down-sampling operation.
    name: str
      The class name.
    """

    def __init__(
        self,
        in_size: Size,
        target_size: Size,
        num_spatial_dims: int,
        operation: Callable,
        channel_axis: Optional[int] = -1,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.channel_axis = channel_axis
        self.operation = operation
        if isinstance(target_size, int):
            self.target_shape = (target_size,) * num_spatial_dims
        elif isinstance(target_size, Sequence) and (len(target_size) == num_spatial_dims):
            self.target_shape = target_size
        else:
            raise ValueError("`target_size` must either be an int or tuple of length "
                             f"{num_spatial_dims} containing ints.")

        # in & out shapes
        if in_size is not None:
            in_size = tuple(in_size)
            self.in_size = in_size
            y = jax.eval_shape(self.update, jax.ShapeDtypeStruct((128,) + in_size, environ.dftype()))
            self.out_size = y.shape[1:]

    def update(self, x):
        """Input-output mapping.

        Parameters
        ----------
        x: Array
          Inputs. Should be a JAX array of shape `(..., dim_1, dim_2, channels)`
          or `(..., dim_1, dim_2)`.
        """
        # channel axis
        channel_axis = self.channel_axis

        if channel_axis:
            if not 0 <= abs(channel_axis) < x.ndim:
                raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
            if channel_axis < 0:
                channel_axis = x.ndim + channel_axis
        # input dimension
        if (x.ndim - (0 if channel_axis is None else 1)) < len(self.target_shape):
            raise ValueError(f"Invalid input dimension. Except >={len(self.target_shape)} "
                             f"dimensions (channel_axis={self.channel_axis}). "
                             f"But got {x.ndim} dimensions.")
        # pooling dimensions
        pool_dims = list(range(x.ndim))
        if channel_axis:
            pool_dims.pop(channel_axis)

        # pooling
        for i, di in enumerate(pool_dims[-len(self.target_shape):]):
            poo_axes = [j for j in range(x.ndim) if j != di]
            op = _generate_vmap(_adaptive_pool1d, poo_axes)
            x = op(x, self.target_shape[i], self.operation)
        return x


class AdaptiveAvgPool1d(_AdaptivePool):
    r"""Applies a 1D adaptive max pooling over an input signal composed of several input planes.

    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Shape:
        - Input: :math:`(N, L_{in}, C)` or :math:`(L_{in}, C)`.
        - Output: :math:`(N, L_{out}, C)` or :math:`(L_{out}, C)`, where
          :math:`L_{out}=\text{output\_size}`.

    Examples:

        >>> import brainstate as brainstate
        >>> # target output size of 5
        >>> m = AdaptiveMaxPool1d(5)
        >>> input = brainstate.random.randn(1, 64, 8)
        >>> output = m(input)
        >>> output.shape
        (1, 5, 8)

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    target_size: int, sequence of int
      The target output shape.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: str
      The class name.
    """
    __module__ = 'brainstate.nn'

    def __init__(self,
                 target_size: Union[int, Sequence[int]],
                 channel_axis: Optional[int] = -1,
                 name: Optional[str] = None,
                 in_size: Optional[Sequence[int]] = None, ):
        super().__init__(in_size=in_size,
                         target_size=target_size,
                         channel_axis=channel_axis,
                         num_spatial_dims=1,
                         operation=jnp.mean,
                         name=name)


class AdaptiveAvgPool2d(_AdaptivePool):
    r"""Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`H_{out} \times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Shape:
        - Input: :math:`(N, H_{in}, W_{in}, C)` or :math:`(H_{in}, W_{in}, C)`.
        - Output: :math:`(N, H_{out}, W_{out}, C)` or :math:`(H_{out}, W_{out}, C)`, where
          :math:`(H_{out}, W_{out})=\text{output\_size}`.

    Examples:

        >>> import brainstate as brainstate
        >>> # target output size of 5x7
        >>> m = AdaptiveMaxPool2d((5, 7))
        >>> input = brainstate.random.randn(1, 8, 9, 64)
        >>> output = m(input)
        >>> output.shape
        (1, 5, 7, 64)
        >>> # target output size of 7x7 (square)
        >>> m = AdaptiveMaxPool2d(7)
        >>> input = brainstate.random.randn(1, 10, 9, 64)
        >>> output = m(input)
        >>> output.shape
        (1, 7, 7, 64)
        >>> # target output size of 10x7
        >>> m = AdaptiveMaxPool2d((None, 7))
        >>> input = brainstate.random.randn(1, 10, 9, 64)
        >>> output = m(input)
        >>> output.shape
        (1, 10, 7, 64)

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    target_size: int, sequence of int
      The target output shape.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: str
      The class name.
    """
    __module__ = 'brainstate.nn'

    def __init__(self,
                 target_size: Union[int, Sequence[int]],
                 channel_axis: Optional[int] = -1,
                 name: Optional[str] = None,

                 in_size: Optional[Sequence[int]] = None, ):
        super().__init__(in_size=in_size,
                         target_size=target_size,
                         channel_axis=channel_axis,
                         num_spatial_dims=2,
                         operation=jnp.mean,
                         name=name)


class AdaptiveAvgPool3d(_AdaptivePool):
    r"""Applies a 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`D_{out} \times H_{out} \times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Shape:
        - Input: :math:`(N, D_{in}, H_{in}, W_{in}, C)` or :math:`(D_{in}, H_{in}, W_{in}, C)`.
        - Output: :math:`(N, D_{out}, H_{out}, W_{out}, C)` or :math:`(D_{out}, H_{out}, W_{out}, C)`,
          where :math:`(D_{out}, H_{out}, W_{out})=\text{output\_size}`.

    Examples:

        >>> import brainstate as brainstate
        >>> # target output size of 5x7x9
        >>> m = AdaptiveMaxPool3d((5, 7, 9))
        >>> input = brainstate.random.randn(1, 8, 9, 10, 64)
        >>> output = m(input)
        >>> output.shape
        (1, 5, 7, 9, 64)
        >>> # target output size of 7x7x7 (cube)
        >>> m = AdaptiveMaxPool3d(7)
        >>> input = brainstate.random.randn(1, 10, 9, 8, 64)
        >>> output = m(input)
        >>> output.shape
        (1, 7, 7, 7, 64)
        >>> # target output size of 7x9x8
        >>> m = AdaptiveMaxPool3d((7, None, None))
        >>> input = brainstate.random.randn(1, 10, 9, 8, 64)
        >>> output = m(input)
        >>> output.shape
        (1, 7, 9, 8, 64)

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    target_size: int, sequence of int
      The target output shape.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: str
      The class name.
    """
    __module__ = 'brainstate.nn'

    def __init__(self,
                 target_size: Union[int, Sequence[int]],
                 channel_axis: Optional[int] = -1,
                 name: Optional[str] = None,
                 in_size: Optional[Sequence[int]] = None, ):
        super().__init__(in_size=in_size,
                         target_size=target_size,
                         channel_axis=channel_axis,
                         num_spatial_dims=3,
                         operation=jnp.mean,
                         name=name)


class AdaptiveMaxPool1d(_AdaptivePool):
    """Adaptive one-dimensional maximum down-sampling.

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    target_size: int, sequence of int
      The target output shape.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: str
      The class name.
    """
    __module__ = 'brainstate.nn'

    def __init__(self,
                 target_size: Union[int, Sequence[int]],
                 channel_axis: Optional[int] = -1,
                 name: Optional[str] = None,
                 in_size: Optional[Sequence[int]] = None, ):
        super().__init__(in_size=in_size,
                         target_size=target_size,
                         channel_axis=channel_axis,
                         num_spatial_dims=1,
                         operation=jnp.max,
                         name=name)


class AdaptiveMaxPool2d(_AdaptivePool):
    """Adaptive two-dimensional maximum down-sampling.

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    target_size: int, sequence of int
      The target output shape.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: str
      The class name.
    """
    __module__ = 'brainstate.nn'

    def __init__(self,
                 target_size: Union[int, Sequence[int]],
                 channel_axis: Optional[int] = -1,
                 name: Optional[str] = None,
                 in_size: Optional[Sequence[int]] = None, ):
        super().__init__(in_size=in_size,
                         target_size=target_size,
                         channel_axis=channel_axis,
                         num_spatial_dims=2,
                         operation=jnp.max,
                         name=name)


class AdaptiveMaxPool3d(_AdaptivePool):
    """Adaptive three-dimensional maximum down-sampling.

    Parameters
    ----------
    in_size: Sequence of int
      The shape of the input tensor.
    target_size: int, sequence of int
      The target output shape.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    name: str
      The class name.
    """
    __module__ = 'brainstate.nn'

    def __init__(self,
                 target_size: Union[int, Sequence[int]],
                 channel_axis: Optional[int] = -1,
                 name: Optional[str] = None,
                 in_size: Optional[Sequence[int]] = None, ):
        super().__init__(in_size=in_size,
                         target_size=target_size,
                         channel_axis=channel_axis,
                         num_spatial_dims=3,
                         operation=jnp.max,
                         name=name)
