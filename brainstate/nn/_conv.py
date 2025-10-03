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

import collections.abc
from typing import Callable, Tuple, Union, Sequence, Optional, TypeVar

import jax
import jax.numpy as jnp

from brainstate._state import ParamState
from brainstate.typing import ArrayLike
from . import _init as init
from ._module import Module
from ._normalizations import weight_standardization

T = TypeVar('T')

__all__ = [
    'Conv1d', 'Conv2d', 'Conv3d',
    'ScaledWSConv1d', 'ScaledWSConv2d', 'ScaledWSConv3d',
]


def to_dimension_numbers(
    num_spatial_dims: int,
    channels_last: bool,
    transpose: bool
) -> jax.lax.ConvDimensionNumbers:
    """
    Create a `lax.ConvDimensionNumbers` for the given inputs.

    This function generates the dimension specification needed for JAX's convolution
    operations based on the number of spatial dimensions and data format.

    Parameters
    ----------
    num_spatial_dims : int
        The number of spatial dimensions (e.g., 1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    channels_last : bool
        If True, the input format is channels-last (e.g., [B, H, W, C] for 2D).
        If False, the input format is channels-first (e.g., [B, C, H, W] for 2D).
    transpose : bool
        If True, creates dimension numbers for transposed convolution.
        If False, creates dimension numbers for standard convolution.

    Returns
    -------
    jax.lax.ConvDimensionNumbers
        A named tuple specifying the dimension layout for lhs (input), rhs (kernel),
        and output of the convolution operation.

    Examples
    --------
    .. code-block:: python

        >>> # For 2D convolution with channels-last format
        >>> dim_nums = to_dimension_numbers(num_spatial_dims=2, channels_last=True, transpose=False)
        >>> print(dim_nums.lhs_spec)  # Input layout: (batch, spatial_1, spatial_2, channel)
        (0, 3, 1, 2)
    """
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
    """
    Replicates entry in `element` `num_replicate` times if needed.

    This utility function ensures that parameters like kernel_size, stride, etc.
    are properly formatted as tuples with the correct length for multi-dimensional
    convolutions.

    Parameters
    ----------
    element : T or Sequence[T]
        The element to replicate. Can be a scalar, string, or sequence.
    num_replicate : int
        The number of times to replicate the element.
    name : str
        The name of the parameter (used for error messages).

    Returns
    -------
    tuple of T
        A tuple containing the replicated elements.

    Raises
    ------
    TypeError
        If the element is a sequence with length not equal to 1 or `num_replicate`.

    Examples
    --------
    .. code-block:: python

        >>> # Replicate a scalar value
        >>> replicate(3, 2, 'kernel_size')
        (3, 3)
        >>>
        >>> # Keep a sequence as is if already correct length
        >>> replicate((3, 5), 2, 'kernel_size')
        (3, 5)
        >>>
        >>> # Replicate a single-element sequence
        >>> replicate([3], 2, 'kernel_size')
        (3, 3)
    """
    if isinstance(element, (str, bytes)) or not isinstance(element, collections.abc.Sequence):
        return (element,) * num_replicate
    elif len(element) == 1:
        return tuple(list(element) * num_replicate)
    elif len(element) == num_replicate:
        return tuple(element)
    else:
        raise TypeError(f"{name} must be a scalar or sequence of length 1 or "
                        f"sequence of length {num_replicate}.")


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
        w_init: Union[Callable, ArrayLike] = init.XavierNormalInit(),
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        name: str = None,
        param_type: type = ParamState,
    ):
        super().__init__(
            in_size=in_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            groups=groups,
            w_mask=w_mask,
            name=name
        )

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
        self.weight = param_type(params)

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
    """
    One-dimensional convolution layer.

    Applies a 1D convolution over an input signal composed of several input planes.
    The input should be a 3D array with the shape of ``[B, L, C]`` where B is batch size,
    L is the sequence length, and C is the number of input channels.

    This layer creates a convolution kernel that is convolved with the layer input
    over a single spatial dimension to produce a tensor of outputs.

    Parameters
    ----------
    in_size : tuple of int
        The input shape without the batch dimension. This argument is important as it is
        used to evaluate the output shape. For Conv1d: (L, C), Conv2d: (H, W, C), Conv3d: (H, W, D, C).
    out_channels : int
        The number of output channels (also called filters or feature maps).
    kernel_size : int or tuple of int
        The shape of the convolutional kernel. For 1D convolution, the kernel size can be
        passed as an integer. For 2D and 3D convolutions, it should be a tuple of integers
        or a single integer (which will be replicated for all spatial dimensions).
    stride : int or tuple of int, optional
        The stride of the convolution. An integer or a sequence of `n` integers, representing
        the inter-window strides along each spatial dimension. Default: 1.
    padding : {'SAME', 'VALID'} or int or tuple of int or sequence of tuple, optional
        The padding strategy. Can be:
        - 'SAME': pads the input so the output has the same shape as input when stride=1
        - 'VALID': no padding
        - int: symmetric padding applied to all spatial dimensions
        - tuple of (low, high): padding for each dimension
        - sequence of tuples: explicit padding for each spatial dimension
        Default: 'SAME'.
    lhs_dilation : int or tuple of int, optional
        The dilation factor for the input. An integer or a sequence of `n` integers, giving
        the dilation factor to apply in each spatial dimension of inputs. Convolution with
        input dilation `d` is equivalent to transposed convolution with stride `d`.
        Default: 1.
    rhs_dilation : int or tuple of int, optional
        The dilation factor for the kernel. An integer or a sequence of `n` integers, giving
        the dilation factor to apply in each spatial dimension of the convolution kernel.
        Convolution with kernel dilation is also known as 'atrous convolution', which increases
        the receptive field without increasing the number of parameters. Default: 1.
    groups : int, optional
        Number of groups for grouped convolution. Controls the connections between inputs and
        outputs. Both `in_channels` and `out_channels` must be divisible by `groups`. When
        groups=1 (default), all inputs are convolved to all outputs. When groups>1, the input
        and output channels are divided into groups, and each group is convolved independently.
        When groups=in_channels, this becomes a depthwise convolution. Default: 1.
    w_init : Callable or ArrayLike, optional
        The initializer for the convolutional kernel weights. Can be an initializer instance
        or a direct array. Default: XavierNormalInit().
    b_init : Callable or ArrayLike or None, optional
        The initializer for the bias. If None, no bias is added. Default: None.
    w_mask : ArrayLike or Callable or None, optional
        An optional mask applied to the weights during forward pass. Useful for implementing
        structured sparsity or custom connectivity patterns. Default: None.
    name : str, optional
        The name of the module. Default: None.
    param_type : type, optional
        The type of parameter state to use. Default: ParamState.

    Attributes
    ----------
    in_size : tuple of int
        The input shape (L, C) without batch dimension.
    out_size : tuple of int
        The output shape (L_out, out_channels) without batch dimension.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple of int
        Size of the convolving kernel.
    weight : ParamState
        The learnable weights (and bias if specified) of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a 1D convolution layer
        >>> conv = bst.nn.Conv1d(in_size=(28, 3), out_channels=16, kernel_size=5)
        >>>
        >>> # Apply to input: batch_size=2, length=28, channels=3
        >>> x = jnp.ones((2, 28, 3))
        >>> y = conv(x)
        >>> print(y.shape)  # (2, 28, 16) with 'SAME' padding
        >>>
        >>> # Without batch dimension
        >>> x_single = jnp.ones((28, 3))
        >>> y_single = conv(x_single)
        >>> print(y_single.shape)  # (28, 16)
        >>>
        >>> # With custom parameters
        >>> conv = bst.nn.Conv1d(
        ...     in_size=(100, 8),
        ...     out_channels=32,
        ...     kernel_size=3,
        ...     stride=2,
        ...     padding='VALID',
        ...     b_init=bst.init.ZeroInit()
        ... )

    Notes
    -----
    The output shape depends on the padding mode:

    - 'SAME': output length = ceil(input_length / stride)
    - 'VALID': output length = ceil((input_length - kernel_size + 1) / stride)

    When groups > 1, the convolution becomes a grouped convolution where input and
    output channels are divided into groups, reducing computational cost.
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 1


class Conv2d(_Conv):
    """
    Two-dimensional convolution layer.

    Applies a 2D convolution over an input signal composed of several input planes.
    The input should be a 4D array with the shape of ``[B, H, W, C]`` where B is batch size,
    H is height, W is width, and C is the number of input channels (channels-last format).

    This layer creates a convolution kernel that is convolved with the layer input
    to produce a tensor of outputs. It is commonly used in computer vision tasks.

    Parameters
    ----------
    in_size : tuple of int
        The input shape without the batch dimension. For Conv2d: (H, W, C) where H is height,
        W is width, and C is the number of input channels. This argument is important as it is
        used to evaluate the output shape.
    out_channels : int
        The number of output channels (also called filters or feature maps). These determine
        the depth of the output feature map.
    kernel_size : int or tuple of int
        The shape of the convolutional kernel. Can be:
        - An integer (e.g., 3): creates a square kernel (3, 3)
        - A tuple of two integers (e.g., (3, 5)): creates a (height, width) kernel
    stride : int or tuple of int, optional
        The stride of the convolution. Controls how much the kernel moves at each step.
        Can be:
        - An integer: same stride for both dimensions
        - A tuple of two integers: (stride_height, stride_width)
        Default: 1.
    padding : {'SAME', 'VALID'} or int or tuple of int or sequence of tuple, optional
        The padding strategy. Options:
        - 'SAME': output spatial size equals input size when stride=1
        - 'VALID': no padding, output size reduced by kernel size
        - int: same symmetric padding for all dimensions
        - (pad_h, pad_w): different padding for each dimension
        - [(pad_h_before, pad_h_after), (pad_w_before, pad_w_after)]: explicit padding
        Default: 'SAME'.
    lhs_dilation : int or tuple of int, optional
        The dilation factor for the input (left-hand side). Controls spacing between input elements.
        A value > 1 inserts zeros between input elements, equivalent to transposed convolution.
        Default: 1.
    rhs_dilation : int or tuple of int, optional
        The dilation factor for the kernel (right-hand side). Also known as atrous convolution
        or dilated convolution. Increases the receptive field without increasing parameters by
        inserting zeros between kernel elements. Useful for capturing multi-scale context.
        Default: 1.
    groups : int, optional
        Number of groups for grouped convolution. Must divide both `in_channels` and `out_channels`.
        - groups=1: standard convolution (all-to-all connections)
        - groups>1: grouped convolution (reduces parameters by factor of groups)
        - groups=in_channels: depthwise convolution (each input channel convolved separately)
        Default: 1.
    w_init : Callable or ArrayLike, optional
        Weight initializer for the convolutional kernel. Can be:
        - An initializer instance (e.g., bst.init.XavierNormal())
        - A callable that returns an array given a shape
        - A direct array matching the kernel shape
        Default: XavierNormalInit().
    b_init : Callable or ArrayLike or None, optional
        Bias initializer. If None, no bias term is added to the output.
        Default: None.
    w_mask : ArrayLike or Callable or None, optional
        Optional weight mask for structured sparsity or custom connectivity. The mask is
        element-wise multiplied with the kernel weights during the forward pass.
        Default: None.
    name : str, optional
        Name identifier for this module instance.
        Default: None.
    param_type : type, optional
        The parameter state class to use for managing learnable parameters.
        Default: ParamState.

    Attributes
    ----------
    in_size : tuple of int
        The input shape (H, W, C) without batch dimension.
    out_size : tuple of int
        The output shape (H_out, W_out, out_channels) without batch dimension.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple of int
        Size of the convolving kernel (height, width).
    weight : ParamState
        The learnable weights (and bias if specified) of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a 2D convolution layer
        >>> conv = bst.nn.Conv2d(in_size=(32, 32, 3), out_channels=64, kernel_size=3)
        >>>
        >>> # Apply to input: batch_size=8, height=32, width=32, channels=3
        >>> x = jnp.ones((8, 32, 32, 3))
        >>> y = conv(x)
        >>> print(y.shape)  # (8, 32, 32, 64) with 'SAME' padding
        >>>
        >>> # Without batch dimension
        >>> x_single = jnp.ones((32, 32, 3))
        >>> y_single = conv(x_single)
        >>> print(y_single.shape)  # (32, 32, 64)
        >>>
        >>> # With custom kernel size and stride
        >>> conv = bst.nn.Conv2d(
        ...     in_size=(224, 224, 3),
        ...     out_channels=128,
        ...     kernel_size=(5, 5),
        ...     stride=2,
        ...     padding='VALID'
        ... )
        >>>
        >>> # Depthwise convolution (groups = in_channels)
        >>> conv = bst.nn.Conv2d(
        ...     in_size=(64, 64, 32),
        ...     out_channels=32,
        ...     kernel_size=3,
        ...     groups=32
        ... )

    Notes
    -----
    The output spatial dimensions depend on the padding mode:

    - 'SAME': output_size = ceil(input_size / stride)
    - 'VALID': output_size = ceil((input_size - kernel_size + 1) / stride)

    Grouped convolution: When groups > 1, the input and output channels are divided
    into groups. Each group is convolved independently, which can significantly reduce
    computational cost while maintaining representational power.
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 2


class Conv3d(_Conv):
    """
    Three-dimensional convolution layer.

    Applies a 3D convolution over an input signal composed of several input planes.
    The input should be a 5D array with the shape of ``[B, H, W, D, C]`` where B is batch size,
    H is height, W is width, D is depth, and C is the number of input channels (channels-last format).

    This layer is commonly used for processing 3D data such as video sequences or
    volumetric medical imaging data.

    Parameters
    ----------
    in_size : tuple of int
        The input shape without the batch dimension. For Conv3d: (H, W, D, C) where H is height,
        W is width, D is depth, and C is the number of input channels. This argument is important
        as it is used to evaluate the output shape.
    out_channels : int
        The number of output channels (also called filters or feature maps). These determine
        the depth of the output feature map.
    kernel_size : int or tuple of int
        The shape of the convolutional kernel. Can be:
        - An integer (e.g., 3): creates a cubic kernel (3, 3, 3)
        - A tuple of three integers (e.g., (3, 5, 5)): creates a (height, width, depth) kernel
    stride : int or tuple of int, optional
        The stride of the convolution. Controls how much the kernel moves at each step.
        Can be:
        - An integer: same stride for all dimensions
        - A tuple of three integers: (stride_h, stride_w, stride_d)
        Default: 1.
    padding : {'SAME', 'VALID'} or int or tuple of int or sequence of tuple, optional
        The padding strategy. Options:
        - 'SAME': output spatial size equals input size when stride=1
        - 'VALID': no padding, output size reduced by kernel size
        - int: same symmetric padding for all dimensions
        - (pad_h, pad_w, pad_d): different padding for each dimension
        - [(pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (pad_d_before, pad_d_after)]: explicit padding
        Default: 'SAME'.
    lhs_dilation : int or tuple of int, optional
        The dilation factor for the input (left-hand side). Controls spacing between input elements.
        A value > 1 inserts zeros between input elements, equivalent to transposed convolution.
        Default: 1.
    rhs_dilation : int or tuple of int, optional
        The dilation factor for the kernel (right-hand side). Also known as atrous convolution
        or dilated convolution. Increases the receptive field without increasing parameters by
        inserting zeros between kernel elements. Particularly useful for 3D data to capture
        larger temporal/spatial context.
        Default: 1.
    groups : int, optional
        Number of groups for grouped convolution. Must divide both `in_channels` and `out_channels`.
        - groups=1: standard convolution (all-to-all connections)
        - groups>1: grouped convolution (significantly reduces parameters and computation for 3D)
        - groups=in_channels: depthwise convolution (each input channel convolved separately)
        Default: 1.
    w_init : Callable or ArrayLike, optional
        Weight initializer for the convolutional kernel. Can be:
        - An initializer instance (e.g., bst.init.XavierNormal())
        - A callable that returns an array given a shape
        - A direct array matching the kernel shape
        Default: XavierNormalInit().
    b_init : Callable or ArrayLike or None, optional
        Bias initializer. If None, no bias term is added to the output.
        Default: None.
    w_mask : ArrayLike or Callable or None, optional
        Optional weight mask for structured sparsity or custom connectivity. The mask is
        element-wise multiplied with the kernel weights during the forward pass.
        Default: None.
    name : str, optional
        Name identifier for this module instance.
        Default: None.
    param_type : type, optional
        The parameter state class to use for managing learnable parameters.
        Default: ParamState.

    Attributes
    ----------
    in_size : tuple of int
        The input shape (H, W, D, C) without batch dimension.
    out_size : tuple of int
        The output shape (H_out, W_out, D_out, out_channels) without batch dimension.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple of int
        Size of the convolving kernel (height, width, depth).
    weight : ParamState
        The learnable weights (and bias if specified) of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a 3D convolution layer for video data
        >>> conv = bst.nn.Conv3d(in_size=(16, 64, 64, 3), out_channels=32, kernel_size=3)
        >>>
        >>> # Apply to input: batch_size=4, frames=16, height=64, width=64, channels=3
        >>> x = jnp.ones((4, 16, 64, 64, 3))
        >>> y = conv(x)
        >>> print(y.shape)  # (4, 16, 64, 64, 32) with 'SAME' padding
        >>>
        >>> # Without batch dimension
        >>> x_single = jnp.ones((16, 64, 64, 3))
        >>> y_single = conv(x_single)
        >>> print(y_single.shape)  # (16, 64, 64, 32)
        >>>
        >>> # For medical imaging with custom parameters
        >>> conv = bst.nn.Conv3d(
        ...     in_size=(32, 32, 32, 1),
        ...     out_channels=64,
        ...     kernel_size=(3, 3, 3),
        ...     stride=2,
        ...     padding='VALID',
        ...     b_init=bst.init.Constant(0.1)
        ... )

    Notes
    -----
    The output spatial dimensions depend on the padding mode:

    - 'SAME': output_size = ceil(input_size / stride)
    - 'VALID': output_size = ceil((input_size - kernel_size + 1) / stride)

    3D convolutions are computationally expensive. Consider using:
    - Smaller kernel sizes
    - Grouped convolutions (groups > 1)
    - Separable convolutions for large-scale applications
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 3


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
        w_init: Union[Callable, ArrayLike] = init.XavierNormalInit(),
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        name: str = None,
        param_type: type = ParamState,
    ):
        super().__init__(
            in_size=in_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            groups=groups,
            w_mask=w_mask,
            name=name,
        )

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
        self.weight = param_type(params)

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
        w = weight_standardization(w, self.eps, params.get('gain', None))
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
    """
    One-dimensional convolution with weight standardization.

    This layer applies weight standardization to the convolutional kernel before
    performing the convolution operation. Weight standardization normalizes the
    weights to have zero mean and unit variance, which can accelerate training
    and improve model performance, especially when combined with group normalization.

    The input should be a 3D array with the shape of ``[B, L, C]`` where B is batch size,
    L is the sequence length, and C is the number of input channels.

    Parameters
    ----------
    in_size : tuple of int
        The input shape without the batch dimension. For Conv1d: (L, C) where L is the sequence
        length and C is the number of input channels. This argument is important as it is used
        to evaluate the output shape.
    out_channels : int
        The number of output channels (also called filters or feature maps). These determine
        the depth of the output feature map.
    kernel_size : int or tuple of int
        The shape of the convolutional kernel. For 1D convolution, can be:
        - An integer (e.g., 5): creates a kernel of size 5
        - A tuple with one integer (e.g., (5,)): equivalent to the above
    stride : int or tuple of int, optional
        The stride of the convolution. Controls how much the kernel moves at each step.
        Default: 1.
    padding : {'SAME', 'VALID'} or int or tuple of int or sequence of tuple, optional
        The padding strategy. Options:
        - 'SAME': output length equals input length when stride=1
        - 'VALID': no padding, output length reduced by kernel size
        - int: symmetric padding
        - (pad_before, pad_after): explicit padding for the sequence dimension
        Default: 'SAME'.
    lhs_dilation : int or tuple of int, optional
        The dilation factor for the input (left-hand side). Controls spacing between input elements.
        A value > 1 inserts zeros between input elements, equivalent to transposed convolution.
        Default: 1.
    rhs_dilation : int or tuple of int, optional
        The dilation factor for the kernel (right-hand side). Also known as atrous convolution
        or dilated convolution. Increases the receptive field without increasing parameters by
        inserting zeros between kernel elements. Useful for capturing long-range dependencies.
        Default: 1.
    groups : int, optional
        Number of groups for grouped convolution. Must divide both `in_channels` and `out_channels`.
        - groups=1: standard convolution (all-to-all connections)
        - groups>1: grouped convolution (reduces parameters by factor of groups)
        - groups=in_channels: depthwise convolution (each input channel convolved separately)
        Default: 1.
    w_init : Callable or ArrayLike, optional
        Weight initializer for the convolutional kernel. Can be:
        - An initializer instance (e.g., bst.init.XavierNormal())
        - A callable that returns an array given a shape
        - A direct array matching the kernel shape
        Default: XavierNormalInit().
    b_init : Callable or ArrayLike or None, optional
        Bias initializer. If None, no bias term is added to the output.
        Default: None.
    ws_gain : bool, optional
        Whether to include a learnable per-channel gain parameter in weight standardization.
        When True, adds a scaling factor that can be learned during training, improving
        model expressiveness. Recommended for most applications.
        Default: True.
    eps : float, optional
        Small constant for numerical stability in weight standardization. Prevents division
        by zero when computing weight standard deviation. Typical values: 1e-4 to 1e-5.
        Default: 1e-4.
    w_mask : ArrayLike or Callable or None, optional
        Optional weight mask for structured sparsity or custom connectivity. The mask is
        element-wise multiplied with the standardized kernel weights during the forward pass.
        Default: None.
    name : str, optional
        Name identifier for this module instance.
        Default: None.
    param_type : type, optional
        The parameter state class to use for managing learnable parameters.
        Default: ParamState.

    Attributes
    ----------
    in_size : tuple of int
        The input shape (L, C) without batch dimension.
    out_size : tuple of int
        The output shape (L_out, out_channels) without batch dimension.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple of int
        Size of the convolving kernel.
    weight : ParamState
        The learnable weights (and bias if specified) of the module.
    eps : float
        Small constant for numerical stability in weight standardization.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a 1D convolution with weight standardization
        >>> conv = bst.nn.ScaledWSConv1d(
        ...     in_size=(100, 16),
        ...     out_channels=32,
        ...     kernel_size=5
        ... )
        >>>
        >>> # Apply to input
        >>> x = jnp.ones((4, 100, 16))
        >>> y = conv(x)
        >>> print(y.shape)  # (4, 100, 32)
        >>>
        >>> # With custom epsilon and no gain
        >>> conv = bst.nn.ScaledWSConv1d(
        ...     in_size=(50, 8),
        ...     out_channels=16,
        ...     kernel_size=3,
        ...     ws_gain=False,
        ...     eps=1e-5
        ... )

    Notes
    -----
    Weight standardization reparameterizes the convolutional weights as:

    .. math::
        \\hat{W} = g \\cdot \\frac{W - \\mu_W}{\\sigma_W + \\epsilon}

    where :math:`\\mu_W` and :math:`\\sigma_W` are the mean and standard deviation
    of the weights, :math:`g` is a learnable gain parameter (if ws_gain=True),
    and :math:`\\epsilon` is a small constant for numerical stability.

    This technique is particularly effective when used with Group Normalization
    instead of Batch Normalization, as it reduces the dependence on batch statistics.

    References
    ----------
    .. [1] Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019).
           Weight Standardization. arXiv preprint arXiv:1903.10520.
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 1


class ScaledWSConv2d(_ScaledWSConv):
    """
    Two-dimensional convolution with weight standardization.

    This layer applies weight standardization to the convolutional kernel before
    performing the convolution operation. Weight standardization normalizes the
    weights to have zero mean and unit variance, improving training dynamics and
    model generalization, particularly in combination with group normalization.

    The input should be a 4D array with the shape of ``[B, H, W, C]`` where B is batch size,
    H is height, W is width, and C is the number of input channels (channels-last format).

    Parameters
    ----------
    in_size : tuple of int
        The input shape without the batch dimension. For Conv2d: (H, W, C) where H is height,
        W is width, and C is the number of input channels. This argument is important as it is
        used to evaluate the output shape.
    out_channels : int
        The number of output channels (also called filters or feature maps). These determine
        the depth of the output feature map.
    kernel_size : int or tuple of int
        The shape of the convolutional kernel. Can be:
        - An integer (e.g., 3): creates a square kernel (3, 3)
        - A tuple of two integers (e.g., (3, 5)): creates a (height, width) kernel
    stride : int or tuple of int, optional
        The stride of the convolution. Controls how much the kernel moves at each step.
        Can be:
        - An integer: same stride for both dimensions
        - A tuple of two integers: (stride_height, stride_width)
        Default: 1.
    padding : {'SAME', 'VALID'} or int or tuple of int or sequence of tuple, optional
        The padding strategy. Options:
        - 'SAME': output spatial size equals input size when stride=1
        - 'VALID': no padding, output size reduced by kernel size
        - int: same symmetric padding for all dimensions
        - (pad_h, pad_w): different padding for each dimension
        - [(pad_h_before, pad_h_after), (pad_w_before, pad_w_after)]: explicit padding
        Default: 'SAME'.
    lhs_dilation : int or tuple of int, optional
        The dilation factor for the input (left-hand side). Controls spacing between input elements.
        A value > 1 inserts zeros between input elements, equivalent to transposed convolution.
        Default: 1.
    rhs_dilation : int or tuple of int, optional
        The dilation factor for the kernel (right-hand side). Also known as atrous convolution
        or dilated convolution. Increases the receptive field without increasing parameters by
        inserting zeros between kernel elements. Useful for semantic segmentation and dense
        prediction tasks.
        Default: 1.
    groups : int, optional
        Number of groups for grouped convolution. Must divide both `in_channels` and `out_channels`.
        - groups=1: standard convolution (all-to-all connections)
        - groups>1: grouped convolution (reduces parameters by factor of groups)
        - groups=in_channels: depthwise convolution (each input channel convolved separately)
        Default: 1.
    w_init : Callable or ArrayLike, optional
        Weight initializer for the convolutional kernel. Can be:
        - An initializer instance (e.g., bst.init.XavierNormal())
        - A callable that returns an array given a shape
        - A direct array matching the kernel shape
        Default: XavierNormalInit().
    b_init : Callable or ArrayLike or None, optional
        Bias initializer. If None, no bias term is added to the output.
        Default: None.
    ws_gain : bool, optional
        Whether to include a learnable per-channel gain parameter in weight standardization.
        When True, adds a scaling factor that can be learned during training, improving
        model expressiveness. Highly recommended when using with Group Normalization.
        Default: True.
    eps : float, optional
        Small constant for numerical stability in weight standardization. Prevents division
        by zero when computing weight standard deviation. Typical values: 1e-4 to 1e-5.
        Default: 1e-4.
    w_mask : ArrayLike or Callable or None, optional
        Optional weight mask for structured sparsity or custom connectivity. The mask is
        element-wise multiplied with the standardized kernel weights during the forward pass.
        Default: None.
    name : str, optional
        Name identifier for this module instance.
        Default: None.
    param_type : type, optional
        The parameter state class to use for managing learnable parameters.
        Default: ParamState.

    Attributes
    ----------
    in_size : tuple of int
        The input shape (H, W, C) without batch dimension.
    out_size : tuple of int
        The output shape (H_out, W_out, out_channels) without batch dimension.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple of int
        Size of the convolving kernel (height, width).
    weight : ParamState
        The learnable weights (and bias if specified) of the module.
    eps : float
        Small constant for numerical stability in weight standardization.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a 2D convolution with weight standardization
        >>> conv = bst.nn.ScaledWSConv2d(
        ...     in_size=(64, 64, 3),
        ...     out_channels=32,
        ...     kernel_size=3
        ... )
        >>>
        >>> # Apply to input
        >>> x = jnp.ones((8, 64, 64, 3))
        >>> y = conv(x)
        >>> print(y.shape)  # (8, 64, 64, 32)
        >>>
        >>> # Combine with custom settings for ResNet-style architecture
        >>> conv = bst.nn.ScaledWSConv2d(
        ...     in_size=(224, 224, 3),
        ...     out_channels=64,
        ...     kernel_size=7,
        ...     stride=2,
        ...     padding='SAME',
        ...     ws_gain=True,
        ...     b_init=bst.init.ZeroInit()
        ... )
        >>>
        >>> # Depthwise separable convolution with weight standardization
        >>> conv = bst.nn.ScaledWSConv2d(
        ...     in_size=(32, 32, 128),
        ...     out_channels=128,
        ...     kernel_size=3,
        ...     groups=128,
        ...     ws_gain=False
        ... )

    Notes
    -----
    Weight standardization reparameterizes the convolutional weights as:

    .. math::
        \\hat{W} = g \\cdot \\frac{W - \\mu_W}{\\sigma_W + \\epsilon}

    where :math:`\\mu_W` and :math:`\\sigma_W` are the mean and standard deviation
    of the weights computed per output channel, :math:`g` is a learnable gain
    parameter (if ws_gain=True), and :math:`\\epsilon` is a small constant.

    Benefits of weight standardization:
    - Reduces internal covariate shift
    - Smooths the loss landscape
    - Works well with Group Normalization
    - Improves training stability with small batch sizes
    - Enables training deeper networks more easily

    References
    ----------
    .. [1] Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019).
           Weight Standardization. arXiv preprint arXiv:1903.10520.
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 2


class ScaledWSConv3d(_ScaledWSConv):
    """
    Three-dimensional convolution with weight standardization.

    This layer applies weight standardization to the convolutional kernel before
    performing the 3D convolution operation. Weight standardization normalizes the
    weights to have zero mean and unit variance, which improves training dynamics
    especially for 3D networks that are typically deeper and more parameter-heavy.

    The input should be a 5D array with the shape of ``[B, H, W, D, C]`` where B is batch size,
    H is height, W is width, D is depth, and C is the number of input channels (channels-last format).

    Parameters
    ----------
    in_size : tuple of int
        The input shape without the batch dimension. For Conv3d: (H, W, D, C) where H is height,
        W is width, D is depth, and C is the number of input channels. This argument is important
        as it is used to evaluate the output shape.
    out_channels : int
        The number of output channels (also called filters or feature maps). These determine
        the depth of the output feature map.
    kernel_size : int or tuple of int
        The shape of the convolutional kernel. Can be:
        - An integer (e.g., 3): creates a cubic kernel (3, 3, 3)
        - A tuple of three integers (e.g., (3, 5, 5)): creates a (height, width, depth) kernel
    stride : int or tuple of int, optional
        The stride of the convolution. Controls how much the kernel moves at each step.
        Can be:
        - An integer: same stride for all dimensions
        - A tuple of three integers: (stride_h, stride_w, stride_d)
        Default: 1.
    padding : {'SAME', 'VALID'} or int or tuple of int or sequence of tuple, optional
        The padding strategy. Options:
        - 'SAME': output spatial size equals input size when stride=1
        - 'VALID': no padding, output size reduced by kernel size
        - int: same symmetric padding for all dimensions
        - (pad_h, pad_w, pad_d): different padding for each dimension
        - [(pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (pad_d_before, pad_d_after)]: explicit padding
        Default: 'SAME'.
    lhs_dilation : int or tuple of int, optional
        The dilation factor for the input (left-hand side). Controls spacing between input elements.
        A value > 1 inserts zeros between input elements, equivalent to transposed convolution.
        Default: 1.
    rhs_dilation : int or tuple of int, optional
        The dilation factor for the kernel (right-hand side). Also known as atrous convolution
        or dilated convolution. Increases the receptive field without increasing parameters by
        inserting zeros between kernel elements. Particularly valuable for 3D to capture
        multi-scale temporal/spatial context efficiently.
        Default: 1.
    groups : int, optional
        Number of groups for grouped convolution. Must divide both `in_channels` and `out_channels`.
        - groups=1: standard convolution (all-to-all connections)
        - groups>1: grouped convolution (critical for reducing 3D conv computational cost)
        - groups=in_channels: depthwise convolution (each input channel convolved separately)
        Default: 1.
    w_init : Callable or ArrayLike, optional
        Weight initializer for the convolutional kernel. Can be:
        - An initializer instance (e.g., bst.init.XavierNormal())
        - A callable that returns an array given a shape
        - A direct array matching the kernel shape
        Default: XavierNormalInit().
    b_init : Callable or ArrayLike or None, optional
        Bias initializer. If None, no bias term is added to the output.
        Default: None.
    ws_gain : bool, optional
        Whether to include a learnable per-channel gain parameter in weight standardization.
        When True, adds a scaling factor that can be learned during training, improving
        model expressiveness. Particularly beneficial for deep 3D networks.
        Default: True.
    eps : float, optional
        Small constant for numerical stability in weight standardization. Prevents division
        by zero when computing weight standard deviation. Typical values: 1e-4 to 1e-5.
        Default: 1e-4.
    w_mask : ArrayLike or Callable or None, optional
        Optional weight mask for structured sparsity or custom connectivity. The mask is
        element-wise multiplied with the standardized kernel weights during the forward pass.
        Default: None.
    name : str, optional
        Name identifier for this module instance.
        Default: None.
    param_type : type, optional
        The parameter state class to use for managing learnable parameters.
        Default: ParamState.

    Attributes
    ----------
    in_size : tuple of int
        The input shape (H, W, D, C) without batch dimension.
    out_size : tuple of int
        The output shape (H_out, W_out, D_out, out_channels) without batch dimension.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple of int
        Size of the convolving kernel (height, width, depth).
    weight : ParamState
        The learnable weights (and bias if specified) of the module.
    eps : float
        Small constant for numerical stability in weight standardization.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a 3D convolution with weight standardization for video
        >>> conv = bst.nn.ScaledWSConv3d(
        ...     in_size=(16, 64, 64, 3),
        ...     out_channels=32,
        ...     kernel_size=3
        ... )
        >>>
        >>> # Apply to input
        >>> x = jnp.ones((4, 16, 64, 64, 3))
        >>> y = conv(x)
        >>> print(y.shape)  # (4, 16, 64, 64, 32)
        >>>
        >>> # For medical imaging with custom parameters
        >>> conv = bst.nn.ScaledWSConv3d(
        ...     in_size=(32, 32, 32, 1),
        ...     out_channels=64,
        ...     kernel_size=(3, 3, 3),
        ...     stride=2,
        ...     ws_gain=True,
        ...     eps=1e-5,
        ...     b_init=bst.init.Constant(0.01)
        ... )
        >>>
        >>> # 3D grouped convolution with weight standardization
        >>> conv = bst.nn.ScaledWSConv3d(
        ...     in_size=(8, 16, 16, 64),
        ...     out_channels=64,
        ...     kernel_size=3,
        ...     groups=8,
        ...     ws_gain=False
        ... )

    Notes
    -----
    Weight standardization reparameterizes the convolutional weights as:

    .. math::
        \\hat{W} = g \\cdot \\frac{W - \\mu_W}{\\sigma_W + \\epsilon}

    where :math:`\\mu_W` and :math:`\\sigma_W` are the mean and standard deviation
    of the weights, :math:`g` is a learnable gain parameter (if ws_gain=True),
    and :math:`\\epsilon` is a small constant for numerical stability.

    For 3D convolutions, weight standardization is particularly beneficial because:
    - 3D networks are typically much deeper and harder to train
    - Reduces sensitivity to weight initialization
    - Improves gradient flow through very deep networks
    - Works well with limited computational resources (small batches)
    - Compatible with Group Normalization for batch-independent normalization

    Applications include video understanding, medical imaging (CT, MRI scans),
    3D object recognition, and temporal sequence modeling.

    References
    ----------
    .. [1] Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019).
           Weight Standardization. arXiv preprint arXiv:1903.10520.
    """
    __module__ = 'brainstate.nn'
    num_spatial_dims: int = 3

