# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
Padding layers for neural networks.

This module provides various padding operations for 1D, 2D, and 3D tensors:
- ReflectionPad: Pads using reflection of the input boundary
- ReplicationPad: Pads using replication of the input boundary
- ZeroPad: Pads with zeros
- ConstantPad: Pads with a constant value
- CircularPad: Pads circularly (wrap around)
"""

import functools
from typing import Union, Sequence, Optional

import jax
import jax.numpy as jnp

from brainstate import environ
from brainstate.typing import Size
from ._module import Module

__all__ = [
    # Reflection padding
    'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d',
    # Replication padding
    'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d',
    # Zero padding
    'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d',
    # Constant padding
    'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
    # Circular padding
    'CircularPad1d', 'CircularPad2d', 'CircularPad3d',
]


def _format_padding(padding: Union[int, Sequence[int]], ndim: int) -> Sequence[tuple]:
    """
    Convert padding specification to format required by jax.numpy.pad.

    Args:
        padding: Padding size(s). Can be:
            - int: same padding for all sides
            - Sequence of length 2*ndim: (left, right) for each dimension
            - Sequence of length ndim: same padding for left and right of each dimension
        ndim: Number of spatial dimensions (1, 2, or 3)

    Returns:
        List of padding tuples for each dimension
    """
    if isinstance(padding, int):
        # Same padding for all sides of all dimensions
        return [(padding, padding) for _ in range(ndim)]

    padding = list(padding)

    if len(padding) == ndim:
        # Same padding for left and right of each dimension
        return [(p, p) for p in padding]
    elif len(padding) == 2 * ndim:
        # Different padding for each side: (left1, right1, left2, right2, ...)
        return [(padding[2 * i], padding[2 * i + 1]) for i in range(ndim)]
    else:
        raise ValueError(f"Padding must have length {ndim} or {2 * ndim}, got {len(padding)}")


# =============================================================================
# Reflection Padding
# =============================================================================

class ReflectionPad1d(Module):
    """
    Pads the input tensor using the reflection of the input boundary.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for both sides
        - Sequence[int] of length 2: (left, right)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ReflectionPad1d(2)
        >>> input = jnp.array([[[1, 2, 3, 4, 5]]])
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 9, 1)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 1)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 2:
            # (length, channels) -> pad only length dimension
            pad_width = [self.padding[0], (0, 0)]
        elif ndim == 3:
            # (batch, length, channels) -> pad only length dimension
            pad_width = [(0, 0), self.padding[0], (0, 0)]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='reflect')


class ReflectionPad2d(Module):
    """
    Pads the input tensor using the reflection of the input boundary.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 2: (height_pad, width_pad)
        - Sequence[int] of length 4: (left, right, top, bottom)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ReflectionPad2d(1)
        >>> input = jnp.ones((1, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 2)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 3:
            # (height, width, channels) -> pad height and width
            pad_width = [self.padding[0], self.padding[1], (0, 0)]
        elif ndim == 4:
            # (batch, height, width, channels) -> pad height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], (0, 0)]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='reflect')


class ReflectionPad3d(Module):
    """
    Pads the input tensor using the reflection of the input boundary.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 3: (depth_pad, height_pad, width_pad)
        - Sequence[int] of length 6: (left, right, top, bottom, front, back)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ReflectionPad3d(1)
        >>> input = jnp.ones((1, 4, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 3)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 4:
            # (depth, height, width, channels) -> pad depth, height and width
            pad_width = [self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        elif ndim == 5:
            # (batch, depth, height, width, channels) -> pad depth, height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='reflect')


# =============================================================================
# Replication Padding
# =============================================================================

class ReplicationPad1d(Module):
    """
    Pads the input tensor using replication of the input boundary.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for both sides
        - Sequence[int] of length 2: (left, right)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ReplicationPad1d(2)
        >>> input = jnp.array([[[1, 2, 3, 4, 5]]])
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 9, 1)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 1)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 2:
            # (length, channels) -> pad only length dimension
            pad_width = [self.padding[0], (0, 0)]
        elif ndim == 3:
            # (batch, length, channels) -> pad only length dimension
            pad_width = [(0, 0), self.padding[0], (0, 0)]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='edge')


class ReplicationPad2d(Module):
    """
    Pads the input tensor using replication of the input boundary.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 2: (height_pad, width_pad)
        - Sequence[int] of length 4: (left, right, top, bottom)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ReplicationPad2d(1)
        >>> input = jnp.ones((1, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 2)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 3:
            # (height, width, channels) -> pad height and width
            pad_width = [self.padding[0], self.padding[1], (0, 0)]
        elif ndim == 4:
            # (batch, height, width, channels) -> pad height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], (0, 0)]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='edge')


class ReplicationPad3d(Module):
    """
    Pads the input tensor using replication of the input boundary.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 3: (depth_pad, height_pad, width_pad)
        - Sequence[int] of length 6: (left, right, top, bottom, front, back)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ReplicationPad3d(1)
        >>> input = jnp.ones((1, 4, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 3)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 4:
            # (depth, height, width, channels) -> pad depth, height and width
            pad_width = [self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        elif ndim == 5:
            # (batch, depth, height, width, channels) -> pad depth, height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='edge')


# =============================================================================
# Zero Padding
# =============================================================================

class ZeroPad1d(Module):
    """
    Pads the input tensor with zeros.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for both sides
        - Sequence[int] of length 2: (left, right)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ZeroPad1d(2)
        >>> input = jnp.array([[[1, 2, 3, 4, 5]]])
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 9, 1)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 1)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 2:
            # (length, channels) -> pad only length dimension
            pad_width = [self.padding[0], (0, 0)]
        elif ndim == 3:
            # (batch, length, channels) -> pad only length dimension
            pad_width = [(0, 0), self.padding[0], (0, 0)]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='constant', constant_values=0)


class ZeroPad2d(Module):
    """
    Pads the input tensor with zeros.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 2: (height_pad, width_pad)
        - Sequence[int] of length 4: (left, right, top, bottom)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ZeroPad2d(1)
        >>> input = jnp.ones((1, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 2)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 3:
            # (height, width, channels) -> pad height and width
            pad_width = [self.padding[0], self.padding[1], (0, 0)]
        elif ndim == 4:
            # (batch, height, width, channels) -> pad height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], (0, 0)]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='constant', constant_values=0)


class ZeroPad3d(Module):
    """
    Pads the input tensor with zeros.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 3: (depth_pad, height_pad, width_pad)
        - Sequence[int] of length 6: (left, right, top, bottom, front, back)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ZeroPad3d(1)
        >>> input = jnp.ones((1, 4, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 3)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 4:
            # (depth, height, width, channels) -> pad depth, height and width
            pad_width = [self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        elif ndim == 5:
            # (batch, depth, height, width, channels) -> pad depth, height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='constant', constant_values=0)


# =============================================================================
# Constant Padding
# =============================================================================

class ConstantPad1d(Module):
    """
    Pads the input tensor with a constant value.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for both sides
        - Sequence[int] of length 2: (left, right)
    value : float, optional
        The constant value to use for padding. Default is 0.
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ConstantPad1d(2, value=3.5)
        >>> input = jnp.array([[[1, 2, 3, 4, 5]]])
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 9, 1)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        value: float = 0,
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 1)
        self.value = value
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 2:
            # (length, channels) -> pad only length dimension
            pad_width = [self.padding[0], (0, 0)]
        elif ndim == 3:
            # (batch, length, channels) -> pad only length dimension
            pad_width = [(0, 0), self.padding[0], (0, 0)]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='constant', constant_values=self.value)


class ConstantPad2d(Module):
    """
    Pads the input tensor with a constant value.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 2: (height_pad, width_pad)
        - Sequence[int] of length 4: (left, right, top, bottom)
    value : float, optional
        The constant value to use for padding. Default is 0.
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ConstantPad2d(1, value=3.5)
        >>> input = jnp.ones((1, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        value: float = 0,
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 2)
        self.value = value
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 3:
            # (height, width, channels) -> pad height and width
            pad_width = [self.padding[0], self.padding[1], (0, 0)]
        elif ndim == 4:
            # (batch, height, width, channels) -> pad height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], (0, 0)]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='constant', constant_values=self.value)


class ConstantPad3d(Module):
    """
    Pads the input tensor with a constant value.

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 3: (depth_pad, height_pad, width_pad)
        - Sequence[int] of length 6: (left, right, top, bottom, front, back)
    value : float, optional
        The constant value to use for padding. Default is 0.
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.ConstantPad3d(1, value=3.5)
        >>> input = jnp.ones((1, 4, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        value: float = 0,
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 3)
        self.value = value
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 4:
            # (depth, height, width, channels) -> pad depth, height and width
            pad_width = [self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        elif ndim == 5:
            # (batch, depth, height, width, channels) -> pad depth, height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='constant', constant_values=self.value)


# =============================================================================
# Circular Padding
# =============================================================================

class CircularPad1d(Module):
    """
    Pads the input tensor using circular padding (wrap around).

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for both sides
        - Sequence[int] of length 2: (left, right)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.CircularPad1d(2)
        >>> input = jnp.array([[[1, 2, 3, 4, 5]]])
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 9, 1)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 1)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 2:
            # (length, channels) -> pad only length dimension
            pad_width = [self.padding[0], (0, 0)]
        elif ndim == 3:
            # (batch, length, channels) -> pad only length dimension
            pad_width = [(0, 0), self.padding[0], (0, 0)]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='wrap')


class CircularPad2d(Module):
    """
    Pads the input tensor using circular padding (wrap around).

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 2: (height_pad, width_pad)
        - Sequence[int] of length 4: (left, right, top, bottom)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.CircularPad2d(1)
        >>> input = jnp.ones((1, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 2)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 3:
            # (height, width, channels) -> pad height and width
            pad_width = [self.padding[0], self.padding[1], (0, 0)]
        elif ndim == 4:
            # (batch, height, width, channels) -> pad height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], (0, 0)]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='wrap')


class CircularPad3d(Module):
    """
    Pads the input tensor using circular padding (wrap around).

    Parameters
    ----------
    padding : int or Sequence[int]
        The size of the padding. Can be:

        - int: same padding for all sides
        - Sequence[int] of length 3: (depth_pad, height_pad, width_pad)
        - Sequence[int] of length 6: (left, right, top, bottom, front, back)
    in_size : Size, optional
        The input size.
    name : str, optional
        The name of the module.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>> pad = brainstate.nn.CircularPad3d(1)
        >>> input = jnp.ones((1, 4, 4, 4, 3))
        >>> output = pad(input)
        >>> print(output.shape)
        (1, 6, 6, 6, 3)
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        in_size: Optional[Size] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.padding = _format_padding(padding, 3)
        if in_size is not None:
            self.in_size = in_size
            y = jax.eval_shape(
                functools.partial(self.update),
                jax.ShapeDtypeStruct(self.in_size, environ.dftype())
            )
            self.out_size = y.shape

    def update(self, x):
        # Add (0, 0) padding for non-spatial dimensions
        ndim = x.ndim
        if ndim == 4:
            # (depth, height, width, channels) -> pad depth, height and width
            pad_width = [self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        elif ndim == 5:
            # (batch, depth, height, width, channels) -> pad depth, height and width
            pad_width = [(0, 0), self.padding[0], self.padding[1], self.padding[2], (0, 0)]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {ndim}D")

        return jnp.pad(x, pad_width, mode='wrap')
