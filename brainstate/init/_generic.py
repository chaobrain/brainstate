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

from typing import Union, Callable, Optional, Sequence

import brainunit as bu
import jax
import numpy as np

from brainstate._state import State
from brainstate._utils import set_module_as
from brainstate.typing import ArrayLike
from ._base import to_size

__all__ = [
    'param',
    'state',
    'noise',
]


def _is_scalar(x):
    return bu.math.isscalar(x)


def are_broadcastable_shapes(shape1, shape2):
    """
    Check if two shapes are broadcastable.

    Parameters:
    - shape1: Tuple[int], the shape of the first array.
    - shape2: Tuple[int], the shape of the second array.

    Returns:
    - bool: True if shapes are broadcastable, False otherwise.
    """
    # Reverse the shapes to compare from the last dimension
    shape1_reversed = shape1[::-1]
    shape2_reversed = shape2[::-1]

    # Iterate over the dimensions of the shorter shape
    for dim1, dim2 in zip(shape1_reversed, shape2_reversed):
        # Check if the dimensions are not equal and neither is 1
        if dim1 != dim2 and 1 not in (dim1, dim2):
            return False

    # If all dimensions are compatible, the shapes are broadcastable
    return True


def _expand_params_to_match_sizes(params, sizes):
    """
    Expand the dimensions of params to match the dimensions of sizes.

    Parameters:
    - params: jax.Array or np.ndarray, the parameter array to be expanded.
    - sizes: tuple[int] or list[int], the target shape dimensions.

    Returns:
    - Expanded params with dimensions matching sizes.
    """
    params_dim = params.ndim
    sizes_dim = len(sizes)
    dim_diff = sizes_dim - params_dim

    # Add new axes to params if it has fewer dimensions than sizes
    for _ in range(dim_diff):
        params = bu.math.expand_dims(params, axis=0)  # Add new axis at the last dimension
    return params


@set_module_as('brainstate.init')
def param(
    parameter: Union[Callable, ArrayLike, State],
    sizes: Union[int, Sequence[int]],
    batch_size: Optional[int] = None,
    allow_none: bool = True,
    allow_scalar: bool = True,
):
    """Initialize parameters.

    Parameters
    ----------
    parameter: callable, ArrayLike, State
      The initialization of the parameter.
      - If it is None, the created parameter will be None.
      - If it is a callable function :math:`f`, the ``f(size)`` will be returned.
      - If it is an instance of :py:class:`init.Initializer``, the ``f(size)`` will be returned.
      - If it is a tensor, then this function check whether ``tensor.shape`` is equal to the given ``size``.
    sizes: int, sequence of int
      The shape of the parameter.
    batch_size: int
      The batch size.
    allow_none: bool
      Whether allow the parameter is None.
    allow_scalar: bool
      Whether allow the parameter is a scalar value.

    Returns
    -------
    param: ArrayType, float, int, bool, None
      The initialized parameter.

    See Also
    --------
    noise, state
    """
    # Check if the parameter is None
    if parameter is None:
        if allow_none:
            return None
        else:
            raise ValueError(f'Expect a parameter with type of float, ArrayType, Initializer, or '
                             f'Callable function, but we got None. ')

    # Check if the parameter is a scalar value
    if allow_scalar and _is_scalar(parameter):
        return parameter

    # Convert sizes to a tuple
    sizes = tuple(to_size(sizes))

    # Check if the parameter is a callable function
    if callable(parameter):
        if batch_size is not None:
            sizes = (batch_size,) + sizes
        return parameter(sizes)
    elif isinstance(parameter, (np.ndarray, jax.Array, bu.Quantity, State)):
        parameter = parameter
    else:
        raise ValueError(f'Unknown parameter type: {type(parameter)}')

    # Check if the shape of the parameter matches the given size
    if not are_broadcastable_shapes(parameter.shape, sizes):
        raise ValueError(f'The shape of the parameter {parameter.shape} does not match with the given size {sizes}')

    # Expand the parameter to match the given batch size
    param_value = parameter.value if isinstance(parameter, State) else parameter
    if batch_size is not None:
        if param_value.ndim <= len(sizes):
            # add a new axis to the params so that it matches the dimensionality of the given shape ``sizes``
            param_value = _expand_params_to_match_sizes(param_value, sizes)
            param_value = bu.math.repeat(
                bu.math.expand_dims(param_value, axis=0),
                batch_size,
                axis=0
            )
        else:
            if param_value.shape[0] != batch_size:
                raise ValueError(f'The batch size of the parameter {param_value.shape[0]} '
                                 f'does not match with the given batch size {batch_size}')
    return type(parameter)(param_value) if isinstance(parameter, State) else param_value


@set_module_as('brainstate.init')
def state(
    init: Union[Callable, jax.typing.ArrayLike],
    sizes: Union[int, Sequence[int]] = None,
    batch_size: Optional[int] = None,
):
    """
    Initialize a :math:`~.State` from a callable function or a data.
    """
    sizes = to_size(sizes)
    if callable(init):
        if sizes is None:
            raise ValueError('"varshape" cannot be None when data is a callable function.')
        sizes = list(sizes)
        if isinstance(batch_size, int):
            sizes.insert(0, batch_size)
        return State(init(sizes))

    else:
        if sizes is not None:
            if bu.math.shape(init) != sizes:
                raise ValueError(f'The shape of "data" {bu.math.shape(init)} does not match with "var_shape" {sizes}')
        if isinstance(batch_size, int):
            batch_size = batch_size
            data = State(
                bu.math.repeat(
                    bu.math.expand_dims(init, axis=0),
                    batch_size,
                    axis=0
                )
            )
        else:
            data = State(init)
    return data


@set_module_as('brainstate.init')
def noise(
    noises: Optional[Union[ArrayLike, Callable]],
    size: Union[int, Sequence[int]],
    num_vars: int = 1,
    noise_idx: int = 0,
) -> Optional[Callable]:
    """Initialize a noise function.

    Parameters
    ----------
    noises: Any
    size: Shape
      The size of the noise.
    num_vars: int
      The number of variables.
    noise_idx: int
      The index of the current noise among all noise variables.

    Returns
    -------
    noise_func: function, None
      The noise function.

    See Also
    --------
    variable_, parameter, delay

    """
    if callable(noises):
        return noises
    elif noises is None:
        return None
    else:
        noises = param(noises, size, allow_none=False)
        if num_vars > 1:
            noises_ = [None] * num_vars
            noises_[noise_idx] = noises
            noises = tuple(noises_)
        return lambda *args, **kwargs: noises
