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

# -*- coding: utf-8 -*-

from brainstate._state import ParamState
from ._module import Module
from functools import partial

import jax
import jax.numpy as jnp

from brainstate.typing import PyTree

__all__ = [
    "count_parameters",
    "clip_grad_norm",
]


def _format_parameter_count(num_params, precision=2):
    if num_params < 1000:
        return str(num_params)

    suffixes = ['', 'K', 'M', 'B', 'T', 'P', 'E']
    magnitude = 0
    while abs(num_params) >= 1000:
        magnitude += 1
        num_params /= 1000.0

    format_string = '{:.' + str(precision) + 'f}{}'
    formatted_value = format_string.format(num_params, suffixes[magnitude])

    # 检查是否接近 1000，如果是，尝试使用更大的基数
    if magnitude < len(suffixes) - 1 and num_params >= 1000 * (1 - 10 ** (-precision)):
        magnitude += 1
        num_params /= 1000.0
        formatted_value = format_string.format(num_params, suffixes[magnitude])

    return formatted_value


def count_parameters(
    module: Module,
    precision: int = 2,
    return_table: bool = False,
):
    """
    Count and display the number of trainable parameters in a neural network model.

    This function iterates through all the parameters of the given model,
    counts the number of parameters for each module, and displays them in a table.
    It also calculates and returns the total number of trainable parameters.

    Parameters:
    -----------
    model : brainstate.nn.Module
        The neural network model for which to count parameters.

    Returns:
    --------
    int
        The total number of trainable parameters in the model.

    Prints:
    -------
    A pretty-formatted table showing the number of parameters for each module,
    followed by the total number of trainable parameters.
    """
    assert isinstance(module, Module), "Input must be a neural network module"  # noqa: E501
    from prettytable import PrettyTable  # noqa: E501
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in module.states(ParamState).items():
        param = parameter.numel()
        table.add_row([name, _format_parameter_count(param, precision=precision)])
        total_params += param
    table.add_row(["Total", _format_parameter_count(total_params, precision=precision)])
    print(table)
    if return_table:
        return table, total_params
    return total_params


def clip_grad_norm(
    grad: PyTree,
    max_norm: float | jax.Array,
    norm_type: int | float | str | None = 2.0,
    return_norm: bool = False,
) -> PyTree | tuple[PyTree, jax.Array]:
    """
    Clip gradient norm of a PyTree of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are scaled if their
    norm exceeds the specified maximum.

    Parameters
    ----------
    grad : PyTree
        A PyTree structure (nested dict, list, tuple, etc.) containing
        JAX arrays representing gradients to be normalized.
    max_norm : float or jax.Array
        Maximum allowed norm of the gradients. If the computed norm
        exceeds this value, gradients will be scaled down proportionally.
    norm_type : int, float, str, or None, optional
        Type of the p-norm to compute. Default is 2.0 (L2 norm).
        Can be:

        - float: p-norm for any p >= 1
        - 'inf' or jnp.inf: infinity norm (maximum absolute value)
        - '-inf' or -jnp.inf: negative infinity norm (minimum absolute value)
        - int: integer p-norm
        - None: defaults to 2.0 (Euclidean norm)
    return_norm : bool, optional
        If True, returns a tuple (clipped_grad, total_norm).
        If False, returns only clipped_grad. Default is False.

    Returns
    -------
    clipped_grad : PyTree
        The input gradient structure with norms clipped to max_norm.
    total_norm : jax.Array, optional
        The computed norm of the gradients before clipping.
        Only returned if return_norm=True.

    Notes
    -----
    The gradient clipping is performed as:

    .. math::
        g_{\\text{clipped}} = g \\cdot \\min\\left(1, \\frac{\\text{max\\_norm}}{\\|g\\|_p}\\right)

    where :math:`\\|g\\|_p` is the p-norm of the concatenated gradient vector.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate 

        >>> # Simple gradient clipping without returning norm
        >>> grads = {'w': jnp.array([3.0, 4.0]), 'b': jnp.array([12.0])}
        >>> clipped_grads = brainstate.nn.clip_grad_norm(grads, max_norm=5.0)
        >>> print(f"Clipped w: {clipped_grads['w']}")
        Clipped w: [1.1538461 1.5384616]

        >>> # Gradient clipping with norm returned
        >>> grads = {'w': jnp.array([3.0, 4.0]), 'b': jnp.array([12.0])}
        >>> clipped_grads, norm = brainstate.nn.clip_grad_norm(grads, max_norm=5.0, return_norm=True)
        >>> print(f"Original norm: {norm:.2f}")
        Original norm: 13.00

        >>> # Using different norm types
        >>> grads = {'layer1': jnp.array([[-2.0, 3.0], [1.0, -4.0]])}
        >>>
        >>> # L2 norm (default)
        >>> clipped_l2, norm_l2 = brainstate.nn.clip_grad_norm(grads, max_norm=3.0, norm_type=2, return_norm=True)
        >>> print(f"L2 norm: {norm_l2:.2f}")
        L2 norm: 5.48
        >>>
        >>> # L1 norm
        >>> clipped_l1, norm_l1 = brainstate.nn.clip_grad_norm(grads, max_norm=5.0, norm_type=1, return_norm=True)
        >>> print(f"L1 norm: {norm_l1:.2f}")
        L1 norm: 10.00
        >>>
        >>> # Infinity norm
        >>> clipped_inf, norm_inf = brainstate.nn.clip_grad_norm(grads, max_norm=2.0, norm_type='inf', return_norm=True)
        >>> print(f"Inf norm: {norm_inf:.2f}")
        Inf norm: 4.00
    """
    if norm_type is None:
        norm_type = 2.0

    # Convert string 'inf' to jnp.inf for compatibility
    if norm_type == 'inf':
        norm_type = jnp.inf
    elif norm_type == '-inf':
        norm_type = -jnp.inf

    # Get all gradient leaves
    grad_leaves = jax.tree.leaves(grad)

    # Handle empty PyTree
    if not grad_leaves:
        if return_norm:
            return grad, jnp.array(0.0)
        return grad

    # Compute norm over flattened gradient values
    norm_fn = partial(jnp.linalg.norm, ord=norm_type)
    flat_grads = jnp.concatenate([g.ravel() for g in grad_leaves])
    total_norm = norm_fn(flat_grads)

    # Compute scaling factor
    clip_factor = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))

    # Apply clipping
    clipped_grad = jax.tree.map(lambda g: g * clip_factor, grad)

    if return_norm:
        return clipped_grad, total_norm
    return clipped_grad
