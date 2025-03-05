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

from __future__ import annotations

__all__ = [
    'spike_bitwise_or',
    'spike_bitwise_and',
    'spike_bitwise_iand',
    'spike_bitwise_not',
    'spike_bitwise_xor',
    'spike_bitwise_ixor',
    'spike_bitwise',
]


def spike_bitwise_or(x, y):
    """
    Perform a bitwise OR operation on spike tensors.

    This function computes the OR operation between two spike tensors.
    The OR operation is implemented using the formula: x + y - x * y,
    which is equivalent to the OR operation for binary values.

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise OR operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation assumes that the input tensors contain binary (0 or 1) values.
        For non-binary inputs, the behavior may not correspond to a true bitwise OR.
    """
    return x + y - x * y


def spike_bitwise_and(x, y):
    """
    Perform a bitwise AND operation on spike tensors.

    This function computes the AND operation between two spike tensors.
    The AND operation is equivalent to element-wise multiplication for binary values.

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise AND operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation is implemented using element-wise multiplication (x * y),
        which is equivalent to the AND operation for binary values.
    """
    return x * y


def spike_bitwise_iand(x, y):
    """
    Perform a bitwise IAND (Inverse AND) operation on spike tensors.

    This function computes the Inverse AND (IAND) operation between two spike tensors.
    IAND is defined as (NOT x) AND y.

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise IAND operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation is implemented using the formula: (1 - x) * y,
        which is equivalent to the IAND operation for binary values.
    """
    return (1 - x) * y


def spike_bitwise_not(x):
    """
    Perform a bitwise NOT operation on spike tensors.

    This function computes the NOT operation on a spike tensor.
    The NOT operation inverts the binary values in the tensor.

    Args:
        x (Tensor): The input spike tensor.

    Returns:
        Tensor: The result of the bitwise NOT operation applied to the input tensor.
               The output tensor has the same shape as the input tensor.

    Note:
        This operation is implemented using the formula: 1 - x,
        which is equivalent to the NOT operation for binary values.
    """
    return 1 - x


def spike_bitwise_xor(x, y):
    """
    Perform a bitwise XOR operation on spike tensors.

    This function computes the XOR operation between two spike tensors.
    XOR is defined as (x OR y) AND NOT (x AND y).

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise XOR operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation is implemented using the formula: x + y - 2 * x * y,
        which is equivalent to the XOR operation for binary values.
    """
    return x + y - 2 * x * y


def spike_bitwise_ixor(x, y):
    """
    Perform a bitwise IXOR (Inverse XOR) operation on spike tensors.

    This function computes the Inverse XOR (IXOR) operation between two spike tensors.
    IXOR is defined as (x AND NOT y) OR (NOT x AND y).

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise IXOR operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation is implemented using the formula: x * (1 - y) + (1 - x) * y,
        which is equivalent to the IXOR operation for binary values.
    """
    return x * (1 - y) + (1 - x) * y


def spike_bitwise(x, y, op: str):
    r"""
    Perform bitwise operations on spike tensors.

    This function applies various bitwise operations on spike tensors based on the specified operation.
    It supports 'or', 'and', 'iand', 'xor', and 'ixor' operations.

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.
        op (str): A string indicating the bitwise operation to perform.
            Supported operations are 'or', 'and', 'iand', 'xor', and 'ixor'.

    Returns:
        Tensor: The result of the bitwise operation applied to the input tensors.

    Raises:
        NotImplementedError: If an unsupported bitwise operation is specified.

    Note:
        The function uses the following mathematical expressions for different operations:
        
        .. math::

           \begin{array}{ccc}
            \hline \text { Mode } & \text { Expression for } \mathrm{g}(\mathrm{x}, \mathrm{y}) & \text { Code for } \mathrm{g}(\mathrm{x}, \mathrm{y}) \\
            \hline \text { ADD } & x+y & x+y \\
            \text { AND } & x \cap y & x \cdot y \\
            \text { IAND } & (\neg x) \cap y & (1-x) \cdot y \\
            \text { OR } & x \cup y & (x+y)-(x \cdot y) \\
            \hline
            \end{array}
    """
    if op == 'or':
        return spike_bitwise_or(x, y)
    elif op == 'and':
        return spike_bitwise_and(x, y)
    elif op == 'iand':
        return spike_bitwise_iand(x, y)
    elif op == 'xor':
        return spike_bitwise_xor(x, y)
    elif op == 'ixor':
        return spike_bitwise_ixor(x, y)
    else:
        raise NotImplementedError(f"Unsupported bitwise operation: {op}.")
