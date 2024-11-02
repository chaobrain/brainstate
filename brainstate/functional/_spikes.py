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
    """Bitwise OR operation for spike tensors."""
    return x + y - x * y


def spike_bitwise_and(x, y):
    """Bitwise AND operation for spike tensors."""
    return x * y


def spike_bitwise_iand(x, y):
    """Bitwise IAND operation for spike tensors."""
    return (1 - x) * y


def spike_bitwise_not(x):
    """Bitwise NOT operation for spike tensors."""
    return 1 - x


def spike_bitwise_xor(x, y):
    """Bitwise XOR operation for spike tensors."""
    return x + y - 2 * x * y


def spike_bitwise_ixor(x, y):
    """Bitwise IXOR operation for spike tensors."""
    return x * (1 - y) + (1 - x) * y


def spike_bitwise(x, y, op: str):
    r"""Bitwise operation for spike tensors.

    .. math::

       \begin{array}{ccc}
        \hline \text { Mode } & \text { Expression for } \mathrm{g}(\mathrm{x}, \mathrm{y}) & \text { Code for } \mathrm{g}(\mathrm{x}, \mathrm{y}) \\
        \hline \text { ADD } & x+y & x+y \\
        \text { AND } & x \cap y & x \cdot y \\
        \text { IAND } & (\neg x) \cap y & (1-x) \cdot y \\
        \text { OR } & x \cup y & (x+y)-(x \cdot y) \\
        \hline
        \end{array}

    Args:
      x: A spike tensor.
      y: A spike tensor.
      op: A string indicating the bitwise operation to perform.
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
