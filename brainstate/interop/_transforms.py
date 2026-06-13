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

"""Invertible array transforms used to bridge differing weight layouts across frameworks.

Every :class:`Transform` defines ``forward`` (brainstate -> foreign) and ``inverse``
(foreign -> brainstate) such that ``inverse(forward(x)) == x``. Expressing each layout
difference as a single invertible object means a layer's correspondence is declared once and
is correct in both conversion directions by construction.

All transforms use :mod:`brainunit.math` for array operations so that ``brainunit.Quantity``
weights are preserved (units carried through transposes/reshapes).
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import brainunit as u
import numpy as np

__all__ = [
    'Transform',
    'Identity',
    'Transpose',
    'PermuteAxes',
    'Reshape',
    'AddScalar',
    'ReorderBlocks',
    'SplitConcat',
    'Chain',
]


class Transform:
    """Base class for an invertible array transform."""

    def forward(self, x):
        """Map a brainstate-layout array to the foreign layout."""
        raise NotImplementedError

    def inverse(self, x):
        """Map a foreign-layout array back to the brainstate layout."""
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}()"


class Identity(Transform):
    """The identity transform (layouts already agree)."""

    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class Transpose(Transform):
    """Permute array axes by ``perm`` (forward) and by its inverse permutation (inverse).

    Parameters
    ----------
    perm : tuple of int
        Axis permutation applied in the forward (brainstate -> foreign) direction.
    """

    def __init__(self, perm: Sequence[int]):
        self.perm = tuple(perm)
        self.inv_perm = tuple(int(i) for i in np.argsort(self.perm))

    def forward(self, x):
        return u.math.transpose(x, self.perm)

    def inverse(self, x):
        return u.math.transpose(x, self.inv_perm)

    def __repr__(self):
        return f"Transpose(perm={self.perm})"


class PermuteAxes(Transpose):
    """Alias of :class:`Transpose` with a name that reads well for conv NHWC<->NCHW reorders."""


class Reshape(Transform):
    """Reshape between two layouts whose target shapes are computed from the input shape.

    Parameters
    ----------
    forward_shape : Callable[[tuple], tuple]
        Given the brainstate-layout shape, returns the foreign-layout shape.
    inverse_shape : Callable[[tuple], tuple]
        Given the foreign-layout shape, returns the brainstate-layout shape.
    """

    def __init__(self,
                 forward_shape: Callable[[Tuple[int, ...]], Tuple[int, ...]],
                 inverse_shape: Callable[[Tuple[int, ...]], Tuple[int, ...]]):
        self._forward_shape = forward_shape
        self._inverse_shape = inverse_shape

    def forward(self, x):
        return u.math.reshape(x, self._forward_shape(tuple(x.shape)))

    def inverse(self, x):
        return u.math.reshape(x, self._inverse_shape(tuple(x.shape)))


class AddScalar(Transform):
    """Add a constant in the forward direction, subtract it in the inverse direction.

    Used to fold a baked-in constant (e.g. the LSTM forget-gate ``+1``) into / out of a bias.

    Parameters
    ----------
    constant : float
        The constant added when going brainstate -> foreign.
    """

    def __init__(self, constant: float):
        self.constant = constant

    def forward(self, x):
        # Wrap the constant in x's unit so a united ``brainunit.Quantity`` bias
        # folds in correctly (raw ``x + constant`` raises on a dimensioned x).
        return u.math.add(x, u.Quantity(self.constant, unit=u.get_unit(x)))

    def inverse(self, x):
        return u.math.subtract(x, u.Quantity(self.constant, unit=u.get_unit(x)))

    def __repr__(self):
        return f"AddScalar({self.constant})"


class ReorderBlocks(Transform):
    """Reorder equal-sized contiguous blocks along an axis.

    Splits the axis into ``len(order)`` equal blocks; the forward output's block ``p`` is the
    input's block ``order[p]``. The inverse uses the inverse permutation. Used for gate
    re-ordering (e.g. brainstate LSTM ``[i, g, f, o]`` <-> flax/torch ``[i, f, g, o]``).

    Parameters
    ----------
    axis : int
        Axis along which blocks are laid out.
    order : tuple of int
        Permutation; forward output block ``p`` = input block ``order[p]``.
    """

    def __init__(self, axis: int, order: Sequence[int]):
        self.axis = axis
        self.order = tuple(order)
        self.inv_order = tuple(int(i) for i in np.argsort(self.order))

    def _apply(self, x, order):
        n = len(order)
        blocks = u.math.split(x, n, axis=self.axis)
        return u.math.concatenate([blocks[i] for i in order], axis=self.axis)

    def forward(self, x):
        return self._apply(x, self.order)

    def inverse(self, x):
        return self._apply(x, self.inv_order)

    def __repr__(self):
        return f"ReorderBlocks(axis={self.axis}, order={self.order})"


class SplitConcat(Transform):
    """Split one array into a tuple of sub-arrays (forward); concatenate back (inverse).

    Unlike the other transforms this maps array <-> tuple-of-arrays, so it is used by the
    dedicated RNN adapters (to un-fuse brainstate's ``(in+h, ...)`` kernel into separate
    input/recurrent kernels) rather than by the generic per-role engine.

    Parameters
    ----------
    axis : int
        Axis along which to split / concatenate.
    sizes : tuple of int
        Sizes of the consecutive chunks along ``axis``.
    """

    def __init__(self, axis: int, sizes: Sequence[int]):
        self.axis = axis
        self.sizes = tuple(sizes)

    def forward(self, x) -> tuple:
        idx = np.cumsum(self.sizes)[:-1].tolist()
        return tuple(u.math.split(x, idx, axis=self.axis))

    def inverse(self, xs) -> object:
        return u.math.concatenate(list(xs), axis=self.axis)

    def __repr__(self):
        return f"SplitConcat(axis={self.axis}, sizes={self.sizes})"


class Chain(Transform):
    """Compose transforms: ``forward`` applies left-to-right, ``inverse`` right-to-left.

    Parameters
    ----------
    *transforms : Transform
        The transforms to compose. ``forward`` applies ``transforms[0]`` first.
    """

    def __init__(self, *transforms: Transform):
        self.transforms = transforms

    def forward(self, x):
        for tf in self.transforms:
            x = tf.forward(x)
        return x

    def inverse(self, x):
        for tf in reversed(self.transforms):
            x = tf.inverse(x)
        return x

    def __repr__(self):
        return f"Chain({', '.join(map(repr, self.transforms))})"
