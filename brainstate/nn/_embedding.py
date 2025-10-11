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

from functools import lru_cache
from typing import Callable, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import core as jax_core

from brainstate._state import ParamState
from brainstate.typing import ArrayLike, Size
from . import init as init
from ._module import Module

__all__ = [
    'Embedding',
]


def _normalize_embedding_size(size: Size) -> Tuple[int, ...]:
    """Convert ``Size`` specifications to a validated tuple of integers."""
    size_array = np.asarray(size)
    if size_array.size == 0:
        raise ValueError('embedding_size must contain at least one dimension.')
    flat = size_array.reshape(-1)
    normalized = tuple(int(dim) for dim in flat)
    if any(dim < 0 for dim in normalized):
        raise ValueError('embedding_size must not contain negative values.')
    return normalized


@lru_cache(maxsize=None)
def _embedding_lookup_fn(
    padding_idx: Optional[int],
    scale_grad_by_freq: bool,
):
    """Return a lookup function with a custom VJP implementing embedding semantics."""

    @jax.custom_vjp
    def _lookup(weight: jax.Array, indices: jax.Array) -> jax.Array:
        indices = jnp.asarray(indices)
        return weight[indices]

    def _lookup_fwd(weight: jax.Array, indices: jax.Array):
        indices = jnp.asarray(indices)
        return weight[indices], (indices, weight.shape)

    def _lookup_bwd(residual, grad_output: jax.Array):
        indices, weight_shape = residual
        grad_output = jnp.asarray(grad_output)
        flat_idx = jnp.ravel(indices)
        if flat_idx.size == 0:
            return jnp.zeros(weight_shape, dtype=grad_output.dtype), None

        flat_idx = jnp.asarray(flat_idx, dtype=jnp.int32)
        grad_flat = grad_output.reshape((flat_idx.shape[0],) + weight_shape[1:])

        if scale_grad_by_freq:
            counts = jnp.bincount(flat_idx, length=weight_shape[0])
            counts = counts.astype(grad_flat.dtype)
            counts = jnp.where(counts == 0, 1.0, counts)
            scale = counts[flat_idx]
            grad_flat = grad_flat / scale.reshape((flat_idx.shape[0],) + (1,) * (grad_flat.ndim - 1))

        if padding_idx is not None:
            pad_value = jnp.asarray(padding_idx, dtype=flat_idx.dtype)
            mask = flat_idx != pad_value
            broadcast_shape = (flat_idx.shape[0],) + (1,) * (grad_flat.ndim - 1)
            grad_flat = grad_flat * mask.reshape(broadcast_shape).astype(grad_flat.dtype)

        grad_weight = jnp.zeros(weight_shape, dtype=grad_output.dtype)
        grad_weight = grad_weight.at[flat_idx].add(grad_flat)
        return grad_weight, None

    _lookup.defvjp(_lookup_fwd, _lookup_bwd)
    return _lookup


def _contains_tracer(tree) -> bool:
    """Return True if the pytree contains any JAX tracer values."""
    return any(isinstance(leaf, jax_core.Tracer) for leaf in jtu.tree_leaves(tree))



class Embedding(Module):
    r"""
    A simple lookup table that stores embeddings of a fixed size.

    This module is commonly used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Parameters
    ----------
    num_embeddings : int
        Size of embedding dictionary. Must be non-negative.
    embedding_size : Size
        Size of each embedding vector. Can be an int or a sequence of ints, and must
        contain only non-negative values.
    embedding_init : Callable or ArrayLike, optional
        The initializer for the embedding lookup table, of shape
        ``(num_embeddings, embedding_size)``. Default is ``LecunUniform()``.
    padding_idx : int, optional
        If specified, the entries at ``padding_idx`` do not contribute to the gradient;
        therefore, the embedding vector at ``padding_idx`` is not updated during training,
        i.e., it remains as a fixed "pad". For a newly constructed Embedding, the embedding
        vector at ``padding_idx`` will default to all zeros. Default is ``None``.
    max_norm : float, optional
        If given, each embedding vector with norm larger than ``max_norm`` is renormalized
        to have norm ``max_norm``. Default is ``None``.
    norm_type : float, optional
        The p of the p-norm to compute for the ``max_norm`` option. Default is ``2.0``.
    scale_grad_by_freq : bool, optional
        If given, this scales gradients by the inverse frequency of the words in
        the mini-batch. Default is ``False``.
    name : str, optional
        The name of the module.
    param_type : type, optional
        The parameter state type to use. Default is ``ParamState``.

    Attributes
    ----------
    num_embeddings : int
        Size of the embedding dictionary.
    embedding_size : tuple[int, ...]
        Size of each embedding vector.
    out_size : tuple[int, ...]
        Output size, same as ``embedding_size``.
    weight : ParamState
        The learnable weights of the module of shape
        ``(num_embeddings, *embedding_size)``.
    padding_idx : int or None
        Index of the padding token.
    max_norm : float or None
        Maximum norm for embedding vectors.
    norm_type : float
        Type of p-norm to compute for max_norm.
    scale_grad_by_freq : bool
        Whether to scale gradients by frequency.
    freeze : bool
        Whether the embedding weights are frozen.

    Examples
    --------
    Create an embedding layer with 10 words and 3-dimensional embeddings:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> embedding = brainstate.nn.Embedding(num_embeddings=10, embedding_size=3)
        >>> embedding.weight.value.shape
        (10, 3)

    Retrieve embeddings for specific indices:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> indices = jnp.array([1, 3, 5])
        >>> output = embedding(indices)
        >>> output.shape
        (3, 3)

    Use with a batch of sequences:

    .. code-block:: python

        >>> # Batch of 2 sequences, each with 4 tokens
        >>> batch_indices = jnp.array([[1, 2, 3, 4],
        ...                            [5, 6, 7, 8]])
        >>> output = embedding(batch_indices)
        >>> output.shape
        (2, 4, 3)

    Use padding_idx to keep padding embeddings fixed:

    .. code-block:: python

        >>> embedding = brainstate.nn.Embedding(num_embeddings=10, embedding_size=3, padding_idx=0)
        >>> # The embedding at index 0 will remain zeros and not be updated during training
        >>> indices = jnp.array([0, 2, 0, 5])
        >>> output = embedding(indices)
        >>> output[0]  # Will be zeros
        Array([0., 0., 0.], dtype=float32)

    Use max_norm to constrain embedding norms:

    .. code-block:: python

        >>> embedding = brainstate.nn.Embedding(num_embeddings=10, embedding_size=3, max_norm=1.0)
        >>> # All embeddings accessed in a forward pass are renormalized to have norm <= 1.0

    Load pretrained embeddings:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> pretrained = jnp.array([[1.0, 2.0, 3.0],
        ...                         [4.0, 5.0, 6.0],
        ...                         [7.0, 8.0, 9.0]])
        >>> embedding = brainstate.nn.Embedding.from_pretrained(pretrained, param_type=brainstate.FakeState)
        >>> embedding.weight.value.shape
        (3, 3)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: Size,
        embedding_init: Union[Callable, ArrayLike] = init.LecunUniform(),
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        freeze: bool = False,
        name: Optional[str] = None,
        param_type: type = ParamState,
    ):
        super().__init__(name=name)

        self.num_embeddings = int(num_embeddings)
        if self.num_embeddings < 0:
            raise ValueError('num_embeddings must not be negative.')

        embedding_size_tuple = _normalize_embedding_size(embedding_size)
        self.embedding_size = embedding_size_tuple
        self.out_size = embedding_size_tuple

        if padding_idx is not None:
            padding_idx = int(padding_idx)
            if padding_idx < 0 or padding_idx >= self.num_embeddings:
                raise ValueError(f'padding_idx must be within [0, {self.num_embeddings}).')
        self.padding_idx = padding_idx

        if max_norm is not None and max_norm <= 0:
            raise ValueError('max_norm must be positive when provided.')
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = bool(scale_grad_by_freq)
        self.freeze = bool(freeze)

        weight_shape = (self.num_embeddings, *self.out_size)
        weight = init.param(embedding_init, weight_shape)

        if self.padding_idx is not None:
            weight = weight.at[self.padding_idx].set(0)

        self.weight = param_type(weight)
        self._lookup = _embedding_lookup_fn(self.padding_idx, self.scale_grad_by_freq)

    @classmethod
    def from_pretrained(
        cls,
        embeddings: ArrayLike,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        freeze: bool = True,
        name: Optional[str] = None,
        param_type: type = ParamState,
    ):
        r"""
        Create an Embedding instance from given 2-dimensional array.

        Parameters
        ----------
        embeddings : ArrayLike
            Array containing weights for the Embedding. First dimension is passed to
            Embedding as ``num_embeddings``, remaining dimensions as ``embedding_size``.
        padding_idx : int, optional
            If specified, the entries at ``padding_idx`` do not contribute to the gradient.
            Default is ``None``.
        max_norm : float, optional
            See module initialization documentation. Default is ``None``.
        norm_type : float, optional
            See module initialization documentation. Default is ``2.0``.
        scale_grad_by_freq : bool, optional
            See module initialization documentation. Default is ``False``.
        freeze : bool, optional
            If ``True``, embeddings are frozen (no gradients). Default is ``True``.
        name : str, optional
            The name of the module.

        Returns
        -------
        Embedding
            An Embedding module with pretrained weights.

        Examples
        --------
        Load pretrained word embeddings:

        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainstate as brainstate
            >>> pretrained = jnp.array([[1.0, 2.0, 3.0],
            ...                         [4.0, 5.0, 6.0],
            ...                         [7.0, 8.0, 9.0]])
            >>> embedding = brainstate.nn.Embedding.from_pretrained(pretrained)
            >>> embedding.weight.value.shape
            (3, 3)
            >>> indices = jnp.array([1])
            >>> embedding(indices)
            Array([[4., 5., 6.]], dtype=float32)
        """
        embeddings = jnp.asarray(embeddings)
        if embeddings.ndim < 2:
            raise ValueError('embeddings must be at least 2-dimensional')

        num_embeddings = embeddings.shape[0]
        embedding_size = embeddings.shape[1:]

        instance = cls(
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            embedding_init=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            freeze=freeze,
            name=name,
            param_type=param_type,
        )

        instance.weight = param_type(jnp.asarray(embeddings))
        return instance

    def update(self, indices: ArrayLike):
        """
        Retrieve embeddings for the given indices.

        Parameters
        ----------
        indices : ArrayLike
            Indices to retrieve embeddings for. Can be any shape.

        Returns
        -------
        ArrayLike
            Embeddings corresponding to the indices, with shape
            ``(*indices.shape, *embedding_size)``.
        """
        indices = jnp.asarray(indices)
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError('Embedding indices must be integers.')

        weight_value = self.weight.value
        effective_weight = weight_value

        if self.max_norm is not None:
            renormed_weight = self._apply_max_norm(weight_value, indices)
            effective_weight = weight_value + jax.lax.stop_gradient(renormed_weight - weight_value)
            if not _contains_tracer(renormed_weight):
                self.weight.value = renormed_weight

        if self.freeze:
            effective_weight = jax.lax.stop_gradient(effective_weight)

        embeddings = self._lookup(effective_weight, indices)
        return embeddings

    def _apply_max_norm(self, weight: jax.Array, indices: jax.Array) -> jax.Array:
        """Apply max_norm constraint to the embedding weights for the given indices."""
        flat_idx = jnp.ravel(indices)
        if flat_idx.size == 0:
            return weight

        flat_idx = jnp.asarray(flat_idx, dtype=jnp.int32)
        if self.padding_idx is not None:
            pad_value = jnp.asarray(self.padding_idx, dtype=flat_idx.dtype)
            flat_idx = flat_idx[flat_idx != pad_value]

        if flat_idx.size == 0:
            return weight

        rows = weight[flat_idx]
        rows_flat = rows.reshape((rows.shape[0], -1))
        row_dtype = rows_flat.dtype

        norms = jnp.linalg.norm(rows_flat, ord=self.norm_type, axis=1, keepdims=True)
        max_norm = jnp.asarray(self.max_norm, dtype=row_dtype)
        eps = jnp.asarray(1e-8, dtype=row_dtype)
        one = jnp.asarray(1.0, dtype=row_dtype)
        scale = jnp.minimum(one, max_norm / (norms + eps))
        rows_scaled = (rows_flat * scale).reshape(rows.shape)

        return weight.at[flat_idx].set(rows_scaled)


