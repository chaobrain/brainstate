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

"""Shared interop machinery: lazy imports, conversion context, the framework-adapter base,
and the *brainstate side* of every conversion.

This module is the single home for brainstate's internal storage layout (which ``State``
attribute and dict key backs each parameter). Framework adapters call the ``build_*`` and
``bst_get_*`` / ``bst_set_*`` helpers here; they never reach into brainstate internals
directly. If a brainstate layer's internal structure changes, only this file changes.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import Optional, Tuple

import jax

import brainstate.nn as bnn
import brainstate.random as brandom
from brainstate.nn import init

from ._errors import MissingDependencyError, MissingShapeError

__all__ = ['Context', 'FrameworkAdapter', 'lazy_import', 'new_key']

_INSTALL_HINTS = {
    'flax': 'pip install flax',
    'equinox': 'pip install equinox',
}


def lazy_import(module_name: str) -> ModuleType:
    """Import an optional framework module, raising :class:`MissingDependencyError` if absent.

    Parameters
    ----------
    module_name : str
        Top-level package to import (e.g. ``"flax"``, ``"equinox"``).

    Returns
    -------
    module
        The imported module.
    """
    top = module_name.split('.')[0]
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        hint = _INSTALL_HINTS.get(top, f'pip install {top}')
        raise MissingDependencyError(top, hint) from e


def new_key() -> jax.Array:
    """Return a throwaway PRNG key from :mod:`brainstate.random` for foreign construction.

    Weights are overwritten immediately after construction, so the key value is irrelevant;
    sourcing it from ``brainstate.random`` keeps RNG usage consistent with the rest of the
    codebase.
    """
    return brandom.split_key()


@dataclass
class Context:
    """Threaded state for a single conversion call.

    Parameters
    ----------
    sample_input : tuple of int, optional
        Unbatched input shape (brainstate convention, no batch dim) used to materialize the
        concrete ``in_size`` of spatial layers (``Conv``, spatial ``BatchNorm``) during import.
    cur_size : tuple of int, optional
        The running input size as the engine walks a ``Sequential`` during import. Advanced to
        each brainstate layer's ``out_size`` after it is built.
    """

    sample_input: Optional[Tuple[int, ...]] = None
    cur_size: Optional[Tuple[int, ...]] = None
    rngs: object = None  # optional user-supplied nnx.Rngs for export construction
    key: object = None   # optional user-supplied PRNG key for equinox export construction

    def require_size(self, layer_name: str) -> Tuple[int, ...]:
        """Return the current spatial input size, or raise :class:`MissingShapeError`."""
        if self.cur_size is None:
            raise MissingShapeError(
                f"Importing `{layer_name}` requires the spatial input shape, but no "
                f"`sample_input` was provided. Pass `sample_input=<a single unbatched example "
                f"or its shape>` to the conversion function."
            )
        return tuple(self.cur_size)


class FrameworkAdapter(ABC):
    """Generic, layer-agnostic structural plumbing for one framework.

    Concrete adapters know how to recurse a model's containers and build a ``Sequential`` in
    their framework. Per-layer parameter read/write lives in the registered mappings, not here.
    """

    name: str

    @abstractmethod
    def is_sequential(self, node) -> bool:
        """Whether ``node`` is a supported sequential stack."""

    @abstractmethod
    def iter_children(self, node):
        """Yield ``(key, child)`` for a sequential node, in forward order."""

    @abstractmethod
    def has_child_modules(self, node) -> bool:
        """Whether ``node`` holds sub-modules (used to flag custom-forward containers)."""

    @abstractmethod
    def layer_type(self, node) -> type:
        """The concrete leaf type used for registry lookup."""

    @abstractmethod
    def build_sequential(self, children: list):
        """Build a framework-native sequential container from converted ``children``."""


# ---------------------------------------------------------------------------
# brainstate side: builders + accessors (the only place brainstate paths live)
# ---------------------------------------------------------------------------

def _zeros_init():
    return init.ZeroInit()


# --- Linear ---------------------------------------------------------------

def build_linear(in_features: int, out_features: int, has_bias: bool):
    """Construct a brainstate ``Linear`` with the given feature sizes and bias flag."""
    return bnn.Linear(in_features, out_features,
                      b_init=_zeros_init() if has_bias else None)


def bst_get_linear(layer):
    """Return ``(weight, bias_or_None)`` of a brainstate ``Linear``/``Conv``."""
    d = layer.weight.value
    return d['weight'], d.get('bias', None)


def bst_set_linear(layer, weight, bias):
    """Write weight (and optional bias) of a brainstate ``Linear``/``Conv``."""
    d = {'weight': weight}
    if bias is not None:
        d['bias'] = bias
    layer.weight.value = d


# --- Embedding ------------------------------------------------------------

def build_embedding(num_embeddings: int, embedding_size: int):
    """Construct a brainstate ``Embedding``."""
    return bnn.Embedding(num_embeddings, embedding_size)


def bst_get_embedding(layer):
    """Return the embedding table array."""
    return layer.weight.value


def bst_set_embedding(layer, table):
    """Write the embedding table array."""
    layer.weight.value = table


# --- Conv -----------------------------------------------------------------

_CONV_CLS = {1: bnn.Conv1d, 2: bnn.Conv2d, 3: bnn.Conv3d}


def build_conv(num_spatial_dims, in_size, out_channels, kernel_size,
               stride, padding, rhs_dilation, groups, has_bias):
    """Construct a brainstate ``Conv{1,2,3}d`` (channels-last)."""
    cls = _CONV_CLS[num_spatial_dims]
    return cls(in_size=tuple(in_size), out_channels=out_channels, kernel_size=kernel_size,
               stride=stride, padding=padding, rhs_dilation=rhs_dilation, groups=groups,
               b_init=_zeros_init() if has_bias else None)


# Conv shares Linear's weight-dict accessors (bst_get_linear / bst_set_linear).


# --- Normalization --------------------------------------------------------

def build_layernorm(in_size, has_scale, has_bias, epsilon):
    return bnn.LayerNorm(tuple(in_size), use_scale=has_scale, use_bias=has_bias, epsilon=epsilon)


def build_rmsnorm(in_size, has_scale, epsilon):
    return bnn.RMSNorm(tuple(in_size), use_scale=has_scale, epsilon=epsilon)


def build_groupnorm(in_size, num_groups, has_scale, has_bias, epsilon):
    return bnn.GroupNorm(tuple(in_size), num_groups=num_groups,
                         use_scale=has_scale, use_bias=has_bias, epsilon=epsilon)


def bst_get_norm(layer, attr, has_offset):
    """Return ``(scale_or_None, offset_or_None)`` from a normalization layer.

    Parameters
    ----------
    layer : Module
        The brainstate normalization layer.
    attr : str
        The ``State`` attribute holding the affine dict (``"weight"`` for LayerNorm/GroupNorm,
        ``"scale"`` for RMSNorm).
    has_offset : bool
        Whether an offset (bias) role is present.
    """
    state = getattr(layer, attr)
    if state is None:
        return None, None
    d = state.value
    scale = d.get('scale', None)
    offset = d.get('bias', None) if has_offset else None
    return scale, offset


def bst_set_norm(layer, attr, scale, offset):
    """Write the affine parameters of a normalization layer."""
    if scale is None and offset is None:
        return
    d = {}
    if scale is not None:
        d['scale'] = scale
    if offset is not None:
        d['bias'] = offset
    getattr(layer, attr).value = d


# --- BatchNorm ------------------------------------------------------------

_BN_CLS = {0: bnn.BatchNorm0d, 1: bnn.BatchNorm1d, 2: bnn.BatchNorm2d, 3: bnn.BatchNorm3d}


def build_batchnorm(num_spatial_dims, in_size, epsilon, momentum, affine):
    """Construct a brainstate ``BatchNorm{1,2,3}d`` (feature axis last)."""
    cls = _BN_CLS[num_spatial_dims]
    return cls(in_size=tuple(in_size), epsilon=epsilon, momentum=momentum, affine=affine)


def bst_get_batchnorm(layer):
    """Return ``(scale, offset, running_mean, running_var)`` (any may be ``None``)."""
    scale = offset = None
    if layer.weight is not None:
        d = layer.weight.value
        scale = d.get('scale', None)
        offset = d.get('bias', None)
    rmean = None if layer.running_mean is None else layer.running_mean.value
    rvar = None if layer.running_var is None else layer.running_var.value
    return scale, offset, rmean, rvar


def bst_set_batchnorm(layer, scale, offset, running_mean, running_var):
    """Write affine + running statistics of a brainstate ``BatchNorm``."""
    if layer.weight is not None:
        d = {}
        if scale is not None:
            d['scale'] = scale
        if offset is not None:
            d['bias'] = offset
        layer.weight.value = d
    if running_mean is not None:
        layer.running_mean.value = running_mean
    if running_var is not None:
        layer.running_var.value = running_var


# --- LSTM -----------------------------------------------------------------

def build_lstm(num_in: int, num_out: int):
    """Construct a brainstate ``LSTMCell``."""
    return bnn.LSTMCell(num_in, num_out)


def bst_get_lstm(layer):
    """Return ``(W, b)`` of the fused LSTM kernel: ``W`` is ``(in+h, 4h)``, ``b`` is ``(4h,)``."""
    d = layer.W.weight.value
    return d['weight'], d['bias']


def bst_set_lstm(layer, weight, bias):
    """Write the fused LSTM kernel."""
    layer.W.weight.value = {'weight': weight, 'bias': bias}


# --- Dropout --------------------------------------------------------------

def build_dropout(prob: float):
    """Construct a brainstate ``Dropout`` (``prob`` = keep probability)."""
    return bnn.Dropout(prob)


# ---------------------------------------------------------------------------
# brainstate recurrent cells that have no equivalence-preserving conversion
# ---------------------------------------------------------------------------

_GRU_REASON = (
    "GRUCell conversion is unsupported: brainstate's GRU uses the Cho-2014 variant (reset "
    "applied before the hidden matmul, `Wh([x, r*h])`) while flax/nnx/equinox use the cuDNN "
    "variant (reset applied after, `r * (W_hn h + b_hn)`). These are mathematically distinct "
    "and cannot be matched by transferring weights."
)
_NO_EQUIV_REASON = (
    "{name} has no equivalence-preserving counterpart in flax.nnx / flax.linen / equinox "
    "(no matching cell, or a differing recurrence formulation), so it is not converted."
)


def register_bst_unsupported():
    """Register brainstate recurrent cells that cannot be converted equivalence-preservingly.

    Idempotent; called by each framework adapter on import.
    """
    from ._registry import register_unsupported_bst
    if hasattr(bnn, 'GRUCell'):
        register_unsupported_bst(bnn.GRUCell, _GRU_REASON)
    for cls_name in ('ValinaRNNCell', 'MGUCell', 'URLSTMCell'):
        cls = getattr(bnn, cls_name, None)
        if cls is not None:
            register_unsupported_bst(cls, _NO_EQUIV_REASON.format(name=cls_name))
