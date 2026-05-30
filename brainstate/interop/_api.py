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

"""Public conversion functions.

Each function lazily imports its framework adapter (which imports ``flax`` / ``equinox`` and
registers that framework's layer mappings on first use), so importing ``brainstate`` adds no
hard dependency on either framework.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ._common import Context
from ._engine import to_bst, to_foreign

__all__ = [
    'from_nnx', 'to_nnx',
    'from_linen', 'to_linen',
    'from_equinox', 'to_equinox',
    'ensure_framework_loaded',
]

_ADAPTERS = {}


def ensure_framework_loaded(framework: str):
    """Import and cache the adapter for ``framework``; raises if the framework is missing."""
    if framework not in _ADAPTERS:
        if framework == 'nnx':
            from ._frameworks._nnx import NnxAdapter
            _ADAPTERS['nnx'] = NnxAdapter()
        elif framework == 'linen':
            from ._frameworks._linen import LinenAdapter
            _ADAPTERS['linen'] = LinenAdapter()
        elif framework == 'equinox':
            from ._frameworks._equinox import EquinoxAdapter
            _ADAPTERS['equinox'] = EquinoxAdapter()
        else:
            raise ValueError(f"Unknown framework: {framework!r}")
    return _ADAPTERS[framework]


def _shape(sample_input):
    if sample_input is None:
        return None
    if hasattr(sample_input, 'shape'):
        return tuple(sample_input.shape)
    return tuple(sample_input)


# ---------------------------------------------------------------------------
# flax.nnx
# ---------------------------------------------------------------------------

def from_nnx(model: Any, *, sample_input: Any = None) -> Any:
    """Convert a ``flax.nnx`` model into an equivalent ``brainstate.nn`` model.

    Parameters
    ----------
    model : flax.nnx.Module
        The source model. Either a single registered layer or an ``nnx`` sequential stack.
    sample_input : array or tuple of int, optional
        A single *unbatched* example (or its shape). Required when the model contains a
        convolution or spatial batch-norm layer, whose brainstate equivalents carry a concrete
        ``in_size``.

    Returns
    -------
    brainstate.nn.Module
        The converted model, weight-equivalent to ``model``.

    Raises
    ------
    MissingDependencyError
        If ``flax`` is not installed.
    UnmappedLayerError, UnsupportedLayerError, UnsupportedStructureError, MissingShapeError
        See :mod:`brainstate.interop` error types.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> from flax import nnx
        >>> src = nnx.Linear(3, 4, rngs=nnx.Rngs(0))
        >>> dst = bst.interop.from_nnx(src)
    """
    adapter = ensure_framework_loaded('nnx')
    s = _shape(sample_input)
    return to_bst(model, adapter, Context(sample_input=s, cur_size=s))


def to_nnx(model: Any, *, rngs: Any = None) -> Any:
    """Convert a ``brainstate.nn`` model into an equivalent ``flax.nnx`` model.

    Parameters
    ----------
    model : brainstate.nn.Module
        The source model.
    rngs : flax.nnx.Rngs, optional
        RNG container used to construct the nnx layers. Construction weights are overwritten
        immediately, so this only matters if you reuse the ``rngs`` afterwards.

    Returns
    -------
    flax.nnx.Module
        The converted model.
    """
    adapter = ensure_framework_loaded('nnx')
    return to_foreign(model, adapter, Context(rngs=rngs))


# ---------------------------------------------------------------------------
# flax.linen
# ---------------------------------------------------------------------------

def from_linen(module: Any, params: Any, *, sample_input: Any = None) -> Any:
    """Convert a ``flax.linen`` module + params into an equivalent ``brainstate.nn`` model.

    Parameters
    ----------
    module : flax.linen.Module
        The linen module describing the architecture.
    params : Mapping
        The variables produced by ``module.init(...)`` (e.g. ``{'params': ..., 'batch_stats':
        ...}``).
    sample_input : array or tuple of int, optional
        Required when the model contains convolution / spatial batch-norm layers.

    Returns
    -------
    brainstate.nn.Module
        The converted model.
    """
    adapter = ensure_framework_loaded('linen')
    s = _shape(sample_input)
    node = adapter.wrap(module, params)
    return to_bst(node, adapter, Context(sample_input=s, cur_size=s))


def to_linen(model: Any) -> Tuple[Any, Any]:
    """Convert a ``brainstate.nn`` model into a ``flax.linen`` ``(module, params)`` pair.

    Parameters
    ----------
    model : brainstate.nn.Module
        The source model.

    Returns
    -------
    tuple
        ``(linen_module, params)`` where ``params`` is a ``FrozenDict`` accepted by
        ``linen_module.apply``.
    """
    adapter = ensure_framework_loaded('linen')
    wrapped = to_foreign(model, adapter, Context())
    return adapter.finalize(wrapped)


# ---------------------------------------------------------------------------
# equinox
# ---------------------------------------------------------------------------

def from_equinox(model: Any, *, sample_input: Any = None) -> Any:
    """Convert an ``equinox`` model into an equivalent ``brainstate.nn`` model.

    Parameters
    ----------
    model : equinox.Module
        The source model. Either a single registered layer or an ``eqx.nn.Sequential``.
    sample_input : array or tuple of int, optional
        Required when the model contains convolution layers.

    Returns
    -------
    brainstate.nn.Module
        The converted model.
    """
    adapter = ensure_framework_loaded('equinox')
    s = _shape(sample_input)
    return to_bst(model, adapter, Context(sample_input=s, cur_size=s))


def to_equinox(model: Any, *, key: Any = None) -> Any:
    """Convert a ``brainstate.nn`` model into an equivalent ``equinox`` model.

    Parameters
    ----------
    model : brainstate.nn.Module
        The source model.
    key : jax PRNG key, optional
        Key used to construct the equinox layers (weights are overwritten immediately).

    Returns
    -------
    equinox.Module
        The converted model.
    """
    adapter = ensure_framework_loaded('equinox')
    return to_foreign(model, adapter, Context(key=key))
