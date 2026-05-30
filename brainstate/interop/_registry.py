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

"""Registry of layer conversion mappings.

A :class:`LayerMapping` ties a brainstate layer type to a foreign layer type and carries the
two conversion closures (``to_bst`` and ``to_foreign``). Co-locating both directions in one
object keeps round-trip correctness reviewable in one place. The registry is populated lazily:
each framework adapter module registers its mappings when it is first imported (which only
happens after the framework itself imports successfully).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

__all__ = [
    'LayerMapping',
    'register_layer_mapping',
    'register_unsupported_bst',
    'register_unsupported_bst_for',
    'register_unsupported_foreign',
    'lookup_import',
    'lookup_export',
    'unsupported_bst_reason',
    'unsupported_foreign_reason',
    'supported_layers',
]


@dataclass
class LayerMapping:
    """A bidirectional conversion mapping for one layer type.

    Parameters
    ----------
    bst_type : type
        The brainstate ``Module`` subclass.
    framework : str
        One of ``"nnx"``, ``"linen"``, ``"equinox"``.
    foreign_type : type
        The foreign layer class.
    to_bst : Callable
        ``(foreign_module, ctx) -> bst.nn.Module``. Builds and weight-fills a brainstate layer.
    to_foreign : Callable
        ``(bst_module, ctx) -> foreign``. Builds and weight-fills the foreign layer. For the
        functional ``linen`` framework the return value is whatever the linen adapter expects
        (a ``(module, params)`` pair handled by the adapter).
    """

    bst_type: type
    framework: str
    foreign_type: type
    to_bst: Callable
    to_foreign: Callable


# framework -> {foreign_type: LayerMapping}
_IMPORT: Dict[str, Dict[type, LayerMapping]] = {}
# (bst_type, framework) -> LayerMapping
_EXPORT: Dict[Tuple[type, str], LayerMapping] = {}
# brainstate types that are deliberately unsupported (any framework): type -> reason
_UNSUPPORTED_BST: Dict[type, str] = {}
# brainstate types unsupported for a *specific* framework: (bst_type, framework) -> reason
_UNSUPPORTED_BST_FW: Dict[Tuple[type, str], str] = {}
# framework -> {foreign_type: reason}
_UNSUPPORTED_FOREIGN: Dict[str, Dict[type, str]] = {}


def register_layer_mapping(mapping: LayerMapping) -> None:
    """Register (or override) a :class:`LayerMapping` in both directions.

    Parameters
    ----------
    mapping : LayerMapping
        The mapping to register.
    """
    _IMPORT.setdefault(mapping.framework, {})[mapping.foreign_type] = mapping
    _EXPORT[(mapping.bst_type, mapping.framework)] = mapping


def register_unsupported_bst(bst_type: type, reason: str) -> None:
    """Mark a brainstate layer type as deliberately unsupported for export (any framework)."""
    _UNSUPPORTED_BST[bst_type] = reason


def register_unsupported_bst_for(framework: str, bst_type: type, reason: str) -> None:
    """Mark a brainstate layer type as unsupported for export to one specific framework."""
    _UNSUPPORTED_BST_FW[(bst_type, framework)] = reason


def register_unsupported_foreign(framework: str, foreign_type: type, reason: str) -> None:
    """Mark a foreign layer type as deliberately unsupported for import."""
    _UNSUPPORTED_FOREIGN.setdefault(framework, {})[foreign_type] = reason


def _lookup_by_mro(table: dict, key_type: type):
    """Look up ``key_type`` allowing subclasses to match a registered base class."""
    if key_type in table:
        return table[key_type]
    for base in key_type.__mro__:
        if base in table:
            return table[base]
    return None


def lookup_import(framework: str, foreign_type: type) -> Optional[LayerMapping]:
    """Return the import mapping for a foreign type, or ``None``."""
    return _lookup_by_mro(_IMPORT.get(framework, {}), foreign_type)


def lookup_export(bst_type: type, framework: str) -> Optional[LayerMapping]:
    """Return the export mapping for a brainstate type + framework, or ``None``."""
    table = {bt: m for (bt, fw), m in _EXPORT.items() if fw == framework}
    return _lookup_by_mro(table, bst_type)


def unsupported_bst_reason(bst_type: type, framework: Optional[str] = None) -> Optional[str]:
    """Return the deliberate-unsupported reason for a brainstate type, or ``None``.

    Checks the framework-specific table first (when ``framework`` is given), then the
    framework-agnostic table.
    """
    if framework is not None:
        for base in bst_type.__mro__:
            if (base, framework) in _UNSUPPORTED_BST_FW:
                return _UNSUPPORTED_BST_FW[(base, framework)]
    for base in bst_type.__mro__:
        if base in _UNSUPPORTED_BST:
            return _UNSUPPORTED_BST[base]
    return None


def unsupported_foreign_reason(framework: str, foreign_type: type) -> Optional[str]:
    """Return the deliberate-unsupported reason for a foreign type, or ``None``."""
    table = _UNSUPPORTED_FOREIGN.get(framework, {})
    for base in foreign_type.__mro__:
        if base in table:
            return table[base]
    return None


def supported_layers(framework: Optional[str] = None) -> Dict[str, list]:
    """List the brainstate layer types with registered conversions.

    Parameters
    ----------
    framework : str, optional
        Restrict to a single framework. If ``None``, report all frameworks.

    Returns
    -------
    dict
        Mapping ``framework -> [brainstate type names]``.
    """
    # ensure adapters are imported so registries are populated
    from . import _api  # noqa: F401  (triggers lazy framework imports on demand)
    frameworks = [framework] if framework else ['nnx', 'linen', 'equinox']
    out = {}
    for fw in frameworks:
        _api.ensure_framework_loaded(fw)
        names = sorted({bt.__name__ for (bt, f) in _EXPORT if f == fw})
        out[fw] = names
    return out
