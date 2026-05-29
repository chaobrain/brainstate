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

"""The flat intermediate representation (IR) of a graph's static structure.

A :class:`GraphDef` records the static skeleton of an object graph as a flat
table of graph-node specs (:class:`NodeSpec`) referenced by integer index,
plus a single root :class:`Edge`. Pytree containers are embedded inline as
:class:`PytreeEdge` and never consume an index. ``State`` leaves live in the
dynamic state mapping and are referenced by the same global index space.

This "hoist graph nodes into an index-keyed table" design is what makes the
representation flat: graph topology (sharing, cycles) is encoded by integer
indices rather than by nested dataclasses, so the whole structure hashes in a
single pass and reconstruction never recurses over graph nodes.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Generic, TypeVar

import jax

from brainstate.typing import PathParts
from brainstate.util import PrettyRepr, PrettyType, PrettyAttr

__all__ = [
    'PytreeType', 'NodeEdge', 'StateEdge', 'StateLeafEdge', 'PytreeEdge',
    'StaticEdge', 'Edge', 'NodeSpec', 'GraphDef',
]

N = TypeVar('N')
Index = int


class PytreeType:
    """Sentinel type marking an inline pytree container in the IR.

    Used as the ``type`` of the pytree node-impl and as ``GraphDef.type`` when
    the root object is a pytree (list/dict/tuple/...) rather than a graph node.
    """


@dataclasses.dataclass(frozen=True)
class NodeEdge:
    """A reference to a graph node by its global index.

    Parameters
    ----------
    index : int
        The global index of the referenced graph node. The same index appears
        on every edge that points at the node, so sharing and cycles are
        encoded by index equality.
    """
    index: int


@dataclasses.dataclass(frozen=True)
class StateEdge:
    """A deduplicated ``State`` leaf, referenced by global index.

    Parameters
    ----------
    index : int
        The global index of the ``State``.
    path : PathParts or None
        The mapping key path on the *defining* occurrence; ``None`` on a shared
        or cyclic back-reference (which carries no state-mapping entry).
    type : type
        The concrete ``State`` subtype (e.g. ``ParamState``).
    """
    index: int
    path: PathParts | None
    type: type


@dataclasses.dataclass(frozen=True)
class StateLeafEdge:
    """A bare ``TreefyState`` leaf (carried directly, never deduplicated).

    Parameters
    ----------
    path : PathParts
        The mapping key path holding the ``TreefyState`` value.
    """
    path: PathParts


@dataclasses.dataclass(frozen=True)
class PytreeEdge:
    """An inline pytree container.

    Parameters
    ----------
    metadata : Any
        The JAX ``PyTreeDef`` (treedef) of one flattening level of the
        container, used to rebuild it on decode.
    fields : tuple of (Key, Edge)
        The ordered child edges keyed by their per-level path key.
    """
    metadata: Any
    fields: tuple


@dataclasses.dataclass(frozen=True)
class StaticEdge:
    """An inline, hashable static value carried directly in the structure.

    Parameters
    ----------
    value : Any
        The static value (must be hashable, as it participates in the cached
        ``GraphDef`` hash / JIT cache key).
    """
    value: Any


Edge = NodeEdge | StateEdge | StateLeafEdge | PytreeEdge | StaticEdge


@dataclasses.dataclass(frozen=True)
class NodeSpec(Generic[N]):
    """The static record for a single **graph node** (never a pytree).

    Parameters
    ----------
    type : type
        The node's Python type.
    index : int
        Its global index in the flat table.
    metadata : Any
        Hashable auxiliary data returned by the node's flatten function
        (``(cls,)`` for a :class:`~brainstate.graph.Node`).
    fields : tuple of (Key, Edge)
        The ordered ``(key, edge)`` pairs describing the node's children.
    """
    type: type
    index: int
    metadata: Any
    fields: tuple


@dataclasses.dataclass(frozen=True, repr=False)
class GraphDef(Generic[N], PrettyRepr):
    """The static structure of an object graph.

    A ``GraphDef`` is a root :class:`Edge` plus a flat tuple of graph-node
    specs ordered ascending by index. It is hashable (the hash is computed once
    and cached) and registered as a JAX static pytree, so it can serve as a
    transform cache key (e.g. for ``jit``).

    Parameters
    ----------
    root : Edge
        The edge for the root object (a ``NodeEdge``, ``PytreeEdge``, etc.).
    node_specs : tuple of NodeSpec
        Graph-node specs, ascending by index.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> model = brainstate.nn.Linear(2, 3)
        >>> graphdef, _ = brainstate.graph.flatten(model)
        >>> graphdef.type is brainstate.nn.Linear
        True
    """

    __module__ = 'brainstate.graph'

    root: Edge
    node_specs: tuple

    def __post_init__(self):
        # Hash is computed lazily on first use and cached. Deferring it means a
        # GraphDef holding an unhashable static value (e.g. a bare ``jax.Array``
        # attribute) can still be constructed and used for split/merge; it only
        # fails if it is actually hashed (e.g. used as a transform cache key).
        object.__setattr__(self, '_cached_hash', None)

    def __hash__(self) -> int:
        h = self._cached_hash
        if h is None:
            try:
                h = hash((self.root, self.node_specs))
            except TypeError as e:
                raise TypeError(
                    "GraphDef is not hashable because it contains an unhashable "
                    "static value. Wrap dynamic arrays in a State (or in "
                    "brainstate.graph.Static if intentionally static)."
                ) from e
            object.__setattr__(self, '_cached_hash', h)
        return h

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is GraphDef
            and self.root == other.root
            and self.node_specs == other.node_specs
        )

    @property
    def type(self) -> type:
        """Type of the root container (the root node type, or :class:`PytreeType`)."""
        if isinstance(self.root, NodeEdge):
            for spec in self.node_specs:
                if spec.index == self.root.index:
                    return spec.type
        return PytreeType

    @property
    def index(self) -> int:
        """Global index of the root node, or ``0`` when the root is a pytree."""
        return self.root.index if isinstance(self.root, NodeEdge) else 0

    def __pretty_repr__(self):
        yield PrettyType(type=type(self))
        yield PrettyAttr('root', self.root)
        yield PrettyAttr('node_specs', self.node_specs)


jax.tree_util.register_static(GraphDef)
