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

"""A clean-room, flat-IR engine for flattening object graphs of ``Node``\\ s and
``State``\\ s into a static structure plus a dynamic state mapping, and back.

The engine is layered in three tiers:

* **Flat IR** (:mod:`._graphdef`) — a :class:`GraphDef` is a root :class:`Edge`
  plus a flat table of :class:`NodeSpec`\\ s. Graph nodes are hoisted into one
  index-keyed table (so sharing and cycles are encoded by integer index), while
  pytree containers are embedded inline. The whole structure hashes in a single
  cached pass and is registered as a JAX static pytree.
* **Traversal kernel** (:mod:`._walk`) — one depth-first primitive backs
  :func:`iter_leaf` / :func:`iter_node` (and ``states`` / ``nodes``), plus the
  node-type registry behind :func:`register_graph_node_type`.
* **Encode / decode** (:mod:`._flatten`) — :func:`flatten` encodes in one
  pre-order pass; :func:`unflatten` decodes in three linear passes (materialize
  states, create node shells, fill), so reconstruction over graph nodes never
  recurses and cycles/sharing resolve regardless of fill order.

Operations (:func:`treefy_split` / :func:`treefy_merge` / :func:`states` /
:func:`nodes` / ...) and graph/pytree conversion (:func:`graph_to_tree` /
:func:`tree_to_graph`) are thin layers on top.
"""

from ._context import (
    split_context,
    merge_context,
)
from ._convert import (
    graph_to_tree,
    tree_to_graph,
    NodeStates,
)
from ._flatten import (
    flatten,
    unflatten,
)
from ._graphdef import (
    GraphDef,
    NodeSpec,
    NodeEdge,
    StateEdge,
    StateLeafEdge,
    PytreeEdge,
    StaticEdge,
)
from ._node import Node
from ._operations import (
    pop_states,
    nodes,
    states,
    treefy_states,
    update_states,
    treefy_split,
    treefy_merge,
    clone,
    graphdef,
)
from ._reftrack import RefMap
from ._walk import (
    register_graph_node_type,
    Static,
    iter_leaf,
    iter_node,
)

__all__ = [
    'Node',
    'graph_to_tree',
    'tree_to_graph',
    'NodeStates',

    'split_context',
    'merge_context',

    'register_graph_node_type',
    'pop_states',
    'nodes',
    'states',
    'treefy_states',
    'update_states',
    'flatten',
    'unflatten',
    'treefy_split',
    'treefy_merge',
    'iter_leaf',
    'iter_node',
    'clone',
    'graphdef',
    'RefMap',
    'GraphDef',
    'NodeSpec',
    'NodeEdge',
    'StateEdge',
    'StateLeafEdge',
    'PytreeEdge',
    'StaticEdge',
    'Static',
]
