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

"""Split/merge contexts that share a reference index across multiple objects.

A :func:`split_context` shares one growing ``ref_index`` across several
:meth:`SplitContext.treefy_split` calls, so references shared *between* the
split objects collapse to the same global index. The paired
:func:`merge_context` shares the inverse ``index_ref`` table across
:meth:`MergeContext.treefy_merge` calls so those references rebuild as one
object. The context stacks are thread-local.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from typing import Any, TypeVar

from typing_extensions import Unpack

from brainstate.typing import Filter
from brainstate.util import NestedDict
from ._flatten import flatten, unflatten
from ._graphdef import GraphDef
from ._operations import _split_state
from ._reftrack import RefMap

__all__ = [
    'split_context',
    'merge_context',
]

Index = int
A = TypeVar('A')


@dataclasses.dataclass
class GraphContext(threading.local):
    """Thread-local stacks of active split/merge contexts.

    Inheriting from ``threading.local`` ensures each thread has its own
    independent context stacks, making nested transforms safe under parallelism.
    """

    ref_index_stack: list[SplitContext] = dataclasses.field(default_factory=list)
    index_ref_stack: list[MergeContext] = dataclasses.field(default_factory=list)


GRAPH_CONTEXT = GraphContext()


@dataclasses.dataclass
class SplitContext:
    """Context for splitting graph nodes, tracking shared references."""

    ref_index: RefMap[Any, Index]

    def treefy_split(self, node: A, *filters: Filter) -> tuple[GraphDef[A], Unpack[tuple[NestedDict, ...]]]:
        graphdef, statetree = flatten(node, self.ref_index)
        state_mappings = _split_state(statetree, filters)
        return graphdef, *state_mappings


@contextlib.contextmanager
def split_context():
    """Context manager for splitting multiple graph nodes sharing a reference index."""
    index_ref: RefMap[Any, Index] = RefMap()
    flatten_ctx = SplitContext(index_ref)
    GRAPH_CONTEXT.ref_index_stack.append(flatten_ctx)
    try:
        yield flatten_ctx, index_ref
    finally:
        GRAPH_CONTEXT.ref_index_stack.pop()
        del flatten_ctx.ref_index


@dataclasses.dataclass
class MergeContext:
    """Context for merging graph nodes, tracking shared references."""

    index_ref: dict[Index, Any]

    def treefy_merge(
        self,
        graphdef: GraphDef[A],
        state_mapping: NestedDict,
        /,
        *state_mappings: NestedDict,
    ) -> A:
        state_mapping = NestedDict.merge(state_mapping, *state_mappings)
        return unflatten(graphdef, state_mapping, index_ref=self.index_ref)


@contextlib.contextmanager
def merge_context():
    """Context manager for merging multiple graph nodes sharing a reference index."""
    index_ref: dict[Index, Any] = {}
    unflatten_ctx = MergeContext(index_ref)
    GRAPH_CONTEXT.index_ref_stack.append(unflatten_ctx)
    try:
        yield unflatten_ctx, dict(unflatten_ctx.index_ref)
    finally:
        GRAPH_CONTEXT.index_ref_stack.pop()
        del unflatten_ctx.index_ref
