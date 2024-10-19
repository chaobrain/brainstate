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

import contextlib
import dataclasses
import threading
from typing import (Any, Tuple, List)

from typing_extensions import Unpack

from brainstate.typing import Filter
from brainstate.util import NestedMapping
from ._graph_operation import (flatten, unflatten, _split_state,
                               GraphDefinition, GraphStateMapping,
                               RefMap, Index, A)

__all__ = [
  'split_context', 'merge_context',
]


@dataclasses.dataclass
class GraphContext(threading.local):
  """
  A context manager for handling complex state updates.
  """
  ref_index_stack: List[SplitContext] = dataclasses.field(default_factory=list)
  index_ref_stack: List[MergeContext] = dataclasses.field(default_factory=list)


GRAPH_CONTEXT = GraphContext()


@dataclasses.dataclass
class SplitContext:
  """
  A context manager for handling graph splitting.
  """
  ref_index: RefMap[Any, Index]

  def split(self, node: A, *filters: Filter) -> Tuple[GraphDefinition[A], Unpack[Tuple[GraphStateMapping, ...]]]:
    graphdef, state_mapping = flatten(node, self.ref_index)
    state_mappings = _split_state(state_mapping, filters)
    return graphdef, *state_mappings


@contextlib.contextmanager
def split_context():
  """
  A context manager for handling graph splitting.
  """
  index_ref: RefMap[Any, Index] = RefMap()
  flatten_ctx = SplitContext(index_ref)
  GRAPH_CONTEXT.ref_index_stack.append(flatten_ctx)

  try:
    yield flatten_ctx
  finally:
    GRAPH_CONTEXT.ref_index_stack.pop()
    del flatten_ctx.ref_index


@dataclasses.dataclass
class MergeContext:
  """
  A context manager for handling graph merging.
  """
  index_ref: dict[Index, Any]

  def merge(
      self,
      graphdef: GraphDefinition[A],
      state_mapping: GraphStateMapping,
      /,
      *state_mappings: GraphStateMapping
  ) -> A:
    state_mapping = NestedMapping.merge(state_mapping, *state_mappings)
    node = unflatten(graphdef, state_mapping, index_ref=self.index_ref)
    return node


@contextlib.contextmanager
def merge_context():
  """
  A context manager for handling graph merging.
  """
  index_ref: dict[Index, Any] = {}

  unflatten_ctx = MergeContext(index_ref)
  GRAPH_CONTEXT.index_ref_stack.append(unflatten_ctx)

  try:
    yield unflatten_ctx
  finally:
    GRAPH_CONTEXT.index_ref_stack.pop()
    del unflatten_ctx.index_ref
