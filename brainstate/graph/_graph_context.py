# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors & 2024 BDP Ecosystem.
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

import jax
import contextlib
import dataclasses
import functools
import threading
from typing import (Any, Tuple, List, overload, Callable, TypeVar, Mapping)

from typing_extensions import Unpack

from brainstate.typing import Filter
from brainstate.util import NestedDict, FrozenDict
from ._graph_operation import (flatten,
                               unflatten,
                               _split_state,
                               GraphDef,
                               RefMap,
                               NodeDef)

__all__ = [
    'split_context',
    'merge_context',
    'update_context',
]

Index = int
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
F = TypeVar('F', bound=Callable)


@dataclasses.dataclass
class GraphContext(threading.local):
    """
    A context manager for handling complex state updates.
    """
    update_context_stacks: dict[str, list[UpdateContext]] = dataclasses.field(default_factory=dict)
    ref_index_stack: List[SplitContext] = dataclasses.field(default_factory=list)
    index_ref_stack: List[MergeContext] = dataclasses.field(default_factory=list)


GRAPH_CONTEXT = GraphContext()


@dataclasses.dataclass
class SplitContext:
    """
    A context manager for handling graph splitting.
    """
    ctxtag: str | None
    ref_index: RefMap[Any, Index]

    def treefy_split(
        self,
        node: A,
        *filters: Filter
    ) -> Tuple[GraphDef[A], Unpack[Tuple[NestedDict, ...]]]:
        ctx = current_update_context(self.ctxtag) if self.ctxtag is not None else None
        graphdef, statetree = flatten(node, self.ref_index)
        state_mappings = _split_state(statetree, filters)
        if ctx is not None:
            if ctx.index_ref is not None and isinstance(graphdef, NodeDef):
                index_to_index = compose_mapping(ctx.index_ref, self.ref_index)
                graphdef = dataclasses.replace(
                    graphdef,
                    index_mapping=FrozenDict(index_to_index)
                )
        return graphdef, *state_mappings


@contextlib.contextmanager
def split_context(ctxtag: str | None = None):
    """
    A context manager for handling graph splitting.
    """
    index_ref: RefMap[Any, Index] = RefMap()
    flatten_ctx = SplitContext(ctxtag, index_ref)
    GRAPH_CONTEXT.ref_index_stack.append(flatten_ctx)

    try:
        yield flatten_ctx
    finally:
        GRAPH_CONTEXT.ref_index_stack.pop()
        if ctxtag is not None:
            ctx = current_update_context(ctxtag)
            ctx.flatten_end(index_ref)
        del flatten_ctx.ref_index
        del flatten_ctx.ctxtag


@dataclasses.dataclass
class MergeContext:
    """
    A context manager for handling graph merging.
    """
    ctxtag: str | None
    index_ref: dict[Index, Any]

    def treefy_merge(
        self,
        graphdef: GraphDef[A],
        state_mapping: NestedDict,
        /,
        *state_mappings: NestedDict
    ) -> A:
        ctx = (
            current_update_context(self.ctxtag)
            if self.ctxtag is not None
            else None
        )
        if (
            ctx is not None
            and isinstance(graphdef, NodeDef)
            and graphdef.index_mapping is not None
        ):
            # outer merge (4), create index_ref_cache
            assert ctx.ref_index is not None
            index_ref_cache = compose_mapping_reversed(
                ctx.ref_index, graphdef.index_mapping
            )
        else:
            # inner merge (2)
            index_ref_cache = None

        state_mapping = NestedDict.merge(state_mapping, *state_mappings)
        node = unflatten(
            graphdef,
            state_mapping,
            index_ref=self.index_ref,
            index_ref_cache=index_ref_cache,
        )
        return node


@contextlib.contextmanager
def merge_context(ctxtag: str | None = None):
    """
    A context manager for handling graph merging.
    """
    index_ref: dict[Index, Any] = {}

    unflatten_ctx = MergeContext(ctxtag, index_ref)
    GRAPH_CONTEXT.index_ref_stack.append(unflatten_ctx)

    try:
        yield unflatten_ctx
    finally:
        GRAPH_CONTEXT.index_ref_stack.pop()
        if ctxtag is not None:
            ctx = current_update_context(ctxtag)
            ctx.unflatten_end(index_ref)
        del unflatten_ctx.index_ref
        del unflatten_ctx.ctxtag


@dataclasses.dataclass
class UpdateContext:
    """
    A context manager for handling complex state updates.
    """

    tag: str
    ref_index: RefMap[Any, Index] | None
    index_ref: dict[Index, Any] | None

    # define hash and eq to make this an opaque object
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, UpdateContext)

    def flatten_end(self, ref_index: RefMap[Any, Index]):
        if self.ref_index is None:
            # outer split (1), store the references
            self.ref_index = ref_index
        else:
            # inner split (3), clear index_ref
            self.index_ref = None

    def unflatten_end(self, index_ref: dict[Index, Any]):
        self.index_ref = index_ref

    @overload
    def split(
        self, graph_node: A, /
    ) -> tuple[GraphDef[A], NestedDict]:
        ...

    @overload
    def split(
        self, graph_node: A, first: Filter, /
    ) -> tuple[GraphDef[A], NestedDict]:
        ...

    @overload
    def split(
        self,
        graph_node: A,
        first: Filter,
        second: Filter,
        /,
        *filters: Filter,
    ) -> tuple[GraphDef[A], NestedDict, Unpack[tuple[NestedDict, ...]]]:
        ...

    def split(
        self, node: A, *filters: Filter
    ) -> tuple[GraphDef[A], NestedDict, Unpack[tuple[NestedDict, ...]]]:
        """
        Split a graph node into a :class:`GraphDef` and one or more :class:`State`s. State is
        a ``Mapping`` from strings or integers to ``Variables``, Arrays or nested States. GraphDef
        contains all the static information needed to reconstruct a ``Module`` graph, it is analogous
        to JAX’s ``PyTreeDef``. :func:`split` is used in conjunction with :func:`merge` to switch
        seamlessly between stateful and stateless representations of the graph.

        Arguments:
          node: graph node to split.
          *filters: some optional filters to group the state into mutually exclusive substates.
        Returns:
          :class:`GraphDef` and one or more :class:`State`'s equal to the number of filters passed. If no
          filters are passed, a single :class:`State` is returned.
        """
        ref_index: RefMap[Any, Index] = RefMap()
        graphdef, state = flatten(node, ref_index)
        states = _split_state(state, filters)

        if (self.index_ref is not None) and isinstance(graphdef, NodeDef):
            index_to_index = compose_mapping(self.index_ref, ref_index)
            graphdef = dataclasses.replace(
                graphdef,
                index_mapping=FrozenDict(index_to_index)
            )

        self.flatten_end(ref_index)

        return graphdef, *states

    def merge(
        self,
        graphdef: GraphDef[A],
        state: NestedDict,
        *states: NestedDict,
    ) -> A:
        """merge"""
        if not isinstance(graphdef, NodeDef):
            raise ValueError(f'Expected a NodeDef instance, but got {type(graphdef)}.' )
        if self.ref_index is None:
            raise ValueError('Cannot merge without ref_index.')

        if graphdef.index_mapping is not None:
            # outer merge (4), create index_ref_cache
            assert self.ref_index is not None
            index_ref_cache = compose_mapping_reversed(
                self.ref_index,
                graphdef.index_mapping
            )
        else:
            # inner merge (2)
            index_ref_cache = None

        state = NestedDict.merge(state, *states)
        index_ref: dict[Index, Any] = {}
        node = unflatten(
            graphdef,
            state,
            index_ref=index_ref,
            index_ref_cache=index_ref_cache
        )

        self.unflatten_end(index_ref)

        return node


jax.tree_util.register_static(UpdateContext)


@dataclasses.dataclass
class UpdateContextManager:
    tag: str

    def __enter__(self):
        ctx = UpdateContext(self.tag, None, None)
        if self.tag not in GRAPH_CONTEXT.update_context_stacks:
            GRAPH_CONTEXT.update_context_stacks[self.tag] = [ctx]
        else:
            GRAPH_CONTEXT.update_context_stacks[self.tag].append(ctx)
        return ctx

    def __exit__(self, *args):
        if self.tag not in GRAPH_CONTEXT.update_context_stacks:
            raise RuntimeError(f'No update context found for tag {self.tag!r}, this is a bug.')
        stack = GRAPH_CONTEXT.update_context_stacks[self.tag]

        ctx = stack.pop()
        # clear references
        del ctx.ref_index
        del ctx.index_ref

        if not stack:
            del GRAPH_CONTEXT.update_context_stacks[self.tag]

    def __call__(self, f: F) -> F:
        @functools.wraps(f)
        def update_context_manager_wrapper(*args, **kwargs):
            with self:
                return f(*args, **kwargs)

        return update_context_manager_wrapper  # type: ignore


def update_context(tag: str):
    """Creates an :class:`UpdateContext` context manager which can be used to handle
    more complex state updates beyond what ``nnx.update`` can handle, including
    updates to static properties and graph structure.

    UpdateContext exposes a ``split`` and ``merge`` API with the same
    signature as ``nnx.split`` / ``nnx.merge`` but performs some bookkeeping
    to have the necessary information in order to perfectly update the input
    objects based on the changes made inside the transform. The UpdateContext
    must call split and merge a total of 4 times, the first
    and last calls happen outside the transform and the second and third calls
    happen inside the transform as shown in the diagram below::


                            idxmap
      (2) merge ─────────────────────────────► split (3)
            ▲                                    │
            │               inside               │
            │. . . . . . . . . . . . . . . . . . │ index_mapping
            │               outside              │
            │                                    ▼
      (1) split──────────────────────────────► merge (4)
                            refmap


    The first call to split ``(1)`` creates a ``refmap`` which keeps track of the
    outer references, and the first call to merge ``(2)`` creates an ``idxmap`` which
    keeps track of the inner references. The second call to split ``(3)`` combines
    the refmap and idxmap to produce the ``index_mapping`` which indicates
    how the outer references map to the inner references. Finally, the last call to
    merge ``(4)`` uses the index_mapping and the refmap to reconstruct the
    output of the transform while reusing/updating the inner references. To avoid
    memory leaks, the idxmap is cleared after ``(3)`` and the refmap is
    cleared after ``(4)``, and both are cleared after the context manager exits.

    Here is a simple example showing the use of ``update_context``::

      >>> import brainstate as bst
      >>> import jax
      ...
      >>> m1 = bst.graph.Dict({})
      >>> with bst.graph.update_context('example') as ctx:
      ...   graphdef, state = ctx.split(m1)
      ...   @jax.jit
      ...   def f(graphdef, state):
      ...     m2 = ctx.merge(graphdef, state)
      ...     m2.a = 1
      ...     m2.ref = m2  # create a reference cycle
      ...     return ctx.split(m2)
      ...   graphdef_out, state_out = f(graphdef, state)
      ...   m3 = ctx.merge(graphdef_out, state_out)
      ...
      >>> assert m1 is m3
      >>> assert m1.a == 1
      >>> assert m1.ref is m1

    Note that ``update_context`` takes in a ``tag`` argument which is used
    primarily as a safety mechanism reduce the risk of accidentally using the
    wrong UpdateContext when using :func:`current_update_context` to access the
    current active context. current_update_context can be used as a way of
    accessing the current active context without having to pass it as a capture::

      >>> m1 = bst.graph.Dict({})
      >>> @jax.jit
      ... def f(graphdef, state):
      ...   ctx = bst.graph.current_update_context('example')
      ...   m2 = ctx.merge(graphdef, state)
      ...   m2.a = 1     # insert static attribute
      ...   m2.ref = m2  # create a reference cycle
      ...   return ctx.split(m2)
      ...
      >>> @bst.graph.update_context('example')
      ... def g(m1):
      ...   ctx = bst.graph.current_update_context('example')
      ...   graphdef, state = ctx.split(m1)
      ...   graphdef_out, state_out = f(graphdef, state)
      ...   return ctx.merge(graphdef_out, state_out)
      ...
      >>> m3 = g(m1)
      >>> assert m1 is m3
      >>> assert m1.a == 1
      >>> assert m1.ref is m1

    As shown in the code above, ``update_context`` can also be used as a
    decorator that creates/activates an UpdateContext context for the
    duration of the function. The context can be accessed using
    :func:`current_update_context`.

    Args:
      tag: A string tag to identify the context.
    """
    return UpdateContextManager(tag)


def current_update_context(tag: str) -> UpdateContext:
    """Returns the current active :class:`UpdateContext` for the given tag."""
    if tag not in GRAPH_CONTEXT.update_context_stacks:
        raise ValueError(f'No update context found for tag {tag!r}.')
    return GRAPH_CONTEXT.update_context_stacks[tag][-1]


def compose_mapping(
    map_ab: Mapping[A, B], map_bc: Mapping[B, C], /
) -> dict[A, C]:
    return {a: map_bc[b] for a, b in map_ab.items() if b in map_bc}


def compose_mapping_reversed(
    map_ab: Mapping[A, B], map_bc: Mapping[B, C], /
) -> dict[C, A]:
    return {map_bc[b]: a for a, b in map_ab.items() if b in map_bc}
