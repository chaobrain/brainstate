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

"""The :class:`Node` base class for mutable graph nodes."""

from __future__ import annotations

from abc import ABCMeta
from copy import deepcopy
from typing import Any, Callable, TypeVar, TYPE_CHECKING

import jax

from brainstate._error import TraceContextError
from brainstate._state import State, TreefyState
from brainstate.typing import Key
from brainstate.util import PrettyObject
from ._operations import treefy_split, treefy_merge
from ._walk import register_graph_node_type

__all__ = ['Node']

G = TypeVar('G', bound='Node')


class GraphNodeMeta(ABCMeta):
    if not TYPE_CHECKING:
        def __call__(cls, *args, **kwargs) -> Any:
            node = cls.__new__(cls, *args, **kwargs)
            node.__init__(*args, **kwargs)
            return node


class Node(PrettyObject, metaclass=GraphNodeMeta):
    """Base class for all mutable graph nodes in the BrainState framework.

    Subclasses are automatically registered with the graph engine, so their
    attributes participate in :func:`~brainstate.graph.treefy_split` /
    :func:`~brainstate.graph.treefy_merge`, state/node introspection, and the
    functional transforms built on top of them. Attributes are flattened in
    alphabetical order; names listed in ``graph_invisible_attrs`` are excluded.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> class MyModel(brainstate.graph.Node):
        ...     def __init__(self):
        ...         self.w = brainstate.ParamState(brainstate.random.randn(3))
        >>> model = MyModel()
        >>> graphdef, params = brainstate.graph.treefy_split(model)
        >>> rebuilt = brainstate.graph.treefy_merge(graphdef, params)
        >>> isinstance(rebuilt, MyModel)
        True
    """

    __module__ = 'brainstate.graph'

    graph_invisible_attrs: tuple[str, ...] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        register_graph_node_type(
            type=cls,
            flatten=_node_flatten,
            set_key=_node_set_key,
            pop_key=_node_pop_key,
            create_empty=_node_create_empty,
            clear=_node_clear,
        )

    def __deepcopy__(self: G, memo=None) -> G:
        graphdef, state = treefy_split(self)
        # ``state`` is a pytree of ``TreefyState`` whose only leaves are the
        # array values; mapping ``deepcopy`` over it copies those values while
        # leaving the (immutable, unpicklable) source-info metadata untouched.
        # A plain ``deepcopy(state)`` would instead try to copy that metadata,
        # which carries a ``jaxlib`` ``Traceback`` and is not picklable. The
        # ``graphdef`` is static structure and safe to share; ``treefy_merge``
        # rebuilds fresh ``State`` objects, so the returned node is independent.
        state = jax.tree.map(deepcopy, state)
        return treefy_merge(graphdef, state)

    def check_valid_context(self, error_msg: Callable[[], str]) -> None:
        """Raise :class:`TraceContextError` if the current trace context is invalid."""
        if not self._trace_state.is_valid():
            raise TraceContextError(error_msg())


# -------------------------------
# Graph node helper functions
# -------------------------------


def _node_flatten(node: Node) -> tuple[list[tuple[str, Any]], tuple[type]]:
    graph_invisible_attrs = getattr(node, 'graph_invisible_attrs', ())
    nodes = sorted(
        (key, value) for key, value in vars(node).items()
        if key not in graph_invisible_attrs
    )
    return nodes, (type(node),)


def _node_set_key(node: Node, key: Key, value: Any) -> None:
    if not isinstance(key, str):
        raise KeyError(f'Invalid key: {key!r}')
    if (
        hasattr(node, key)
        and isinstance(state := getattr(node, key), State)
        and isinstance(value, TreefyState)
    ):
        state.update_from_ref(value)
    else:
        setattr(node, key, value)


def _node_pop_key(node: Node, key: Key) -> Any:
    if not isinstance(key, str):
        raise KeyError(f'Invalid key: {key!r}')
    return vars(node).pop(key)


def _node_create_empty(static: tuple[type[G], ...]) -> G:
    node_type, = static
    return object.__new__(node_type)


def _node_clear(node: Node) -> None:
    vars(node).clear()
