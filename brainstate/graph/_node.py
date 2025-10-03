# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors.
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

from abc import ABCMeta
from copy import deepcopy
from typing import Any, Type, TypeVar, Tuple, TYPE_CHECKING

from brainstate._state import State, TreefyState
from brainstate.typing import Key
from brainstate.util._pretty_pytree import PrettyObject
from ._operation import register_graph_node_type, treefy_split, treefy_merge

__all__ = [
    'Node',
]

G = TypeVar('G', bound='Node')
A = TypeVar('A')


class GraphNodeMeta(ABCMeta):
    if not TYPE_CHECKING:
        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            node = cls.__new__(cls, *args, **kwargs)
            node.__init__(*args, **kwargs)
            return node


class Node(PrettyObject, metaclass=GraphNodeMeta):
    """
    Base class for all graph nodes.

    This class provides the following functionalities:

    - Register the node type with the graph tool.
    - Prevent mutation of the node from different trace level.
    - Provide a pretty repr for the node.
    - Provide a treescope repr for the node.
    - Deepcopy the node.

    """
    __module__ = 'brainstate.graph'

    graph_invisible_attrs = ()

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
        graphdef = deepcopy(graphdef)
        state = deepcopy(state)
        return treefy_merge(graphdef, state)


# -------------------------------
# Graph Definition
# -------------------------------


def _node_flatten(node: Node) -> Tuple[Tuple[Tuple[str, Any], ...], Tuple[Type]]:
    graph_invisible_attrs = getattr(node, 'graph_invisible_attrs', ())
    # graph_invisible_attrs = tuple(graph_invisible_attrs) + ('_trace_state',)
    nodes = sorted(
        (key, value) for key, value in vars(node).items()
        if (key not in graph_invisible_attrs)
    )
    return nodes, (type(node),)


def _node_set_key(node: Node, key: Key, value: Any) -> None:
    if not isinstance(key, str):
        raise KeyError(f'Invalid key: {key!r}')
    elif (
        hasattr(node, key)
        and isinstance(state := getattr(node, key), State)
        and isinstance(value, TreefyState)
    ):
        state.update_from_ref(value)
    else:
        setattr(node, key, value)


def _node_pop_key(node: Node, key: Key):
    if not isinstance(key, str):
        raise KeyError(f'Invalid key: {key!r}')
    return vars(node).pop(key)


def _node_create_empty(static: tuple[Type[G],]) -> G:
    node_type, = static
    node = object.__new__(node_type)
    return node


def _node_clear(node: Node):
    module_vars = vars(node)
    module_vars.clear()
