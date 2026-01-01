# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import functools
from typing import Callable, TypeVar, Any

import jax

from brainstate._state import StateTraceStack
from brainstate.graph import graph_to_tree, tree_to_graph

__all__ = [
    'eval_shape',
]

A = TypeVar('A')


def eval_shape(
    f: Callable[..., A],
    *args: Any,
    **kwargs: Any,
) -> A:
    """
    Evaluate the shape of the output of a function.
    """
    from brainstate.random import DEFAULT
    def check_state(st):
        if st is not DEFAULT and id(st) not in find_state_ids:
            st.raise_error_with_source_info(
                ValueError('')
            )

    @functools.wraps(f)
    def _eval_shape_fn(*args_, **kwargs_):
        args_, kwargs_ = tree_to_graph((args_, kwargs_))
        with StateTraceStack(check_read=check_state) as stack:
            out_ = f(*args_, **kwargs_)
        return graph_to_tree(out_)[0]

    (args, kwargs), find_states = graph_to_tree((args, kwargs))
    find_state_ids = set(id(s) for s in find_states.values())
    out = jax.eval_shape(_eval_shape_fn, *args, **kwargs)
    return tree_to_graph(out)
