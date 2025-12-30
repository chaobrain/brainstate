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

from brainstate.graph._convert import graph_to_tree, tree_to_graph

A = TypeVar('A')

__all__ = [
    'model_eval_shape',
]


def model_eval_shape(
    f: Callable[..., A],
    *args: Any,
    **kwargs: Any,
) -> A:
    """
    Evaluate the shape of the output of a function.
    """

    @functools.wraps(f)
    def _eval_shape_fn(*args_, **kwargs_):
        args_, kwargs_ = tree_to_graph((args_, kwargs_))
        out_ = f(*args_, **kwargs_)
        return graph_to_tree(out_)[0]

    args, kwargs = graph_to_tree((args, kwargs))[0]
    out = jax.eval_shape(_eval_shape_fn, *args, **kwargs)
    return tree_to_graph(out)
