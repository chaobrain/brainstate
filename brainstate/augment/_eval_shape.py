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

import functools
from typing import Any, TypeVar, Callable, Sequence, Union

import jax

from brainstate.graph import graph_to_tree, tree_to_graph
from brainstate.random import DEFAULT, RandomState
from ._random import restore_rngs

__all__ = [
    'eval_shape',
]

A = TypeVar('A')


def eval_shape(
    fn: Callable[..., A],
    *args: Any,
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
    **kwargs: Any,
) -> A:
    """
    Compute the shape/dtype of ``fn`` without any FLOPs.

    Here's an example::

        >>> import brainstate as bst
        >>> class MLP:
        ...     def __init__(self, n_in, n_mid, n_out):
        ...         self.dense1 = bst.nn.Linear(n_in, n_mid)
        ...         self.dense2 = bst.nn.Linear(n_mid, n_out)

        >>> r = bst.augment.eval_shape(lambda: MLP(1, 2, 3))
        >>> r
        MLP(
          dense1=Linear(
            in_size=(1,),
            out_size=(2,),
            w_mask=None,
            weight=ParamState(
              value={'bias': ShapeDtypeStruct(shape=(2,), dtype=float32), 'weight': ShapeDtypeStruct(shape=(1, 2), dtype=float32)}
            )
          ),
          dense2=Linear(
            in_size=(2,),
            out_size=(3,),
            w_mask=None,
            weight=ParamState(
              value={'bias': ShapeDtypeStruct(shape=(3,), dtype=float32), 'weight': ShapeDtypeStruct(shape=(2, 3), dtype=float32)}
            )
          )
        )

    Args:
        fn: The function whose output shape should be evaluated.
        *args: a positional argument tuple of arrays, scalars, or (nested) standard
              Python containers (tuples, lists, dicts, namedtuples, i.e. pytrees) of
              those types. Since only the ``shape`` and ``dtype`` attributes are
              accessed, one can use :class:`jax.ShapeDtypeStruct` or another container
              that duck-types as ndarrays (note however that duck-typed objects cannot
              be namedtuples because those are treated as standard Python containers).
        **kwargs: a keyword argument dict of arrays, scalars, or (nested) standard
              Python containers (pytrees) of those types. As in ``args``, array values
              need only be duck-typed to have ``shape`` and ``dtype`` attributes.
        rngs: a :class:`RandomState` or a sequence of :class:`RandomState` objects
                representing the random number generators to use. If not provided, the
                default random number generator will be used.

    Returns:
        out: a nested PyTree containing :class:`jax.ShapeDtypeStruct` objects as leaves.


    """

    @functools.wraps(fn)
    @restore_rngs(rngs=rngs)
    def _eval_shape_fn(*args_, **kwargs_):
        args_, kwargs_ = tree_to_graph((args_, kwargs_))
        out = fn(*args_, **kwargs_)
        return graph_to_tree(out)

    args, kwargs = graph_to_tree((args, kwargs))
    out = jax.eval_shape(_eval_shape_fn, *args, **kwargs)
    return tree_to_graph(out)
