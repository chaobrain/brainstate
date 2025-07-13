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

import functools
from typing import Any, TypeVar, Callable, Sequence, Union

import jax

from brainstate import random
from brainstate.graph import Node, flatten, unflatten
from ._random import restore_rngs

__all__ = [
    'abstract_init',
]

A = TypeVar('A')


def abstract_init(
    fn: Callable[..., A],
    *args: Any,
    rngs: Union[random.RandomState, Sequence[random.RandomState]] = random.DEFAULT,
    **kwargs: Any,
) -> A:
    """
    Compute the shape/dtype of ``fn`` without any FLOPs.

    Here's an example::

        >>> import brainstate
        >>> class MLP:
        ...     def __init__(self, n_in, n_mid, n_out):
        ...         self.dense1 = brainstate.nn.Linear(n_in, n_mid)
        ...         self.dense2 = brainstate.nn.Linear(n_mid, n_out)

        >>> r = brainstate.augment.abstract_init(lambda: MLP(1, 2, 3))
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
        out = fn(*args_, **kwargs_)
        assert isinstance(out, Node), 'The output of the function must be Node'
        graph_def, treefy_states = flatten(out)
        return graph_def, treefy_states

    graph_def_, treefy_states_ = jax.eval_shape(_eval_shape_fn, *args, **kwargs)
    return unflatten(graph_def_, treefy_states_)
