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

import functools
from typing import Any, TypeVar, Callable, Sequence, Union

import jax

from brainstate import random
from brainstate._utils import set_module_as
from brainstate.graph import Node, flatten, unflatten
from ._random import restore_rngs

__all__ = [
    'abstract_init',
]

A = TypeVar('A')


@set_module_as('brainstate.transform')
def abstract_init(
    fn: Callable[..., A],
    *args: Any,
    rngs: Union[random.RandomState, Sequence[random.RandomState]] = random.DEFAULT,
    **kwargs: Any,
) -> A:
    """
    Compute the shape/dtype of ``fn`` without any FLOPs.

    This function evaluates the shape and dtype of the output of a function without
    actually executing the computational operations. It's particularly useful for
    initializing neural network models to understand their structure and parameter
    shapes without performing expensive computations.

    Parameters
    ----------
    fn : callable
        The function whose output shape should be evaluated.
    *args
        Positional argument tuple of arrays, scalars, or (nested) standard
        Python containers (tuples, lists, dicts, namedtuples, i.e. pytrees) of
        those types. Since only the ``shape`` and ``dtype`` attributes are
        accessed, one can use :class:`jax.ShapeDtypeStruct` or another container
        that duck-types as ndarrays (note however that duck-typed objects cannot
        be namedtuples because those are treated as standard Python containers).
    rngs : RandomState or sequence of RandomState, default random.DEFAULT
        A :class:`RandomState` or a sequence of :class:`RandomState` objects
        representing the random number generators to use. If not provided, the
        default random number generator will be used.
    **kwargs
        Keyword argument dict of arrays, scalars, or (nested) standard
        Python containers (pytrees) of those types. As in ``args``, array values
        need only be duck-typed to have ``shape`` and ``dtype`` attributes.

    Returns
    -------
    A
        A nested PyTree containing :class:`jax.ShapeDtypeStruct` objects as leaves,
        representing the structure and shape/dtype information of the function output.

    Examples
    --------
    Basic usage with neural network initialization:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> class MLP:
        ...     def __init__(self, n_in, n_mid, n_out):
        ...         self.dense1 = brainstate.nn.Linear(n_in, n_mid)
        ...         self.dense2 = brainstate.nn.Linear(n_mid, n_out)
        >>>
        >>> # Get shape information without actual computation
        >>> model_shape = brainstate.transform.abstract_init(lambda: MLP(1, 2, 3))

    With function arguments:

    .. code-block:: python

        >>> def create_model(input_size, hidden_size, output_size):
        ...     return brainstate.nn.Sequential([
        ...         brainstate.nn.Linear(input_size, hidden_size),
        ...         brainstate.nn.ReLU(),
        ...         brainstate.nn.Linear(hidden_size, output_size)
        ...     ])
        >>>
        >>> # Abstract initialization with arguments
        >>> model_shape = brainstate.transform.abstract_init(
        ...     create_model, 784, 256, 10
        ... )

    Using custom random number generators:

    .. code-block:: python

        >>> import brainstate.random as random
        >>>
        >>> # Create custom RNG
        >>> rng = random.RandomState(42)
        >>>
        >>> def init_with_custom_weights():
        ...     return brainstate.nn.Linear(10, 5)
        >>>
        >>> model_shape = brainstate.transform.abstract_init(
        ...     init_with_custom_weights, rngs=rng
        ... )

    Evaluating function with array inputs:

    .. code-block:: python

        >>> def model_forward(x):
        ...     layer = brainstate.nn.Linear(x.shape[-1], 128)
        ...     return layer(x)
        >>>
        >>> # Use ShapeDtypeStruct to represent input without actual data
        >>> input_shape = jax.ShapeDtypeStruct((32, 784), jnp.float32)
        >>> output_shape = brainstate.transform.abstract_init(model_forward, input_shape)
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
