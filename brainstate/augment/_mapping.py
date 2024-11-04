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

import dataclasses
import functools
from typing import Any, TypeVar, Callable, Hashable, Sequence, Iterable, Mapping, Tuple, Union, Optional

import jax

from brainstate.graph import (NodeStates, graph_to_tree, tree_to_graph, update_context)
from brainstate.graph._graph_convert import clear_non_graph_nodes
from brainstate.random import DEFAULT, RandomState
from brainstate.typing import Missing, Filter
from brainstate.util import NestedDict
from ._random import restore_rngs

__all__ = [
    'StateAxes',
    'vmap',
    'pmap',
]

AxisName = Hashable
F = TypeVar("F", bound=Callable)
Index = int
Carry = TypeVar("Carry")


class StateAxes:
    """
    A class to represent the axes of a state.

    This class is used to control how graph nodes like Modules are vectorized or
    parallelized by specifying the axes to be applied to substates of the graph
    node given a Filter.

    Args:
        filter_axes: A mapping from filters to axes. The axes can be an index, a carry or None.

    """

    def __init__(
        self,
        filter_axes: Union[Mapping[Filter, Index | Carry | None], Iterable[Tuple[Filter, Index | Carry | None]]],
    ):
        iterable = filter_axes.items() if isinstance(filter_axes, Mapping) else filter_axes
        self._filters = tuple(filter_ for filter_, _ in iterable)
        self._axes = tuple(axis for _, axis in iterable)

    @property
    def filters(self) -> Tuple[Filter, ...]:
        return self._filters

    @property
    def axes(self) -> Tuple[Index | Carry | None, ...]:
        return self._axes

    def __repr__(self):
        return f'StateAxes({dict(self.items())})'

    def items(self):
        return zip(self.filters, self.axes)

    def __eq__(self, other):
        return isinstance(other, StateAxes) and self.filters == other.filters and self.axes == other.axes

    def __hash__(self):
        return hash((self.filters, self.axes))


def _map_split_fn(ctx, path, prefix, x):
    if isinstance(prefix, StateAxes):
        return NodeStates.from_split(*ctx.treefy_split(x, *prefix.filters), metadata=prefix)
    return NodeStates.from_split(*ctx.treefy_split(x), metadata=prefix)


@dataclasses.dataclass(eq=False)
class MapFn:
    f: Callable[..., Any]
    in_axes: Any
    out_axes: Any
    ctxtag: str

    def __post_init__(self):
        functools.update_wrapper(self, self.f)

    def __call__(self, *pure_args: Tuple[Any, ...]):
        # pytree to graph
        args = tree_to_graph(pure_args, ctxtag=self.ctxtag)

        # call the function
        out = self.f(*args)

        # graph to pytree
        args_out = clear_non_graph_nodes(args)
        pure_args_out, pure_out = graph_to_tree(
            (args_out, out),
            prefix=(self.in_axes, self.out_axes),
            split_fn=_map_split_fn,
            ctxtag=self.ctxtag,
        )
        return pure_args_out, pure_out


def _map_transform(
    ctxtag,
    transform,
    f: F,
    *,
    in_axes: Optional[int | Sequence[Any]] = 0,
    out_axes: Any = 0,
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
    **transform_kwargs,
):
    # jax in axes
    jax_in_axes = jax.tree.map(
        lambda x: NodeStates.from_prefixes(x.axes, metadata=x) if isinstance(x, StateAxes) else x,
        in_axes,
    )

    # jax out axes
    jax_out_axes = jax.tree.map(
        lambda x: NodeStates.from_prefixes(x.axes, metadata=x) if isinstance(x, StateAxes) else x,
        out_axes,
    )

    # mapped function
    mapped_fn = transform(
        MapFn(f, in_axes, out_axes, ctxtag),
        in_axes=jax_in_axes,
        out_axes=(jax_in_axes, jax_out_axes),
        **transform_kwargs
    )

    @functools.wraps(f)
    @restore_rngs(rngs=rngs)  # restore the random key of default random number generator
    @update_context(ctxtag)
    def map_wrapper(*args):
        # graph to pytree
        pure_args = graph_to_tree(args, prefix=in_axes, split_fn=_map_split_fn, ctxtag=ctxtag)

        # vmap with pytree
        pure_args_out, pure_out = mapped_fn(*pure_args)

        # pytree to graph
        _args_out, out = tree_to_graph((pure_args_out, pure_out), ctxtag=ctxtag)
        return out

    return map_wrapper  # type: ignore


def vmap(
    fn: F | Missing = Missing(),
    *,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # brainstate specific arguments
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
) -> F | Callable[[F], F]:
    """
    Vectorizing map. Creates a function which maps ``fun`` over argument axes.

    The transformation :func:`vmap` is designed to work with ``pygraph`` structure
    defined in the ``brainstate`` library. It is used to vectorize functions by
    pushing the mapped axis down into primitive operations.

    More information please see `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__.


    These are several example usage::

        >>> import brainstate as bst
        >>> import jax.numpy as jnp

        >>> model = bst.nn.Linear(2, 3)
        >>> x = jnp.ones((5, 2))

        >>> @bst.augment.vmap(in_axes=(None, 0), out_axes=0)
        ... def forward(model, x):
        ...     return model(x)

        >>> y = forward(model, x)
        >>> print(y.shape)
        (5, 3)

    Another example with a more complex model::

        >>> class LinearEnsemble(bst.nn.Module):
        ...     def __init__(self, n: int):
        ...         super().__init__()
        ...         self.n = n
        ...         self.w = bst.ParamState(bst.random.random((n, 2, 3)))

        >>> model = LinearEnsemble(5)
        >>> x = jnp.ones((2,))

        >>> @bst.augment.vmap(in_axes=(0, None), out_axes=0)
        ... def forward(model, x):
        ...     return jnp.dot(x, model.w.value)

        >>> y = forward(model, x)
        >>> print(y.shape)
        (5, 3)

    To control how different types of states are vectorized, ``StateAxes``
    can be passed to ``in_axes`` and ``out_axes`` specifying the axes to be
    applied to each substate given a filter. The following example shows how to
    share the parameters between the ensemble members which keeping different
    batch statistics and dropout random state::

        >>> class Foo(bst.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.a = bst.ParamState(jnp.arange(4))
        ...         self.b = bst.ShortTermState(jnp.arange(4))

        >>> state_axes = bst.augment.StateAxes({bst.ParamState: 0, bst.ShortTermState: None})
        >>> @bst.augment.vmap(in_axes=(state_axes,), out_axes=0)
        ... def mul(foo):
        ...     return foo.a.value * foo.b.value

        >>> model = Foo()
        >>> y = mul(model)
        >>> print(y.shape)
        (4, 4)

    Args:
        fn: Function to be mapped over additional axes.
        in_axes: An integer, None, or sequence of values specifying which input
          array axes to map over.

          If each positional argument to ``fun`` is an array, then ``in_axes`` can
          be an integer, a None, or a tuple of integers and Nones with length equal
          to the number of positional arguments to ``fun``. An integer or ``None``
          indicates which array axis to map over for all arguments (with ``None``
          indicating not to map any axis), and a tuple indicates which axis to map
          for each corresponding positional argument. Axis integers must be in the
          range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
          dimensions (axes) of the corresponding input array.

          If the positional arguments to ``fun`` are container (pytree) types, ``in_axes``
          must be a sequence with length equal to the number of positional arguments to
          ``fun``, and for each argument the corresponding element of ``in_axes`` can
          be a container with a matching pytree structure specifying the mapping of its
          container elements. In other words, ``in_axes`` must be a container tree prefix
          of the positional argument tuple passed to ``fun``. See this link for more detail:
          https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees

          Either ``axis_size`` must be provided explicitly, or at least one
          positional argument must have ``in_axes`` not None. The sizes of the
          mapped input axes for all mapped positional arguments must all be equal.

          Arguments passed as keywords are always mapped over their leading axis
          (i.e. axis index 0).

          See below for examples.

        out_axes: An integer, None, or (nested) standard Python container
          (tuple/list/dict) thereof indicating where the mapped axis should appear
          in the output. All outputs with a mapped axis must have a non-None
          ``out_axes`` specification. Axis integers must be in the range ``[-ndim,
          ndim)`` for each output array, where ``ndim`` is the number of dimensions
          (axes) of the array returned by the :func:`vmap`-ed function, which is one
          more than the number of dimensions (axes) of the corresponding array
          returned by ``fun``.
        axis_name: Optional, a hashable Python object used to identify the mapped
          axis so that parallel collectives can be applied.
        axis_size: Optional, an integer indicating the size of the axis to be
          mapped. If not provided, the mapped axis size is inferred from arguments.
        spmd_axis_name: Optional, a hashable Python object or tuple of hashable
            Python objects used to identify the mapped axis so that parallel collectives
            can be applied. This is used to specify multiple axes to be mapped over
            in a nested :func:`vmap` call. The length of the tuple must match the
            number of nested :func:`vmap` calls. The first element of the tuple
            corresponds to the outermost :func:`vmap` call, the second element to
            the next outermost, and so on. If the tuple is not provided, the
            ``axis_name`` is used for all nested :func:`vmap` calls.
        rngs: Optional, a random number generator or sequence of random number
            generators to be used in the mapped function. These random number
            generators are restored their random key after the mapped function is
            executed.

    Returns:
        Batched/vectorized version of ``fun`` with arguments that correspond to
        those of ``fun``, but with extra array axes at positions indicated by
        ``in_axes``, and a return value that corresponds to that of ``fun``, but
        with extra array axes at positions indicated by ``out_axes``.

    """
    if isinstance(fn, Missing):
        return functools.partial(
            vmap,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            rngs=rngs,
        )  # type: ignore[return-value]

    return _map_transform(
        'vmap',  # ctxtag
        jax.vmap,
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
        rngs=rngs
    )


def pmap(
    fn: Callable[[NestedDict, ...], Any] | Missing = Missing(),
    axis_name: Optional[AxisName] = None,
    *,
    in_axes: Any = 0,
    out_axes: Any = 0,
    static_broadcasted_argnums: int | Iterable[int] = (),
    devices: Optional[Sequence[jax.Device]] = None,  # noqa: F811
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: int | Iterable[int] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    # brainstate specific arguments
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
) -> Callable[[F], F] | F:
    """
    Parallel map with support for collective operations.

    The purpose of :py:func:`pmap` is to express single-program multiple-data
    (SPMD) programs. Applying :py:func:`pmap` to a function will compile the
    function with XLA (similarly to :py:func:`jit`), then execute it in parallel
    on XLA devices, such as multiple GPUs or multiple TPU cores. Semantically it
    is comparable to :py:func:`vmap` because both transformations map a function
    over array axes, but where :py:func:`vmap` vectorizes functions by pushing the
    mapped axis down into primitive operations, :py:func:`pmap` instead replicates
    the function and executes each replica on its own XLA device in parallel.

    The mapped axis size must be less than or equal to the number of local XLA
    devices available, as returned by :py:func:`jax.local_device_count()` (unless
    ``devices`` is specified, see below). For nested :py:func:`pmap` calls, the
    product of the mapped axis sizes must be less than or equal to the number of
    XLA devices.

    More information please see `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__.

    If there are 4 XLA devices available, the following example will execute
    the function in parallel on each device::


        >>> import brainstate as bst
        >>> import jax.numpy as jnp

        >>> model = bst.nn.Linear(2, 3)
        >>> x = jnp.ones((4, 2))

        >>> @bst.augment.vmap(in_axes=(None, 0), out_axes=0)
        ... def forward(model, x):
        ...     return model(x)

        >>> y = forward(model, x)
        >>> print(y.shape)
        (4, 3)

    Another example with a more complex model::

        >>> class LinearEnsemble(bst.nn.Module):
        ...     def __init__(self, n: int):
        ...         super().__init__()
        ...         self.n = n
        ...         self.w = bst.ParamState(bst.random.random((n, 2, 3)))

        >>> model = LinearEnsemble(4)
        >>> x = jnp.ones((2,))

        >>> @bst.augment.vmap(in_axes=(0, None), out_axes=0)
        ... def forward(model, x):
        ...     return jnp.dot(x, model.w.value)

        >>> y = forward(model, x)
        >>> print(y.shape)
        (4, 3)

    To control how different types of states are vectorized, ``StateAxes``
    can be passed to ``in_axes`` and ``out_axes`` specifying the axes to be
    applied to each substate given a filter. The following example shows how to
    share the parameters between the ensemble members which keeping different
    batch statistics and dropout random state::

        >>> class Foo(bst.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.a = bst.ParamState(jnp.arange(4))
        ...         self.b = bst.ShortTermState(jnp.arange(4))

        >>> state_axes = bst.augment.StateAxes({bst.ParamState: 0, bst.ShortTermState: None})
        >>> @bst.augment.vmap(in_axes=(state_axes,), out_axes=0)
        ... def mul(foo):
        ...     return foo.a.value * foo.b.value

        >>> model = Foo()
        >>> y = mul(model)
        >>> print(y.shape)
        (4, 4)


    Args:
      fn: Function to be mapped over argument axes. Its arguments and return
        value should be arrays, scalars, or (nested) standard Python containers
        (tuple/list/dict) thereof. Positional arguments indicated by
        ``static_broadcasted_argnums`` can be anything at all, provided they are
        hashable and have an equality operation defined.
      axis_name: Optional, a hashable Python object used to identify the mapped
        axis so that parallel collectives can be applied.
      in_axes: A non-negative integer, None, or nested Python container thereof
        that specifies which axes of positional arguments to map over. Arguments
        passed as keywords are always mapped over their leading axis (i.e. axis
        index 0). See :py:func:`vmap` for details.
      out_axes: A non-negative integer, None, or nested Python container thereof
        indicating where the mapped axis should appear in the output. All outputs
        with a mapped axis must have a non-None ``out_axes`` specification
        (see :py:func:`vmap`).
      static_broadcasted_argnums: An int or collection of ints specifying which
        positional arguments to treat as static (compile-time constant).
        Operations that only depend on static arguments will be constant-folded.
        Calling the pmapped function with different values for these constants
        will trigger recompilation. If the pmapped function is called with fewer
        positional arguments than indicated by ``static_broadcasted_argnums`` then
        an error is raised. Each of the static arguments will be broadcasted to
        all devices. Arguments that are not arrays or containers thereof must be
        marked as static. Defaults to ().

        Static arguments must be hashable, meaning both ``__hash__`` and
        ``__eq__`` are implemented, and should be immutable.

      devices: This is an experimental feature and the API is likely to change.
        Optional, a sequence of Devices to map over. (Available devices can be
        retrieved via jax.devices()). Must be given identically for each process
        in multi-process settings (and will therefore include devices across
        processes). If specified, the size of the mapped axis must be equal to
        the number of devices in the sequence local to the given process. Nested
        :py:func:`pmap` s with ``devices`` specified in either the inner or outer
        :py:func:`pmap` are not yet supported.
      backend: This is an experimental feature and the API is likely to change.
        Optional, a string representing the XLA backend. 'cpu', 'gpu', or 'tpu'.
      axis_size: Optional; the size of the mapped axis.
      donate_argnums: Specify which positional argument buffers are "donated" to
        the computation. It is safe to donate argument buffers if you no longer need
        them once the computation has finished. In some cases XLA can make use of
        donated buffers to reduce the amount of memory needed to perform a
        computation, for example recycling one of your input buffers to store a
        result. You should not reuse buffers that you donate to a computation, JAX
        will raise an error if you try to.
        Note that donate_argnums only work for positional arguments, and keyword
        arguments will not be donated.

        For more details on buffer donation see the
        `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
      global_arg_shapes: Optional; a tuple of tuples of integers representing the
        shapes of the global arguments. These are arguments that are not replicated
        across devices, but are broadcasted to all devices. The tuple should have
        the same length as the number of global arguments, and each inner tuple
        should have the same length as the corresponding argument. The shapes of
        the global arguments must be the same on all devices.
      rngs: Optional, a random number generator or sequence of random number
        generators to be used in the mapped function. These random number
        generators are restored their random key after the mapped function is
        executed.

    Returns:
      A parallelized version of ``fun`` with arguments that correspond to those of
      ``fun`` but with extra array axes at positions indicated by ``in_axes`` and
      with output that has an additional leading array axis (with the same size).

    """

    if isinstance(fn, Missing):
        return functools.partial(
            pmap,
            axis_name=axis_name,
            in_axes=in_axes,
            out_axes=out_axes,
            static_broadcasted_argnums=static_broadcasted_argnums,
            devices=devices,
            backend=backend,
            axis_size=axis_size,
            donate_argnums=donate_argnums,
            global_arg_shapes=global_arg_shapes,
            rngs=rngs,
        )  # type: ignore[return-value]

    return _map_transform(
        'pmap',  # ctxtag
        jax.pmap,
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        static_broadcasted_argnums=static_broadcasted_argnums,
        devices=devices,
        backend=backend,
        axis_size=axis_size,
        donate_argnums=donate_argnums,
        global_arg_shapes=global_arg_shapes,
        rngs=rngs,
    )
