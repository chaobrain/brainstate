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
from typing import Any, TypeVar, Callable, Hashable, Sequence, Iterable, Tuple, Union, Optional, Dict, List

import jax
from jax.interpreters.batching import BatchTracer

from brainstate._state import State, StateTraceStack, catch_new_states
from brainstate.compile._loop_collect_return import scan
from brainstate.random import DEFAULT, RandomState
from brainstate.typing import Missing, Filter
from brainstate.util import NestedDict, BrainStateError
from ._random import restore_rngs

__all__ = [
    'vmap',
    'pmap',
    'map',
    'vmap_new_states',
]

AxisName = Hashable
F = TypeVar("F", bound=Callable)
AxisToState = Dict[int, List[State]]
StateToAxis = Dict[State, int]


class BatchAxisError(BrainStateError):
    pass


def _flatten_in_out_states(
    in_states: Dict[int, Dict] | Any = None,
) -> Tuple[AxisToState, StateToAxis]:
    if in_states is None:
        return dict(), dict()
    if isinstance(in_states, dict):
        keys = tuple(in_states.keys())
        values = tuple(in_states.values())
        is_axis_in_states = (
            all([isinstance(key, int) for key in keys]) and
            all([isinstance(value, dict) for value in values])
        )
    else:
        is_axis_in_states = False
    if is_axis_in_states:
        axis_to_states = {key: list(value.values()) for key, value in in_states.items()}
        state_to_axis = {}
        for key, value in in_states.items():
            for state in value.values():
                state_to_axis[state] = key
        return axis_to_states, state_to_axis
    else:
        in_states = jax.tree.leaves(in_states)
        axis_to_states = {0: list(in_states)}
        state_to_axis = {state: 0 for state in in_states}
        return axis_to_states, state_to_axis


def _vmap_transform(
    f: F,
    *,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    in_states: Dict[int, Dict] | Any | None = None,
    out_states: Dict[int, Dict] | Any | None = None,
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
    **transform_kwargs,
):
    axis_to_in_states, in_state_to_axis = _flatten_in_out_states(in_states)
    axis_to_out_states, out_state_to_axis = _flatten_in_out_states(out_states)
    for _in_state, _axis in in_state_to_axis.items():
        if _in_state in out_state_to_axis:
            _out_axis = out_state_to_axis[_in_state]
            if _out_axis != _axis:
                _in_state.raise_error_with_source_info(
                    BatchAxisError(
                        f"State {_in_state} has been mapped to axis {_axis} in in_states, "
                        f"However, it is mapped to axis {_out_axis} in out_states."
                    )
                )
        else:
            out_state_to_axis[_in_state] = _axis
            if _axis not in axis_to_out_states:
                axis_to_out_states[_axis] = []
            axis_to_out_states[_axis].append(_in_state)
    if isinstance(rngs, RandomState):
        rngs = (rngs,)
    rng_ids = set([id(rng) for rng in rngs])

    @functools.wraps(f)
    def new_fn(in_states_, args):
        # restore state values
        for i, states in enumerate(axis_to_in_states.values()):
            for state, state_val in zip(states, in_states_[i]):
                state.restore_value(state_val)

        # call the function
        with StateTraceStack() as stack:
            outs = f(*args)

        # analyze
        for state in stack.get_write_states():
            leaves = jax.tree.leaves(state.value)
            if isinstance(leaves[0], BatchTracer) and state not in out_state_to_axis:
                if isinstance(state, RandomState) and id(state) in rng_ids:
                    continue
                state.raise_error_with_source_info(
                    BatchAxisError(
                        f"The value of State {state} is batched, but it is not in the out_states."
                    )
                )

        out_states_ = [
            [state.value for state in states]
            for axis, states in axis_to_out_states.items()
        ]
        return out_states_, outs

    @functools.wraps(f)
    def vmapped_fn(*args):
        # vmapping
        in_state_vals = [
            [st.value for st in states]
            for axis, states in axis_to_in_states.items()
        ]
        in_axes_st = list(axis_to_in_states.keys())
        out_axes_st = list(axis_to_out_states.keys())
        if len(in_axes_st) == 0:
            in_axes_st = 0
        if len(out_axes_st) == 0:
            out_axes_st = 0
        out_state_vals, outs = restore_rngs(
            jax.vmap(
                new_fn,
                in_axes=(in_axes_st, in_axes),
                out_axes=(out_axes_st, out_axes),
                **transform_kwargs
            ),
            rngs=rngs
        )(in_state_vals, args)

        # restore mapped state values
        for i, states in enumerate(axis_to_out_states.values()):
            for state, st_val in zip(states, out_state_vals[i]):
                state.restore_value(st_val)
        return outs

    return vmapped_fn


def vmap(
    fn: F | Missing = Missing(),
    *,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # brainstate specific arguments
    in_states: Dict[int, Dict] | Any | None = None,
    out_states: Dict[int, Dict] | Any | None = None,
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

        >>> class Model(bst.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>
        >>>         self.a = bst.ShortTermState(bst.random.randn(5))
        >>>         self.b = bst.ShortTermState(bst.random.randn(5))
        >>>         self.c = bst.State(bst.random.randn(1))

        >>>     def __call__(self, *args, **kwargs):
        >>>         self.c.value = self.a.value * self.b.value
        >>>         return self.c.value + 1.

        >>> model = Model()

        >>> r = bst.augment.vmap(
        >>>     model,
        >>>     in_states=model.states(bst.ShortTermState),
        >>>     out_states=model.c
        >>> )()

    Args:
        fn: Function to be mapped over additional axes.
        in_axes: An integer, None, or sequence of values specifying which input
          array axes to map over.
        out_axes: An integer, None, or (nested) standard Python container
          (tuple/list/dict) thereof indicating where the mapped axis should appear
          in the output.
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
        in_states: Optional, the :class:`State` objects to be mapped over in the inputs.
        out_states: Optional, the :class:`State` objects to be mapped over in the outputs.

    Returns:
        Batched/vectorized version of ``fun`` with arguments that correspond to
        those of ``fun``, but with extra array axes at positions indicated by
        ``in_axes``, and a return value that corresponds to that of ``fun``, but
        with extra array axes at positions indicated by ``out_axes``.

    """

    if isinstance(fn, Missing):
        return functools.partial(
            _vmap_transform,
            in_axes=in_axes,
            out_axes=out_axes,
            in_states=in_states,
            out_states=out_states,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            rngs=rngs,
        )  # type: ignore[return-value]

    return _vmap_transform(
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        in_states=in_states,
        out_states=out_states,
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

    return restore_rngs(
        jax.pmap(
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
        ),
        rngs=rngs
    )


def _batch_and_remainder(x, batch_size: int):
    leaves, tree_def = jax.tree.flatten(x)

    scan_leaves = []
    remainder_leaves = []

    length = None
    for leaf in leaves:
        if length is None:
            length = leaf.shape[0]
        if length != leaf.shape[0]:
            raise ValueError(f"All inputs must have the same length. Got {length} and {leaf.shape[0]}.")

    num_batches, num_remainder = divmod(length, batch_size)
    for leaf in leaves:
        total_batch_elems = num_batches * batch_size
        scan_leaves.append(leaf[:total_batch_elems].reshape(num_batches, batch_size, *leaf.shape[1:]))
        if num_remainder:
            remainder_leaves.append(leaf[total_batch_elems:])

    scan_tree = tree_def.unflatten(scan_leaves)
    if num_remainder:
        remainder_tree = tree_def.unflatten(remainder_leaves)
        return scan_tree, remainder_tree
    else:
        return scan_tree, None


def map(
    f,
    *xs,
    batch_size: int | None = None,
):
    """
    Map a function over leading array axes.

    Like Python's builtin map, except inputs and outputs are in the form of
    stacked arrays. Consider using the :func:`~jax.vmap` transform instead, unless you
    need to apply a function element by element for reduced memory usage or
    heterogeneous computation with other control flow primitives.

    When ``xs`` is an array type, the semantics of :func:`~map` are given by this
    Python implementation::

        def map(f, *xs):
            return np.stack([f(*x) for x in xs])

    Like :func:`~scan`, :func:`~map` is implemented in terms of JAX primitives so
    many of the same advantages over a Python loop apply: ``xs`` may be an
    arbitrary nested pytree type, and the mapped computation is compiled only
    once.

    If ``batch_size`` is provided, the computation is executed in batches of that size
    and parallelized using :func:`~jax.vmap`. This can be used as either a more performant
    version of ``map`` or as a memory-efficient version of ``vmap``. If the axis is not
    divisible by the batch size, the remainder is processed in a separate ``vmap`` and
    concatenated to the result.

        >>> import jax.numpy as jnp
        >>> x = jnp.ones((10, 3, 4))
        >>> def f(x):
        ...   print('inner shape:', x.shape)
        ...   return x + 1
        >>> y = map(f, x, batch_size=3)
        inner shape: (3, 4)
        inner shape: (3, 4)
        >>> y.shape
        (10, 3, 4)

    In the example above, "inner shape" is printed twice, once while tracing the batched
    computation and once while tracing the remainder computation.

    Args:
        f: a Python function to apply element-wise over the first axis or axes of
            ``xs``.
        xs: values over which to map along the leading axis.
        batch_size: (optional) integer specifying the size of the batch for each step to execute
            in parallel.

    Returns:
        Mapped values.
    """
    if batch_size is not None:
        scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)
        g = lambda _, x: ((), vmap(f)(*x))
        _, scan_ys = scan(g, (), scan_xs)
        if remainder_xs is None:
            ys = jax.tree.map(lambda x: flatten_(x), scan_ys)
        else:
            remainder_ys = vmap(f)(*remainder_xs)
            ys = jax.tree.map(
                lambda x, y: jax.lax.concatenate([flatten_(x), y], dimension=0),
                scan_ys,
                remainder_ys,
            )
    else:
        g = lambda _, x: ((), f(*x))
        _, ys = scan(g, (), xs)
    return ys


def flatten_(x):
    return x.reshape(-1, *x.shape[2:])


def _vmap_new_states_transform(
    fun: Callable[..., Any],
    *,
    # -- normal jax.vmap arguments -- #
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # -- brainstate specific arguments -- #
    tag: str | None = None,
    state_to_exclude: Filter | None = None,
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
):
    if isinstance(rngs, RandomState):
        rngs = (rngs,)
    assert axis_size is not None, "axis_size must be provided for vmap_new_states."

    @vmap(
        in_axes=(0, in_axes),
        out_axes=(out_axes, 0),
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
        rngs=rngs
    )
    def new_fun(keys, args):
        # set random keys
        assert len(keys) == axis_size, "The length of keys must be equal to axis_size."
        for key, rng in zip(keys, rngs):
            rng.set_key(key)

        # call the function
        with catch_new_states(tag=tag, states_to_exclude=state_to_exclude) as catcher:
            out = fun(*args)

        # get vmap state values
        vmap_state_vals = catcher.get_state_values()
        return out, vmap_state_vals

    @functools.wraps(fun)
    def vmapped_fn(*args):
        # vmapping
        with catch_new_states(states_to_exclude=state_to_exclude) as catcher:
            outs, vmap_state_vals = new_fun([rng.split_key(axis_size) for rng in rngs], args)
            vmap_states = catcher.get_states()

        # restore vmapped state values
        for st_val, st in zip(vmap_state_vals, vmap_states):
            st.restore_value(st_val)
        return outs

    return vmapped_fn


def vmap_new_states(
    fun: Callable = Missing(),
    *,
    # -- normal jax.vmap arguments -- #
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # -- brainstate specific arguments -- #
    tag: str | None = None,
    states_to_exclude: Sequence[int] = (),
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
):
    """
    Vectorize a function over new states created within it.

    This function applies JAX's vmap transformation to newly created states
    during the function's execution. It allows for more
    flexible vectorization in the context of stateful computations.






    Args:
        fun (Callable, optional): The function to be vectorized. Defaults to Missing().
        in_axes (int | None | Sequence[Any], optional): Specification of input axes for vectorization. Defaults to 0.
        out_axes (Any, optional): Specification of output axes after vectorization. Defaults to 0.
        axis_name (AxisName, optional): Name of the axis being vectorized over. Defaults to None.
        axis_size (int, optional): Size of the axis being vectorized over. Defaults to None.
        spmd_axis_name (AxisName | tuple[AxisName, ...], optional): Name(s) of SPMD axis/axes. Defaults to None.
        tag (str, optional): A tag to identify specific states. Defaults to None.
        states_to_exclude (Sequence[int], optional): Indices of states to exclude from vectorization. Defaults to ().
        rngs (Union[RandomState, Sequence[RandomState]], optional): Random number generator(s) to use.
            Defaults to :class:`DEFAULT`.

    Returns:
        Callable: A vectorized version of the input function that handles new state creation.
    """
    if isinstance(fun, Missing):
        return functools.partial(
            _vmap_new_states_transform,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            tag=tag,
            states_to_exclude=states_to_exclude,
            rngs=rngs,
        )
    else:
        return _vmap_new_states_transform(
            fun,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            tag=tag,
            state_to_exclude=states_to_exclude,
            rngs=rngs,
        )
