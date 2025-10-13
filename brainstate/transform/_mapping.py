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
from typing import (
    Any,
    TypeVar,
    Callable,
    Hashable,
    Sequence,
    Iterable,
    Tuple,
    Union,
    Optional,
    Dict
)

import jax

from brainstate._compatible_import import Device
from brainstate._state import catch_new_states
from brainstate._utils import set_module_as
from brainstate.typing import Missing, Filter
from brainstate.util import NestedDict
from ._loop_collect_return import scan
from ._make_jaxpr import StatefulMapping

__all__ = [
    'vmap',
    'pmap',
    'map',
    'vmap_new_states',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable


@set_module_as('brainstate.transform')
def vmap(
    fn: F | Missing = Missing(),
    *,
    # --- normal jax.vmap arguments --- #
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # --- brainstate specific arguments --- #
    state_in_axes: Union[Dict[AxisName, Filter], Filter] = None,
    state_out_axes: Union[Dict[AxisName, Filter], Filter] = None,
    unexpected_out_state_mapping: str = 'raise',
) -> StatefulMapping | Callable[[F], StatefulMapping]:
    """
    Vectorize a callable while preserving BrainState state semantics.

    This helper mirrors :func:`jax.vmap` but routes execution through
    :class:`~brainstate.transform.StatefulMapping` so that reads and writes to
    :class:`~brainstate.State` instances (including newly created random states)
    are tracked correctly across the mapped axis. The returned object can be used
    directly or as a decorator when ``fn`` is omitted.

    Parameters
    ----------
    fn : callable, optional
        Function to be vectorised. If omitted, the function acts as a decorator.
    in_axes : int | None | sequence, default 0
        Mapping specification for positional arguments, following the semantics
        of :func:`jax.vmap`.
    out_axes : any, default 0
        Placement of the mapped axis in the result. Must broadcast with the
        structure of the outputs.
    axis_name : hashable, optional
        Name for the mapped axis so that collective primitives (e.g. ``lax.psum``)
        can target it.
    axis_size : int, optional
        Explicit size of the mapped axis. If omitted, the size is inferred from
        the arguments.
    spmd_axis_name : hashable or tuple[hashable], optional
        Axis labels used when the transformed function is itself executed inside
        another SPMD transform (e.g. nested :func:`vmap` or :func:`pmap`).
    state_in_axes : dict[AxisName, Filter] or Filter, optional
        Filters identifying which :class:`State` objects should be batched on
        input. Passing a single filter is shorthand for ``{0: filter}``. Filters
        are converted with :func:`brainstate.util.filter.to_predicate`.
    state_out_axes : dict[AxisName, Filter] or Filter, optional
        Filters describing how written states are scattered back across the
        mapped axis. Semantics mirror ``state_in_axes``.
    unexpected_out_state_mapping : {'raise', 'warn', 'ignore'}, default 'raise'
        Policy when a state is written during the mapped call but not matched by
        ``state_out_axes``. ``'raise'`` propagates a :class:`BatchAxisError`,
        ``'warn'`` emits a warning, and ``'ignore'`` silently accepts the state.

    Returns
    -------
    StatefulMapping or callable
        If ``fn`` is supplied, returns a :class:`StatefulMapping` instance that
        behaves like ``fn`` but with batch semantics. Otherwise a decorator is
        returned.

    Raises
    ------
    ValueError
        If axis sizes are inconsistent or cannot be inferred.
    BatchAxisError
        If a state write violates ``state_out_axes`` and the policy is ``'raise'``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>> from brainstate.util.filter import OfType
        >>>
        >>> counter = bst.ShortTermState(jnp.array(0.0))
        >>>
        >>> @bst.transform.vmap(
        ...     in_axes=0,
        ...     out_axes=0,
        ...     state_in_axes={0: OfType(bst.ShortTermState)},
        ...     state_out_axes={0: OfType(bst.ShortTermState)},
        ... )
        ... def accumulate(x):
        ...     counter.value = counter.value + x
        ...     return counter.value
        >>>
        >>> xs = jnp.arange(3.0)
        >>> accumulate(xs)
        Array([0., 1., 3.], dtype=float32)
        >>> counter.value
        Array(3., dtype=float32)

    See Also
    --------
    brainstate.transform.StatefulMapping : Underlying state-aware mapping helper.
    pmap : Parallel mapping variant for multiple devices.
    vmap_new_states : Vectorize newly created states within ``fn``.
    """

    if isinstance(fn, Missing):
        return functools.partial(
            vmap,
            in_axes=in_axes,
            out_axes=out_axes,
            state_in_axes=state_in_axes,
            state_out_axes=state_out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            unexpected_out_state_mapping=unexpected_out_state_mapping,
        )  # type: ignore[return-value]

    return StatefulMapping(
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        state_in_axes=state_in_axes,
        state_out_axes=state_out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        unexpected_out_state_mapping=unexpected_out_state_mapping,
        mapping_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name),
        name='vmap'
    )


@set_module_as('brainstate.transform')
def pmap(
    fn: Callable[[NestedDict, ...], Any] | Missing = Missing(),
    axis_name: Optional[AxisName] = None,
    *,
    in_axes: Any = 0,
    out_axes: Any = 0,
    static_broadcasted_argnums: int | Iterable[int] = (),
    devices: Optional[Sequence[Device]] = None,  # noqa: F811
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: int | Iterable[int] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    # --- brainstate specific arguments --- #
    state_in_axes: Union[Dict[AxisName, Filter], Filter] = None,
    state_out_axes: Union[Dict[AxisName, Filter], Filter] = None,
    unexpected_out_state_mapping: str = 'raise',
) -> Callable[[F], F] | F:
    """
    Parallel mapping with state-aware semantics across devices.

    This function mirrors :func:`jax.pmap` but integrates with
    :class:`~brainstate.transform.StatefulMapping` so that
    :class:`~brainstate.State` objects (including random states) are replicated
    and restored correctly on every device. When ``fn`` is omitted the function
    can be used as a decorator.

    Parameters
    ----------
    fn : callable, optional
        Function to execute in SPMD style. If omitted, a decorator is returned.
    axis_name : hashable, optional
        Name for the mapped axis used by collective primitives.
    in_axes : any, default 0
        Axis mapping for positional arguments, identical to :func:`jax.pmap`.
    out_axes : any, default 0
        Placement of the mapped axis in the outputs.
    static_broadcasted_argnums : int or iterable[int], default ()
        Indices of positional arguments to treat as compile-time constants.
    devices : sequence[Device], optional
        Explicit device list to map over. Must be identical on every host in
        multi-host setups.
    backend : str, optional
        Backend identifier (``'cpu'``, ``'gpu'``, or ``'tpu'``).
    axis_size : int, optional
        Size of the mapped axis. Defaults to ``len(devices)`` or the local device
        count when ``devices`` is ``None``.
    donate_argnums : int or iterable[int], default ()
        Positional arguments whose buffers may be donated to the computation.
    global_arg_shapes : tuple[tuple[int, ...], ...], optional
        Shapes for globally distributed arguments (i.e. arguments not replicated
        across devices).
    state_in_axes : dict[AxisName, Filter] or Filter, optional
        Filters indicating which states should be treated as device-mapped inputs.
    state_out_axes : dict[AxisName, Filter] or Filter, optional
        Filters describing how state writes are scattered back to devices.
    unexpected_out_state_mapping : {'raise', 'warn', 'ignore'}, default 'raise'
        Policy applied when a state write is not covered by ``state_out_axes``.

    Returns
    -------
    StatefulMapping or callable
        If ``fn`` is provided, returns a :class:`StatefulMapping` executing ``fn``
        over devices. Otherwise returns a decorator that produces such an object.

    Raises
    ------
    ValueError
        If ``axis_size`` or argument shapes are inconsistent.
    BatchAxisError
        If an unexpected state write occurs and the policy is ``'raise'``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>> from brainstate.util.filter import OfType
        >>>
        >>> weights = bst.ParamState(jnp.ones((4,)))
        >>>
        >>> @bst.transform.pmap(
        ...     axis_name='devices',
        ...     in_axes=0,
        ...     out_axes=0,
        ...     state_in_axes={0: OfType(bst.ParamState)},
        ...     state_out_axes={0: OfType(bst.ParamState)},
        ... )
        ... def update(delta):
        ...     weights.value = weights.value + delta
        ...     return weights.value
        >>>
        >>> deltas = jnp.arange(jax.local_device_count() * 4.).reshape(
        ...     jax.local_device_count(), 4
        ... )
        >>> updated = update(deltas)
        >>> updated.shape
        (jax.local_device_count(), 4)

    See Also
    --------
    jax.pmap : Underlying JAX primitive.
    vmap : Single-host vectorisation with the same state semantics.
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
            unexpected_out_state_mapping=unexpected_out_state_mapping,
        )  # type: ignore[return-value]

    return StatefulMapping(
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        state_in_axes=state_in_axes,
        state_out_axes=state_out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        mapping_fn=functools.partial(
            jax.pmap,
            static_broadcasted_argnums=static_broadcasted_argnums,
            devices=devices,
            backend=backend,
            donate_argnums=donate_argnums,
            global_arg_shapes=global_arg_shapes,
        ),
        unexpected_out_state_mapping=unexpected_out_state_mapping,
        name='pmap'
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


@set_module_as('brainstate.transform')
def map(
    f,
    *xs,
    batch_size: int | None = None,
):
    """
    Apply a Python function over the leading axis of one or more pytrees.

    Compared with :func:`jax.vmap`, this helper executes sequentially by default
    (via :func:`jax.lax.scan`), making it useful when auto-vectorisation is
    impractical or when memory usage must be reduced. Providing ``batch_size``
    enables chunked evaluation that internally leverages :func:`vmap` to improve
    throughput while keeping peak memory bounded.

    Parameters
    ----------
    f : callable
        Function applied element-wise across the leading dimension. Its return
        value must be a pytree whose leaves can be stacked along axis ``0``.
    *xs : Any
        Positional pytrees sharing the same length along their leading axis.
    batch_size : int, optional
        Size of vectorised blocks. When given, ``map`` first processes full
        batches using :func:`vmap` then handles any remainder sequentially.

    Returns
    -------
    Any
        PyTree matching the structure of ``f``'s outputs with results stacked
        along the leading dimension.

    Raises
    ------
    ValueError
        If the inputs do not share the same leading length.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainstate.transform import map
        >>>
        >>> xs = jnp.arange(6).reshape(6, 1)
        >>>
        >>> def normalize(row):
        ...     return row / (1.0 + jnp.linalg.norm(row))
        >>>
        >>> stacked = map(normalize, xs, batch_size=2)
        >>> stacked.shape
        (6, 1)

    See Also
    --------
    vmap : Vectorised mapping with automatic batching.
    jax.lax.scan : Primitive used for the sequential fallback.
    """
    if batch_size is not None:
        scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)
        g = lambda _, x: ((), vmap(f)(*x))
        _, scan_ys = scan(g, (), scan_xs)
        if remainder_xs is None:
            ys = jax.tree.map(lambda x: _flatten(x), scan_ys)
        else:
            remainder_ys = vmap(f)(*remainder_xs)
            ys = jax.tree.map(
                lambda x, y: jax.lax.concatenate([_flatten(x), y], dimension=0),
                scan_ys,
                remainder_ys,
            )
    else:
        g = lambda _, x: ((), f(*x))
        _, ys = scan(g, (), xs)
    return ys


def _flatten(x):
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
    state_tag: str | None = None,
    state_to_exclude: Filter | None = None,
    state_in_axes: Union[Dict[AxisName, Filter], Filter] = None,
    state_out_axes: Union[Dict[AxisName, Filter], Filter] = None,
    unexpected_out_state_mapping: str = 'raise',
):
    # TODO: How about nested call ``vmap_new_states``?
    if isinstance(axis_size, int) and axis_size <= 0:
        raise ValueError(f"axis_size must be greater than 0, got {axis_size}.")

    @vmap(
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
        state_in_axes=state_in_axes,
        state_out_axes=state_out_axes,
        unexpected_out_state_mapping=unexpected_out_state_mapping,
    )
    def new_fun(args):
        # call the function
        with catch_new_states(state_tag=state_tag, state_to_exclude=state_to_exclude) as catcher:
            out = fun(*args)

        # get vmap state values
        vmap_state_vals = catcher.get_state_values()
        return out, vmap_state_vals

    @functools.wraps(fun)
    def vmapped_fn(*args):
        with catch_new_states(state_to_exclude=state_to_exclude) as catcher:
            outs, vmap_state_vals = new_fun(args)
            vmap_states = catcher.get_states()

        # restore vmapped state values
        assert len(vmap_state_vals) == len(vmap_states), (
            f'Internal Error: state count mismatch. {len(vmap_state_vals)} != {len(vmap_states)}'
        )
        for st_val, st in zip(vmap_state_vals, vmap_states):
            st.restore_value(st_val)
            # ------------------------------------------------
            # --- this is CRUCIAL to avoid jax tracing leakage
            # ------------------------------------------------
            st.decrease_stack_level()
        return outs

    return vmapped_fn


@set_module_as('brainstate.transform')
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
    state_tag: str | None = None,
    state_to_exclude: Filter = None,
    state_in_axes: Union[Dict[AxisName, Filter], Filter] = None,
    state_out_axes: Union[Dict[AxisName, Filter], Filter] = None,
    unexpected_out_state_mapping: str = 'raise',
):
    """
    Vectorise a function that creates new BrainState states on the fly.

    The helper wraps :func:`vmap` but also captures states instantiated inside
    ``fun`` via :func:`brainstate._state.catch_new_states`. Newly created states
    are materialised for each batch element and restored after execution so that
    their side effects persist exactly once. When ``fun`` is omitted the helper
    can be used as a decorator.

    Parameters
    ----------
    fun : callable, optional
        Function to transform. If omitted, :func:`vmap_new_states` returns a
        decorator expecting ``fun``.
    in_axes : int | None | sequence, default 0
        Mapping specification for positional arguments, following
        :func:`jax.vmap` semantics.
    out_axes : any, default 0
        Placement of the mapped axis in the outputs.
    axis_name : hashable, optional
        Name of the mapped axis for collective primitives.
    axis_size : int, optional
        Explicit size of the mapped axis. Must be positive when provided.
    spmd_axis_name : hashable or tuple[hashable], optional
        Axis labels used when nesting inside other SPMD transforms.
    state_tag : str, optional
        Tag used to limit which newly created states are tracked.
    state_to_exclude : Filter, optional
        Filter describing states that should *not* participate in the mapping.
    state_in_axes : dict[AxisName, Filter] or Filter, optional
        Filters indicating which existing states are batched on input.
    state_out_axes : dict[AxisName, Filter] or Filter, optional
        Filters describing how written states are scattered over the mapped axis.
    unexpected_out_state_mapping : {'raise', 'warn', 'ignore'}, default 'raise'
        Behaviour when a state write is not covered by ``state_out_axes``.

    Returns
    -------
    callable
        A function with vectorised semantics that also mirrors new state
        creation across the mapped axis.

    Raises
    ------
    ValueError
        If ``axis_size`` is provided and is not strictly positive.
    BatchAxisError
        If unexpected state writes occur and the policy is ``'raise'``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as bst
        >>> import jax.numpy as jnp
        >>> from brainstate.transform import vmap_new_states
        >>>
        >>> @vmap_new_states(in_axes=0, out_axes=0)
        ... def forward(x):
        ...     scratch = bst.ShortTermState(jnp.array(0.0), tag='scratch')
        ...     scratch.value = scratch.value + x
        ...     return scratch.value
        >>>
        >>> forward(jnp.arange(3.0))
        Array([0., 1., 2.], dtype=float32)

    See Also
    --------
    vmap : State-aware vectorisation for existing states.
    catch_new_states : Context manager used internally to intercept state creation.
    """
    if isinstance(fun, Missing):
        return functools.partial(
            _vmap_new_states_transform,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            state_tag=state_tag,
            state_to_exclude=state_to_exclude,
            state_in_axes=state_in_axes,
            state_out_axes=state_out_axes,
            unexpected_out_state_mapping=unexpected_out_state_mapping,
        )
    else:
        return _vmap_new_states_transform(
            fun,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            state_tag=state_tag,
            state_to_exclude=state_to_exclude,
            state_in_axes=state_in_axes,
            unexpected_out_state_mapping=unexpected_out_state_mapping,
        )
