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
import inspect
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Callable, Dict, Optional, Tuple, Union, TypeVar

import jax
import jax.numpy as jnp

from brainstate._compatible_import import Device
from brainstate._state import State, NonBatchState, catch_new_states, StateTraceStack, TRACE_CONTEXT
from brainstate._utils import set_module_as
from brainstate.typing import Missing, Filter
from brainstate.util import NestedDict, filter
from ._loop_collect_return import scan
from ._mapping_core import (
    normalize_state_axes,
    state_map_transform,
    unwind_new_state_levels,
    _import_rand_state,
)

__all__ = [
    'StatefulMapping',
    'vmap2',
    'vmap2_new_states',
    'pmap2',
    'pmap2_new_states',
    'map',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable

#: Tag marking a state that must stay replicated (un-batched, axis ``None``)
#: when initialized through :func:`vmap2_new_states` / :func:`pmap2_new_states`.
INIT_NO_BATCHING = 'INIT_NO_BATCHING'


def _ensure_tuple(x) -> Tuple:
    """Coerce ``int`` / iterable / ``None`` into a tuple (for static arg specs)."""
    if x is None:
        return ()
    if isinstance(x, int):
        return (x,)
    return tuple(x)


class StatefulMapping:
    """
    State-aware mapping wrapper built on the shared :mod:`brainstate.transform`
    mapping engine.

    ``StatefulMapping`` augments a JAX mapping primitive (``jax.vmap`` or
    ``jax.pmap``) with awareness of :class:`~brainstate.State` instances. It
    tracks reads and writes across the mapped axis, splits random states so each
    lane is seeded independently, scatters batched writes back to the right
    axis, and restores the global RNG so randomness is consumed exactly once per
    call. It is normally constructed by :func:`brainstate.transform.vmap2` or
    :func:`brainstate.transform.pmap2`, but can be used directly for custom
    mapping primitives.

    Parameters
    ----------
    fun : callable
        Callable to wrap. May read and write closed-over
        :class:`~brainstate.State` objects.
    in_axes : int, tuple of int, or None, default 0
        Mapped-axis alignment per positional argument, following
        :func:`jax.vmap` semantics. ``None`` marks an argument as broadcast.
    out_axes : int, tuple of int, or None, default 0
        Placement of the mapped axis in the return value.
    state_in_axes : dict, Filter, State, or iterable of State, optional
        Which states participate as batched inputs. A dict maps axis identifiers
        to selectors; a bare selector is shorthand for ``{0: selector}``.
        Selectors may be :mod:`brainstate.util.filter` filters, a single
        :class:`~brainstate.State` instance, or an iterable of instances.
    state_out_axes : dict, Filter, State, or iterable of State, optional
        Which written states are scattered back along the mapped axis, with the
        same selector semantics as ``state_in_axes``.
    unexpected_out_state_mapping : {'auto', 'raise', 'warn', 'ignore'}, default 'auto'
        Policy for states written with a batched value but not covered by
        ``state_out_axes``. ``'auto'`` scatters them at their detected axis,
        ``'raise'`` raises a :class:`~brainstate._error.BatchAxisError`,
        ``'warn'`` scatters them with a warning, and ``'ignore'`` scatters them
        silently.
    static_argnums : int or iterable of int, default ()
        Positional arguments treated as compile-time constants for caching.
    static_argnames : str or iterable of str, default ()
        Keyword arguments treated as compile-time constants for caching.
    axis_env : sequence, optional
        Retained for backward compatibility; mapping primitives manage the axis
        environment via ``axis_name``.
    return_only_write : bool, default True
        Retained for backward compatibility.
    axis_size : int, optional
        Explicit size of the mapped axis. Inferred from arguments/states when
        omitted.
    axis_name : hashable, optional
        Name of the mapped axis for collective primitives.
    name : str, optional
        Diagnostic identifier.
    mapping_fn : callable, default ``jax.vmap``
        Mapping primitive accepting ``in_axes``/``out_axes``/``axis_size``/
        ``axis_name``.
    mapping_kwargs : dict, optional
        Extra keyword arguments forwarded to ``mapping_fn``.

    Attributes
    ----------
    origin_fun : callable
        The wrapped callable.
    in_axes, out_axes : int, tuple, or None
        Argument/return mapping specification.
    state_in_axes, state_out_axes : dict
        Normalized ``{axis: predicate}`` state selectors.
    axis_size : int or None
        Mapped axis size, if explicitly provided.
    axis_name : hashable or None
        Axis identifier forwarded to collectives.
    mapping_fn : callable
        The underlying mapping primitive.

    Notes
    -----
    Random states are always split along the mapped axis and restored
    afterwards; this cannot be disabled. Plans (state groupings, batch sizes)
    are cached per abstract argument signature so repeated calls with the same
    structure avoid re-tracing.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> from brainstate.util.filter import OfType
        >>>
        >>> counter = brainstate.ShortTermState(jnp.zeros(3))
        >>>
        >>> def accumulate(x):
        ...     counter.value = counter.value + x
        ...     return counter.value
        >>>
        >>> batched = brainstate.transform.StatefulMapping(
        ...     accumulate,
        ...     in_axes=0,
        ...     out_axes=0,
        ...     state_in_axes={0: OfType(brainstate.ShortTermState)},
        ...     state_out_axes={0: OfType(brainstate.ShortTermState)},
        ...     name="batched_accumulate",
        ... )
        >>>
        >>> batched(jnp.asarray([1., 2., 3.]))
        Array([1., 2., 3.], dtype=float32)
        >>> counter.value
        Array([1., 2., 3.], dtype=float32)
    """
    __module__ = "brainstate.transform"

    def __init__(
        self,
        fun: Callable,
        in_axes: Union[int, Tuple[int, ...], None] = 0,
        out_axes: Union[int, Tuple[int, ...], None] = 0,
        state_in_axes: Optional[Union[Dict[AxisName, Filter], Filter]] = None,
        state_out_axes: Optional[Union[Dict[AxisName, Filter], Filter]] = None,
        unexpected_out_state_mapping: str = 'auto',
        # JIT specific parameters (kept for backward compatibility)
        static_argnums: Union[int, Iterable[int]] = (),
        static_argnames: Union[str, Iterable[str]] = (),
        axis_env: Optional[Sequence[tuple[Hashable, int]]] = None,
        return_only_write: bool = True,
        # mapping specific parameters
        axis_size: Optional[int] = None,
        axis_name: AxisName | None = None,
        name: Optional[str] = None,
        # mapping function
        mapping_fn: Callable = jax.vmap,
        mapping_kwargs: Dict = None
    ):
        self.origin_fun = fun
        self.name = name
        self.in_axes = in_axes
        self.out_axes = out_axes

        # store raw specs (for the engine) and normalized predicates (as attributes)
        self._state_in_axes_spec = state_in_axes
        self._state_out_axes_spec = state_out_axes
        self.state_in_axes = normalize_state_axes(state_in_axes)
        self.state_out_axes = normalize_state_axes(state_out_axes)

        self.axis_size = axis_size
        self.axis_name = axis_name
        self.mapping_fn = mapping_fn
        self.mapping_kwargs = dict() if mapping_kwargs is None else mapping_kwargs
        self.unexpected_out_state_mapping = unexpected_out_state_mapping

        self.static_argnums = _ensure_tuple(static_argnums)
        self.static_argnames = _ensure_tuple(static_argnames)
        self.axis_env = axis_env
        self.return_only_write = return_only_write

        # build the cached, engine-backed wrapper once
        self._wrapped = state_map_transform(
            fun,
            in_axes=in_axes,
            out_axes=out_axes,
            state_in_axes=state_in_axes,
            state_out_axes=state_out_axes,
            axis_size=axis_size,
            axis_name=axis_name,
            mapping_fn=mapping_fn,
            mapping_kwargs=self.mapping_kwargs,
            unexpected_out_state_mapping=unexpected_out_state_mapping,
            static_argnums=self.static_argnums,
            static_argnames=self.static_argnames,
            name=name or 'StatefulMapping',
        )

    def __call__(self, *args, **kwargs):
        """Execute the state-aware mapping on the given arguments.

        Parameters
        ----------
        *args
            Positional arguments to map over (per ``in_axes``).
        **kwargs
            Keyword arguments. Broadcast to every mapped lane (not mapped).

        Returns
        -------
        Any
            The mapped result, with state side effects applied.
        """
        return self._wrapped(*args, **kwargs)


@set_module_as('brainstate.transform')
def vmap2(
    fn: F | Missing = Missing(),
    *,
    # --- normal jax.vmap arguments --- #
    in_axes: Optional[int | Sequence[Any]] = 0,
    out_axes: Any = 0,
    axis_name: Optional[AxisName] = None,
    axis_size: Optional[int] = None,
    spmd_axis_name: Optional[AxisName | Tuple[AxisName, ...]] = None,
    # --- brainstate specific arguments --- #
    state_in_axes: Union[Dict[AxisName, Filter], Filter] = None,
    state_out_axes: Union[Dict[AxisName, Filter], Filter] = None,
    unexpected_out_state_mapping: str = 'auto',
) -> StatefulMapping | Callable[[F], StatefulMapping]:
    """
    Vectorize a callable while preserving BrainState state semantics.

    This mirrors :func:`jax.vmap` but routes execution through
    :class:`~brainstate.transform.StatefulMapping` so reads and writes to
    :class:`~brainstate.State` instances (including random states) are tracked
    across the mapped axis. The returned object can be used directly or as a
    decorator when ``fn`` is omitted.

    Parameters
    ----------
    fn : callable, optional
        Function to vectorise. If omitted, the function acts as a decorator.
    in_axes : int | None | sequence, default 0
        Mapping specification for positional arguments.
    out_axes : any, default 0
        Placement of the mapped axis in the result.
    axis_name : hashable, optional
        Name for the mapped axis so collective primitives can target it.
    axis_size : int, optional
        Explicit mapped axis size. Inferred from arguments when omitted.
    spmd_axis_name : hashable or tuple, optional
        Axis labels for nested SPMD transforms.
    state_in_axes : dict, Filter, State, or iterable of State, optional
        Selectors for states batched on input. A bare selector means ``{0: ...}``.
    state_out_axes : dict, Filter, State, or iterable of State, optional
        Selectors for written states scattered back along the mapped axis.
    unexpected_out_state_mapping : {'auto', 'raise', 'warn', 'ignore'}, default 'auto'
        Policy for states written with a batched value but not declared in
        ``state_out_axes``. The default ``'auto'`` infers the output axis from
        the detected batch dimension.

    Returns
    -------
    StatefulMapping or callable
        A :class:`StatefulMapping` if ``fn`` is supplied, otherwise a decorator.

    Raises
    ------
    ValueError
        If axis sizes are inconsistent or cannot be inferred.
    BatchAxisError
        If a state write violates ``state_out_axes`` under the ``'raise'`` policy.

    See Also
    --------
    brainstate.transform.StatefulMapping : Underlying state-aware mapping helper.
    brainstate.transform.pmap2 : Multi-device parallel variant.
    brainstate.transform.vmap : Declaration-based vectorisation (now a shim).

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> from brainstate.util.filter import OfType
        >>>
        >>> counter = brainstate.ShortTermState(jnp.zeros(3))
        >>>
        >>> @brainstate.transform.vmap2(
        ...     in_axes=0,
        ...     out_axes=0,
        ...     state_in_axes={0: OfType(brainstate.ShortTermState)},
        ...     state_out_axes={0: OfType(brainstate.ShortTermState)},
        ... )
        ... def accumulate(x):
        ...     counter.value = counter.value + x
        ...     return counter.value
        >>>
        >>> accumulate(jnp.asarray([1., 2., 3.]))
        Array([1., 2., 3.], dtype=float32)
        >>> counter.value
        Array([1., 2., 3.], dtype=float32)
    """

    if isinstance(fn, Missing):
        return functools.partial(
            vmap2,
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
        name='vmap2'
    )


@set_module_as('brainstate.transform')
def pmap2(
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
    unexpected_out_state_mapping: str = 'auto',
) -> Callable[[F], F] | F:
    """
    Parallel-map a callable across devices with state-aware semantics.

    This mirrors :func:`jax.pmap` but integrates with
    :class:`~brainstate.transform.StatefulMapping` so :class:`~brainstate.State`
    objects (including random states) are replicated, split, and restored
    correctly on every device. When ``fn`` is omitted a decorator is returned.

    Parameters
    ----------
    fn : callable, optional
        Function to execute in SPMD style. If omitted, a decorator is returned.
    axis_name : hashable, optional
        Name for the mapped axis used by collective primitives.
    in_axes : any, default 0
        Axis mapping for positional arguments.
    out_axes : any, default 0
        Placement of the mapped axis in the outputs.
    static_broadcasted_argnums : int or iterable of int, default ()
        Indices of positional arguments treated as compile-time constants.
    devices : sequence of Device, optional
        Explicit device list to map over.
    backend : str, optional
        Backend identifier (``'cpu'``, ``'gpu'``, or ``'tpu'``).
    axis_size : int, optional
        Size of the mapped axis. Defaults to the local device count.
    donate_argnums : int or iterable of int, default ()
        Positional arguments whose buffers may be donated.
    global_arg_shapes : tuple of tuple of int, optional
        Shapes for globally distributed arguments.
    state_in_axes : dict, Filter, State, or iterable of State, optional
        Selectors for states treated as device-mapped inputs.
    state_out_axes : dict, Filter, State, or iterable of State, optional
        Selectors for state writes scattered back across devices.
    unexpected_out_state_mapping : {'auto', 'raise', 'warn', 'ignore'}, default 'auto'
        Policy for state writes not covered by ``state_out_axes``.

    Returns
    -------
    StatefulMapping or callable
        A :class:`StatefulMapping` executing ``fn`` over devices, or a decorator.

    See Also
    --------
    jax.pmap : Underlying JAX primitive.
    brainstate.transform.vmap2 : Single-host vectorisation with the same semantics.
    """

    if isinstance(fn, Missing):
        return functools.partial(
            pmap2,
            axis_name=axis_name,
            in_axes=in_axes,
            out_axes=out_axes,
            static_broadcasted_argnums=static_broadcasted_argnums,
            devices=devices,
            backend=backend,
            axis_size=axis_size,
            donate_argnums=donate_argnums,
            global_arg_shapes=global_arg_shapes,
            state_in_axes=state_in_axes,
            state_out_axes=state_out_axes,
            unexpected_out_state_mapping=unexpected_out_state_mapping,
        )  # type: ignore[return-value]

    # Only forward pmap keyword arguments that the installed jax.pmap accepts.
    # ``global_arg_shapes`` was removed in recent JAX releases.
    pmap_kwargs = dict(
        static_broadcasted_argnums=static_broadcasted_argnums,
        devices=devices,
        backend=backend,
        donate_argnums=donate_argnums,
        global_arg_shapes=global_arg_shapes,
    )
    _supported = set(inspect.signature(jax.pmap).parameters)
    pmap_kwargs = {k: v for k, v in pmap_kwargs.items() if k in _supported}

    return StatefulMapping(
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        state_in_axes=state_in_axes,
        state_out_axes=state_out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        mapping_fn=functools.partial(jax.pmap, **pmap_kwargs),
        unexpected_out_state_mapping=unexpected_out_state_mapping,
        name='pmap2'
    )


def _batch_and_remainder(x, batch_size: int):
    """Split a pytree into full batches and a remainder along the leading axis."""
    leaves, tree_def = jax.tree.flatten(x)

    scan_leaves = []
    remainder_leaves = []

    length = None
    for leaf in leaves:
        if length is None:
            length = leaf.shape[0]
        if length != leaf.shape[0]:
            raise ValueError(f"All inputs must have the same length. Got {length} and {leaf.shape[0]}.")

    if length is None:
        raise ValueError("map requires at least one array input to determine the leading length.")

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


def _flatten(x):
    """Flatten the first two dimensions of an array."""
    return x.reshape(-1, *x.shape[2:])


def _validate_leading_lengths(xs) -> int:
    """Validate that every leaf in ``xs`` shares the same leading length."""
    leaves = jax.tree.leaves(xs)
    if not leaves:
        raise ValueError("map requires at least one array input.")
    length = leaves[0].shape[0]
    for leaf in leaves:
        if leaf.shape[0] != length:
            raise ValueError(
                f"All inputs to map must share the same leading length; "
                f"got {length} and {leaf.shape[0]}."
            )
    return length


@set_module_as('brainstate.transform')
def map(
    f,
    *xs,
    batch_size: int | None = None,
):
    """
    Apply a function over the leading axis of one or more pytrees.

    Compared with :func:`jax.vmap`, this helper executes sequentially by default
    (via :func:`brainstate.transform.scan`), which keeps peak memory low. When
    ``batch_size`` is given, full batches are processed with :func:`vmap2` and
    any remainder is handled separately, trading memory for throughput.

    Parameters
    ----------
    f : callable
        Function applied across the leading dimension. Its return value must be a
        pytree whose leaves can be stacked along axis ``0``.
    *xs : Any
        Positional pytrees sharing the same leading length.
    batch_size : int, optional
        Size of vectorised blocks. When given, ``map`` processes full batches
        with :func:`vmap2` and then handles any remainder.

    Returns
    -------
    Any
        A pytree matching the structure of ``f``'s outputs, stacked along the
        leading dimension.

    Raises
    ------
    ValueError
        If the inputs do not share the same leading length.

    See Also
    --------
    brainstate.transform.vmap2 : Vectorised mapping with automatic batching.
    brainstate.transform.scan : Primitive used for the sequential path.

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
        >>> map(normalize, xs, batch_size=2).shape
        (6, 1)
    """
    _validate_leading_lengths(xs)

    if batch_size is not None:
        if not (isinstance(batch_size, int) and batch_size > 0):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
        vmapped = vmap2(f)
        scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)
        g = lambda _, x: ((), vmapped(*x))
        _, scan_ys = scan(g, (), scan_xs)
        if remainder_xs is None:
            ys = jax.tree.map(lambda x: _flatten(x), scan_ys)
        else:
            remainder_ys = vmap2(f)(*remainder_xs)
            ys = jax.tree.map(
                lambda x, y: jax.lax.concatenate([_flatten(x), y], dimension=0),
                scan_ys,
                remainder_ys,
            )
    else:
        g = lambda _, x: ((), f(*x))
        _, ys = scan(g, (), xs)
    return ys


# ============================================================================ #
# New-state initialization mapping (vmap2_new_states / pmap2_new_states)
# ============================================================================ #


def _build_new_state_resolver(state_out_axes):
    """Build an ordered ``[(axis, predicate), ...]`` resolver for new states.

    Resolution priority (first match wins):

    1. axis ``None`` -- :class:`~brainstate.NonBatchState` (plus any user-supplied
       ``None`` selector). These states are replicated, not batched.
    2. user-specified axes (other than ``0``), in declaration order.
    3. axis ``0`` -- everything else (default batched axis).

    Returns
    -------
    ordered : list of (axis, predicate)
        Resolver entries in priority order.
    axes_order : list
        Unique axis identifiers in priority order, used to build the mapped
        ``out_axes``.
    """
    if state_out_axes is None:
        state_out_axes = dict()
    if not isinstance(state_out_axes, dict):
        state_out_axes = {0: state_out_axes}
    user = {k: filter.to_predicate(v) for k, v in state_out_axes.items()}

    nonbatch = filter.to_predicate(NonBatchState)
    no_batch_tag = filter.to_predicate(INIT_NO_BATCHING)
    ordered = []

    if None in user:
        user_none = user[None]
        ordered.append(
            (None, lambda p, s, a=user_none, b=nonbatch, c=no_batch_tag: a(p, s) or b(p, s) or c(p, s))
        )
    else:
        ordered.append((None, lambda p, s, b=nonbatch, c=no_batch_tag: b(p, s) or c(p, s)))

    for axis, pred in user.items():
        if axis is None or axis == 0:
            continue
        ordered.append((axis, pred))

    if 0 in user:
        ordered.append((0, user[0]))
    else:
        ordered.append((0, filter.to_predicate(...)))  # Everything (catch-all)

    axes_order = list(dict.fromkeys(axis for axis, _ in ordered))
    return ordered, axes_order


def _resolve_new_state_axis(st, ordered):
    for axis, pred in ordered:
        if pred(tuple(), st):
            return axis
    return 0


def _map_new_states(
    behavior: str,
    module: 'Module',
    init_kwargs: Dict,
    state_tag: str = None,
    axis_size: int = None,
    state_out_axes: Dict[int, Filter] = None,
    axis_name: AxisName | None = None,
    spmd_axis_name: AxisName | Tuple[AxisName, ...] | None = None,
):
    """Initialize a module's states under a mapping transform.

    Unlike the general :func:`state_map_transform` engine (whose plan is keyed to
    pre-existing state objects), state initialization creates *fresh* state
    objects on every invocation. This routine therefore uses a single mapping
    pass: an eager probe discovers the random states ``init_all_states`` touches,
    those are split per lane, and a single ``jax.vmap`` / ``jax.pmap`` pass
    creates the batched states and returns their values for scatter-back.

    Parameters
    ----------
    behavior : {'vmap', 'pmap'}
        Which mapping primitive to use.
    module : Module
        Module whose ``init_all_states`` creates the states to map.
    init_kwargs : dict
        Keyword arguments forwarded to ``module.init_all_states``.
    state_tag : str, optional
        Tag applied to the newly created states.
    axis_size : int, optional
        Mapped axis size. Defaults to ``jax.local_device_count()`` for ``pmap``.
    state_out_axes : dict or Filter, optional
        Output-axis selectors (see :func:`_build_new_state_resolver`).
    axis_name : hashable, optional
        Mapped-axis name for collectives.
    spmd_axis_name : hashable or tuple, optional
        SPMD axis label (``vmap`` only).

    Returns
    -------
    dict[Any, list[State]]
        Mapping of axis identifier to the list of created (now batched) states.
    """
    RandomState = _import_rand_state()
    base_level = TRACE_CONTEXT.get_trace_stack_level()
    ordered, axes_order = _build_new_state_resolver(state_out_axes)

    if axis_size is None:
        if behavior == 'pmap':
            axis_size = jax.local_device_count()
        else:
            raise ValueError("vmap2_new_states requires an explicit axis_size.")

    # --- eager probe: discover the random states init touches ------------- #
    rng_states = []
    probe_trace = StateTraceStack(name='new_states_probe')

    def probe_hook(state):
        if isinstance(state, RandomState):
            rng_states.append(state)
            return state.split_key()
        return state._value

    probe_trace.set_new_arg(probe_hook)
    try:
        with probe_trace:
            module.init_all_states(**init_kwargs)
    finally:
        probe_trace.recovery_original_values()
    # de-duplicate random states, preserving order
    _seen = set()
    rng_states = [r for r in rng_states if not (id(r) in _seen or _seen.add(id(r)))]

    # --- main pass: create batched states under the mapping primitive ----- #
    state_box: Dict[Any, list] = {}

    def init_fn(rng_keys):
        for rng, key in zip(rng_states, rng_keys):
            rng.restore_value(key)
        with catch_new_states() as catcher:
            module.init_all_states(**init_kwargs)
        grouped_vals = defaultdict(list)
        grouped_states = defaultdict(list)
        for st in catcher.get_states():
            axis = _resolve_new_state_axis(st, ordered)
            grouped_vals[axis].append(st.value)
            grouped_states[axis].append(st)
        state_box.clear()
        state_box.update(grouped_states)
        return tuple(grouped_vals.get(k, []) for k in axes_order)

    rng_keys = [rng.split_key(axis_size) for rng in rng_states]
    rng_backups = [rng.split_key() for rng in rng_states]

    if behavior == 'vmap':
        primitive = functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name)
    elif behavior == 'pmap':
        primitive = jax.pmap
    else:
        raise ValueError(f"Invalid behavior {behavior!r}; must be 'vmap' or 'pmap'.")

    with catch_new_states(state_tag):
        mapped = primitive(
            init_fn,
            in_axes=(0 if len(rng_states) else None,),
            out_axes=tuple(axes_order),
            axis_size=axis_size,
            axis_name=axis_name,
        )
        tuple_vals = mapped(rng_keys)

    # restore the global RNG once
    for rng, key in zip(rng_states, rng_backups):
        rng.restore_value(key)

    # scatter batched values into the freshly created state objects
    dict_vmap_states: Dict[Any, list] = defaultdict(list)
    for axis, vals in zip(axes_order, tuple_vals):
        states = state_box.get(axis, [])
        for st, val in zip(states, vals):
            st.restore_value(val)
            dict_vmap_states[axis].append(st)

    # unwind trace levels so the new states are usable in the outer scope
    all_new_states = [st for states in dict_vmap_states.values() for st in states]
    unwind_new_state_levels(all_new_states, base_level)

    return dict(dict_vmap_states)


@set_module_as('brainstate.transform')
def vmap2_new_states(
    module: 'Module',
    init_kwargs: Dict,
    state_tag: str = None,
    axis_size: int = None,
    state_out_axes: Dict[int, Filter] = None,
    spmd_axis_name: AxisName | Tuple[AxisName, ...] | None = None,
):
    """
    Initialize and vectorize newly created states within a module.

    Wraps ``module.init_all_states(**init_kwargs)`` in a :func:`vmap2`-style
    transform, executes it ``axis_size`` times in parallel, and restores the
    vectorized values back onto the freshly created state objects. Random states
    are split per lane, so random initializers produce a distinct draw for every
    batch member.

    Parameters
    ----------
    module : Module
        Module whose ``init_all_states`` creates the states to vectorize.
    init_kwargs : dict
        Keyword arguments forwarded to ``module.init_all_states``.
    state_tag : str, optional
        Tag applied to the newly created states.
    axis_size : int
        Size of the vectorization axis. Required.
    state_out_axes : dict[int, Filter] or Filter, optional
        Output-axis selectors. ``None`` (default) batches every state on axis
        ``0`` except :class:`~brainstate.NonBatchState`, which is replicated on
        axis ``None``.
    spmd_axis_name : hashable or tuple, optional
        SPMD axis label forwarded to the underlying ``jax.vmap``.

    Returns
    -------
    dict[Any, list[State]]
        Mapping of axis identifier to the lists of vectorized states.

    Raises
    ------
    ValueError
        If ``axis_size`` is not provided.

    See Also
    --------
    brainstate.transform.vmap2 : Vectorize a callable with state semantics.
    brainstate.transform.pmap2_new_states : Multi-device variant.
    brainstate.NonBatchState : Marker for states that should be replicated.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> class Counter(brainstate.nn.Module):
        ...     def init_state(self):
        ...         self.count = brainstate.ShortTermState(jnp.zeros(()))
        >>>
        >>> module = Counter()
        >>> _ = brainstate.transform.vmap2_new_states(
        ...     module, init_kwargs={}, axis_size=5
        ... )
        >>> module.count.value.shape
        (5,)
    """
    return _map_new_states(
        'vmap',
        module,
        init_kwargs,
        state_tag=state_tag,
        axis_size=axis_size,
        state_out_axes=state_out_axes,
        spmd_axis_name=spmd_axis_name,
    )


@set_module_as('brainstate.transform')
def pmap2_new_states(
    module: 'Module',
    init_kwargs: Dict,
    state_tag: str = None,
    axis_size: int = None,
    state_out_axes: Dict[int, Filter] = None,
    axis_name: AxisName | None = None,
):
    """
    Initialize and parallelize newly created states across devices.

    Wraps ``module.init_all_states(**init_kwargs)`` in a :func:`pmap2`-style
    transform, executes it across ``axis_size`` devices, and restores the
    device-distributed values back onto the freshly created state objects.
    Random states are split per device.

    Parameters
    ----------
    module : Module
        Module whose ``init_all_states`` creates the states to parallelize.
    init_kwargs : dict
        Keyword arguments forwarded to ``module.init_all_states``.
    state_tag : str, optional
        Tag applied to the newly created states.
    axis_size : int, optional
        Number of devices to map over. Defaults to ``jax.local_device_count()``.
    state_out_axes : dict[int, Filter] or Filter, optional
        Output-axis selectors. ``None`` (default) shards every state on axis
        ``0`` except :class:`~brainstate.NonBatchState`, which is replicated.
    axis_name : hashable, optional
        Mapped-axis name for collective primitives.

    Returns
    -------
    dict[Any, list[State]]
        Mapping of axis identifier to the lists of parallelized states.

    See Also
    --------
    brainstate.transform.pmap2 : Parallel mapping across devices.
    brainstate.transform.vmap2_new_states : Single-device variant.
    jax.pmap : Underlying JAX parallel mapping primitive.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> class ParallelCounter(brainstate.nn.Module):
        ...     def init_state(self):
        ...         self.count = brainstate.ShortTermState(jnp.zeros(()))
        >>>
        >>> module = ParallelCounter()
        >>> _ = brainstate.transform.pmap2_new_states(
        ...     module, init_kwargs={}, axis_size=jax.local_device_count()
        ... )
        >>> module.count.value.shape
        (jax.local_device_count(),)
    """
    return _map_new_states(
        'pmap',
        module,
        init_kwargs,
        state_tag=state_tag,
        axis_size=axis_size,
        state_out_axes=state_out_axes,
        axis_name=axis_name,
    )
