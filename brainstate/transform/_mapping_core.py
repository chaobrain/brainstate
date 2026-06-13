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

"""
Shared mapping engine for BrainState batching transforms.

This module hosts the single state-aware mapping engine used by both the
legacy :func:`brainstate.transform.vmap` / :func:`vmap_new_states` and the
modern :func:`brainstate.transform.vmap2` / :func:`pmap2` families.

The engine is built on the same ``StatefulFunction`` / ``make_jaxpr``
foundation as the rest of :mod:`brainstate.transform` (``jit``, ``grad``,
``scan``, ``cond``, ...). It threads :class:`~brainstate.State` values
*explicitly* through the underlying JAX mapping primitive (``jax.vmap`` or
``jax.pmap``) with concrete ``in_axes`` / ``out_axes``. The only place JAX
batch internals are inspected is a *read-only* probe of
``BatchTracer.batch_dim`` used to auto-detect the output batch axis of states
that the user did not explicitly declare.

Cold calls perform up to three passes:

1. **probe** -- a pass under a :class:`~brainstate.StateTraceStack` whose
   ``new_arg`` hook strips the mapped axis from matched input states and splits
   random states. This enumerates every touched state and classifies the
   batched-input states by filter/instance predicate. When ``axis_name`` is
   ``None`` the probe runs *eagerly* (one concrete lane); functions ``jax.vmap``
   can handle cannot branch on the mapped values, so that single lane follows
   the same control-flow path every lane would. When ``axis_name`` is set the
   probe is instead *traced* under ``jax.make_jaxpr(..., axis_env=[(axis_name,
   size)])`` so that collectives (``jax.lax.psum``, ``jax.lax.axis_index``, ...)
   have their axis name bound and stage correctly rather than raising
   ``NameError: unbound axis name``.
2. **discovery** -- a real ``jax.vmap`` pass that reads each written state's
   ``BatchTracer.batch_dim`` to learn the output batch axis. **Skipped** when
   the function writes no states (read-only / stateless), avoiding an extra
   trace.
3. **execution** -- the actual ``jax.vmap`` / ``jax.pmap`` call that scatters
   batched state writes back along the discovered axes.

Dynamic keyword arguments are mapped over axis 0 (matching ``jax.vmap``);
keyword arguments named in ``static_argnames`` are closed over as compile-time
constants and broadcast unchanged.

Caveats
-------
* **Repeated execution of** ``f`` **on cold calls (B9).** A cold call runs ``f``
  up to three times (eager probe + discovery + execution). This re-runs not only
  JAX-staged host callbacks (``jax.debug.print``, ``jax.experimental.io_callback``)
  but also *ordinary Python-level* side effects in ``f`` -- ``print``, list
  appends, counters, logging, RNG drawn via the host ``random`` module, etc. Such
  effects are observed multiple times on the first call with a given argument
  structure. Warm calls (same argument structure) reuse the cached plan and run
  only the execution pass. Keep ``f`` free of observable Python side effects, or
  expect them once per pass on cold calls.

* **Zero-placeholder probe / value-dependent control flow (B4).** During the
  eager probe, matched batched-input states are fed trace-free ``numpy`` zero
  placeholders (see :func:`_deaxed_like`), not their real per-lane values. This
  keeps the probe leak-free, but means ``f`` must not change *which states it
  touches* based on the numeric contents of a batched state -- e.g.
  ``if some_state.value.sum() > 0: other_state.value = ...``. The probe walks a
  single control-flow path determined by zeros, so a different path taken at real
  values would mis-enumerate the touched states. This mirrors ``jax.vmap``'s own
  rule that you cannot branch Python control flow on mapped values; branching on
  unmapped/broadcast state values is fine.

* **Global-state mutation / thread-safety (B10).** The engine temporarily mutates
  the closed-over :class:`~brainstate.State` objects in place (restoring lane
  values during the probe/discovery passes and snapshotting/restoring around
  them). It is therefore **not safe** to invoke two mapped functions that share
  the same ``State`` objects concurrently from multiple threads; the passes will
  race on those states. Concurrency across *disjoint* state sets is fine.
"""

import functools
import warnings
import weakref
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, TypeVar

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate._compatible_import import BatchTracer
from brainstate._error import BatchAxisError
from brainstate._state import State, StateTraceStack, TRACE_CONTEXT, NonBatchState
from brainstate.util import filter as filter_module
from ._make_jaxpr import StatefulFunction, get_arg_cache_key

try:
    from jax.api_util import flatten_axes
except ImportError:  # pragma: no cover - older/newer jax layout
    from jax._src.api_util import flatten_axes

__all__ = [
    # Phase 1 -- shared helpers (re-exported by _mapping1)
    '_flatten_in_out_states',
    '_remove_axis',
    '_compile_stateful_function',
    '_get_batch_size',
    '_format_state_axes',
    # Phase 2 -- axis normalization
    'normalize_state_axes',
    'make_identity_predicate',
    'coerce_axis_value_to_predicate',
    # Phase 3 -- rng + stack level
    'split_rng_keys',
    'unwind_new_state_levels',
    # Phase 3b -- new-state output-axis resolver
    'INIT_NO_BATCHING',
    '_build_new_state_resolver',
    '_resolve_new_state_axis',
    # Phase 4 -- engine
    'leaf_batch_dim',
    'StateMapPlan',
    'LiveStateMapPlan',
    'state_map_transform',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable
AxisToState = Dict[int, List[State]]
StateToAxis = Dict[State, int]

_rand = None

# Audit #3: states already warned about an undeclared 'auto' read-modify-write
# whose leading dim does not match the batch size (so it is scattered, gaining an
# axis). A WeakSet keeps the warning one-time-per-state without pinning the state
# alive or leaking ids after garbage collection.
_AUTO_RMW_SCATTER_WARNED: "weakref.WeakSet" = weakref.WeakSet()


def _import_rand_state():
    global _rand
    if _rand is None:
        from brainstate.random import RandomState
        _rand = RandomState
    return _rand


# ============================================================================ #
# Phase 1 -- shared helpers (moved verbatim from _mapping1, re-exported back)
# ============================================================================ #


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


def _remove_axis(x, axis: int):
    if not isinstance(axis, int):
        raise TypeError(f"Expected the mapped axis to be an integer, but got {type(axis)}.")
    # A non-array (scalar) leaf has no ``ndim``/``shape``; touching them yields an
    # opaque ``AttributeError``. Report which leaf cannot carry a mapped axis.
    ndim = getattr(x, 'ndim', None)
    if ndim is None:
        raise ValueError(
            f"Cannot map axis {axis} over a non-array leaf of type "
            f"{type(x).__name__}: it has no array dimensions. Mapped inputs and "
            f"states must be arrays; use a None axis to broadcast scalars."
        )
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Mapped axis {axis} is out of bounds for array of shape {x.shape}.")
    return x[tuple(slice(None, None, None) if i != axis else 0 for i in range(ndim))]


def _deaxed_like(x, axis: int):
    """Trace-free, de-batched placeholder used during state probing.

    Returns a value with the same per-lane shape/dtype/unit as
    ``_remove_axis(x, axis)``, but built from ``numpy`` so it can never be
    bound to a (possibly nested) trace. The probe only needs each state's
    structure to discover what is touched; the concrete values are recovered
    afterwards via ``StateTraceStack.recovery_original_values``. Slicing the
    live value instead (the former behaviour) produced a tracer whenever the
    state was first read inside a nested control-flow primitive, and that
    tracer escaped its trace.
    """
    if axis < 0:
        axis += jnp.ndim(x)
    shape = tuple(s for i, s in enumerate(jnp.shape(x)) if i != axis)
    mantissa = np.zeros(shape, dtype=x.dtype)
    unit = u.get_unit(x)
    return mantissa if unit.is_unitless else u.Quantity(mantissa, unit=unit)


def _compile_stateful_function(
    stateful_fn: StatefulFunction,
    in_axes: Tuple[Any, int | None | Tuple],
    args: Tuple[Any, Tuple],
):
    """Strip mapped axes and build a per-signature cache key for a stateful fn.

    ``in_axes`` and ``args`` are each 2-tuples -- ``(state_in_axes, arg_in_axes)``
    and ``(state_vals, args)`` respectively, unpacked below -- *not* a bare int /
    flat tuple as an older annotation implied.

    .. note::

        Compatibility shim (B7). The current :func:`state_map_transform` engine
        does **not** use this helper -- it derives its plan cache key via
        :func:`~brainstate.transform._make_jaxpr.get_arg_cache_key` directly.
        This function is retained because it is part of the module's public
        surface (listed in ``__all__``) and exercised by existing tests; do not
        delete it without a deprecation cycle.
    """
    in_axes_st, in_axes = in_axes
    state_vals, args = args

    # check in_axes
    if isinstance(in_axes, tuple) and len(in_axes) != len(args):
        raise ValueError(
            "vmap in_axes must be an int, None, or a tuple of entries corresponding "
            "to the positional arguments passed to the function, "
            f"but got {len(in_axes)=}, {len(args)=}"
        )

    # check state_vals
    if len(state_vals) > 0:
        state_vals = [jax.tree.map(lambda x: _remove_axis(x, axis), vals)
                      for vals, axis in zip(state_vals, in_axes_st)]
    else:
        state_vals = []

    if isinstance(in_axes, int):
        args = jax.tree.map(lambda x: _remove_axis(x, in_axes), args)
    elif isinstance(in_axes, tuple):
        args = tuple([
            arg
            if in_axis is None else
            jax.tree.map(lambda x: _remove_axis(x, in_axis), arg)
            for arg, in_axis in zip(args, in_axes)
        ])
    stateful_fn.make_jaxpr(state_vals, args)
    return stateful_fn.get_arg_cache_key(state_vals, args)


def _get_batch_size(
    args: Tuple,
    in_axes,
    in_states: AxisToState,
    axis_size: Optional[int] = None,
    kwargs: Optional[Dict] = None,
) -> int:
    batch_sizes = []

    # Batch size from positional args (supports pytree-prefix in_axes).
    if in_axes is not None and len(args):
        in_axes_tuple = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        args_flat, in_tree = jax.tree.flatten(tuple(args))
        axes_flat = flatten_axes("vmap in_axes", in_tree, in_axes_tuple)
        for leaf, ax in zip(args_flat, axes_flat):
            if ax is not None:
                batch_sizes.append(leaf.shape[ax])

    # Batch size from dynamic kwargs (always mapped over axis 0, like jax.vmap).
    if kwargs:
        for v in kwargs.values():
            for leaf in jax.tree.leaves(v):
                batch_sizes.append(leaf.shape[0])

    # Batch size from in_states.
    if in_states is not None:
        for axis, states in in_states.items():
            if axis is None:
                continue
            for state in states:
                state_leaves = jax.tree.leaves(state.value)
                if len(state_leaves):
                    batch_sizes.append(state_leaves[0].shape[axis])

    if len(batch_sizes) == 0:
        if axis_size is None:
            raise ValueError(
                "Unable to determine the mapped axis size: 'in_axes' has no "
                "non-None entry, no mapped keyword arguments were given, and no "
                "batched states were found. Specify 'axis_size'."
            )
        return axis_size

    # When axis_size is given explicitly it must agree with the size inferred from
    # arguments/states; otherwise the mapping primitive fails late with an opaque
    # XLA buffer-size error and the RNG split is sized wrong (audit #4).
    if axis_size is not None and axis_size not in set(batch_sizes):
        raise ValueError(
            f"axis_size={axis_size} conflicts with the mapped axis size(s) "
            f"{sorted(set(batch_sizes))} inferred from arguments/states. Either "
            f"omit axis_size or make it match the mapped inputs."
        )

    # Ensure all batch sizes are consistent.
    if len(set(batch_sizes)) > 1:
        raise ValueError(f"Inconsistent batch sizes found: {set(batch_sizes)}")
    return batch_sizes[0]


def _format_state_axes(
    in_states, out_states,
):
    axis_to_in_states, in_state_to_axis = _flatten_in_out_states(in_states)
    axis_to_out_states, out_state_to_axis = _flatten_in_out_states(out_states)
    for _in_state, _axis in in_state_to_axis.items():
        if _in_state in out_state_to_axis:
            _out_axis = out_state_to_axis[_in_state]
            if _out_axis != _axis:
                _in_state.raise_error_with_source_info(
                    BatchAxisError(
                        f"State {_in_state} has been mapped to axis {_axis} in 'in_states', "
                        f"However, it is mapped to axis {_out_axis} in 'out_states'."
                    )
                )
        else:
            out_state_to_axis[_in_state] = _axis
            if _axis not in axis_to_out_states:
                axis_to_out_states[_axis] = []
            axis_to_out_states[_axis].append(_in_state)

    return axis_to_in_states, in_state_to_axis, axis_to_out_states, out_state_to_axis


# ============================================================================ #
# Phase 2 -- axis specification normalization
# ============================================================================ #


def make_identity_predicate(states) -> Callable[[Tuple, State], bool]:
    """Build a predicate matching exactly the given :class:`State` instances.

    Parameters
    ----------
    states : State or iterable of State
        The state instances to match by object identity.

    Returns
    -------
    callable
        A ``predicate(path, state)`` returning ``True`` when ``state`` is one of
        the provided instances (compared with :func:`id`).
    """
    if isinstance(states, State):
        states = [states]
    ids = set(id(st) for st in states)

    def predicate(path, state):
        return id(state) in ids

    return predicate


def coerce_axis_value_to_predicate(value) -> Callable[[Tuple, State], bool]:
    """Coerce a single ``state_*_axes`` value into a predicate.

    Accepts a :class:`State` instance, an iterable of states, or any
    :mod:`brainstate.util.filter` ``Filter`` (type, tag string, ``...``,
    predicate, etc.).
    """
    if isinstance(value, State):
        return make_identity_predicate(value)
    if isinstance(value, (list, tuple, set)) and all(isinstance(v, State) for v in value) and len(value) > 0:
        return make_identity_predicate(value)
    return filter_module.to_predicate(value)


def normalize_state_axes(spec) -> Dict[Any, Callable[[Tuple, State], bool]]:
    """Normalize a ``state_in_axes`` / ``state_out_axes`` specification.

    The returned mapping is always ``{axis: predicate}``. Supported inputs:

    - ``None`` -> ``{}`` (no states selected).
    - a ``dict`` mapping axes to ``Filter`` / ``State`` / iterables of states.
    - a bare ``Filter`` / ``State`` / iterable of states -> ``{0: predicate}``.

    Parameters
    ----------
    spec : None, dict, Filter, State, or iterable of State
        The user supplied axis specification.

    Returns
    -------
    dict
        Mapping of axis identifier (``int`` or ``None``) to predicate.
    """
    if spec is None:
        return dict()
    if isinstance(spec, dict):
        return {axis: coerce_axis_value_to_predicate(value) for axis, value in spec.items()}
    return {0: coerce_axis_value_to_predicate(spec)}


# ============================================================================ #
# Phase 3 -- RNG splitting + stack-level unwinding
# ============================================================================ #


def split_rng_keys(rng_states, batch_size: int):
    """Split each random state into ``batch_size`` per-lane keys.

    Parameters
    ----------
    rng_states : sequence of RandomState
        Random states discovered during the mapped call.
    batch_size : int
        Number of mapped lanes.

    Returns
    -------
    keys : tuple
        For every random state, an array of ``batch_size`` keys (one per lane).
    backups : tuple
        A single advanced key per random state, used to restore the global RNG
        after the mapped call so randomness is consumed exactly once.
    """
    keys = tuple(rng.split_key(batch_size) for rng in rng_states)
    backups = tuple(rng.split_key() for rng in rng_states)
    return keys, backups


def unwind_new_state_levels(states, base_level: int) -> None:
    """Unwind the trace stack level of newly created states.

    Each state created inside a mapping transform records a ``stack_level``
    equal to the number of active :class:`StateTraceStack` contexts at creation
    time. To make the state usable in the *outer* scope after the transform
    returns, its level must be decreased back to ``base_level`` (the number of
    trace contexts active *before* the transform was entered). This is a
    delta-based unwind -- no magic numbers -- so it is correct regardless of how
    many trace contexts the transform nested.

    Parameters
    ----------
    states : iterable of State
        Newly created states to unwind.
    base_level : int
        The trace stack level captured before entering the transform.
    """
    for st in states:
        delta = st.stack_level - base_level
        for _ in range(max(delta, 0)):
            st.decrease_stack_level()


# ============================================================================ #
# Phase 3b -- new-state output-axis resolver (shared by the new-states paths)
# ============================================================================ #

# Tag marking states that must be replicated (not batched) when created inside a
# new-states mapping transform. Shared so both ``_mapping1`` and ``_mapping2``
# resolve it identically; re-exported by ``_mapping2`` for backward-compatible
# imports (e.g. ``brainstate.nn._delay``).
INIT_NO_BATCHING = 'INIT_NO_BATCHING'


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
    user = {k: filter_module.to_predicate(v) for k, v in state_out_axes.items()}

    nonbatch = filter_module.to_predicate(NonBatchState)
    no_batch_tag = filter_module.to_predicate(INIT_NO_BATCHING)
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
        ordered.append((0, filter_module.to_predicate(...)))  # Everything (catch-all)

    axes_order = list(dict.fromkeys(axis for axis, _ in ordered))
    return ordered, axes_order


def _resolve_new_state_axis(st, ordered):
    """Return the output axis for ``st`` per a resolver from :func:`_build_new_state_resolver`."""
    for axis, pred in ordered:
        if pred(tuple(), st):
            return axis
    return 0


# ============================================================================ #
# Phase 4 -- unified discovery + execution engine
# ============================================================================ #


def _remove_axis_tree(value, axis: int):
    """Strip ``axis`` from every leaf of a pytree value."""
    return jax.tree.map(lambda x: _remove_axis(x, axis), value)


def _strip_args(args: Tuple, in_axes) -> Tuple:
    """Remove the mapped axis from positional arguments.

    Supports ``in_axes`` as an int, ``None``, a per-argument tuple, or an
    arbitrary pytree prefix (per-leaf axes), matching :func:`jax.vmap`.
    """
    if in_axes is None:
        return args
    in_axes_tuple = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
    args_flat, in_tree = jax.tree.flatten(tuple(args))
    axes_flat = flatten_axes("vmap in_axes", in_tree, in_axes_tuple)
    stripped = [leaf if ax is None else _remove_axis(leaf, ax)
                for leaf, ax in zip(args_flat, axes_flat)]
    return tuple(jax.tree.unflatten(in_tree, stripped))


def _split_kwargs(kwargs, static_argnames):
    """Split kwargs into dynamic (mapped over axis 0) and static (broadcast)."""
    dyn = {k: v for k, v in kwargs.items() if k not in static_argnames}
    static = {k: v for k, v in kwargs.items() if k in static_argnames}
    return dyn, static


def _normalize_static_argnums(static_argnums, n_args):
    """Resolve ``static_argnums`` to a frozenset of non-negative indices.

    Negative indices are counted from the end (as in :func:`jax.jit`); an index
    outside ``range(n_args)`` raises a clear :class:`ValueError` instead of the
    opaque ``IndexError`` that would surface later when the argument is sliced.
    """
    resolved = set()
    for i in static_argnums:
        j = i + n_args if i < 0 else i
        if not (0 <= j < n_args):
            raise ValueError(
                f"static_argnums={tuple(static_argnums)} refers to positional "
                f"argument {i}, but the function was called with {n_args} "
                f"positional argument(s)."
            )
        resolved.add(j)
    return frozenset(resolved)


def _close_static_argnums(f, in_axes, static_argnums, args):
    """Bake the positional ``static_argnums`` into ``f`` (``jax.jit`` parity).

    Static positional arguments are compile-time constants: they are neither
    traced nor mapped. We drop them from the argument list handed to the mapping
    primitive and close over them in a thin wrapper that re-inserts them at their
    original positions -- mirroring how ``static_argnames`` keyword arguments are
    already handled. This avoids asking the primitive to map a non-array constant
    (which fails with ``'<type>' object has no attribute 'ndim'``).

    Parameters
    ----------
    f : callable
        The user function.
    in_axes : int | None | tuple
        Positional-argument batch-axis specification.
    static_argnums : frozenset of int
        Already-normalized (non-negative, in-range) static positional indices.
    args : tuple
        Full positional arguments for this call.

    Returns
    -------
    tuple
        ``(f_closed, dyn_args, dyn_in_axes)``. ``dyn_args`` excludes the static
        positions; ``dyn_in_axes`` drops the matching entries when ``in_axes`` is
        a per-argument tuple (an ``int`` / ``None`` axis is returned unchanged,
        since it applies uniformly to whatever positional arguments remain).
    """
    if not static_argnums:
        return f, args, in_axes
    n = len(args)
    static_vals = {i: args[i] for i in static_argnums}
    dyn_args = tuple(a for i, a in enumerate(args) if i not in static_argnums)

    @functools.wraps(f)
    def f_closed(*dyn, **kwargs):
        it = iter(dyn)
        full = [static_vals[i] if i in static_vals else next(it) for i in range(n)]
        return f(*full, **kwargs)

    if isinstance(in_axes, tuple) and len(in_axes) == n:
        dyn_in_axes = tuple(ax for i, ax in enumerate(in_axes) if i not in static_argnums)
    else:
        dyn_in_axes = in_axes
    return f_closed, dyn_args, dyn_in_axes


def _strip_kwargs(dyn_kwargs):
    """Remove the leading mapped axis from each dynamic kwarg.

    :func:`jax.vmap` maps keyword arguments over axis 0; this mirrors that for
    the eager/traced probe pass.
    """
    return {k: _remove_axis_tree(v, 0) for k, v in dyn_kwargs.items()}


def leaf_batch_dim(value) -> Optional[int]:
    """Return the common ``batch_dim`` of a (possibly batched) pytree value.

    Reads ``BatchTracer.batch_dim`` for batched leaves. Returns ``None`` when no
    leaf is batched. Raises :class:`BatchAxisError` when leaves disagree.
    """
    leaves = jax.tree.leaves(value)
    dims = set(leaf.batch_dim if isinstance(leaf, BatchTracer) else None for leaf in leaves)
    if len(dims) == 0:
        return None
    if len(dims) != 1:
        raise BatchAxisError(
            f"State has inconsistent batch dimensions across its leaves: {dims}. "
            "All leaves must share the same batch dimension."
        )
    return dims.pop()


def _leaf_axis_size(value, axis: int) -> Optional[int]:
    """Size of ``value``'s first leaf along ``axis`` (``None`` if out of range).

    Used by the ``'auto'`` policy to decide whether an undeclared batched write is
    really a per-lane read-modify-write input: if the state's prior value is
    already sized like the batch along the detected axis, it is treated as a
    batched input rather than scattered (see :func:`_build_plan`).
    """
    leaves = jax.tree.leaves(value, is_leaf=u.math.is_quantity)
    if not leaves:
        return None
    shape = getattr(leaves[0], 'shape', None)
    if shape is None:
        return None
    if axis < 0:
        axis += len(shape)
    if 0 <= axis < len(shape):
        return shape[axis]
    return None


class LiveStateMapPlan:
    """A plan with its states resolved to strong references.

    Valid for the duration of a single mapped call. Holding an instance keeps
    the plan's states alive while the mapping primitive runs. Produced by
    :meth:`StateMapPlan.materialize` (warm path) or :func:`_build_plan`
    (cold path), and consumed by :func:`_execute_plan`.

    Attributes
    ----------
    in_groups : list of (axis, list[State])
        Batched input states grouped by their input axis (random states
        excluded).
    rng_states : list of RandomState
        Random states split along the mapped axis.
    out_groups : list of (axis, list[State])
        States whose writes are scattered back along the mapped axis.
    oth_out_states : list of State
        Written states that are broadcast (not batched) on output and therefore
        restored from a single representative lane.
    auto_in_candidate_ids : frozenset of int
        Object ids of undeclared ``'auto'`` read-modify-write states whose
        per-lane promotion is re-evaluated against the live value on every call
        (audit #1). These states already live in :attr:`out_groups`; the ids are
        matched there in :func:`_execute_plan`.
    """

    __slots__ = ('in_groups', 'rng_states', 'out_groups', 'oth_out_states', 'auto_in_candidate_ids')

    def __init__(self, in_groups, rng_states, out_groups, oth_out_states, auto_in_candidate_ids=frozenset()):
        self.in_groups = in_groups
        self.rng_states = rng_states
        self.out_groups = out_groups
        self.oth_out_states = oth_out_states
        self.auto_in_candidate_ids = frozenset(auto_in_candidate_ids)


class StateMapPlan:
    """Cached plan describing how states are batched for a mapped call.

    The plan holds only :class:`weakref.ref` references to its states, never
    strong ones. This is deliberate (audit finding **B3**): the warm-path cache
    is keyed purely on the *argument* signature
    (:func:`~brainstate.transform._make_jaxpr.get_arg_cache_key`), which carries
    no State identity. If the caller recreates its states -- e.g. a second
    ``init_all_states()`` builds a fresh module -- a warm cache hit would
    otherwise route writes onto the orphaned originals. By holding weakrefs, the
    engine can notice when the original states have been garbage-collected
    (:meth:`materialize` returns ``None``) and rebuild the plan against the new
    states.

    A plan therefore never keeps its states alive on its own; callers
    materialize a live, strong-ref :class:`LiveStateMapPlan` for the duration of
    a single mapped call.

    .. note::

        Detection relies on the originals actually being collected. If the
        caller keeps a strong reference to a stale state elsewhere, its weakref
        stays alive and the plan is *not* invalidated; writes still land on the
        stale object. Recreating states under a cached mapped function is best
        avoided -- rebuild the mapped wrapper alongside the states.

    Attributes
    ----------
    in_groups : list of (axis, list[weakref.ref[State]])
        Batched input states grouped by their input axis (random states
        excluded).
    rng_states : list of weakref.ref[RandomState]
        Random states split along the mapped axis.
    out_groups : list of (axis, list[weakref.ref[State]])
        States whose writes are scattered back along the mapped axis.
    oth_out_states : list of weakref.ref[State]
        Written states that are broadcast (not batched) on output and therefore
        restored from a single representative lane.
    """

    __slots__ = ('in_groups', 'rng_states', 'out_groups', 'oth_out_states', 'auto_in_candidate_ids')

    def __init__(self, in_groups, rng_states, out_groups, oth_out_states, auto_in_candidate_ids=frozenset()):
        self.in_groups = [(axis, [weakref.ref(st) for st in states]) for axis, states in in_groups]
        self.rng_states = [weakref.ref(st) for st in rng_states]
        self.out_groups = [(axis, [weakref.ref(st) for st in states]) for axis, states in out_groups]
        self.oth_out_states = [weakref.ref(st) for st in oth_out_states]
        # Plain ids (no weakref): these states are a subset of out_groups, whose
        # weakrefs already gate staleness; on materialize the same live objects
        # are dereferenced, so their ids still match.
        self.auto_in_candidate_ids = frozenset(auto_in_candidate_ids)

    @classmethod
    def from_live(cls, live: 'LiveStateMapPlan') -> 'StateMapPlan':
        """Snapshot a :class:`LiveStateMapPlan` as a weakref-backed plan for caching."""
        return cls(live.in_groups, live.rng_states, live.out_groups, live.oth_out_states,
                   live.auto_in_candidate_ids)

    def materialize(self) -> Optional['LiveStateMapPlan']:
        """Resolve all weakrefs into a strong-ref :class:`LiveStateMapPlan`.

        Returns
        -------
        LiveStateMapPlan or None
            The live plan if every referenced state is still alive, otherwise
            ``None`` -- signalling the plan is stale and must be rebuilt.
        """

        def deref(refs):
            out = []
            for r in refs:
                st = r()
                if st is None:
                    return None
                out.append(st)
            return out

        in_groups = []
        for axis, refs in self.in_groups:
            states = deref(refs)
            if states is None:
                return None
            in_groups.append((axis, states))
        rng_states = deref(self.rng_states)
        if rng_states is None:
            return None
        out_groups = []
        for axis, refs in self.out_groups:
            states = deref(refs)
            if states is None:
                return None
            out_groups.append((axis, states))
        oth_out_states = deref(self.oth_out_states)
        if oth_out_states is None:
            return None
        return LiveStateMapPlan(in_groups, rng_states, out_groups, oth_out_states,
                                self.auto_in_candidate_ids)


def _probe_axis_size(args, in_axes, axis_size, kwargs=None):
    """A positive axis size for binding ``axis_env`` during the traced probe.

    Only used to bind the named axis so collectives can be staged; its exact
    value does not affect which states are discovered or how inputs are
    classified.
    """
    if axis_size is not None:
        return int(axis_size)
    try:
        return int(_get_batch_size(args, in_axes, {}, axis_size, kwargs))
    except Exception:
        return 2


class _ReadTrackingTrace(StateTraceStack):
    """Probe trace that records *genuine* reads (value-getter accesses).

    :attr:`StateTraceStack.been_writen` cannot tell a read-modify-write apart
    from a pure write: :meth:`StateTraceStack.write_its_value` calls
    :meth:`StateTraceStack.read_its_value` internally the first time it sees a
    state, so every written state looks "read". This subclass flags the window in
    which that internal read happens, so only reads triggered by the value getter
    (``state.value``) are counted as genuine. :func:`_build_plan` uses the result
    to classify an undeclared ``'auto'`` batched write as a per-lane
    read-modify-write (promotable) versus a pure output (scatter only).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.genuine_read_ids: set = set()
        self._in_write_value = False

    def read_its_value(self, state) -> None:
        if not self._in_write_value:
            self.genuine_read_ids.add(id(state))
        super().read_its_value(state)

    def write_its_value(self, state) -> None:
        self._in_write_value = True
        try:
            super().write_its_value(state)
        finally:
            self._in_write_value = False


def _probe_states(f, args, kwargs, in_predicates, in_axes, name,
                  axis_name=None, axis_size=None, static_argnames=()):
    """Enumerate touched states and classify batched inputs.

    When ``axis_name`` is set, the probe runs under :func:`jax.make_jaxpr` with
    the named axis bound via ``axis_env`` so collectives (``psum``,
    ``axis_index``, ...) can be traced during state discovery (mirrors the
    ``shard_map`` approach). Otherwise it runs eagerly.

    Returns
    -------
    state_trace : StateTraceStack
        Trace recording every touched state (after value recovery).
    dim_to_in_states : dict[int, list[State]]
        Batched input states grouped by axis (``None`` axis treated as
        broadcast and therefore omitted).
    rng_states : list[RandomState]
        Random states encountered during the call.
    seen_in_ids : set[int]
        Object ids of states classified as batched inputs.
    genuine_read_ids : set[int]
        Object ids of states whose value getter was actually invoked (genuine
        reads), used to tell read-modify-write states from pure writes.

    Notes
    -----
    Matched batched-input states are fed trace-free ``numpy`` zero placeholders
    (:func:`_deaxed_like`) rather than their real per-lane values, so the probe
    only observes each state's *structure*, never its contents (B4). A function
    whose set of touched states depends on the numeric value of a batched state
    (value-dependent Python control flow) is therefore unsupported -- the probe
    follows the single path implied by zeros. See the module-level *Caveats*.
    """
    RandomState = _import_rand_state()
    dyn_kwargs, static_kwargs = _split_kwargs(kwargs, static_argnames)
    stripped = _strip_args(args, in_axes)
    stripped_kwargs = _strip_kwargs(dyn_kwargs)
    dim_to_in_states: Dict[int, List[State]] = defaultdict(list)
    rng_states: List[Any] = []
    seen_in_ids = set()

    def hook(state):
        if isinstance(state, RandomState):
            rng_states.append(state)
            return state.split_key()
        for axis, pred in in_predicates.items():
            if pred(tuple(), state):
                if axis is None:
                    # explicitly broadcast input -- leave value untouched
                    return state._value
                dim_to_in_states[axis].append(state)
                seen_in_ids.add(id(state))
                return jax.tree.map(lambda x: _deaxed_like(x, axis), state._value, is_leaf=u.math.is_quantity)
        return state._value

    state_trace = _ReadTrackingTrace(name=name)
    state_trace.set_new_arg(hook)
    try:
        if axis_name is None:
            with state_trace:
                f(*stripped, **stripped_kwargs, **static_kwargs)
        else:
            probe_size = _probe_axis_size(args, in_axes, axis_size, dyn_kwargs)

            def _probe_body(*probe_args):
                with state_trace:
                    f(*probe_args, **stripped_kwargs, **static_kwargs)
                return jnp.zeros(())

            jax.make_jaxpr(_probe_body, axis_env=[(axis_name, probe_size)])(*stripped)
    finally:
        state_trace.recovery_original_values()
    return state_trace, dict(dim_to_in_states), rng_states, seen_in_ids, state_trace.genuine_read_ids


def _detect_out_dims(f, args, kwargs, in_groups, rng_states, write_states,
                     batch_size, in_axes, axis_size, axis_name, static_argnames=()):
    """Detect the output batch dimension of every written state.

    Runs a real ``jax.vmap`` discovery pass (always ``vmap`` -- dimensions are
    identical under ``pmap``) and reads ``BatchTracer.batch_dim`` for each
    written state. All touched state values are snapshotted beforehand and
    restored afterwards so the discovery pass has no observable side effects.
    Dynamic keyword arguments are threaded as a mapped input (axis 0) to match
    :func:`jax.vmap`.
    """
    dyn_kwargs, static_kwargs = _split_kwargs(kwargs, static_argnames)
    write_set_ids = {id(st) for st in write_states}
    in_group_axes = [axis for axis, _ in in_groups]
    if len(in_group_axes) == 0:
        in_group_axes = 0
    in_group_vals = [[st.value for st in states] for _, states in in_groups]

    # snapshot every state we will touch so the pass is side-effect free
    touched = list(write_states)
    for _, states in in_groups:
        touched.extend(states)
    touched.extend(rng_states)
    # de-duplicate while keeping objects
    snap_states = list({id(st): st for st in touched}.values())
    snapshots = [(st, st.value) for st in snap_states]

    rng_vals = [st.split_key(batch_size) for st in rng_states]

    detected: Dict[int, Optional[int]] = {}

    def disc_fn(args_, kwargs_, rng_keys, group_vals):
        for st, key in zip(rng_states, rng_keys):
            st.restore_value(key)
        for (axis, states), vals in zip(in_groups, group_vals):
            for st, v in zip(states, vals):
                st.restore_value(v)
        # plain watcher (no new_arg): the user function runs under a raw
        # jax.vmap here, and state writes of vmap tracers are legitimate —
        # an active StateTraceStack keeps the tracer-write guard quiet
        with StateTraceStack(name='state_map:detect'):
            f(*args_, **kwargs_, **static_kwargs)
        for st in write_states:
            detected[id(st)] = leaf_batch_dim(st.value)
        return jnp.zeros(())

    try:
        jax.vmap(
            disc_fn,
            in_axes=(in_axes, 0, 0 if len(rng_states) else None, in_group_axes),
            out_axes=None,
            axis_size=axis_size,
            axis_name=axis_name,
        )(args, dyn_kwargs, rng_vals, in_group_vals)
    finally:
        for st, val in snapshots:
            st.restore_value(val)

    # states that are written but were never assigned a dim (e.g. short-circuit)
    for st in write_states:
        detected.setdefault(id(st), None)
    return detected, write_set_ids


def _build_plan(
    f, args, kwargs,
    in_predicates, out_predicates,
    in_axes, axis_size, axis_name,
    unexpected_out_state_mapping, name, static_argnames=(),
    out_decl_name='state_out_axes',
    out_decl_extra=" or set unexpected_out_state_mapping to 'auto', 'warn', or 'ignore'",
):
    """Probe + discover + assemble a :class:`StateMapPlan`.

    ``out_decl_name`` / ``out_decl_extra`` let a caller phrase the undeclared-write
    error in its own vocabulary -- the legacy ``vmap`` shim declares states via
    ``out_states`` and has no policy knob, so it passes ``out_decl_name='out_states'``
    and an empty ``out_decl_extra`` (audit #5).
    """
    RandomState = _import_rand_state()

    state_trace, dim_to_in_states, rng_states, seen_in_ids, genuine_read_ids = _probe_states(
        f, args, kwargs, in_predicates, in_axes, name,
        axis_name=axis_name, axis_size=axis_size, static_argnames=static_argnames,
    )
    rng_set_ids = {id(st) for st in rng_states}
    write_states = [st for st in state_trace.get_write_states() if id(st) not in rng_set_ids]

    in_groups = sorted(dim_to_in_states.items(), key=lambda kv: kv[0])
    in_state_to_axis = {id(st): axis for axis, states in in_groups for st in states}

    dyn_kwargs = _split_kwargs(kwargs, static_argnames)[0]
    batch_size = _get_batch_size(args, in_axes, dict(in_groups), axis_size, dyn_kwargs)

    if len(write_states):
        detected, _ = _detect_out_dims(
            f, args, kwargs, in_groups, rng_states, write_states,
            batch_size, in_axes, axis_size, axis_name, static_argnames=static_argnames,
        )
    else:
        # No state writes -> nothing to scatter; the assembly loop below treats
        # missing detections as None, so we can skip the discovery vmap entirely.
        detected = {}

    # assemble output groups in deterministic trace order
    out_axis_groups: Dict[int, List[State]] = defaultdict(list)
    oth_out_states: List[State] = []
    # #1: ids of undeclared read-modify-write states whose per-lane promotion is
    # decided at execution time against the live value (see the 'auto' branch and
    # _execute_plan). Deferring the decision makes warm calls match cold calls.
    auto_in_candidate_ids: set = set()

    def _match_out_axis(st):
        for axis, pred in out_predicates.items():
            if pred(tuple(), st):
                return True, axis
        return False, None

    for st in state_trace.states:
        if id(st) in rng_set_ids:
            continue
        in_axis = in_state_to_axis.get(id(st))
        is_write = id(st) in {id(w) for w in write_states}
        det = detected.get(id(st))

        if in_axis is not None:
            # batched input -> also a batched output at the same axis
            matched, out_axis = _match_out_axis(st)
            if matched and out_axis is not None and out_axis != in_axis:
                st.raise_error_with_source_info(
                    BatchAxisError(
                        f"State {st} is batched on input axis {in_axis} but "
                        f"state_out_axes maps it to axis {out_axis}."
                    )
                )
            out_axis_groups[in_axis].append(st)
            continue

        matched, out_axis = _match_out_axis(st)
        if matched:
            if out_axis is None:
                if is_write:
                    oth_out_states.append(st)
            else:
                if det is not None:
                    out_axis_groups[out_axis].append(st)
                elif is_write:
                    oth_out_states.append(st)
            continue

        if is_write and det is not None:
            # an undeclared batched write -- apply policy
            if unexpected_out_state_mapping == 'raise':
                st.raise_error_with_source_info(
                    BatchAxisError(
                        f"State\n {st} \nwas written with a batched value on axis {det} but is "
                        f"not covered by {out_decl_name}. Declare it in {out_decl_name}{out_decl_extra}."
                    )
                )
            elif unexpected_out_state_mapping == 'warn':
                warnings.warn(
                    f"State\n {st} \nwas written with a batched value on axis {det} but is "
                    "not covered by state_out_axes; scattering it automatically.",
                    UserWarning,
                )
                out_axis_groups[det].append(st)
            elif unexpected_out_state_mapping == 'auto':
                # #1: scatter the write at its detected axis. If the state is also
                # genuinely read (read-modify-write), record it as a per-lane
                # promotion candidate: _execute_plan re-checks, on every call,
                # whether the live value is already batch-sized along this axis and
                # if so feeds it per lane instead of broadcasting. Deferring this
                # decision (instead of baking it into the cached plan) keeps warm
                # calls in lock-step with cold calls, so the value no longer gains a
                # new leading axis on every warm call.
                out_axis_groups[det].append(st)
                if id(st) in genuine_read_ids:
                    auto_in_candidate_ids.add(id(st))
                    # #3: when the live leading dim does not match the batch size,
                    # the state is scattered (gaining an axis), which is rarely the
                    # intent for a read-modify-write buffer. Surface the engine's
                    # otherwise-silent choice once per state.
                    cur = _leaf_axis_size(st.value, det)
                    if cur != batch_size and st not in _AUTO_RMW_SCATTER_WARNED:
                        _AUTO_RMW_SCATTER_WARNED.add(st)
                        warnings.warn(
                            f"State {st} is written with a batched value on axis {det} "
                            f"under the 'auto' policy but is not declared in "
                            f"state_in_axes/state_out_axes, and its current leading size "
                            f"({cur}) does not match the mapped size ({batch_size}); it is "
                            f"being scattered, which adds a new leading axis. If it is a "
                            f"per-lane read-modify-write buffer, pre-shape it to the mapped "
                            f"size or declare it via state_in_axes/state_out_axes to make "
                            f"the intent explicit.",
                            UserWarning,
                        )
            elif unexpected_out_state_mapping == 'ignore':
                out_axis_groups[det].append(st)
            else:
                raise ValueError(
                    "Invalid value for unexpected_out_state_mapping: "
                    f"{unexpected_out_state_mapping!r}. Must be 'auto', 'raise', 'warn', or 'ignore'."
                )
        elif is_write:
            # broadcast write -- restore from a single lane
            oth_out_states.append(st)

    out_groups = sorted(out_axis_groups.items(), key=lambda kv: kv[0])
    # Return a live (strong-ref) plan for immediate execution; the caller
    # snapshots it as a weakref-backed StateMapPlan for the warm cache (B3). The
    # auto read-modify-write candidates are promoted per-call in _execute_plan.
    return LiveStateMapPlan(in_groups, rng_states, out_groups, oth_out_states,
                            frozenset(auto_in_candidate_ids))


class _PlanStaleError(Exception):
    """Internal: the function's write set diverged from the cached plan.

    Raised inside the execution pass when ``f`` writes a state the plan does
    not cover (e.g. Python control flow on an unmapped state changed branch
    between calls). Caught by ``state_map_transform`` to rebuild the plan and
    retry exactly once.
    """


def _execute_plan(plan: LiveStateMapPlan, f, args, kwargs, in_axes, out_axes,
                  axis_size, axis_name, mapping_fn, mapping_kwargs, static_argnames=()):
    """Run the cached plan through the mapping primitive and restore states.

    ``plan`` must be a :class:`LiveStateMapPlan` (strong references); it keeps
    the plan's states alive for the duration of the call.
    """
    dyn_kwargs, static_kwargs = _split_kwargs(kwargs, static_argnames)
    in_groups = plan.in_groups
    rng_states = plan.rng_states
    out_groups = plan.out_groups
    oth_out_states = plan.oth_out_states

    # Batch size from declared inputs + args (before any per-call promotion).
    batch_size = _get_batch_size(args, in_axes, dict(in_groups), axis_size, dyn_kwargs)

    # #1: re-evaluate undeclared 'auto' read-modify-write promotion against the
    # LIVE state value on every call so warm calls match cold calls. A candidate
    # (already present in out_groups) whose current leading size equals the batch
    # is fed per lane this call instead of broadcast; otherwise it is left to
    # scatter. Promotion only folds in states already sized to the batch, so
    # ``batch_size`` is unaffected and need not be recomputed.
    if plan.auto_in_candidate_ids:
        already_in = {id(st) for _, states in in_groups for st in states}
        promote: Dict[int, List[State]] = defaultdict(list)
        for axis, states in out_groups:
            for st in states:
                if (id(st) in plan.auto_in_candidate_ids
                        and id(st) not in already_in
                        and _leaf_axis_size(st.value, axis) == batch_size):
                    promote[axis].append(st)
        if promote:
            merged: Dict[int, List[State]] = defaultdict(list)
            for axis, states in in_groups:
                merged[axis].extend(states)
            for axis, states in promote.items():
                merged[axis].extend(states)
            in_groups = sorted(merged.items(), key=lambda kv: kv[0])

    in_group_axes = [axis for axis, _ in in_groups]
    if len(in_group_axes) == 0:
        in_group_axes = 0
    out_group_axes = [axis for axis, _ in out_groups]
    if len(out_group_axes) == 0:
        out_group_axes = 0

    rng_keys, rng_backups = split_rng_keys(rng_states, batch_size)

    in_group_vals = [[st.value for st in states] for _, states in in_groups]

    # every state the plan accounts for; a write outside this set means the
    # plan no longer matches the function (M2)
    expected_write_ids = (
        {id(st) for st in rng_states}
        | {id(st) for _, states in in_groups for st in states}
        | {id(st) for _, states in out_groups for st in states}
        | {id(st) for st in oth_out_states}
    )

    def fn_to_map(args_, kwargs_, rng_keys_, group_vals):
        for st, key in zip(rng_states, rng_keys_):
            st.restore_value(key)
        for (axis, states), vals in zip(in_groups, group_vals):
            for st, v in zip(states, vals):
                st.restore_value(v)
        # Observe writes without altering values (no ``new_arg`` hook): states
        # created inside ``f`` register above this stack and stay invisible.
        watcher = StateTraceStack(name='state_map:watch')
        with watcher:
            out = f(*args_, **kwargs_, **static_kwargs)
        stale = [
            (st, org) for st, written, org in
            zip(watcher.states, watcher.been_writen, watcher.original_state_values)
            if written and id(st) not in expected_write_ids
        ]
        if stale:
            # roll the unplanned states back to their first-seen values so the
            # aborted trace cannot leak tracers into them
            for st, org in stale:
                st.restore_value(org)
            raise _PlanStaleError(
                f"The cached mapping plan no longer matches the write set of "
                f"'{getattr(f, '__name__', f)}': {len(stale)} written state(s) "
                f"are not covered by the plan."
            )
        out_group_vals = [[st.value for st in states] for _, states in out_groups]
        oth_vals = [st.value for st in oth_out_states]
        return out, out_group_vals, oth_vals

    mapped_fn = mapping_fn(
        fn_to_map,
        in_axes=(in_axes, 0, 0 if len(rng_states) else None, in_group_axes),
        out_axes=(out_axes, out_group_axes, None),
        axis_size=axis_size,
        axis_name=axis_name,
        **mapping_kwargs,
    )

    # snapshot every plan state so a failed execution pass cannot leave
    # tracers behind (M1; the probe/discovery passes already restore)
    touched = list(rng_states) + list(oth_out_states)
    for _, states in in_groups:
        touched.extend(states)
    for _, states in out_groups:
        touched.extend(states)
    snapshots = [(st, st.value) for st in {id(st): st for st in touched}.values()]

    try:
        out, out_group_vals, oth_vals = mapped_fn(args, dyn_kwargs, rng_keys, in_group_vals)
    except Exception:
        for st, val in snapshots:
            st.restore_value(val)
        for rng, key in zip(rng_states, rng_backups):
            rng.restore_value(key)
        raise

    # scatter batched output state values
    for (axis, states), vals in zip(out_groups, out_group_vals):
        for st, v in zip(states, vals):
            st.restore_value(v)
    # restore broadcast output state values
    for st, v in zip(oth_out_states, oth_vals):
        st.restore_value(v)
    # restore the global RNG once (randomness consumed exactly once per call)
    for rng, key in zip(rng_states, rng_backups):
        rng.restore_value(key)
    return out


def state_map_transform(
    f: Callable,
    *,
    in_axes: int | None | Tuple = 0,
    out_axes: Any = 0,
    state_in_axes=None,
    state_out_axes=None,
    axis_size: Optional[int] = None,
    axis_name: AxisName | None = None,
    mapping_fn: Callable = jax.vmap,
    mapping_kwargs: Optional[Dict] = None,
    unexpected_out_state_mapping: str = 'auto',
    static_argnums=(),
    static_argnames=(),
    name: Optional[str] = None,
    out_decl_name: str = 'state_out_axes',
    out_decl_extra: str = " or set unexpected_out_state_mapping to 'auto', 'warn', or 'ignore'",
):
    """Build a state-aware mapped version of ``f``.

    This is the shared engine behind :func:`vmap`, :func:`vmap2`, :func:`pmap2`
    and friends. See the module docstring for the cold/warm pass model.

    Parameters
    ----------
    f : callable
        Function to map. May read and write closed-over :class:`State` objects.
    in_axes, out_axes : int | None | tuple
        Argument/return batch-axis specification, as in :func:`jax.vmap`.
    state_in_axes, state_out_axes : None | dict | Filter | State | iterable
        State batching specification. Normalized via
        :func:`normalize_state_axes` into ``{axis: predicate}``.
    axis_size : int, optional
        Explicit mapped axis size.
    axis_name : hashable, optional
        Name of the mapped axis for collectives.
    mapping_fn : callable, default ``jax.vmap``
        Mapping primitive accepting ``in_axes``/``out_axes``/``axis_size``/
        ``axis_name``.
    mapping_kwargs : dict, optional
        Extra keyword arguments forwarded to ``mapping_fn``.
    unexpected_out_state_mapping : {'auto', 'raise', 'warn', 'ignore'}
        Policy for states written with a batched value but not declared in
        ``state_out_axes``.
    static_argnums, static_argnames : int/str or iterable
        Positional/keyword arguments treated as compile-time constants
        (``jax.jit`` parity). They key the per-signature plan cache and are
        closed over -- neither traced nor mapped -- so a ``static_argnums``
        position is excluded from ``in_axes`` mapping entirely (its ``in_axes``
        entry, if any, is ignored). Negative ``static_argnums`` count from the
        end; an out-of-range index raises :class:`ValueError`.
    name : str, optional
        Diagnostic name.

    Returns
    -------
    callable
        A wrapped function with the same calling convention as ``f``.
    """
    in_predicates = normalize_state_axes(state_in_axes)
    out_predicates = normalize_state_axes(state_out_axes)
    mapping_kwargs = dict() if mapping_kwargs is None else mapping_kwargs
    name = name or getattr(f, '__name__', 'state_map')

    if isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    cache: Dict[Any, StateMapPlan] = dict()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if isinstance(in_axes, tuple) and len(in_axes) != len(args):
            raise ValueError(
                "vmap in_axes must be an int, None, or a tuple of entries corresponding "
                "to the positional arguments passed to the function, but got "
                f"{len(in_axes)} in_axes entries for {len(args)} positional arguments."
            )
        # #8: positional ``static_argnums`` are compile-time constants (jit
        # parity). Normalize/validate the indices, key the cache on their values
        # (via get_arg_cache_key), then close over them so they are neither
        # traced nor mapped -- the engine below sees a function of only the
        # remaining dynamic positional arguments.
        static_nums = _normalize_static_argnums(static_argnums, len(args))
        cache_key = get_arg_cache_key(static_nums, static_argnames, args, kwargs)
        cf, cargs, caxes = _close_static_argnums(f, in_axes, static_nums, args)
        plan = cache.get(cache_key, None)
        # B3: a cached plan holds only weakrefs; materialize() returns None if
        # any of its states was garbage-collected (e.g. the caller rebuilt its
        # module / re-ran init_all_states), in which case we re-plan against the
        # live states instead of writing onto the orphaned originals.
        live = plan.materialize() if plan is not None else None
        if live is None:
            live = _build_plan(
                cf, cargs, kwargs,
                in_predicates, out_predicates,
                caxes, axis_size, axis_name,
                unexpected_out_state_mapping, name,
                static_argnames=static_argnames,
                out_decl_name=out_decl_name, out_decl_extra=out_decl_extra,
            )
            cache[cache_key] = StateMapPlan.from_live(live)
        try:
            return _execute_plan(
                live, cf, cargs, kwargs, caxes, out_axes,
                axis_size, axis_name, mapping_fn, mapping_kwargs,
                static_argnames=static_argnames,
            )
        except _PlanStaleError:
            # the write set diverged from the cached plan (e.g. Python control
            # flow on an unmapped state changed branch between calls): drop the
            # plan, re-probe, and retry exactly once. A second divergence
            # (write set changing within a single call) is surfaced as-is.
            live = _build_plan(
                cf, cargs, kwargs,
                in_predicates, out_predicates,
                caxes, axis_size, axis_name,
                unexpected_out_state_mapping, name,
                static_argnames=static_argnames,
                out_decl_name=out_decl_name, out_decl_extra=out_decl_extra,
            )
            cache[cache_key] = StateMapPlan.from_live(live)
            return _execute_plan(
                live, cf, cargs, kwargs, caxes, out_axes,
                axis_size, axis_name, mapping_fn, mapping_kwargs,
                static_argnames=static_argnames,
            )

    wrapped.__brainstate_state_map__ = True
    return wrapped
