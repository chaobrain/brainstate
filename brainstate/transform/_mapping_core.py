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

1. **probe** -- an eager pass under a :class:`~brainstate.StateTraceStack`
   whose ``new_arg`` hook strips the mapped axis from matched input states and
   splits random states. This enumerates every touched state and classifies
   the batched-input states by filter/instance predicate. It is safe for any
   function ``jax.vmap`` can handle, because such functions cannot branch on
   the mapped values (so the single concrete lane the probe executes follows
   the same control-flow path every lane would).
2. **discovery** -- a real ``jax.vmap`` pass that reads each written state's
   ``BatchTracer.batch_dim`` to learn the output batch axis.
3. **execution** -- the actual ``jax.vmap`` / ``jax.pmap`` call that scatters
   batched state writes back along the discovered axes.

Warm calls (same argument structure) reuse the cached plan and perform only
the execution pass.
"""

import functools
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp

from brainstate._compatible_import import BatchTracer
from brainstate._error import BatchAxisError
from brainstate._state import State, StateTraceStack, TRACE_CONTEXT
from brainstate.util import filter as filter_module
from ._make_jaxpr import StatefulFunction, get_arg_cache_key

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
    # Phase 4 -- engine
    'leaf_batch_dim',
    'StateMapPlan',
    'state_map_transform',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable
AxisToState = Dict[int, List[State]]
StateToAxis = Dict[State, int]

_rand = None


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
    assert isinstance(axis, int), f"Expected axis to be an integer, but got {type(axis)}"
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise IndexError(f"Axis {axis} is out of bounds for array of shape {x.shape}")
    return x[tuple(slice(None, None, None) if i != axis else 0 for i in range(x.ndim))]


def _compile_stateful_function(
    stateful_fn: StatefulFunction,
    in_axes: int | Tuple[int, ...],
    args: Tuple
):
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
    in_axes: int | Tuple[int, ...],
    in_states: AxisToState,
    axis_size: Optional[int] = None,
) -> int:
    batch_sizes = []

    # Check batch size from args and in_axes
    if isinstance(in_axes, int):
        in_axes = (in_axes,) * len(args)
    if in_axes is not None:
        for arg, in_axis in zip(args, in_axes):
            if in_axis is not None:
                arg_leaves = jax.tree.leaves(arg)
                if arg_leaves:
                    batch_sizes.append(arg_leaves[0].shape[in_axis])

    # Check batch size from in_states
    if in_states is not None:
        for axis, states in in_states.items():
            if axis is None:
                continue
            for state in states:
                state_leaves = jax.tree.leaves(state.value)
                if len(state_leaves):
                    batch_sizes.append(state_leaves[0].shape[axis])

    if len(batch_sizes) == 0:
        assert axis_size is not None, (
            "Unable to determine batch size. Please provide the 'axis_size' argument."
        )
        return axis_size
    else:
        # Ensure all batch sizes are consistent
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
# Phase 4 -- unified discovery + execution engine
# ============================================================================ #


def _remove_axis_tree(value, axis: int):
    """Strip ``axis`` from every leaf of a pytree value."""
    return jax.tree.map(lambda x: _remove_axis(x, axis), value)


def _strip_args(args: Tuple, in_axes) -> Tuple:
    """Remove the mapped axis from positional arguments (per ``in_axes``)."""
    if in_axes is None:
        return args
    if isinstance(in_axes, int):
        return tuple(jax.tree.map(lambda x: _remove_axis(x, in_axes), a) for a in args)
    # tuple / list of per-argument axes
    return tuple(
        a if ax is None else jax.tree.map(lambda x: _remove_axis(x, ax), a)
        for a, ax in zip(args, in_axes)
    )


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


class StateMapPlan:
    """Cached plan describing how states are batched for a mapped call.

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
    """

    __slots__ = ('in_groups', 'rng_states', 'out_groups', 'oth_out_states')

    def __init__(self, in_groups, rng_states, out_groups, oth_out_states):
        self.in_groups = in_groups
        self.rng_states = rng_states
        self.out_groups = out_groups
        self.oth_out_states = oth_out_states


def _probe_states(f, args, kwargs, in_predicates, in_axes, name):
    """Eagerly enumerate touched states and classify batched inputs.

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
    """
    RandomState = _import_rand_state()
    stripped = _strip_args(args, in_axes)
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
                return _remove_axis_tree(state._value, axis)
        return state._value

    state_trace = StateTraceStack(name=name)
    state_trace.set_new_arg(hook)
    try:
        with state_trace:
            f(*stripped, **kwargs)
    finally:
        state_trace.recovery_original_values()
    return state_trace, dict(dim_to_in_states), rng_states, seen_in_ids


def _detect_out_dims(f, args, kwargs, in_groups, rng_states, write_states, batch_size, in_axes, axis_size, axis_name):
    """Detect the output batch dimension of every written state.

    Runs a real ``jax.vmap`` discovery pass (always ``vmap`` -- dimensions are
    identical under ``pmap``) and reads ``BatchTracer.batch_dim`` for each
    written state. All touched state values are snapshotted beforehand and
    restored afterwards so the discovery pass has no observable side effects.
    """
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

    def disc_fn(args_, rng_keys, group_vals):
        for st, key in zip(rng_states, rng_keys):
            st.restore_value(key)
        for (axis, states), vals in zip(in_groups, group_vals):
            for st, v in zip(states, vals):
                st.restore_value(v)
        f(*args_, **kwargs)
        for st in write_states:
            detected[id(st)] = leaf_batch_dim(st.value)
        return jnp.zeros(())

    try:
        jax.vmap(
            disc_fn,
            in_axes=(in_axes, 0 if len(rng_states) else None, in_group_axes),
            out_axes=None,
            axis_size=axis_size,
            axis_name=axis_name,
        )(args, rng_vals, in_group_vals)
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
    unexpected_out_state_mapping, name,
):
    """Probe + discover + assemble a :class:`StateMapPlan`."""
    RandomState = _import_rand_state()

    state_trace, dim_to_in_states, rng_states, seen_in_ids = _probe_states(
        f, args, kwargs, in_predicates, in_axes, name
    )
    rng_set_ids = {id(st) for st in rng_states}
    write_states = [st for st in state_trace.get_write_states() if id(st) not in rng_set_ids]

    in_groups = sorted(dim_to_in_states.items(), key=lambda kv: kv[0])
    in_state_to_axis = {id(st): axis for axis, states in in_groups for st in states}

    batch_size = _get_batch_size(args, in_axes, dict(in_groups), axis_size)

    detected, _ = _detect_out_dims(
        f, args, kwargs, in_groups, rng_states, write_states,
        batch_size, in_axes, axis_size, axis_name,
    )

    # assemble output groups in deterministic trace order
    out_axis_groups: Dict[int, List[State]] = defaultdict(list)
    oth_out_states: List[State] = []

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
                        "not covered by state_out_axes. Declare it in state_out_axes or set "
                        "unexpected_out_state_mapping to 'auto', 'warn', or 'ignore'."
                    )
                )
            elif unexpected_out_state_mapping == 'warn':
                warnings.warn(
                    f"State\n {st} \nwas written with a batched value on axis {det} but is "
                    "not covered by state_out_axes; scattering it automatically.",
                    UserWarning,
                )
                out_axis_groups[det].append(st)
            elif unexpected_out_state_mapping in ('ignore', 'auto'):
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
    return StateMapPlan(in_groups, rng_states, out_groups, oth_out_states)


def _execute_plan(plan: StateMapPlan, f, args, kwargs, in_axes, out_axes,
                  axis_size, axis_name, mapping_fn, mapping_kwargs):
    """Run the cached plan through the mapping primitive and restore states."""
    in_groups = plan.in_groups
    rng_states = plan.rng_states
    out_groups = plan.out_groups
    oth_out_states = plan.oth_out_states

    in_group_axes = [axis for axis, _ in in_groups]
    if len(in_group_axes) == 0:
        in_group_axes = 0
    out_group_axes = [axis for axis, _ in out_groups]
    if len(out_group_axes) == 0:
        out_group_axes = 0

    batch_size = _get_batch_size(args, in_axes, dict(in_groups), axis_size)
    rng_keys, rng_backups = split_rng_keys(rng_states, batch_size)

    in_group_vals = [[st.value for st in states] for _, states in in_groups]

    def fn_to_map(args_, rng_keys_, group_vals):
        for st, key in zip(rng_states, rng_keys_):
            st.restore_value(key)
        for (axis, states), vals in zip(in_groups, group_vals):
            for st, v in zip(states, vals):
                st.restore_value(v)
        out = f(*args_, **kwargs)
        out_group_vals = [[st.value for st in states] for _, states in out_groups]
        oth_vals = [st.value for st in oth_out_states]
        return out, out_group_vals, oth_vals

    mapped_fn = mapping_fn(
        fn_to_map,
        in_axes=(in_axes, 0 if len(rng_states) else None, in_group_axes),
        out_axes=(out_axes, out_group_axes, None),
        axis_size=axis_size,
        axis_name=axis_name,
        **mapping_kwargs,
    )
    out, out_group_vals, oth_vals = mapped_fn(args, rng_keys, in_group_vals)

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
        Positional/keyword arguments treated as compile-time constants when
        building the per-signature cache key.
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
        cache_key = get_arg_cache_key(static_argnums, static_argnames, args, kwargs)
        plan = cache.get(cache_key, None)
        if plan is None:
            plan = _build_plan(
                f, args, kwargs,
                in_predicates, out_predicates,
                in_axes, axis_size, axis_name,
                unexpected_out_state_mapping, name,
            )
            cache[cache_key] = plan
        return _execute_plan(
            plan, f, args, kwargs, in_axes, out_axes,
            axis_size, axis_name, mapping_fn, mapping_kwargs,
        )

    wrapped.__brainstate_state_map__ = True
    return wrapped
