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

from __future__ import annotations

import functools
from collections import defaultdict
from typing import TypeVar, Callable, Dict, Hashable, List, Any, Sequence, Optional

import jax

from brainstate._state import State, StateTraceStack, TRACE_CONTEXT, catch_new_states
from brainstate.typing import Missing
from brainstate.util.filter import Filter

# Shared mapping helpers now live in ``_mapping_core`` so that the legacy
# ``vmap`` / ``vmap_new_states`` and the modern ``vmap2`` / ``pmap2`` families
# converge on a single implementation. They are re-exported here unchanged so
# existing imports (and tests) of ``brainstate.transform._mapping1`` keep
# working.
from ._mapping_core import (  # noqa: E402
    _import_rand_state,
    _get_batch_size,
    _format_state_axes,
    _strip_args,
    make_identity_predicate,
    state_map_transform,
    unwind_new_state_levels,
    _build_new_state_resolver,
    _resolve_new_state_axis,
    _new_state_probe,
)

__all__ = [
    'vmap',
    'vmap_new_states',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable
AxisToState = Dict[int, List[State]]
StateToAxis = Dict[State, int]


def _states_to_predicate_axes(formatted_axis_to_states: AxisToState):
    """Convert ``{axis: [State, ...]}`` into ``{axis: identity_predicate}``.

    The legacy ``vmap`` selects states by *declaration* (explicit instances),
    whereas the shared engine selects by predicate. Identity predicates bridge
    the two: each declared axis maps to a predicate matching exactly the
    declared instances.
    """
    return {
        axis: make_identity_predicate(states)
        for axis, states in formatted_axis_to_states.items()
        if len(states) > 0
    }


def _vmap_transform(
    f: F,
    *,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    in_states: Dict[int, Dict] | Any | None = None,
    out_states: Dict[int, Dict] | Any | None = None,
    axis_size: Optional[int] = None,
    axis_name: AxisName | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
):
    """Declaration-based ``vmap`` implemented as a shim over the shared engine.

    The explicit ``in_states`` / ``out_states`` declarations are converted to
    identity predicates and handed to :func:`state_map_transform`. The
    ``'raise'`` policy is forced so that a state written with a batched value but
    not declared in ``out_states`` raises a :class:`BatchAxisError`, preserving
    the historical contract. Keyword arguments are rejected, also for
    historical compatibility.
    """
    if isinstance(in_axes, list):
        # canonicalize list -> tuple (see jax-ml/jax#2367)
        in_axes = tuple(in_axes)

    # format + validate state axes (raises on in/out axis mismatch)
    (
        axis_to_in_states,
        in_state_to_axis,
        axis_to_out_states,
        out_state_to_axis,
    ) = _format_state_axes(in_states, out_states)

    state_in_axes = _states_to_predicate_axes(axis_to_in_states)
    state_out_axes = _states_to_predicate_axes(axis_to_out_states)

    engine_fn = state_map_transform(
        f,
        in_axes=in_axes,
        out_axes=out_axes,
        state_in_axes=state_in_axes,
        state_out_axes=state_out_axes,
        axis_size=axis_size,
        axis_name=axis_name,
        mapping_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name),
        unexpected_out_state_mapping='raise',
        name='vmap',
        # #5: speak the legacy vmap vocabulary (in_states/out_states, no policy
        # knob) in the undeclared-write error instead of engine internals.
        out_decl_name='out_states',
        out_decl_extra='',
    )

    @functools.wraps(f)
    def vmapped_fn(*args, **kwargs):
        if len(kwargs):
            raise NotImplementedError(
                "Keyword arguments `f(**kwargs)` are not supported in brainstate.transform.vmap"
            )
        return engine_fn(*args)

    return vmapped_fn


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
    in_states: Dict[int, Dict] | Any | None = None,
    out_states: Dict[int, Dict] | Any | None = None,
) -> F | Callable[[F], F]:
    """
    Vectorize a callable while preserving BrainState state semantics.

    This is the declaration-based vectorization API: states that participate
    in the mapped axis are declared explicitly through ``in_states`` and
    ``out_states`` (by :class:`~brainstate.State` instance). It is implemented
    as a thin shim over the shared mapping engine that also powers
    :func:`~brainstate.transform.vmap2`; the declared states are converted to
    identity selectors internally.

    Compared with :func:`~brainstate.transform.vmap2`, this entry point keeps
    the historical contract: a state written with a batched value but not
    declared in ``out_states`` raises a
    :class:`~brainstate._error.BatchAxisError` (rather than being inferred
    automatically), and keyword arguments are not supported.

    Parameters
    ----------
    fn : callable, optional
        Function to vectorize. If omitted, the function acts as a decorator.
    in_axes : int, None, or sequence, default 0
        Mapped-axis alignment per positional argument, following
        :func:`jax.vmap` semantics. ``None`` marks an argument as broadcast.
    out_axes : Any, default 0
        Placement of the mapped axis in the return value.
    axis_name : hashable, optional
        Name for the mapped axis so collective primitives can target it.
    axis_size : int, optional
        Explicit mapped-axis size. Inferred from arguments/states when omitted.
    spmd_axis_name : hashable or tuple of hashable, optional
        Axis labels for nested SPMD transforms.
    in_states : dict, State, or iterable of State, optional
        States batched on input, declared by instance. A dict maps axis
        identifiers to states; a bare state (or iterable of states) is
        shorthand for ``{0: ...}``.
    out_states : dict, State, or iterable of State, optional
        States whose writes are scattered back along the mapped axis, with the
        same declaration semantics as ``in_states``.

    Returns
    -------
    callable
        The vectorized function if ``fn`` is supplied, otherwise a decorator.

    Raises
    ------
    BatchAxisError
        If a state is written with a batched value but not declared in
        ``out_states``.
    NotImplementedError
        If keyword arguments are passed to the vectorized function.

    See Also
    --------
    brainstate.transform.vmap2 : Filter/predicate-based vectorization with
        automatic output-axis inference.
    brainstate.transform.vmap_new_states : Vectorize states created inside the
        function.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> counter = brainstate.ShortTermState(jnp.zeros(3))
        >>>
        >>> @brainstate.transform.vmap(
        ...     in_axes=0,
        ...     in_states=counter,
        ...     out_states=counter,
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
            _vmap_transform,
            in_axes=in_axes,
            out_axes=out_axes,
            in_states=in_states,
            out_states=out_states,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
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
    )


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
    in_states: Dict[int, Dict] | Any | None = None,
    out_states: Dict[int, Dict] | Any | None = None,
):
    # TODO: How about nested call ``vmap_new_states``?
    if isinstance(axis_size, int) and axis_size <= 0:
        raise ValueError(f"axis_size must be greater than 0, got {axis_size}.")

    if in_states is not None or out_states is not None:
        raise ValueError(
            "vmap_new_states does not use 'in_states'/'out_states': it maps over "
            "states *created inside* the function, not pre-existing ones. Use "
            "brainstate.transform.vmap (which declares in_states/out_states) to "
            "vectorize over pre-existing states."
        )

    RandomState = _import_rand_state()

    if isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    @functools.wraps(fun)
    def vmapped_fn(*args, **kwargs):
        if len(kwargs):
            raise NotImplementedError(
                "Keyword arguments `f(**kwargs)` are not supported in "
                "brainstate.transform.vmap_new_states"
            )

        base_level = TRACE_CONTEXT.get_trace_stack_level()
        stripped = _strip_args(args, in_axes)

        # --- eager probe: discover the random states ``fun`` touches -------- #
        rng_states: List[Any] = []
        probe_trace = StateTraceStack(name='vmap_new_states_probe')

        def probe_hook(state):
            if isinstance(state, RandomState):
                rng_states.append(state)
                return state.split_key()
            return state._value

        probe_trace.set_new_arg(probe_hook)
        try:
            # ``_new_state_probe`` marks this as the discovery pass: the states
            # created here are thrown away, so one-shot consumers (e.g. graph
            # compilers keyed off the created state objects) can defer to the
            # real mapped pass via ``in_new_state_probe()``.
            with _new_state_probe():
                with catch_new_states(state_to_exclude=state_to_exclude):
                    with probe_trace:
                        fun(*stripped)
        finally:
            probe_trace.recovery_original_values()
        # B7: defensive de-dup. The ``new_arg`` hook fires at most once per
        # distinct State (StateTraceStack guards on state id in ``read_its_value``),
        # so duplicates should not occur; kept to stay robust if that ever changes.
        _seen = set()
        rng_states = [r for r in rng_states if not (id(r) in _seen or _seen.add(id(r)))]

        # --- single mapped pass over ``fun`` ------------------------------- #
        batch_size = _get_batch_size(args, in_axes, {}, axis_size)
        rng_keys = [rng.split_key(batch_size) for rng in rng_states]
        rng_backups = [rng.split_key() for rng in rng_states]

        # B6: group new states by output axis exactly like vmap2_new_states so a
        # NonBatchState (or INIT_NO_BATCHING-tagged state) created inside ``fun``
        # is replicated (axis None) rather than scattered at axis 0. There is no
        # ``state_out_axes`` here, so the resolver yields just [None, 0].
        ordered, axes_order = _build_new_state_resolver(None)
        new_states_box: Dict[Any, List[State]] = {}

        def new_fun(args_, rng_keys_):
            for rng, key in zip(rng_states, rng_keys_):
                rng.restore_value(key)
            with catch_new_states(state_tag=state_tag, state_to_exclude=state_to_exclude) as catcher:
                # plain watcher (no new_arg): ``fun`` runs under a raw
                # jax.vmap here, and state writes of vmap tracers are
                # legitimate — an active StateTraceStack keeps the
                # tracer-write guard quiet (the extra stack level is undone
                # by unwind_new_state_levels, which is delta-based)
                with StateTraceStack(name='vmap_new_states:run'):
                    out = fun(*args_)
            grouped_vals: Dict[Any, List] = defaultdict(list)
            grouped_states: Dict[Any, List] = defaultdict(list)
            for st in catcher.get_states():
                axis = _resolve_new_state_axis(st, ordered)
                grouped_vals[axis].append(st.value)
                grouped_states[axis].append(st)
            new_states_box.clear()
            new_states_box.update(grouped_states)
            return out, tuple(grouped_vals.get(k, []) for k in axes_order)

        try:
            with catch_new_states(state_to_exclude=state_to_exclude):
                mapped = jax.vmap(
                    new_fun,
                    in_axes=(in_axes, 0 if len(rng_states) else None),
                    out_axes=(out_axes, tuple(axes_order)),
                    axis_size=axis_size,
                    axis_name=axis_name,
                    spmd_axis_name=spmd_axis_name,
                )
                outs, grouped_out_vals = mapped(args, rng_keys)
        finally:
            # restore the global RNG once -- also on failure, so a crashed
            # mapped pass cannot leave key tracers in the random states
            for rng, key in zip(rng_states, rng_backups):
                rng.restore_value(key)

        # restore vmapped new-state values + unwind trace levels (avoids leakage)
        all_new_states: List[State] = []
        for axis, vals in zip(axes_order, grouped_out_vals):
            for st, st_val in zip(new_states_box.get(axis, []), vals):
                st.restore_value(st_val)
                all_new_states.append(st)
        unwind_new_state_levels(all_new_states, base_level)
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
    state_tag: str | None = None,
    state_to_exclude: Filter = None,
    in_states: Dict[int, Dict] | Any | None = None,
    out_states: Dict[int, Dict] | Any | None = None,
) -> Callable:
    """
    Vectorize a function over the new states it creates.

    Unlike :func:`~brainstate.transform.vmap`, which maps over states that
    already exist, this transform maps over states *created during* the call
    to ``fun`` (for example, parameters allocated inside a module's
    initializer). Each mapped lane creates its own copy of every new state, and
    random states are split per lane so randomly initialized values differ
    across the mapped axis. It is implemented as a single mapping pass over the
    shared engine helpers.

    .. note::

        ``fun`` is executed **twice** -- an eager probe (to discover the random
        states it touches) plus the mapped pass -- so it must be idempotent and
        free of un-rolled-back side effects. :class:`~brainstate.NonBatchState`
        (and ``INIT_NO_BATCHING``-tagged) states created inside ``fun`` are
        replicated rather than batched.

    Parameters
    ----------
    fun : callable, optional
        Function to vectorize. If omitted, this acts as a decorator.
    in_axes : int, None, or sequence, default 0
        Mapped-axis alignment per positional argument, following
        :func:`jax.vmap` semantics. ``None`` marks an argument as broadcast.
    out_axes : Any, default 0
        Placement of the mapped axis in the return value.
    axis_name : hashable, optional
        Name for the mapped axis so collective primitives can target it.
    axis_size : int, optional
        Explicit mapped-axis size. Inferred from the inputs when omitted.
    spmd_axis_name : hashable or tuple of hashable, optional
        Axis labels for nested SPMD transforms.
    state_tag : str, optional
        Tag applied to the newly created states so they can be retrieved later.
    state_to_exclude : Filter, optional
        Selector for new states that should be left untouched (not vectorized).
    in_states, out_states : dict, State, or iterable of State, optional
        Not supported by this transform. ``vmap_new_states`` maps over states
        *created inside* ``fun``, so there are no pre-existing states to declare.
        Passing either raises :class:`ValueError`; use
        :func:`~brainstate.transform.vmap` to vectorize over pre-existing states.

    Returns
    -------
    callable
        A vectorized version of ``fun`` that handles new-state creation, or a
        decorator if ``fun`` is omitted.

    Raises
    ------
    ValueError
        If ``axis_size`` is provided but not a positive integer, or if
        ``in_states`` / ``out_states`` is provided (use
        :func:`~brainstate.transform.vmap` for pre-existing states).
    NotImplementedError
        If keyword arguments are passed to the vectorized function.

    See Also
    --------
    brainstate.transform.vmap : Vectorize over pre-existing states.
    brainstate.transform.vmap2_new_states : Module-oriented new-state mapping.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> @brainstate.transform.vmap_new_states(in_axes=0, axis_size=4)
        ... def build(x):
        ...     scratch = brainstate.ShortTermState(jnp.zeros(()))
        ...     scratch.value = scratch.value + x
        ...     return scratch.value
        >>>
        >>> build(jnp.arange(4.))
        Array([0., 1., 2., 3.], dtype=float32)
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
            in_states=in_states,
            out_states=out_states,
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
            in_states=in_states,
            out_states=out_states,
        )
