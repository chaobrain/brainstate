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

import functools
from typing import TypeVar, Callable, Dict, Hashable, List, Any, Tuple, Sequence, Optional

import jax

from brainstate._compatible_import import BatchTracer
from brainstate._error import BatchAxisError
from brainstate._state import State, catch_new_states
from brainstate.typing import Missing
from brainstate.util.filter import Filter
from ._make_jaxpr import StatefulFunction

__all__ = [
    'vmap',
    'vmap_new_states',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable
AxisToState = Dict[int, List[State]]
StateToAxis = Dict[State, int]

# Shared mapping helpers now live in ``_mapping_core`` so that the legacy
# ``vmap`` / ``vmap_new_states`` and the modern ``vmap2`` / ``pmap2`` families
# converge on a single implementation. They are re-exported here unchanged so
# existing imports (and tests) of ``brainstate.transform._mapping1`` keep
# working.
from ._mapping_core import (  # noqa: E402
    _import_rand_state,
    _flatten_in_out_states,
    _remove_axis,
    _compile_stateful_function,
    _get_batch_size,
    _format_state_axes,
)


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
    RandomState = _import_rand_state()

    # format state axes
    (
        axis_to_in_states,
        in_state_to_axis,
        axis_to_out_states,
        out_state_to_axis
    ) = _format_state_axes(in_states, out_states)

    # check in_axes
    if isinstance(in_axes, list):
        # To be a tree prefix of the positional args tuple, in_axes can never be a
        # list: if in_axes is not a leaf, it must be a tuple of trees. However,
        # in cases like these users expect tuples and lists to be treated
        # essentially interchangeably, so we canonicalize lists to tuples here
        # rather than raising an error. https://github.com/jax-ml/jax/issues/2367
        in_axes = tuple(in_axes)

    def _vmap_fn_for_compilation(in_vmap_state_vals, args):
        """
        Compile a function for vectorized mapping (vmap) with state restoration.

        This internal function is used to prepare a function for vectorized mapping
        by restoring state values before calling the original function.

        Args:
            in_vmap_state_vals (List[List]): A nested list containing the state values
                to be restored. The outer list corresponds to different axes, while
                the inner lists contain the state values for each axis.
            args (Tuple): The arguments to be passed to the original function after
                state restoration.

        Returns:
            Any: The result of calling the original function 'f' with the restored
            state and provided arguments.
        """
        # restore state values
        for i, states in enumerate(axis_to_in_states.values()):
            for state, state_val in zip(states, in_vmap_state_vals[i]):
                state.restore_value(state_val)

        # call the function
        return f(*args)

    def _set_axis_env(batch_size):
        axis_env = None if axis_name is None else [(axis_name, batch_size)]
        stateful_fn.axis_env = axis_env

    # stateful function
    stateful_fn = StatefulFunction(_vmap_fn_for_compilation, name='vmap')

    @functools.wraps(f)
    def new_fn_for_vmap(
        rng_keys,
        in_state_vmap_vals,
        in_state_oth_vals,
        args,
    ):
        """
        Wrapper function for vectorized mapping (vmap) that handles state restoration and function execution.

        This function restores state values, random number generators (RNGs), and other state values
        before calling the original function. It then processes the outputs and prepares them for
        vectorized mapping.

        Args:
            rng_keys (Sequence): Random number generator keys for each mapped instance.
            in_state_vmap_vals (Sequence[Sequence]): Input state values for vectorized mapping,
                organized by axis.
            in_state_oth_vals (Sequence): Other input state values not involved in vectorized mapping.
            args (Tuple): Arguments to be passed to the original function.

        Returns:
            Tuple: A tuple containing four elements:
                - out_rng_keys (List): Updated RNG keys after function execution.
                - out_state_vmap_vals (List[List]): Output state values for vectorized mapping,
                  organized by axis.
                - out_state_oth_vals (List): Other output state values not involved in vectorized mapping.
                - outs: The output of the original function call.

        Raises:
            AssertionError: If there's a mismatch in the number of states, state values, or RNG keys.
            BatchAxisError: If a state value is batched but not included in out_states.
        """
        # restore vmapping state values
        for i, states in enumerate(axis_to_in_states.values()):
            assert len(states) == len(in_state_vmap_vals[i]), (
                f"The number of states in axis {i} should be equal to the number "
                f"of state values, but got {len(states)} and {len(in_state_vmap_vals[i])}."
            )
            for state, state_val in zip(states, in_state_vmap_vals[i]):
                state.restore_value(state_val)

        # restore rngs
        cache_key = stateful_fn.get_arg_cache_key(in_state_vmap_vals, args)
        state_trace = stateful_fn.get_state_trace_by_cache(cache_key)
        rngs = state_trace.state_subset(RandomState)
        rng_sets = set(rngs)
        assert len(rngs) == len(rng_keys), (
            f"The number of random states in the function should be equal to the number "
            f"of random keys, but got {len(rngs)} and {len(rng_keys)}."
        )
        for rng, key in zip(rngs, rng_keys):
            rng.restore_value(key)

        # restore other state values
        oth_in_state = [
            st for st in state_trace.states
            if st not in in_state_to_axis and st not in rng_sets
        ]
        assert len(oth_in_state) == len(in_state_oth_vals), (
            f"The number of states in 'in_states' should be equal to the number "
            f"of state values, but got {len(oth_in_state)} and {len(in_state_oth_vals)}."
        )
        for state, state_val in zip(oth_in_state, in_state_oth_vals):
            state.restore_value(state_val)

        # call the function
        outs = stateful_fn.jaxpr_call_auto(in_state_vmap_vals, args)

        # analyze vmapping axis error
        for state in state_trace.get_write_states():
            leaves = jax.tree.leaves(state.value)
            if (
                any([isinstance(leaf, BatchTracer) and (leaf.batch_dim is not None) for leaf in leaves])
                and state not in out_state_to_axis
            ):
                if isinstance(state, RandomState) and state in rng_sets:
                    continue
                state.raise_error_with_source_info(
                    BatchAxisError(f"The value of State {state} is batched, "
                                   f"but it is not in the out_states.")
                )

        # out state values for vmapping
        out_state_vmap_vals = [
            [state.value for state in states]
            for axis, states in axis_to_out_states.items()
        ]
        out_state_oth_vals = [
            st.value for st in state_trace.states
            if st not in out_state_to_axis and st not in rng_sets
        ]
        out_rng_keys = [rng.value for rng in rngs]
        return out_rng_keys, out_state_vmap_vals, out_state_oth_vals, outs

    @functools.wraps(f)
    def vmapped_fn(*args, **kwargs):
        if len(kwargs):
            raise NotImplementedError(
                "Keyword arguments `f(**kwargs)` are not supported in brainstate.transform.vmap"
            )

        # in states values
        in_state_map_vals = [
            [st.value for st in states]
            for axis, states in axis_to_in_states.items()
        ]
        st_in_axes = list(axis_to_in_states.keys())
        if len(st_in_axes) == 0:
            st_in_axes = 0

        # compile stateful function
        batch_size = None
        if axis_name is not None:
            batch_size = _get_batch_size(args, in_axes, axis_to_in_states, axis_size)
            _set_axis_env(batch_size)
        cache_key = _compile_stateful_function(
            stateful_fn,
            (st_in_axes, in_axes),
            (in_state_map_vals, args)
        )

        # random keys
        state_trace = stateful_fn.get_state_trace_by_cache(cache_key)
        rngs = state_trace.state_subset(RandomState)
        rng_sets = set(rngs)
        if len(rngs):
            # batch size
            if batch_size is None:
                batch_size = _get_batch_size(args, in_axes, axis_to_in_states, axis_size)
            rng_keys = tuple(rng.split_key(batch_size) for rng in rngs)
            rng_backup = tuple(rng.split_key() for rng in rngs)
        else:
            rng_keys = tuple()
            rng_backup = tuple()

        # in states other values
        in_state_oth_vals = [
            st.value
            for st in state_trace.states
            if st not in in_state_to_axis and st not in rng_sets
        ]

        # out state axis
        st_out_axes = list(axis_to_out_states.keys())
        if len(st_out_axes) == 0:
            st_out_axes = 0

        # --- vmapping --- #
        fn = jax.vmap(
            new_fn_for_vmap,
            in_axes=(0, st_in_axes, None, in_axes),
            out_axes=(0, st_out_axes, None, out_axes),
            axis_size=axis_size,
            axis_name=axis_name,
            spmd_axis_name=spmd_axis_name,
        )
        _, out_state_map_vals, out_state_oth_vals, outs = fn(
            rng_keys, in_state_map_vals, in_state_oth_vals, args
        )

        # restore mapped state values
        for i, states in enumerate(axis_to_out_states.values()):
            assert len(states) == len(out_state_map_vals[i]), (
                f"The number of states in axis {i} should be equal to the number "
                f"of state values, but got {len(states)} and {len(out_state_map_vals[i])}."
            )
            for state, st_val in zip(states, out_state_map_vals[i]):
                state.restore_value(st_val)

        # restore other state values
        out_oth_states = [
            st for st in state_trace.states
            if st not in out_state_to_axis and st not in rng_sets
        ]
        assert len(out_oth_states) == len(out_state_oth_vals), (
            f"The number of states in 'out_states' should be equal to the number "
            f"of state values, but got {len(out_oth_states)} and {len(out_state_oth_vals)}."
        )
        for state, st_val in zip(out_oth_states, out_state_oth_vals):
            state.restore_value(st_val)

        # restore random keys
        for rng, key in zip(rngs, rng_backup):
            rng.restore_value(key)
        return outs

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

    @vmap(
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
        in_states=in_states,
        out_states=out_states,
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
        # vmapping
        with catch_new_states(state_to_exclude=state_to_exclude) as catcher:
            outs, vmap_state_vals = new_fun(args)
            vmap_states = catcher.get_states()

        # restore vmapped state values
        for st_val, st in zip(vmap_state_vals, vmap_states):
            st.restore_value(st_val)
            # ------------------------------------------------
            # --- this is CRUCIAL to avoid jax tracing leakage
            # ------------------------------------------------
            st.decrease_stack_level()
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
        state_tag (str, optional): A tag to identify specific states. Defaults to None.
        state_to_exclude (Sequence[int], optional): Indices of states to exclude from vectorization. Defaults to ().

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
