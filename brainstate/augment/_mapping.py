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
    Dict,
    List
)

import jax
from jax.interpreters.batching import BatchTracer

from brainstate._compatible_import import Device
from brainstate._state import State, catch_new_states
from brainstate.compile import scan, StatefulFunction
from brainstate.random import RandomState, DEFAULT
from brainstate.typing import Missing, Filter
from brainstate.util import NestedDict, BrainStateError
from ._random import restore_rngs

__all__ = [
    'vmap',
    'pmap',
    'map',
    'vmap_new_states',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable
AxisToState = Dict[int, List[State]]
StateToAxis = Dict[State, int]


class BatchAxisError(BrainStateError):
    """
    Exception raised for errors related to batch axis operations.

    This custom exception is used to indicate errors that occur during
    batch processing or vectorization operations, particularly in the
    context of state management in the BrainState framework.

    Inherits from:
        BrainStateError: The base error class for BrainState-related exceptions.
    """
    pass


def _flatten_in_out_states(
    in_states: Dict[int, Dict] | Any = None,
) -> Tuple[AxisToState, StateToAxis]:
    """
    Flattens and organizes input or output states into axis-based mappings.

    This function processes the input or output states, converting them into two
    dictionary representations: one mapping axes to states, and another mapping
    states to axes. It handles both structured (Dict[int, Dict]) and unstructured
    input formats.

    Args:
        in_states (Dict[int, Dict] | Any, optional): The input or output states to be
            flattened. Can be a nested dictionary structure where the outer keys are
            axes and inner dictionaries contain states, or any other structure
            containing states. Defaults to None.

    Returns:
        Tuple[AxisToState, StateToAxis]: A tuple containing two dictionaries:
            - AxisToState: Maps axes (int) to lists of states.
            - StateToAxis: Maps individual states to their corresponding axes (int).

    Note:
        If in_states is None, empty dictionaries are returned for both mappings.
        If in_states is not in the expected Dict[int, Dict] format, all states are
        assigned to axis 0.
    """
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
    """
    Remove a specified axis from an array or nested structure.

    This function removes a specified axis from an array or nested structure,
    adjusting the shape and structure of the output accordingly.

    Args:
        x (Any): The input array or nested structure to remove the axis from.
        axis (int): The axis to remove from the input.

    Returns:
        Any: The output array or nested structure with the specified axis removed.
    """
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
    """
    Compile a stateful function with specified input axes and arguments.

    This function prepares and compiles a stateful function for vectorized mapping (vmap)
    by adjusting the input arguments based on the specified axes and then generating
    the function's JAX program representation (jaxpr).

    Args:
        stateful_fn (StatefulFunction): The stateful function to be compiled.
        in_axes (int | Tuple[int, ...]): Specifies which axes of the input arguments
            to map over. Can be a single integer (same for all args) or a tuple of integers.
        args (Tuple): The input arguments to the function.

    Raises:
        ValueError: If the length of in_axes tuple doesn't match the number of arguments.

    Returns:
        None. The function modifies the stateful_fn in-place by calling make_jaxpr.
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
            arg if in_axis is None else _remove_axis(arg, in_axis)
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
    """
    Determine the batch size from input arguments, axes, and states.

    This function calculates the batch size by examining the shapes of input arguments
    and states along specified axes. It ensures consistency across all inputs.

    Args:
        args (Tuple): The input arguments to the function being vectorized.
        in_axes (int | Tuple[int, ...]): The axes along which to vectorize for each argument.
            Can be a single integer (same for all args) or a tuple of integers.
        in_states (AxisToState): A dictionary mapping axes to lists of states.

    Returns:
        int: The determined batch size.

    Raises:
        ValueError: If unable to determine batch size or if inconsistent batch sizes are found.
    """
    batch_sizes = []

    # Check batch size from args and in_axes
    if isinstance(in_axes, int):
        in_axes = (in_axes,) * len(args)
    for arg, in_axis in zip(args, in_axes):
        if in_axis is not None:
            arg_leaves = jax.tree.leaves(arg)
            if arg_leaves:
                batch_sizes.append(arg_leaves[0].shape[in_axis])

    # Check batch size from in_states
    if in_states is not None:
        for axis, states in in_states.items():
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
    in_states,
    out_states,
):
    """
    Format and validate the axes of input and output states.

    This function processes the input and output states, ensuring consistency
    between their axis mappings. It also handles cases where a state appears
    in the input but not in the output.

    Args:
        in_states: The input states to be formatted. Can be a dictionary mapping
                   axes to states, or any other structure containing states.
        out_states: The output states to be formatted. Can be a dictionary mapping
                    axes to states, or any other structure containing states.

    Returns:
        A tuple containing four elements:
        - axis_to_in_states (dict): Mapping of axes to input states.
        - in_state_to_axis (dict): Mapping of input states to their axes.
        - axis_to_out_states (dict): Mapping of axes to output states.
        - out_state_to_axis (dict): Mapping of output states to their axes.

    Raises:
        BatchAxisError: If there's an inconsistency between the axis mappings
                        of input and output states.
    """
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
    """
    Transforms a function for vectorized mapping (vmap) with state management.

    This internal function applies vectorized mapping to the input function while
    handling state management for input and output states. It supports custom
    axis specifications for both inputs and outputs.

    Args:
        f (F): The function to be transformed for vectorized mapping.
        in_axes (int | None | Sequence[Any]): Specifies which axes of the input
            arguments to map over. Default is 0.
        out_axes (Any): Specifies where the mapped axis should appear in the output.
            Default is 0.
        in_states (Dict[int, Dict] | Any | None): Specifies the input states and
            their corresponding axes for mapping. Default is None.
        out_states (Dict[int, Dict] | Any | None): Specifies the output states and
            their corresponding axes for mapping. Default is None.
        **transform_kwargs: Additional keyword arguments for the transformation.

    Returns:
        Callable: A new function that applies vectorized mapping to the input
        function while managing states.
    """

    # TODO: support jax.disable_jit()

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
        state_trace = stateful_fn.get_state_trace(cache_key)
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
        """
        Applies vectorized mapping (vmap) to the input function while managing state.

        This function handles the vectorization process, including state management,
        random number generation, and function compilation. It prepares the input
        states, compiles the stateful function, manages random number generators,
        applies the vmap transformation, and restores the output states.

        Args:
            *args: Variable length argument list containing the input arguments
                   to be passed to the vectorized function.

        Returns:
            Any: The output of the vectorized function after applying vmap and
                 managing states.

        Note:
            This function assumes the existence of several helper functions and
            data structures (e.g., axis_to_in_states, in_state_to_axis) which
            should be defined in the broader context.
        """
        if len(kwargs):
            raise NotImplementedError(
                "Keyword arguments `f(**kwargs)` are not supported in brainstate.augment.vmap"
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
        state_trace = stateful_fn.get_state_trace(cache_key)
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
    """
    Vectorizing map. Creates a function which maps ``fun`` over argument axes.

    The transformation :func:`vmap` is designed to work with ``pygraph`` structure
    defined in the ``brainstate`` library. It is used to vectorize functions by
    pushing the mapped axis down into primitive operations.

    More information please see `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__.

    These are several example usage::

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp

        >>> class Model(brainstate.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>
        >>>         self.a = brainstate.ShortTermState(brainstate.random.randn(5))
        >>>         self.b = brainstate.ShortTermState(brainstate.random.randn(5))
        >>>         self.c = brainstate.State(brainstate.random.randn(1))

        >>>     def __call__(self, *args, **kwargs):
        >>>         self.c.value = self.a.value * self.b.value
        >>>         return self.c.value + 1.

        >>> model = Model()

        >>> r = brainstate.augment.vmap(
        >>>     model,
        >>>     in_states=model.states(brainstate.ShortTermState),
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
