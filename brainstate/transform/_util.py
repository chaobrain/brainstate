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

from functools import wraps
from typing import Sequence, Tuple, Hashable, Union, Callable, Dict, Any

import jax

from brainstate._state import StateTraceStack
from brainstate.typing import PyTree
from ._make_jaxpr import StatefulFunction


def wrap_single_fun_in_multi_branches(
    stateful_fun: StatefulFunction,
    merged_state_trace: StateTraceStack,
    read_state_vals: Sequence[PyTree | None],
    return_states: bool = True,
    cache_key: Hashable = None,
):
    """
    Wrap a stateful function for use in multi-branch control flow.

    This function creates a wrapper that allows a stateful function to be used
    in control flow operations where multiple functions share state. It manages
    state values by extracting only the states needed by this specific function
    from a merged state trace.

    Parameters
    ----------
    stateful_fun : StatefulFunction
        The stateful function to be wrapped.
    merged_state_trace : StateTraceStack
        The merged state trace containing all states from multiple functions.
    read_state_vals : sequence of PyTree or None
        The original read state values for all states in the merged trace.
    return_states : bool, default True
        Whether to return updated state values along with the function output.

    Returns
    -------
    callable
        A wrapped function that can be used in multi-branch control flow.

    Examples
    --------
    Usage in conditional execution:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create states
        >>> state1 = brainstate.State(jnp.array([1.0]))
        >>> state2 = brainstate.State(jnp.array([2.0]))
        >>>
        >>> def branch_fn(x):
        ...     state1.value *= x
        ...     return state1.value + state2.value
        >>>
        >>> # During compilation, this wrapper allows the function
        >>> # to work with merged state traces from multiple branches
        >>> sf = brainstate.transform.StatefulFunction(branch_fn)
        >>> # wrapped_fn = wrap_single_fun_in_multi_branches(sf, merged_trace, read_vals)
    """
    state_ids_belong_to_this_fun = {id(st): st for st in stateful_fun.get_states_by_cache(cache_key)}

    @wraps(stateful_fun.fun)
    def wrapped_branch(write_state_vals, *operands):
        # "write_state_vals" should have the same length as "merged_state_trace.states"
        assert len(merged_state_trace.states) == len(write_state_vals) == len(read_state_vals)

        # get all state values needed for this function, which is a subset of "write_state_vals"
        st_vals_for_this_fun = []
        for write, st, val_w, val_r in zip(merged_state_trace.been_writen,
                                           merged_state_trace.states,
                                           write_state_vals,
                                           read_state_vals):
            if id(st) in state_ids_belong_to_this_fun:
                st_vals_for_this_fun.append(val_w if write else val_r)

        # call this function
        new_state_vals, out = stateful_fun.jaxpr_call(st_vals_for_this_fun, *operands)
        assert len(new_state_vals) == len(st_vals_for_this_fun)

        if return_states:
            # get all written state values
            new_state_vals = {id(st): val for st, val in
                              zip(stateful_fun.get_states_by_cache(cache_key), new_state_vals)}
            write_state_vals = tuple([
                (new_state_vals[id(st)] if id(st) in state_ids_belong_to_this_fun else w_val)
                if write else None
                for write, st, w_val in zip(merged_state_trace.been_writen,
                                            merged_state_trace.states,
                                            write_state_vals)
            ])
            return write_state_vals, out
        return out

    return wrapped_branch


def wrap_single_fun_in_multi_branches_while_loop(
    stateful_fun: StatefulFunction,
    merged_state_trace: StateTraceStack,
    read_state_vals: Sequence[PyTree | None],
    return_states: bool = True,
    cache_key: Hashable = None,
):
    """
    Wrap a stateful function for use in while loop control flow.

    This function creates a wrapper specifically designed for while loop operations
    where multiple functions share state. It manages state values by extracting only
    the states needed by this specific function from a merged state trace, with
    special handling for the loop's init_val structure.

    Parameters
    ----------
    stateful_fun : StatefulFunction
        The stateful function to be wrapped.
    merged_state_trace : StateTraceStack
        The merged state trace containing all states from multiple functions.
    read_state_vals : sequence of PyTree or None
        The original read state values for all states in the merged trace.
    return_states : bool, default True
        Whether to return updated state values along with the function output.

    Returns
    -------
    callable
        A wrapped function that can be used in while loop control flow.

    Examples
    --------
    Usage in while loop operations:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create states
        >>> counter = brainstate.State(jnp.array([0]))
        >>> accumulator = brainstate.State(jnp.array([0.0]))
        >>>
        >>> def cond_fn(val):
        ...     return counter.value < 10
        >>>
        >>> def body_fn(val):
        ...     counter.value += 1
        ...     accumulator.value += val
        ...     return val * 2
        >>>
        >>> # During compilation, this wrapper allows the functions
        >>> # to work with merged state traces in while loops
        >>> sf_cond = brainstate.transform.StatefulFunction(cond_fn)
        >>> sf_body = brainstate.transform.StatefulFunction(body_fn)
        >>> # wrapped_cond = wrap_single_fun_in_multi_branches_while_loop(sf_cond, ...)
        >>> # wrapped_body = wrap_single_fun_in_multi_branches_while_loop(sf_body, ...)
    """
    state_ids_belong_to_this_fun = {id(st): st for st in stateful_fun.get_states_by_cache(cache_key)}

    @wraps(stateful_fun.fun)
    def wrapped_branch(init_val):
        write_state_vals, init_val = init_val
        # "write_state_vals" should have the same length as "merged_state_trace.states"
        assert len(merged_state_trace.states) == len(write_state_vals) == len(read_state_vals)

        # get all state values needed for this function, which is a subset of "write_state_vals"
        st_vals_for_this_fun = []
        for write, st, val_w, val_r in zip(merged_state_trace.been_writen,
                                           merged_state_trace.states,
                                           write_state_vals,
                                           read_state_vals):
            if id(st) in state_ids_belong_to_this_fun:
                st_vals_for_this_fun.append(val_w if write else val_r)

        # call this function
        new_state_vals, out = stateful_fun.jaxpr_call(st_vals_for_this_fun, init_val)
        assert len(new_state_vals) == len(st_vals_for_this_fun)

        if return_states:
            # get all written state values
            new_state_vals = {id(st): val for st, val in
                              zip(stateful_fun.get_states_by_cache(cache_key), new_state_vals)}
            write_state_vals = tuple([
                (new_state_vals[id(st)] if id(st) in state_ids_belong_to_this_fun else w_val)
                if write else None
                for write, st, w_val in zip(merged_state_trace.been_writen,
                                            merged_state_trace.states,
                                            write_state_vals)
            ])
            return write_state_vals, out
        return out

    return wrapped_branch


def wrap_single_fun(
    stateful_fun: StatefulFunction,
    been_writen: Sequence[bool],
    read_state_vals: Tuple[PyTree | None],
):
    """
    Wrap a stateful function for use in scan operations.

    This function creates a wrapper specifically designed for scan operations.
    It manages state values by combining written and read states, calls the
    stateful function, and returns only the written states along with the
    carry and output values.

    Parameters
    ----------
    stateful_fun : StatefulFunction
        The stateful function to be wrapped for scan operations.
    been_writen : sequence of bool
        Boolean flags indicating which states have been written to.
    read_state_vals : tuple of PyTree or None
        The original read state values for all states.

    Returns
    -------
    callable
        A wrapped function that can be used in scan operations with proper
        state management.

    Examples
    --------
    Usage in scan operations:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create states
        >>> state1 = brainstate.State(jnp.array([0.0]))
        >>> state2 = brainstate.State(jnp.array([1.0]))
        >>>
        >>> def scan_fn(carry, x):
        ...     state1.value += x  # This state will be written
        ...     result = carry + state1.value + state2.value  # state2 is only read
        ...     return result, result ** 2
        >>>
        >>> # During compilation, this wrapper allows the function
        >>> # to work properly in scan operations
        >>> sf = brainstate.transform.StatefulFunction(scan_fn)
        >>> # wrapped_fn = wrap_single_fun(sf, been_written_flags, read_values)
        >>>
        >>> # The wrapped function handles state management automatically
        >>> xs = jnp.arange(5.0)
        >>> init_carry = 0.0
        final_carry, ys = brainstate.transform.scan(scan_fn, init_carry, xs)
    """

    @wraps(stateful_fun.fun)
    def wrapped_fun(new_carry, inputs):
        writen_state_vals, carry = new_carry
        assert len(been_writen) == len(writen_state_vals) == len(read_state_vals)

        # collect all written and read states
        state_vals = [
            written_val if written else read_val
            for written, written_val, read_val in zip(been_writen, writen_state_vals, read_state_vals)
        ]

        # call the jaxpr
        state_vals, (carry, out) = stateful_fun.jaxpr_call(state_vals, carry, inputs)

        # only return the written states
        writen_state_vals = tuple([val if written else None for written, val in zip(been_writen, state_vals)])

        # return
        return (writen_state_vals, carry), out

    return wrapped_fun


def warp_grad_fn(
    fn: Callable,
    argnums: Union[int, Sequence[int]],
    args: Sequence[Any],
    kwargs: Dict,
):
    args = tuple(args)

    if isinstance(argnums, int):
        @wraps(fn)
        def new_fn(dyn_args):
            new_args = list(args)
            new_args[argnums] = dyn_args
            return fn(*new_args, **kwargs)

        assert argnums < len(args), f"argnum {argnums} is out of range {len(args)}"
        return new_fn, args[argnums]

    else:

        @wraps(fn)
        def new_fn(dyn_args):
            assert len(dyn_args) == len(argnums)
            new_args = list(args)
            for i, argnum in enumerate(argnums):
                new_args[argnum] = dyn_args[i]
            return fn(*new_args, **kwargs)

        argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        params = []
        for i in argnums:
            assert i < len(args), f"argnum {i} is out of range {len(args)}"
            params.append(args[i])
        return new_fn, params


def tree_random_split(rng_key, target=None, treedef=None):
    """
    Split key for a key for every leaf.

    Args:
        rng_key (jax.Array): A JAX PRNG key.
        target (PyTree, optional): A pytree to infer the tree structure from.
                                   Required if `treedef` is not provided.
        treedef (TreeDef, optional): An explicit tree structure. If provided, `target` is ignored.

    Returns:
        PyTree: A pytree of PRNG keys with the same structure as `target` or `treedef`.
    """
    if treedef is None:
        treedef = jax.tree.structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)
