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

from functools import wraps
from typing import Sequence, Tuple

from brainstate._state import StateTraceStack
from brainstate.typing import PyTree
from ._make_jaxpr import StatefulFunction


def write_back_state_values(
    state_trace: StateTraceStack,
    read_state_vals: Sequence[PyTree],
    write_state_vals: Sequence[PyTree],
):
    assert len(state_trace.states) == len(state_trace.been_writen) == len(read_state_vals) == len(write_state_vals)
    for st, write, val_r, val_w in zip(state_trace.states, state_trace.been_writen, read_state_vals, write_state_vals):
        if write:
            st.value = val_w
        else:
            st.restore_value(val_r)


def wrap_single_fun_in_multi_branches(
    stateful_fun: StatefulFunction,
    merged_state_trace: StateTraceStack,
    read_state_vals: Sequence[PyTree | None],
    return_states: bool = True
):
    state_ids_belong_to_this_fun = {id(st): st for st in stateful_fun.get_states()}

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
            new_state_vals = {id(st): val for st, val in zip(stateful_fun.get_states(), new_state_vals)}
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
    return_states: bool = True
):
    state_ids_belong_to_this_fun = {id(st): st for st in stateful_fun.get_states()}

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
            new_state_vals = {id(st): val for st, val in zip(stateful_fun.get_states(), new_state_vals)}
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
    been_writen: Tuple[bool],
    read_state_vals: Tuple[PyTree | None],
):
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
