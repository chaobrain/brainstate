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

from functools import partial
from typing import Tuple, Dict, NamedTuple, Sequence, Any, Callable, Hashable, List

import jax
from jax._src.core import ClosedJaxpr

from brainstate._compatible_import import Jaxpr, Var
from brainstate._state import State
from brainstate.transform._ir_inline import inline_jit
from brainstate.transform._make_jaxpr import StatefulFunction, get_arg_cache_key
from ._compiler import compile
from ._data import CompiledGraph, Group, Projection, Input, Output
from ._utils import _is_connection

__all__ = [
    'ParsedResults',
    'parse',
]


class ParsedResults(NamedTuple):
    static_argnames: Sequence
    static_argnums: Sequence
    cache_fn: Callable
    cache_key: Hashable
    out_treedef: Any
    jaxpr: Jaxpr
    in_states: Sequence[State]
    out_states: Sequence[State]
    invar_to_state: Dict[Var, State]
    outvar_to_state: Dict[Var, State]
    state_to_invars: Dict[State, Sequence[Var]]
    state_to_outvars: Dict[State, Sequence[Var]]
    compiled: CompiledGraph

    def run(self, *args, mode: str = 'compiled', **kwargs) -> Any:
        if mode == 'compiled':
            return self.run_compiled_graph(*args, **kwargs)

        elif mode == 'original':
            return self.run_original_jaxpr(*args, **kwargs)

        elif mode == 'debug':
            result = self.run_original_jaxpr(*args, **kwargs)
            compiled = self.run_compiled_graph(*args, **kwargs)
            return result, compiled

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run_original_jaxpr(self, *args, **kwargs) -> Any:
        return self._run_impl(lambda *data: jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *data), *args, **kwargs)

    def run_compiled_graph(self, *args, **kwargs) -> Any:
        return self._run_impl(self._run_graph, *args, **kwargs)

    def _run_impl(self, impl, *args, **kwargs) -> Any:
        # data check
        if self.cache_fn(*args, **kwargs) != self.cache_key:
            raise ValueError("Cache key mismatch. The function has been called with different arguments.")

        # inputs
        in_state_val = [st.value for st in self.in_states]
        kwargs = {k: v for k, v in kwargs.items() if k not in self.static_argnames}  # remove static kwargs
        args = tuple(args[i] for i in range(len(args)) if i not in self.static_argnums)
        args = jax.tree.flatten((args, kwargs, in_state_val))[0]

        # run jaxpr
        jaxpr_outs = impl(*args)

        # outputs
        out, new_state_vals = self.out_treedef.unflatten(jaxpr_outs)
        if len(new_state_vals) != len(self.out_states):
            raise ValueError(f'State length mismatch in output: expected '
                             f'{len(self.out_states)} states, got {len(new_state_vals)}')
        for st, val in zip(self.out_states, new_state_vals):
            st.restore_value(val)
        return out

    def _run_graph(self, *args) -> Any:
        """
        Run the network using the compiled graph structure.

        This executes components in call_graph, maintaining a variable environment.
        """
        # Build variable environment: Var -> value mapping
        var_env = {}

        # Step 1: Initialize environment with input arguments
        self._initialize_var_env(var_env, args)

        # Step 2: Execute components in call_graph
        for component in self.compiled.call_graph:
            if isinstance(component, Input):
                self._execute_input(component, var_env)
            elif isinstance(component, Group):
                self._execute_group(component, var_env)
            elif isinstance(component, Projection):
                self._execute_projection(component, var_env)
            elif isinstance(component, Output):
                self._execute_output(component, var_env)

        # Step 3: Collect outputs from environment
        outputs = self._collect_outputs(var_env)

        return outputs

    def _initialize_var_env(self, var_env: Dict[Var, Any], args: Tuple) -> None:
        """Initialize variable environment with input arguments and state values."""
        # Map to jaxpr invars
        assert len(args) == len(self.jaxpr.jaxpr.invars), (
            f"Argument count mismatch: expected {len(self.jaxpr.jaxpr.invars)}, got {len(args)}"
        )
        for var, val in zip(self.jaxpr.jaxpr.invars, args):
            var_env[var] = val

    def _execute_input(self, input_comp: Input, var_env: Dict[Var, Any]) -> None:
        """Execute an Input component."""
        # Gather input values from environment
        input_vals = [var_env[var] for var in input_comp.jaxpr.jaxpr.invars]

        # Execute the input jaxpr
        results = jax.core.eval_jaxpr(input_comp.jaxpr.jaxpr, input_comp.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(input_comp.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _execute_group(self, group: Group, var_env: Dict[Var, Any]) -> None:
        """Execute a Group component."""
        # Gather input values from environment
        input_vals = []
        for var in group.jaxpr.jaxpr.invars:
            if var not in var_env:
                raise RuntimeError(
                    f"Variable {var} not found in environment when executing {group.name}"
                )
            input_vals.append(var_env[var])

        # Execute the group jaxpr
        results = jax.core.eval_jaxpr(group.jaxpr.jaxpr, group.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(group.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _execute_projection(self, projection: Projection, var_env: Dict[Var, Any]) -> None:
        """Execute a Projection component."""
        # Gather input values from environment
        input_vals = []
        for var in projection.jaxpr.jaxpr.invars:
            if var in var_env:
                input_vals.append(var_env[var])
            else:
                # This might be a constvar, check in the original jaxpr
                if var in self.jaxpr.jaxpr.constvars:
                    # Find the index and use the corresponding const
                    idx = self.jaxpr.jaxpr.constvars.index(var)
                    input_vals.append(self.jaxpr.consts[idx])
                else:
                    raise RuntimeError(f"Variable {var} not found in environment or constvars")

        # Execute the projection jaxpr
        results = jax.core.eval_jaxpr(projection.jaxpr.jaxpr, projection.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(projection.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _execute_output(self, output: Output, var_env: Dict[Var, Any]) -> None:
        """Execute an Output component."""
        # Gather input values from environment
        input_vals = [var_env[var] for var in output.jaxpr.jaxpr.invars]

        # Execute the output jaxpr
        results = jax.core.eval_jaxpr(output.jaxpr.jaxpr, output.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(output.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _collect_outputs(self, var_env: Dict[Var, Any]) -> Any:
        """Collect output values from the variable environment."""
        output_vals = []
        for var in self.jaxpr.jaxpr.outvars:
            if var not in var_env:
                raise RuntimeError(f"Output variable {var} not found in environment")
            output_vals.append(var_env[var])

        # Return single value or tuple based on output count
        if len(output_vals) == 1:
            return output_vals[0]
        else:
            return tuple(output_vals)


def parse(
    stateful_fn: StatefulFunction,
    jit_inline: bool = True,
) -> Callable[..., ParsedResults]:
    assert isinstance(stateful_fn, StatefulFunction), "stateful_fn must be an instance of StatefulFunction"
    assert stateful_fn.return_only_write, (
        "Parser currently only supports stateful functions that return only write states. "
    )

    def call(*args, **kwargs):
        # jaxpr
        jaxpr = stateful_fn.get_jaxpr(*args, **kwargs)
        if jit_inline:
            jaxpr = inline_jit(jaxpr, _is_connection)

        # Build state mappings
        in_states = stateful_fn.get_states(*args, **kwargs)
        out_states = stateful_fn.get_write_states(*args, **kwargs)
        state_mapping = _build_state_mapping(jaxpr, in_states, out_states)

        # Compile the SNN
        compiled_snn = compile(
            closed_jaxpr=jaxpr,
            in_states=in_states,
            out_states=out_states,
            invar_to_state=state_mapping['invar_to_state'],
            outvar_to_state=state_mapping['outvar_to_state'],
            state_to_invars=state_mapping['state_to_invars'],
            state_to_outvars=state_mapping['state_to_outvars'],
        )
        cache_fn = partial(get_arg_cache_key, stateful_fn.static_argnums, stateful_fn.static_argnames)
        cache_key = stateful_fn.get_arg_cache_key(*args, **kwargs)

        return ParsedResults(
            static_argnums=stateful_fn.static_argnums,
            static_argnames=stateful_fn.static_argnames,
            out_treedef=stateful_fn.get_out_treedef_by_cache(cache_key),
            cache_fn=cache_fn,
            cache_key=cache_key,
            jaxpr=jaxpr,
            in_states=in_states,
            out_states=out_states,
            invar_to_state=state_mapping['invar_to_state'],
            outvar_to_state=state_mapping['outvar_to_state'],
            state_to_invars=state_mapping['state_to_invars'],
            state_to_outvars=state_mapping['state_to_outvars'],
            compiled=compiled_snn,
        )

    return call


def _build_state_mapping(
    closed_jaxpr: ClosedJaxpr,
    in_states: List[State],
    out_states: List[State],
) -> Dict:
    # --- validations ---
    if not isinstance(closed_jaxpr, ClosedJaxpr):
        raise TypeError(f"closed_jaxpr must be a ClosedJaxpr, got {type(closed_jaxpr)}")

    if not all(isinstance(s, State) for s in in_states):
        bad = [type(s) for s in in_states if not isinstance(s, State)]
        raise TypeError(f"in_states must contain only State instances, got {bad}")

    if not all(isinstance(s, State) for s in out_states):
        bad = [type(s) for s in out_states if not isinstance(s, State)]
        raise TypeError(f"out_states must contain only State instances, got {bad}")

    missing_out = [s for s in out_states if s not in in_states]
    if missing_out:
        raise ValueError(
            f"All out_states must be present in in_states. Missing: {[repr(s) for s in missing_out]}"
        )

    # empty initialization
    invar_to_state = dict()
    state_to_invars = dict()
    outvar_to_state = dict()
    state_to_outvars = dict()

    # Extract the actual jaxpr from ClosedJaxpr
    jaxpr = closed_jaxpr.jaxpr

    # input states <---> input variables #
    # ---------------------------------- #

    # Get state structure information
    in_state_vals = [state.value for state in in_states]
    in_state_avals, in_state_tree = jax.tree.flatten(in_state_vals)
    n_inp_before_states = len(jaxpr.invars) - len(in_state_avals)

    # Map state tree to invars and outvars
    # Input variables: the last len(state_avals) invars correspond to states
    state_tree_invars = jax.tree.unflatten(in_state_tree, jaxpr.invars[n_inp_before_states:])

    # Build mappings using the tree structure
    # This ensures proper correspondence between states and their JAXpr variables
    assert len(in_states) == len(state_tree_invars), "Mismatch between number of input states and state tree invars"
    for state, invar in zip(in_states, state_tree_invars):
        # Always flatten the tree structure to get individual variables
        invar_leaves = jax.tree.leaves(invar)

        # Store the relationships
        for var in invar_leaves:
            invar_to_state[var] = state

        # Store the reverse mappings
        state_to_invars[state] = invar_leaves

    # output states <---> output variables #
    # ------------------------------------ #

    # Get state structure information
    out_state_vals = [state.value for state in out_states]
    out_state_avals, out_state_tree = jax.tree.flatten(out_state_vals)
    n_out_before_states = len(jaxpr.outvars) - len(out_state_avals)

    # Output variables: after the main outputs, the rest correspond to state updates
    state_tree_outvars = jax.tree.unflatten(out_state_tree, jaxpr.outvars[n_out_before_states:])
    assert len(out_states) == len(state_tree_outvars), \
        'Mismatch between number of output states and state tree outvars'

    # Build mappings using the tree structure
    # This ensures proper correspondence between states and their JAXpr variables
    for state, outvar in zip(out_states, state_tree_outvars):
        # Always flatten the tree structure to get individual variables
        outvar_leaves = jax.tree.leaves(outvar)

        # Store the relationships
        for var in outvar_leaves:
            outvar_to_state[var] = state
        state_to_outvars[state] = outvar_leaves

    return {
        'invar_to_state': invar_to_state,
        'state_to_invars': state_to_invars,
        'outvar_to_state': outvar_to_state,
        'state_to_outvars': state_to_outvars,
        'in_states': in_states,
        'out_states': out_states,
        'hidden_states': [s for s in out_states],
    }
