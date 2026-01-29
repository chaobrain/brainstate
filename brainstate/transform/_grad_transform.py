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
from typing import Union, Callable, Dict, Sequence, Optional, Any, Tuple, TypeVar, Iterator, List

import jax
import jax.numpy as jnp

from brainstate._compatible_import import DropVar, Literal
from brainstate._state import State
from brainstate.transform._make_jaxpr import StatefulFunction, make_jaxpr
from brainstate.transform._unvmap import unvmap
from brainstate.typing import PyTree
from brainstate.util import PrettyType, PrettyAttr, PrettyRepr

__all__ = [
    'GradientTransform',
]

A = TypeVar('A')
Gradient = PyTree
LossValue = PyTree
AuxData = PyTree
TransformFn = Callable


def _check_for_nan(x) -> Tuple[bool, int, Optional[Any]]:
    """
    Check if an array contains NaN or Inf values.

    Parameters
    ----------
    x : array-like
        The array to check for NaN/Inf values.

    Returns
    -------
    tuple
        A tuple of (has_bad, bad_count, bad_indices) where:
        - has_bad: bool indicating if any NaN/Inf values exist
        - bad_count: number of NaN/Inf values found
        - bad_indices: indices of NaN/Inf values (None if none found)
    """
    if not hasattr(x, 'dtype'):
        return False, 0, None
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return False, 0, None
    # Check for both NaN and Inf
    bad_mask = jnp.isnan(x) | jnp.isinf(x)
    has_bad = bool(jnp.any(bad_mask))
    if has_bad:
        bad_count = int(jnp.sum(bad_mask))
        # Handle scalar arrays (0d arrays)
        if x.ndim == 0:
            bad_indices = ()
        else:
            bad_indices = jnp.where(bad_mask)
        return True, bad_count, bad_indices
    return False, 0, None


def _check_pytree_for_nan(pytree, name: str = "") -> Tuple[bool, List[Dict]]:
    """
    Check an entire pytree for NaN values.

    Parameters
    ----------
    pytree : PyTree
        The pytree to check for NaN values.
    name : str, optional
        A name to identify the pytree in reports.

    Returns
    -------
    tuple
        A tuple of (has_nan, results) where:
        - has_nan: bool indicating if any NaN values exist in the pytree
        - results: list of dicts with details about each leaf containing NaN
    """
    results = []
    leaves = jax.tree.leaves(pytree)
    for i, leaf in enumerate(leaves):
        has_nan, count, indices = _check_for_nan(leaf)
        if has_nan:
            results.append(
                {
                    'leaf_index': i,
                    'nan_count': count,
                    'indices': indices,
                    'shape': getattr(leaf, 'shape', None),
                    'dtype': getattr(leaf, 'dtype', None),
                }
            )
    return len(results) > 0, results


def _eval_jaxpr_with_nan_check(jaxpr, consts, *args) -> Tuple[List, List[Dict], List[str]]:
    """
    Evaluate a jaxpr equation by equation, checking for NaN after each operation.

    This function implements a custom jaxpr interpreter that evaluates each
    primitive operation and checks if NaN values are introduced in the outputs.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to evaluate.
    consts : sequence
        The constant values for the jaxpr.
    *args
        The input arguments for the jaxpr.

    Returns
    -------
    tuple
        A tuple of (outputs, nan_report, all_eqn_strs) where:
        - outputs: list of output values from the jaxpr evaluation
        - nan_report: list of dicts with NaN detection info for each equation
          that first introduced NaN values
        - all_eqn_strs: list of equation strings for all equations
    """
    env = {}
    nan_report = []
    all_eqn_strs = []  # Collect all equation strings for context display

    # Bind constants to their variables
    for var, val in zip(jaxpr.constvars, consts):
        env[var] = val

    # Bind input arguments to their variables
    for var, val in zip(jaxpr.invars, args):
        env[var] = val

    # Evaluate each equation
    for eqn_idx, eqn in enumerate(jaxpr.eqns):
        all_eqn_strs.append(str(eqn))  # Store equation string for context
        # Get input values for this equation
        invals = [env[v] if not isinstance(v, Literal) else v.val for v in eqn.invars]

        # Check inputs for NaN (to track propagation vs. introduction)
        input_has_nan, _ = _check_pytree_for_nan(invals, f"eqn_{eqn_idx}_inputs")

        # Evaluate the primitive
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]

        # Check outputs for NaN
        output_has_nan, output_nan_details = _check_pytree_for_nan(outvals, f"eqn_{eqn_idx}_outputs")

        # If NaN appeared in output but wasn't in input, record it
        if output_has_nan and not input_has_nan:
            nan_report.append({
                'eqn_index': eqn_idx,
                'primitive': eqn.primitive.name,
                'input_shapes': [getattr(v, 'shape', None) for v in invals],
                'output_shapes': [getattr(v, 'shape', None) for v in outvals],
                'input_values': invals,  # Include actual input values for debugging
                'nan_details': output_nan_details,
                'equation_str': str(eqn),
                'source_info': getattr(eqn, 'source_info', None),
            })

        # Store outputs in environment
        for var, val in zip(eqn.outvars, outvals):
            if not isinstance(var, DropVar):
                env[var] = val

    # Get final outputs
    outputs = [env[v] if not isinstance(v, Literal) else v.val for v in jaxpr.outvars]
    return outputs, nan_report, all_eqn_strs


def _format_nan_report(
    nan_report: List[Dict],
    total_eqns: int,
    all_eqn_strs: Optional[List[str]] = None,
    context_window: int = 5
) -> str:
    """
    Format a NaN/Inf report into a human-readable string with context window.

    Parameters
    ----------
    nan_report : list
        List of dicts containing NaN/Inf detection information.
    total_eqns : int
        Total number of equations that were evaluated.
    all_eqn_strs : list of str, optional
        List of all equation strings for context display.
    context_window : int, default 5
        Number of equations to show before and after each NaN/Inf source.

    Returns
    -------
    str
        A formatted string describing where NaN/Inf values were detected.
    """
    if not nan_report:
        return f"No NaN/Inf detected in {total_eqns} equations."

    lines = [f"NaN/Inf detected! Found in {len(nan_report)} equation(s) out of {total_eqns}:"]

    for info in nan_report:
        eqn_idx = info['eqn_index']
        phase = info.get('phase', 'unknown')

        # Show context header
        lines.append(f"\n=== [{phase.upper()}] Context around Equation {eqn_idx} ({info['primitive']}) ===")

        # Show context window if equation strings are available
        if all_eqn_strs:
            start_idx = max(0, eqn_idx - context_window)
            end_idx = min(total_eqns, eqn_idx + context_window + 1)

            for i in range(start_idx, end_idx):
                marker = "  <-- NaN/Inf introduced here" if i == eqn_idx else ""
                lines.append(f"  Equation {i}: {all_eqn_strs[i]}{marker}")

        # Show details about the NaN/Inf source
        lines.append(f"\n  Primitive: {info['primitive']}")
        lines.append(f"  Input shapes: {info['input_shapes']}")
        lines.append(f"  Output shapes: {info['output_shapes']}")

        # Format source info nicely if available
        source_info = info.get('source_info')
        if source_info is not None:
            try:
                # Try to get traceback from source_info
                if hasattr(source_info, 'traceback') and source_info.traceback:
                    tb = source_info.traceback()
                    if tb:
                        lines.append(f"  Source: {tb[-1] if tb else 'unknown'}")
            except Exception:
                pass  # Skip if source info extraction fails

        # Show input values that led to NaN/Inf (truncated for large arrays)
        for i, (shape, val) in enumerate(zip(info['input_shapes'], info['input_values'])):
            if hasattr(val, 'size') and val.size <= 10:
                lines.append(f"  Input {i} value: {val}")
            elif hasattr(val, 'size'):
                lines.append(
                    f"  Input {i} value (truncated): shape={shape}, min={float(jnp.min(val)):.4g}, max={float(jnp.max(val)):.4g}")

    return '\n'.join(lines)


# =============================================================================
# JIT-compatible NaN/Inf detection functions
# =============================================================================

def _check_nan_jit_compatible(values) -> jax.Array:
    """
    Check for NaN/Inf in a pytree, JIT-compatible.

    Parameters
    ----------
    values : PyTree
        The pytree of values to check for NaN/Inf.

    Returns
    -------
    jax.Array
        A scalar boolean indicating if any NaN/Inf values exist.
    """
    leaves = jax.tree.leaves(values)
    has_bad = jnp.array(False)
    for leaf in leaves:
        if hasattr(leaf, 'dtype') and jnp.issubdtype(leaf.dtype, jnp.floating):
            has_bad = has_bad | jnp.any(jnp.isnan(leaf) | jnp.isinf(leaf))
    return has_bad


def _nan_error_callback(name, jaxpr_info=None):
    """
    Callback to raise error with detailed NaN/Inf analysis.

    This function is called via jax.debug.callback when NaN/Inf is detected.
    If jaxpr_info is provided, it performs detailed equation-by-equation analysis.

    Parameters
    ----------
    name : str
        A descriptive name for the computation (e.g., "gradient computation").
    jaxpr_info : dict, optional
        Contains {'jaxpr': jaxpr, 'consts': consts, 'flat_args': flat_args}
        for detailed equation-by-equation analysis.
    """
    # Try detailed jaxpr analysis if info is available
    jaxpr = jaxpr_info['jaxpr']
    consts = jaxpr_info['consts']
    flat_args = jaxpr_info['flat_args']

    outputs, nan_report, all_eqn_strs = _eval_jaxpr_with_nan_check(jaxpr, consts, *flat_args)

    if nan_report:
        for item in nan_report:
            item['phase'] = 'gradient'
        report = _format_nan_report(nan_report, len(jaxpr.eqns), all_eqn_strs)
        raise RuntimeError(f"NaN/Inf detected in {name}:\n{report}")


class GradientTransform(PrettyRepr):
    """
    Automatic Differentiation Transformations for the ``State`` system.

    This class implements gradient transformations for functions that operate on State objects.
    It allows for flexible configuration of gradient computation with respect to specified states
    and function arguments.

    Parameters
    ----------
    target : callable
        The function to be transformed.
    transform : callable
        The transformation function to apply.
    grad_states : State, sequence of State, or dict of State, optional
        States to compute gradients for.
    argnums : int or sequence of int, optional
        Indices of arguments to differentiate with respect to.
    return_value : bool, default False
        Whether to return the function's value along with gradients.
    has_aux : bool, default False
        Whether the function returns auxiliary data.
    transform_params : dict, optional
        Additional parameters for the transformation function.
    check_states : bool, default True
        Whether to check that all grad_states are found in the function.
    debug_nan : bool, default False
        Whether to enable NaN debugging. When True, raises RuntimeError with
        detailed diagnostics if NaN is detected during gradient computation.

    Attributes
    ----------
    target : callable
        The function to be transformed.
    stateful_target : StatefulFunction
        A wrapper around the target function for state management.
    raw_argnums : int, sequence of int, or None
        The original argnums specified by the user.
    true_argnums : int or tuple of int
        The adjusted argnums used internally.
    return_value : bool
        Whether to return the function's value along with gradients.
    has_aux : bool
        Whether the function returns auxiliary data.

    Examples
    --------
    Basic gradient computation with states:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create states
        >>> weight = brainstate.State(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        >>> bias = brainstate.State(jnp.array([0.5, -0.5]))
        >>>
        >>> def loss_fn(x):
        ...     y = x @ weight.value + bias.value
        ...     return jnp.sum(y ** 2)
        >>>
        >>> # Create gradient transform
        >>> grad_transform = brainstate.transform.GradientTransform(
        ...     target=loss_fn,
        ...     transform=jax.grad,
        ...     grad_states=[weight, bias]
        ... )
        >>>
        >>> # Compute gradients
        >>> x = jnp.array([1.0, 2.0])
        >>> grads = grad_transform(x)

    With function arguments and auxiliary data:

    .. code-block:: python

        >>> def loss_fn_with_aux(x, scale):
        ...     y = x @ weight.value + bias.value
        ...     loss = jnp.sum((y * scale) ** 2)
        ...     return loss, {"predictions": y, "scale": scale}
        >>>
        >>> grad_transform = brainstate.transform.GradientTransform(
        ...     target=loss_fn_with_aux,
        ...     transform=jax.grad,
        ...     grad_states=[weight, bias],
        ...     argnums=[0, 1],  # gradient w.r.t x and scale
        ...     has_aux=True,
        ...     return_value=True
        ... )
        >>>
        >>> grads, loss_value, aux_data = grad_transform(x, 2.0)
    """

    __module__ = "brainstate.transform"

    def __init__(
        self,
        target: Callable,
        transform: TransformFn,
        grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
        argnums: Optional[Union[int, Sequence[int]]] = None,
        return_value: bool = False,
        has_aux: bool = False,
        transform_params: Optional[Dict[str, Any]] = None,
        check_states: bool = True,
        debug_nan: bool = False,
    ):
        """
        Initialize a ``GradientTransform`` instance.

        Parameters
        ----------
        target : callable
            The function to be transformed.
        transform : callable
            The transformation function to apply.
        grad_states : State, sequence of State, or dict of State, optional
            States to compute gradients for.
        argnums : int or sequence of int, optional
            Indices of arguments to differentiate with respect to.
        return_value : bool, default False
            Whether to return the function's value along with gradients.
        has_aux : bool, default False
            Whether the function returns auxiliary data.
        transform_params : dict, optional
            Additional parameters for the transformation function.
        check_states : bool, default True
            Whether to check that all grad_states are found in the function.
        debug_nan : bool, default False
            Whether to enable NaN debugging. When True, the gradient computation
            is evaluated equation-by-equation, and if NaN is detected, a RuntimeError
            is raised with detailed information about which primitive operation
            first introduced the NaN values.

        Raises
        ------
        TypeError
            If any grad_states are not State instances.
        """
        # gradient variables
        if isinstance(grad_states, dict):
            grad_states = {k: v for k, v in grad_states.items()}
        self._grad_states, self._grad_tree = jax.tree.flatten(grad_states, is_leaf=lambda x: isinstance(x, State))
        self._grad_state_ids = [id(v) for v in self._grad_states]
        self._grad_id_to_state = {id(v): v for v in self._grad_states}
        if any(not isinstance(v, State) for v in self._grad_states):
            raise TypeError("All grad_states must be State instances.")
        self.check_states = check_states

        # parameters
        if argnums is None and len(self._grad_states) == 0:
            argnums = 0
        if argnums is None:
            assert len(self._grad_states) > 0
            _argnums = 0
        elif isinstance(argnums, int):
            _argnums = (0, argnums + 2) if len(self._grad_states) > 0 else (argnums + 2)
        else:
            assert isinstance(argnums, (tuple, list))
            _argnums = tuple(a + 2 for a in argnums)
            if len(self._grad_states) > 0:
                _argnums = (0,) + _argnums
        self.raw_argnums = argnums
        self.true_argnums = _argnums
        self.return_value = return_value
        self.has_aux = has_aux
        self.debug_nan = debug_nan

        # target
        assert callable(target), "The target should be a callable object."
        self.target = target
        self.stateful_target = StatefulFunction(target, name='gradient', return_only_write=True)

        # transform
        grad_setting = dict() if transform_params is None else transform_params
        if self.has_aux:
            self._transform = transform(
                self._fun_with_aux, argnums=self.true_argnums, has_aux=True, **grad_setting
            )
        else:
            self._transform = transform(
                self._fun_without_aux, argnums=self.true_argnums, has_aux=True, **grad_setting
            )

    def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
        yield PrettyType(self.__class__.__name__)
        yield PrettyAttr("target", self.target)
        yield PrettyAttr("grad_states", self._grad_states)
        yield PrettyAttr("grad_tree", self._grad_tree)
        yield PrettyAttr("argnums", self.raw_argnums)
        yield PrettyAttr("return_value", self.return_value)
        yield PrettyAttr("has_aux", self.has_aux)
        yield PrettyAttr("transform", self._transform)

    def _split_state_vals(self, state_trace):
        """
        Split state values into gradient and non-gradient states.

        Args:
            state_trace: The state trace containing all states.

        Returns:
            Tuple[Dict, Dict]: A tuple of dictionaries containing gradient and non-gradient state values.
        """
        grad_vals = dict()
        other_vals = dict()
        all_ids = set(self._grad_state_ids)
        for st in state_trace.states:
            id_ = id(st)
            if id_ in all_ids:
                grad_vals[id_] = st.value
                all_ids.remove(id_)
            else:
                other_vals[id_] = st.value
        if len(all_ids):
            if self.check_states:
                err = (
                    f"Some states are not found in the state trace when performing gradient transformations.\n "
                    f"You can turn off this check by setting `check_states=False` in "
                    f"GradientTransform` initialization. Missing states: "
                )
                for i, id_ in enumerate(all_ids):
                    st = self._grad_id_to_state[id_]
                    st.raise_error_with_source_info(ValueError(err + str(st)))
            else:
                id2state = {id(st): st for st in self._grad_states}
                for id_ in all_ids:
                    grad_vals[id_] = id2state[id_].value

        return grad_vals, other_vals

    def _merge_state_vals(self, grad_vals: Dict, other_vals: Dict, state_trace):
        """
        Merge gradient and non-gradient state values back into a single list.

        Args:
            grad_vals (Dict): Dictionary of gradient state values.
            other_vals (Dict): Dictionary of non-gradient state values.
            state_trace: The state trace containing all states.

        Returns:
            List: A list of merged state values.
        """
        res = []
        for st in state_trace.states:
            id_ = id(st)
            if id_ in self._grad_state_ids:
                res.append(grad_vals[id_])
            else:
                res.append(other_vals[id_])
        return res

    def _call_target(self, grad_vals: Dict, other_vals: Dict, *args, **kwargs):
        """
        Call the target function with the given state values and arguments.

        Args:
            grad_vals (Dict): Dictionary of gradient state values.
            other_vals (Dict): Dictionary of non-gradient state values.
            *args: Positional arguments to pass to the target function.
            **kwargs: Keyword arguments to pass to the target function.

        Returns:
            Tuple: A tuple containing updated state values and the function output.
        """
        state_trace = self.stateful_target.get_state_trace(*args, **kwargs, compile_if_miss=True)
        state_vals = self._merge_state_vals(grad_vals, other_vals, state_trace)
        write_state_vals, out = self.stateful_target.jaxpr_call(state_vals, *args, **kwargs)
        return write_state_vals, out

    def _fun_with_aux(self, grad_vals: Dict, other_vals: Dict, *args, **kwargs):
        """
        Wrapper function for target functions that return auxiliary data.

        Args:
            grad_vals (Dict): Dictionary of gradient state values.
            other_vals (Dict): Dictionary of non-gradient state values.
            *args: Positional arguments to pass to the target function.
            **kwargs: Keyword arguments to pass to the target function.

        Returns:
            Tuple: A tuple containing the primary output and a tuple of (all outputs, updated state values).
        """
        # Users should return the auxiliary data like::
        # >>> # 1. example of return one data
        # >>> return scalar_loss, data
        # >>> # 2. example of return multiple data
        # >>> return scalar_loss, (data1, data2, ...)
        write_state_vals, outs = self._call_target(grad_vals, other_vals, *args, **kwargs)
        assert isinstance(outs, (tuple, list))
        return outs[0], (outs, write_state_vals)

    def _fun_without_aux(self, grad_vals: Dict, other_vals: Dict, *args, **kwargs):
        """
        Wrapper function for target functions that do not return auxiliary data.

        Args:
            grad_vals (Dict): Dictionary of gradient state values.
            other_vals (Dict): Dictionary of non-gradient state values.
            *args: Positional arguments to pass to the target function.
            **kwargs: Keyword arguments to pass to the target function.

        Returns:
            Tuple: A tuple containing the output and a tuple of (output, updated state values).
        """
        write_state_vals, out = self._call_target(grad_vals, other_vals, *args, **kwargs)
        return out, (out, write_state_vals)

    def _return(self, rets, read_state_vals, state_trace):
        """
        Process and format the return values from the gradient computation.

        Args:
            rets: The raw results from the gradient computation.
            state_trace: The state trace containing all states.

        Returns:
            Union[Gradient, Tuple]: The processed gradient results, potentially including function value and/or auxiliary data.
        """
        # unpack the return values
        grads, (outputs, write_state_vals) = rets

        # assign new values to the states
        state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)

        # check returned grads
        if len(self._grad_states) > 0:
            grads_of_states = grads if self.raw_argnums is None else grads[0]
            grads_of_states = [grads_of_states[st_id] for st_id in self._grad_state_ids]
            if self.raw_argnums is None:
                grads = self._grad_tree.unflatten(grads_of_states)
            else:
                var_grads = self._grad_tree.unflatten(grads_of_states)
                arg_grads = grads[1] if isinstance(self.raw_argnums, int) else grads[1:]
                grads = (var_grads, arg_grads)

        # check returned value
        if self.return_value:
            # check aux
            if self.has_aux:
                return grads, outputs[0], outputs[1]
            else:
                return grads, outputs
        else:
            # check aux
            if self.has_aux:
                return grads, outputs[1]
            else:
                return grads

    def __call__(
        self, *args, **kwargs
    ) -> (
        Gradient |
        Tuple[Gradient, LossValue] |
        Tuple[Gradient, AuxData] |
        Tuple[Gradient, LossValue, AuxData]
    ):
        """
        Compute gradients by calling the transformed function.

        Parameters
        ----------
        *args
            Positional arguments to pass to the target function.
        **kwargs
            Keyword arguments to pass to the target function.

        Returns
        -------
        Gradient or tuple
            The computed gradients, potentially including function value and/or auxiliary data.
            The exact return structure depends on the settings of return_value and has_aux.
        """
        # compute the model
        self.stateful_target.make_jaxpr(*args, **kwargs)
        cache = self.stateful_target.get_arg_cache_key(*args, **kwargs)

        # apply the gradient transformation
        state_trace = self.stateful_target.get_state_trace_by_cache(cache)
        read_state_vals = state_trace.get_read_state_values(True)
        grad_vals, other_vals = self._split_state_vals(state_trace)

        # Compute gradients (JIT-compatible)
        rets = self._transform(grad_vals, other_vals, *args, **kwargs)
        grads = rets[0]

        if self.debug_nan:
            # Check for NaN/Inf using JIT-compatible operations
            has_nan = _check_nan_jit_compatible(grads)

            # Capture jaxpr for detailed analysis (jaxpr and consts are static, safe to capture)
            def grad_fn(gv, ov, *a, **kw):
                return self._transform(gv, ov, *a, **kw)

            grad_jaxpr = jax.make_jaxpr(grad_fn)(grad_vals, other_vals, *args, **kwargs)
            # st_fn = StatefulFunction(grad_fn).make_jaxpr(grad_vals, other_vals, *args, **kwargs)
            # grad_jaxpr = st_fn.get_jaxpr(grad_vals, other_vals, *args, **kwargs)

            # Flatten args for passing through the callback
            flat_args, _ = jax.tree.flatten((grad_vals, other_vals, args, kwargs))

            # Create callback that receives flat_args as concrete values
            def error_callback(flat_args_concrete, consts):
                jaxpr_info = {
                    'jaxpr': grad_jaxpr.jaxpr,  # Static, captured in closure
                    'consts': consts,  # Static, captured in closure
                    'flat_args': flat_args_concrete  # Concrete values from callback
                }
                _nan_error_callback("gradient computation", jaxpr_info)

            # Pack grads and flat_args together as operand
            flat_args_unvmapped = jax.tree.map(functools.partial(unvmap, op='none'), flat_args)

            # Use jax.lax.cond - pass flat_args through so they become concrete in callback
            # This is compatible with jax.jit
            jax.lax.cond(
                unvmap(has_nan, op='any'),
                lambda operand: jax.debug.callback(error_callback, *operand),
                lambda operand: None,
                (flat_args_unvmapped, grad_jaxpr.consts),
            )

        # analyze and return the results
        res = self._return(rets, read_state_vals, state_trace)
        return res
