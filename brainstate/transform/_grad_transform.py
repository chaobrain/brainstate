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

from typing import Union, Callable, Dict, Sequence, Optional, Any, Tuple, TypeVar, Iterator

import jax

from brainstate._state import State
from brainstate.transform._make_jaxpr import StatefulFunction
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
                err = f"Some states are not found in the state trace when performing gradient transformations.\n "
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

        # TODO: support jax.disable_jit()

        # compute the model
        self.stateful_target.make_jaxpr(*args, **kwargs)
        cache = self.stateful_target.get_arg_cache_key(*args, **kwargs)

        # apply the gradient transformation
        state_trace = self.stateful_target.get_state_trace_by_cache(cache)
        read_state_vals = state_trace.get_read_state_values(True)
        rets = self._transform(*self._split_state_vals(state_trace), *args, **kwargs)

        # analyze and return the results
        res = self._return(rets, read_state_vals, state_trace)
        return res
