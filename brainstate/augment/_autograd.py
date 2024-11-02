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

"""
Gradient transformations are relatively simple compared to ``vmap`` or ``pmap`` augmentations.
This is because the gradient transformations are not using the Jaxpr, instead, most of them are
computed in the Python level. However, there is an exception, the ``checkpoint`` transformation,
which has been moved into the ``compile`` module.

The wrapped gradient transformations here are made possible by using the following ideas:
1. All the states to compute the gradients should be known before the transformation.
   There must be provided through the ``grad_states`` argument in any of the gradient transformations.
2. The states that have been written in the function should be collected and updated after the function call.
   We record these states during the function call and updated them after the function call.

"""

from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Union, Callable, Dict, Sequence, Optional, Any, Tuple, TypeVar

import jax
from jax import numpy as jnp
from jax._src.api import _vjp
from jax.api_util import argnums_partial
from jax.extend import linear_util

from brainstate._state import State, StateTraceStack
from brainstate._utils import set_module_as
from brainstate.typing import PyTree, Missing

__all__ = [
    'vector_grad', 'grad', 'jacrev', 'jacfwd', 'jacobian', 'hessian',
]

A = TypeVar('A')
Gradient = PyTree
LossValue = PyTree
AuxData = PyTree


def _isgeneratorfunction(fun):
    # re-implemented here because of https://bugs.python.org/issue33261
    while inspect.ismethod(fun):
        fun = fun.__func__
    while isinstance(fun, partial):
        fun = fun.func
    return inspect.isfunction(fun) and bool(fun.__code__.co_flags & inspect.CO_GENERATOR)


def _check_callable(fun):
    # In Python 3.10+, the only thing stopping us from supporting staticmethods
    # is that we can't take weak references to them, which the C++ JIT requires.
    if isinstance(fun, staticmethod):
        raise TypeError(f"staticmethod arguments are not supported, got {fun}")
    if not callable(fun):
        raise TypeError(f"Expected a callable value, got {fun}")
    if _isgeneratorfunction(fun):
        raise TypeError(f"Expected a function, got a generator function: {fun}")


def functional_vector_grad(func, argnums=0, return_value: bool = False, has_aux: bool = False):
    """
     Compute the gradient of a vector with respect to the input.
     """
    _check_callable(func)

    @wraps(func)
    def grad_fun(*args, **kwargs):
        f = linear_util.wrap_init(func, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
        if has_aux:
            y, vjp_fn, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        else:
            y, vjp_fn = _vjp(f_partial, *dyn_args, has_aux=False)
        leaves, tree = jax.tree.flatten(y)
        tangents = jax.tree.unflatten(tree, [jnp.ones(l.shape, dtype=l.dtype) for l in leaves])
        grads = vjp_fn(tangents)
        if isinstance(argnums, int):
            grads = grads[0]
        if has_aux:
            return (grads, y, aux) if return_value else (grads, aux)
        else:
            return (grads, y) if return_value else grads

    return grad_fun


def _jacrev(fun, argnums=0, holomorphic=False, allow_int=False, has_aux=False, return_value=False):
    @wraps(fun)
    def fun_wrapped(*args, **kwargs):
        if has_aux:
            y, aux = fun(*args, **kwargs)
            if return_value:
                return y, (y, aux)
            else:
                return y, aux
        else:
            y = fun(*args, **kwargs)
            if return_value:
                return y, y
            else:
                return y, None

    transform = jax.jacrev(fun_wrapped, argnums=argnums, holomorphic=holomorphic, allow_int=allow_int, has_aux=True)

    @wraps(fun)
    def jacfun(*args, **kwargs):
        jac, aux = transform(*args, **kwargs)
        if return_value:
            return (jac, aux[0], aux[1]) if has_aux else (jac, aux)
        else:
            return (jac, aux) if has_aux else jac

    return jacfun


def _jacfwd(fun, argnums=0, holomorphic=False, has_aux=False, return_value=False):
    @wraps(fun)
    def fun_wrapped(*args, **kwargs):
        if has_aux:
            y, aux = fun(*args, **kwargs)
            if return_value:
                return y, (y, aux)
            else:
                return y, aux
        else:
            y = fun(*args, **kwargs)
            if return_value:
                return y, y
            else:
                return y, None

    transform = jax.jacfwd(fun_wrapped, argnums=argnums, holomorphic=holomorphic, has_aux=True)

    @wraps(fun)
    def jacfun(*args, **kwargs):
        jac, aux = transform(*args, **kwargs)
        if return_value:
            return (jac, aux[0], aux[1]) if has_aux else (jac, aux)
        else:
            return (jac, aux) if has_aux else jac

    return jacfun


class GradientTransform(object):
    """
    Automatic Differentiation Transformations for the ``State`` system.
    """
    __module__ = "brainstate.augment"

    def __init__(
        self,
        target: Callable,
        transform: Callable,
        grad_states: Any,
        argnums: Optional[Union[int, Sequence[int]]],
        return_value: bool,
        has_aux: bool,
        transform_params: Optional[Dict[str, Any]] = None,
    ):
        # gradient variables
        if isinstance(grad_states, dict):
            grad_states = {k: v for k, v in grad_states.items()}
        self._grad_states, self._grad_tree = jax.tree.flatten(grad_states)
        if any(not isinstance(v, State) for v in self._grad_states):
            raise TypeError("All grad_states must be State instances.")

        # parameters
        if argnums is None and len(self._grad_states) == 0:
            argnums = 0
        if argnums is None:
            assert len(self._grad_states) > 0
            _argnums = 0
        elif isinstance(argnums, int):
            _argnums = (0, argnums + 1) if len(self._grad_states) > 0 else (argnums + 1)
        else:
            assert isinstance(argnums, (tuple, list))
            _argnums = tuple(a + 1 for a in argnums)
            if len(self._grad_states) > 0:
                _argnums = (0,) + _argnums
        self._nonvar_argnums = argnums
        self._argnums = _argnums
        self._return_value = return_value
        self._has_aux = has_aux

        # target
        self.target = target

        # transform
        self._states_to_be_written: Tuple[State, ...] = None
        _grad_setting = dict() if transform_params is None else transform_params
        if self._has_aux:
            self._transform = transform(self._fun_with_aux, argnums=self._argnums, has_aux=True, **_grad_setting)
        else:
            self._transform = transform(self._fun_without_aux, argnums=self._argnums, has_aux=True, **_grad_setting)

    def __repr__(self):
        name = self.__class__.__name__
        format_ref = (f'{name}(target={self.target}, \n' +
                      f'{" " * len(name)} num_of_grad_vars={len(self._grad_states)}, \n'
                      f'{" " * len(name)} num_of_dyn_vars={len(self._states_to_be_written)})')
        return format_ref

    def __call_target(self, *args, **kwargs):
        if self._states_to_be_written is None:
            with StateTraceStack() as stack:
                output = self.target(*args, **kwargs)
                grad_ids = set([id(v) for v in self._grad_states])
                self._states_to_be_written = [st for st in stack.get_write_states() if id(st) not in grad_ids]
        else:
            output = self.target(*args, **kwargs)
        return output

    def _fun_with_aux(self, grad_values: tuple, *args, **kwargs):
        for v, d in zip(self._grad_states, grad_values):
            v._value = d
        # Users should return the auxiliary data like::
        # >>> # 1. example of return one data
        # >>> return scalar_loss, data
        # >>> # 2. example of return multiple data
        # >>> return scalar_loss, (data1, data2, ...)
        outs = self.__call_target(*args, **kwargs)
        # outputs: [0] is the value for gradient,
        #          [1] is other values for return
        assert self._states_to_be_written is not None, "The states to be written should be collected."
        return outs[0], (outs, [v.value for v in self._grad_states], [v.value for v in self._states_to_be_written])

    def _fun_without_aux(self, grad_values: tuple, *args, **kwargs):
        for v, d in zip(self._grad_states, grad_values):
            v._value = d
        # Users should return the scalar value like this::
        # >>> return scalar_loss
        out = self.__call_target(*args, **kwargs)
        assert self._states_to_be_written is not None, "The states to be written should be collected."
        return out, (out, [v.value for v in self._grad_states], [v.value for v in self._states_to_be_written])

    def __return(self, rets):
        grads, (outputs, new_grad_vals, new_dyn_vals) = rets
        for i, val in enumerate(new_grad_vals):
            self._grad_states[i].value = val
        for i, val in enumerate(new_dyn_vals):
            self._states_to_be_written[i].value = val

        # check returned grads
        if len(self._grad_states) > 0:
            if self._nonvar_argnums is None:
                grads = self._grad_tree.unflatten(grads)
            else:
                var_grads = self._grad_tree.unflatten(grads[0])
                arg_grads = grads[1] if isinstance(self._nonvar_argnums, int) else grads[1:]
                grads = (var_grads, arg_grads)

        # check returned value
        if self._return_value:
            # check aux
            if self._has_aux:
                return grads, outputs[0], outputs[1]
            else:
                return grads, outputs
        else:
            # check aux
            if self._has_aux:
                return grads, outputs[1]
            else:
                return grads

    def __call__(
        self, *args, **kwargs
    ) -> Gradient | Tuple[Gradient, LossValue] | Tuple[Gradient, AuxData] | Tuple[Gradient, LossValue, AuxData]:
        rets = self._transform([v.value for v in self._grad_states], *args, **kwargs)
        return self.__return(rets)


@set_module_as("brainstate.augment")
def grad(
    fun: Callable = Missing(),
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    holomorphic: Optional[bool] = False,
    allow_int: Optional[bool] = False,
    reduce_axes: Optional[Sequence[str]] = (),
    has_aux: Optional[bool] = None,
    return_value: Optional[bool] = False,
) -> GradientTransform | Callable[[Callable], GradientTransform]:
    """
    Compute the gradient of a scalar-valued function with respect to its arguments.

    Args:
      fun: callable. the scalar-valued function to be differentiated.
      reduce_axes: (Sequence[str]) optional. Specifies the axes to reduce over when
        differentiating with respect to array-valued arguments. The default, (),
        means to differentiate each element of the output with respect to each
        element of the argument. If the argument is an array, this argument controls
        how many axes the output of grad has.
      allow_int: (bool) optional. Whether to allow differentiating with respect to
        integer valued inputs. The gradient of an integer input will have a trivial
        vector-space dtype (float0). Default False.
      holomorphic: (bool) optional. Whether fun is promised to be holomorphic.
        Default False.
      grad_states: (State, Sequence[State], Dict[str, State]) optional. The variables
        in fun to take their gradients.
      fun: the scalar-valued function to be differentiated.
      argnums: (int or tuple of ints) optional. Specifies which positional
        argument(s) to differentiate with respect to.
      has_aux: (bool) optional. Indicates whether fun returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      return_value: (bool) optional. Indicates whether to return the value of the
        function along with the gradient. Default False.

    Returns:
      A function which computes the gradient of fun. The function takes the same
      arguments as `fun`, but returns the gradient instead. If `has_aux` is True,
      the function returns a pair where the first element is the gradient and the
      second element is the auxiliary data. If `return_value` is True, the function
      returns a pair where the first element is the gradient and the second element
      is the value of the function.

    """
    if isinstance(fun, Missing):
        def transform(fun) -> GradientTransform:
            return GradientTransform(target=fun,
                                     transform=jax.grad,
                                     grad_states=grad_states,
                                     argnums=argnums,
                                     return_value=return_value,
                                     has_aux=False if has_aux is None else has_aux,
                                     transform_params=dict(holomorphic=holomorphic,
                                                           allow_int=allow_int,
                                                           reduce_axes=reduce_axes))

        return transform

    return GradientTransform(target=fun,
                             transform=jax.grad,
                             grad_states=grad_states,
                             argnums=argnums,
                             return_value=return_value,
                             has_aux=False if has_aux is None else has_aux,
                             transform_params=dict(holomorphic=holomorphic,
                                                   allow_int=allow_int,
                                                   reduce_axes=reduce_axes))


@set_module_as("brainstate.augment")
def vector_grad(
    func: Callable = Missing(),
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    return_value: bool = False,
    has_aux: Optional[bool] = None,
) -> GradientTransform | Callable[[Callable], GradientTransform]:
    """Take vector-valued gradients for function ``func``.

    Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_,
    `brainpy.math.jacrev <./brainpy.math.autograd.jacrev.html>`_ and
    `brainpy.math.jacfwd <./brainpy.math.autograd.jacfwd.html>`_,
    the returns in this function are different for different argument settings.

    1. When "grad_states" is None
      - "has_aux=False" + "return_value=False" => ``arg_grads``.
      - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
    2. When "grad_states" is not None and "argnums" is None
      - "has_aux=False" + "return_value=False" => ``var_grads``.
      - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
    3. When "grad_states" is not None and "argnums" is not None
      - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
      - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
      - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
      - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.


    Parameters
    ----------
    func: Callable
      Function whose gradient is to be computed.
    grad_states : optional, ArrayType, sequence of ArrayType, dict
      The variables in ``func`` to take their gradients.
    has_aux: optional, bool
      Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    return_value : bool
      Whether return the loss value.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).

    Returns
    -------
    func : GradientTransform
      The vector gradient function.
    """

    if isinstance(func, Missing):
        def transform(fun) -> GradientTransform:
            return GradientTransform(target=fun,
                                     transform=functional_vector_grad,
                                     grad_states=grad_states,
                                     argnums=argnums,
                                     return_value=return_value,
                                     has_aux=False if has_aux is None else has_aux)

        return transform

    else:
        return GradientTransform(target=func,
                                 transform=functional_vector_grad,
                                 grad_states=grad_states,
                                 argnums=argnums,
                                 return_value=return_value,
                                 has_aux=False if has_aux is None else has_aux)


@set_module_as("brainstate.augment")
def jacrev(
    fun: Callable,
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> GradientTransform:
    """
    Extending automatic Jacobian (reverse-mode) of ``func`` to classes.

    This function extends the JAX official ``jacrev`` to make automatic jacobian
    computation on functions and class functions. Moreover, it supports returning
    value ("return_value") and returning auxiliary data ("has_aux").

    Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_, the returns are
    different for different argument settings in ``brainpy.math.jacrev``.

    1. When "grad_states" is None
      - "has_aux=False" + "return_value=False" => ``arg_grads``.
      - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
    2. When "grad_states" is not None and "argnums" is None
      - "has_aux=False" + "return_value=False" => ``var_grads``.
      - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
    3. When "grad_states" is not None and "argnums" is not None
      - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
      - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
      - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
      - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.

    Parameters
    ----------
    fun: Function whose Jacobian is to be computed.
    grad_states : optional, ArrayType, sequence of ArrayType, dict
      The variables in ``func`` to take their gradients.
    has_aux: optional, bool
      Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    return_value : bool
      Whether return the loss value.
    argnums: Optional, integer or sequence of integers.
      Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool.
      Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool.
      Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

    Returns
    -------
    fun: GradientTransform
      The transformed object.
    """
    return GradientTransform(target=fun,
                             transform=_jacrev,
                             grad_states=grad_states,
                             argnums=argnums,
                             return_value=return_value,
                             has_aux=False if has_aux is None else has_aux,
                             transform_params=dict(holomorphic=holomorphic,
                                                   allow_int=allow_int))


jacobian = jacrev


@set_module_as("brainstate.augment")
def jacfwd(
    func: Callable,
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: bool = False,
    holomorphic: bool = False,
) -> GradientTransform:
    """Extending automatic Jacobian (forward-mode) of ``func`` to classes.

    This function extends the JAX official ``jacfwd`` to make automatic jacobian
    computation on functions and class functions. Moreover, it supports returning
    value ("return_value") and returning auxiliary data ("has_aux").

    Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_, the returns are
    different for different argument settings in ``brainpy.math.jacfwd``.

    1. When "grad_states" is None
      - "has_aux=False" + "return_value=False" => ``arg_grads``.
      - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
    2. When "grad_states" is not None and "argnums" is None
      - "has_aux=False" + "return_value=False" => ``var_grads``.
      - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
    3. When "grad_states" is not None and "argnums" is not None
      - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
      - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
      - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
      - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.

    Parameters
    ----------
    func: Function whose Jacobian is to be computed.
    grad_states : optional, ArrayType, sequence of ArrayType, dict
      The variables in ``func`` to take their gradients.
    has_aux: optional, bool
      Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    return_value : bool
      Whether return the loss value.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

    Returns
    -------
    obj: GradientTransform
      The transformed object.
    """

    return GradientTransform(target=func,
                             transform=_jacfwd,
                             grad_states=grad_states,
                             argnums=argnums,
                             return_value=return_value,
                             has_aux=False if has_aux is None else has_aux,
                             transform_params=dict(holomorphic=holomorphic))


@set_module_as("brainstate.augment")
def hessian(
    func: Callable,
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: bool = False,
    return_value: bool = False,
    holomorphic: bool = False,
) -> GradientTransform:
    """
    Hessian of ``func`` as a dense array.

    Parameters
    ----------
    func : callable
      Function whose Hessian is to be computed.  Its arguments at positions
      specified by ``argnums`` should be arrays, scalars, or standard Python
      containers thereof. It should return arrays, scalars, or standard Python
      containers thereof.
    grad_states : optional, ArrayCollector, sequence of ArrayType
      The variables required to compute their gradients.
    argnums: Optional, integer or sequence of integers
      Specifies which positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic : bool
      Indicates whether ``fun`` is promised to be holomorphic. Default False.
    return_value : bool
      Whether return the hessian values.

    Returns
    -------
    obj: ObjectTransform
      The transformed object.
    """
    return GradientTransform(target=func,
                             transform=jax.hessian,
                             grad_states=grad_states,
                             argnums=argnums,
                             return_value=return_value,
                             has_aux=False if has_aux is None else has_aux,
                             transform_params=dict(holomorphic=holomorphic))
