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

from functools import wraps
from typing import Union, Callable, Dict, Sequence, Optional

import brainunit as u
import jax

from brainstate._state import State
from brainstate._utils import set_module_as
from ._grad_transform import GradientTransform

__all__ = [
    'jacrev', 'jacfwd', 'jacobian',
]


def _jacrev(
    fn,
    argnums=0,
    holomorphic=False,
    allow_int=False,
    has_aux=False,
    unit_aware=False,
):
    @wraps(fn)
    def fun_wrapped(*args, **kwargs):
        if has_aux:
            y, aux = fn(*args, **kwargs)
            return y, aux
        else:
            y = fn(*args, **kwargs)
            return y, None

    if unit_aware:
        transform = u.autograd.jacrev(
            fun_wrapped, argnums=argnums, holomorphic=holomorphic,
            allow_int=allow_int, has_aux=True
        )
    else:
        transform = jax.jacrev(
            fun_wrapped, argnums=argnums, holomorphic=holomorphic,
            allow_int=allow_int, has_aux=True
        )

    @wraps(fn)
    def jacfun(*args, **kwargs):
        jac, aux = transform(*args, **kwargs)
        return (jac, aux) if has_aux else jac

    return jacfun


def _jacfwd(
    fn,
    argnums=0,
    holomorphic=False,
    has_aux=False,
    unit_aware=False,
):
    @wraps(fn)
    def fn_wrapped(*args, **kwargs):
        if has_aux:
            y, aux = fn(*args, **kwargs)
            return y, aux
        else:
            y = fn(*args, **kwargs)
            return y, None

    if unit_aware:
        transform = u.autograd.jacfwd(fn_wrapped, argnums=argnums, holomorphic=holomorphic, has_aux=True)
    else:
        transform = jax.jacfwd(fn_wrapped, argnums=argnums, holomorphic=holomorphic, has_aux=True)

    @wraps(fn)
    def jacfun(*args, **kwargs):
        jac, aux = transform(*args, **kwargs)
        return (jac, aux) if has_aux else jac

    return jacfun


@set_module_as("brainstate.transform")
def jacrev(
    fun: Callable,
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    unit_aware: bool = False,
    check_states: bool = True,
) -> GradientTransform:
    """
    Extending automatic Jacobian (reverse-mode) of ``func`` to classes.

    This function extends the JAX official ``jacrev`` to make automatic jacobian
    computation on functions and class functions. Moreover, it supports returning
    value ("return_value") and returning auxiliary data ("has_aux").


    1. When ``grad_states`` is None

        - ``has_aux=False`` + ``return_value=False`` => ``arg_grads``.
        - ``has_aux=True`` + ``return_value=False`` => ``(arg_grads, aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``(arg_grads, loss_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``(arg_grads, loss_value, aux_data)``.
    2. When ``grad_states`` is not None and ``argnums`` is None

        - ``has_aux=False`` + ``return_value=False`` => ``var_grads``.
        - ``has_aux=True`` + ``return_value=False`` => ``(var_grads, aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``(var_grads, loss_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``(var_grads, loss_value, aux_data)``.
    3. When ``grad_states`` is not None and ``argnums`` is not None

        - ``has_aux=False`` + ``return_value=False`` => ``(var_grads, arg_grads)``.
        - ``has_aux=True`` + ``return_value=False`` => ``((var_grads, arg_grads), aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``((var_grads, arg_grads), loss_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``((var_grads, arg_grads), loss_value, aux_data)``.

    Parameters
    ----------
    fun: Callable
        Function whose Jacobian is to be computed.
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
    unit_aware: (bool) optional. Whether to return the gradient in the unit-aware
        mode. Default False.
    check_states: bool
          Whether to check the states in ``grad_states``. Default True.

    Returns
    -------
    fun: GradientTransform
      The transformed object.
    """
    return GradientTransform(
        target=fun,
        transform=_jacrev,
        grad_states=grad_states,
        argnums=argnums,
        return_value=return_value,
        has_aux=False if has_aux is None else has_aux,
        transform_params=dict(holomorphic=holomorphic,
                              allow_int=allow_int,
                              unit_aware=unit_aware, ),
        check_states=check_states
    )


jacobian = jacrev


@set_module_as("brainstate.transform")
def jacfwd(
    func: Callable,
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: bool = False,
    holomorphic: bool = False,
    unit_aware: bool = False,
    check_states: bool = True,
) -> GradientTransform:
    """Extending automatic Jacobian (forward-mode) of ``func`` to classes.

    This function extends the JAX official ``jacfwd`` to make automatic jacobian
    computation on functions and class functions. Moreover, it supports returning
    value ("return_value") and returning auxiliary data ("has_aux").


    1. When ``grad_states`` is None

        - ``has_aux=False`` + ``return_value=False`` => ``arg_grads``.
        - ``has_aux=True`` + ``return_value=False`` => ``(arg_grads, aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``(arg_grads, loss_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``(arg_grads, loss_value, aux_data)``.
    2. When ``grad_states`` is not None and ``argnums`` is None

        - ``has_aux=False`` + ``return_value=False`` => ``var_grads``.
        - ``has_aux=True`` + ``return_value=False`` => ``(var_grads, aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``(var_grads, loss_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``(var_grads, loss_value, aux_data)``.
    3. When ``grad_states`` is not None and ``argnums`` is not None

        - ``has_aux=False`` + ``return_value=False`` => ``(var_grads, arg_grads)``.
        - ``has_aux=True`` + ``return_value=False`` => ``((var_grads, arg_grads), aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``((var_grads, arg_grads), loss_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``((var_grads, arg_grads), loss_value, aux_data)``.


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
    unit_aware: (bool) optional. Whether to return the gradient in the unit-aware
        mode. Default False.
    check_states: bool
      Whether to check the states in ``grad_states``. Default True.

    Returns
    -------
    obj: GradientTransform
      The transformed object.
    """

    return GradientTransform(
        target=func,
        transform=_jacfwd,
        grad_states=grad_states,
        argnums=argnums,
        return_value=return_value,
        has_aux=False if has_aux is None else has_aux,
        transform_params=dict(holomorphic=holomorphic, unit_aware=unit_aware),
        check_states=check_states
    )

