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

from typing import Union, Callable, Dict, Sequence, Optional

import brainunit as u
import jax

from brainstate._state import State
from brainstate._utils import set_module_as
from ._grad_transform import GradientTransform

__all__ = [
    'hessian',
]


@set_module_as("brainstate.transform")
def hessian(
    func: Callable,
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    return_value: bool = False,
    holomorphic: bool = False,
    has_aux: Optional[bool] = None,
    unit_aware: bool = False,
    check_states: bool = True,
) -> GradientTransform:
    """
    Hessian of ``func`` as a dense array.


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
    has_aux: Optional, bool
        Indicates whether ``fun`` returns a pair where the first element is considered
        the output of the mathematical function to be differentiated and the second
        element is auxiliary data. Default False.
    unit_aware: (bool) optional. Whether to return the gradient in the unit-aware
        mode. Default False.
    check_states: bool
      Whether to check the states in ``grad_states``. Default True.

    Returns
    -------
    obj: ObjectTransform
      The transformed object.
    """
    return GradientTransform(
        target=func,
        transform=u.autograd.hessian if unit_aware else jax.hessian,
        grad_states=grad_states,
        argnums=argnums,
        return_value=return_value,
        has_aux=False if has_aux is None else has_aux,
        transform_params=dict(holomorphic=holomorphic),
        check_states=check_states
    )
