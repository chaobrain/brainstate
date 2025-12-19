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

from functools import partial
from typing import Union, Callable, Dict, Sequence, Optional

import brainunit as u
import jax
import jax.numpy as jnp

from brainstate._state import State
from brainstate._utils import set_module_as
from brainstate.typing import Missing, SeedOrKey
from ._grad_transform import GradientTransform
from ._util import warp_grad_fn, tree_random_split

__all__ = [
    'vector_grad', 'grad', 'fwd_grad',
]


@set_module_as("brainstate.transform")
def grad(
    fun: Callable = Missing(),
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    holomorphic: Optional[bool] = False,
    allow_int: Optional[bool] = False,
    has_aux: Optional[bool] = None,
    return_value: Optional[bool] = False,
    unit_aware: bool = False,
    check_states: bool = True,
) -> GradientTransform | Callable[[Callable], GradientTransform]:
    """
    Compute the gradient of a scalar-valued function with respect to its arguments.


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
    fun : callable, optional
        The scalar-valued function to be differentiated.
    grad_states : State, sequence of State, or dict of State, optional
        The variables in fun to take their gradients.
    argnums : int or sequence of int, optional
        Specifies which positional argument(s) to differentiate with respect to.
    holomorphic : bool, default False
        Whether fun is promised to be holomorphic.
    allow_int : bool, default False
        Whether to allow differentiating with respect to
        integer valued inputs. The gradient of an integer input will have a trivial
        vector-space dtype (float0).
    has_aux : bool, optional
        Indicates whether fun returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.
    return_value : bool, default False
        Indicates whether to return the value of the
        function along with the gradient.
    unit_aware : bool, default False
        Whether to return the gradient in the unit-aware mode.
    check_states : bool, default True
        Whether to check that all grad_states are found in the function.

    Returns
    -------
    GradientTransform or callable
        A function which computes the gradient of fun. The function takes the same
        arguments as `fun`, but returns the gradient instead. If `has_aux` is True,
        the function returns a pair where the first element is the gradient and the
        second element is the auxiliary data. If `return_value` is True, the function
        returns a pair where the first element is the gradient and the second element
        is the value of the function.

    Examples
    --------
    Basic gradient computation:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Simple function gradient
        >>> def f(x):
        ...     return jnp.sum(x ** 2)
        >>>
        >>> grad_f = brainstate.transform.grad(f)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> gradient = grad_f(x)

    Gradient with respect to states:

    .. code-block:: python

        >>> # Create states
        >>> weight = brainstate.State(jnp.array([1.0, 2.0]))
        >>> bias = brainstate.State(jnp.array([0.5]))
        >>>
        >>> def loss_fn(x):
        ...     prediction = jnp.dot(x, weight.value) + bias.value
        ...     return prediction ** 2
        >>>
        >>> # Compute gradients with respect to states
        >>> grad_fn = brainstate.transform.grad(loss_fn, grad_states=[weight, bias])
        >>> x = jnp.array([1.0, 2.0])
        >>> state_grads = grad_fn(x)

    With auxiliary data and return value:

    .. code-block:: python

        >>> def loss_with_aux(x):
        ...     prediction = jnp.dot(x, weight.value) + bias.value
        ...     loss = prediction ** 2
        ...     return loss, {"prediction": prediction}
        >>>
        >>> grad_fn = brainstate.transform.grad(
        ...     loss_with_aux,
        ...     grad_states=[weight, bias],
        ...     has_aux=True,
        ...     return_value=True
        ... )
        >>> grads, loss_value, aux_data = grad_fn(x)
    """
    if isinstance(fun, Missing):
        def transform(fun) -> GradientTransform:
            return GradientTransform(
                target=fun,
                transform=u.autograd.grad if unit_aware else jax.grad,
                grad_states=grad_states,
                argnums=argnums,
                return_value=return_value,
                has_aux=False if has_aux is None else has_aux,
                transform_params=dict(holomorphic=holomorphic, allow_int=allow_int),
                check_states=check_states
            )

        return transform

    return GradientTransform(
        target=fun,
        transform=u.autograd.grad if unit_aware else jax.grad,
        grad_states=grad_states,
        argnums=argnums,
        return_value=return_value,
        has_aux=False if has_aux is None else has_aux,
        transform_params=dict(holomorphic=holomorphic, allow_int=allow_int),
        check_states=check_states
    )


@set_module_as("brainstate.transform")
def vector_grad(
    func: Callable = Missing(),
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    return_value: bool = False,
    has_aux: Optional[bool] = None,
    unit_aware: bool = False,
    check_states: bool = True,
) -> GradientTransform | Callable[[Callable], GradientTransform]:
    """
    Take vector-valued gradients for function ``func``.

    Same as :py:func:`grad`, :py:func:`jacrev`, and :py:func:`jacfwd`,
    the returns in this function are different for different argument settings.


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
    func : callable, optional
        Function whose gradient is to be computed.
    grad_states : State, sequence of State, or dict of State, optional
        The variables in ``func`` to take their gradients.
    argnums : int or sequence of int, optional
        Specifies which positional argument(s) to differentiate with respect to.
    return_value : bool, default False
        Whether to return the loss value.
    has_aux : bool, optional
        Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.
    unit_aware : bool, default False
        Whether to return the gradient in the unit-aware mode.
    check_states : bool, default True
        Whether to check that all grad_states are found in the function.

    Returns
    -------
    GradientTransform or callable
        The vector gradient function.

    Examples
    --------
    Basic vector gradient computation:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Vector-valued function
        >>> def f(x):
        ...     return jnp.array([x[0]**2, x[1]**3, x[0]*x[1]])
        >>>
        >>> vector_grad_f = brainstate.transform.vector_grad(f)
        >>> x = jnp.array([2.0, 3.0])
        >>> gradients = vector_grad_f(x)  # Shape: (3, 2)

    With states:

    .. code-block:: python

        >>> params = brainstate.State(jnp.array([1.0, 2.0]))
        >>>
        >>> def model(x):
        ...     return jnp.array([
        ...         x * params.value[0],
        ...         x**2 * params.value[1]
        ...     ])
        >>>
        >>> vector_grad_fn = brainstate.transform.vector_grad(
        ...     model, grad_states=[params]
        ... )
        >>> x = 3.0
        >>> param_grads = vector_grad_fn(x)
    """

    if isinstance(func, Missing):
        def transform(fun) -> GradientTransform:
            return GradientTransform(
                target=fun,
                transform=partial(u.autograd.vector_grad, unit_aware=unit_aware),
                grad_states=grad_states,
                argnums=argnums,
                return_value=return_value,
                has_aux=False if has_aux is None else has_aux,
                check_states=check_states
            )

        return transform

    else:
        return GradientTransform(
            target=func,
            transform=partial(u.autograd.vector_grad, unit_aware=unit_aware),
            grad_states=grad_states,
            argnums=argnums,
            return_value=return_value,
            has_aux=False if has_aux is None else has_aux,
            check_states=check_states
        )


def _fwd_grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    tangent_size: Optional[int] = None,
    has_aux: bool = False,
    drct_der_clip: Optional[float] = None,
    key: SeedOrKey = None,
) -> Callable:
    tangent_size = () if tangent_size is None else (tangent_size,)
    from brainstate.random._seed import split_key

    def wrapper(*args, **kwargs):
        f_partial, params = warp_grad_fn(fun, argnums, args, kwargs)
        v = jax.tree.map(
            lambda x, k: jax.random.normal(k, tangent_size + x.shape, x.dtype),
            params,
            tree_random_split(split_key() if key is None else key, params)
        )
        if len(tangent_size) == 0:
            r = jax.jvp(f_partial, (params,), (v,), has_aux=has_aux)
            if has_aux:
                loss_value, drct_drv, aux = r
            else:
                loss_value, drct_drv = r
            if drct_der_clip is not None:
                drct_drv = jnp.clip(drct_drv, -drct_der_clip, drct_der_clip)
            grads = jax.tree.map(lambda v_leaf: v_leaf * drct_drv, v)
            return (grads, aux) if has_aux else grads
        elif len(tangent_size) == 1:
            r = jax.vmap(lambda x: jax.jvp(f_partial, (params,), (x,), has_aux=has_aux))(v)
            if has_aux:
                loss_value, drct_drv, aux = r
                aux = jax.tree.map(lambda x: x[0], aux)
            else:
                loss_value, drct_drv = r
            loss_value = loss_value[0]
            if drct_der_clip is not None:
                drct_drv = jnp.clip(drct_drv, -drct_der_clip, drct_der_clip)
            grads = jax.tree.map(lambda v_leaf: jax.vmap(jnp.multiply)(v_leaf, drct_drv).mean(axis=0), v)
            return (grads, aux) if has_aux else grads
        else:
            raise ValueError(f"Only support tangent_size of 0 or 1, but got {tangent_size}")

    return wrapper


@set_module_as("brainstate.transform")
def fwd_grad(
    func: Callable = Missing(),
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    return_value: bool = False,
    has_aux: Optional[bool] = None,
    tangent_size: Optional[int] = None,
    drct_der_clip: Optional[float] = None,
    key: SeedOrKey = None,
) -> GradientTransform | Callable[[Callable], GradientTransform]:
    """
    Take forward first-order gradients for function ``func``.

    Same as :py:func:`grad`, :py:func:`jacrev`, and :py:func:`jacfwd`,
    the returns in this function are different for different argument settings.

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
    func : callable, optional
        Function whose gradient is to be computed.
    grad_states : State, sequence of State, or dict of State, optional
        The variables in ``func`` to take their gradients.
    argnums : int or sequence of int, optional
        Specifies which positional argument(s) to differentiate with respect to.
    return_value : bool, default False
        Whether to return the loss value.
    has_aux : bool, optional
        Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.
    unit_aware : bool, default False
        Whether to return the gradient in the unit-aware mode.
    check_states : bool, default True
        Whether to check that all grad_states are found in the function.

    Returns
    -------
    GradientTransform or callable
        The vector gradient function.

    Examples
    --------
    Basic vector gradient computation:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Vector-valued function
        >>> def f(x):
        ...     return jnp.array([x[0]**2, x[1]**3, x[0]*x[1]])
        >>>
        >>> vector_grad_f = brainstate.transform.vector_grad(f)
        >>> x = jnp.array([2.0, 3.0])
        >>> gradients = vector_grad_f(x)  # Shape: (3, 2)

    With states:

    .. code-block:: python

        >>> params = brainstate.State(jnp.array([1.0, 2.0]))
        >>>
        >>> def model(x):
        ...     return jnp.array([
        ...         x * params.value[0],
        ...         x**2 * params.value[1]
        ...     ])
        >>>
        >>> vector_grad_fn = brainstate.transform.vector_grad(
        ...     model, grad_states=[params]
        ... )
        >>> x = 3.0
        >>> param_grads = vector_grad_fn(x)
    """

    if isinstance(func, Missing):
        def transform(fun) -> GradientTransform:
            return GradientTransform(
                target=fun,
                transform=_fwd_grad,
                grad_states=grad_states,
                argnums=argnums,
                return_value=return_value,
                has_aux=False if has_aux is None else has_aux,
                transform_params=dict(
                    tangent_size=tangent_size,
                    drct_der_clip=drct_der_clip,
                    key=key,
                )
            )

        return transform

    else:
        return GradientTransform(
            target=func,
            transform=_fwd_grad,
            grad_states=grad_states,
            argnums=argnums,
            return_value=return_value,
            has_aux=False if has_aux is None else has_aux,
            transform_params=dict(
                tangent_size=tangent_size,
                drct_der_clip=drct_der_clip,
                key=key,
            )
        )
