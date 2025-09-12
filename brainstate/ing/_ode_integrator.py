# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Callable

import brainunit as u
import jax.numpy as jnp
from jax.tree_util import tree_map

from brainstate import environ
from brainstate.augment import vector_grad
from brainstate.typing import PyTree, ArrayLike

__all__ = [
    'ode_euler_step',
    'ode_rk2_step',
    'ode_rk3_step',
    'ode_rk4_step',
    'ode_expeuler_step',
]

DT = ArrayLike
ODE = Callable[[PyTree, float, ...], PyTree]


def ode_euler_step(
    f: ODE,
    y: PyTree,
    t: DT,
    *args
):
    r"""
    Explicit Euler step for ordinary differential equations.

    Implements a single forward Euler step for ODEs of the form

    .. math::

        \frac{dy}{dt} = f(y, t), \qquad y_{n+1} = y_n + \Delta t\, f(y_n, t_n).

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree`` that computes
        the time-derivative at ``(y, t)``.
    y : PyTree
        Current state at time ``t``. Any JAX-compatible pytree.
    t : float
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}`` after one Euler step.

    Notes
    -----
    - First-order accurate with local truncation error :math:`\mathcal{O}(\Delta t)`.
    - Minimal cost, but least accurate among the provided ODE solvers.
    """
    dt = environ.get_dt()
    k1 = f(y, t, *args)
    return tree_map(lambda x, _k1: x + dt * _k1, y, k1)


def ode_rk2_step(
    f: ODE,
    y: PyTree,
    t: DT,
    *args
):
    r"""
    Second-order Runge–Kutta (RK2) step for ODEs.

    The classical RK2 (Heun/midpoint) method performs two function evaluations:

    .. math::

        k_1 = f(y_n, t_n),\quad
        k_2 = f\big(y_n + \Delta t\,k_1,\ t_n + \Delta t\big),\\
        y_{n+1} = y_n + \tfrac{\Delta t}{2}\,(k_1 + k_2).

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK2 step.

    Notes
    -----
    Second-order accurate with local truncation error :math:`\mathcal{O}(\Delta t^2)`.
    """
    dt = environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(tree_map(lambda x, k: x + dt * k, y, k1), t + dt, *args)
    return tree_map(lambda x, _k1, _k2: x + dt / 2 * (_k1 + _k2), y, k1, k2)


def ode_rk3_step(
    f: ODE,
    y: PyTree,
    t: DT,
    *args
):
    r"""
    Third-order Runge–Kutta (RK3) step for ODEs.

    A common RK3 scheme uses three stages:

    .. math::

        k_1 = f(y_n, t_n),\quad
        k_2 = f\big(y_n + \tfrac{\Delta t}{2}k_1,\ t_n + \tfrac{\Delta t}{2}\big),\\
        k_3 = f\big(y_n - \Delta t\,k_1 + 2\Delta t\,k_2,\ t_n + \Delta t\big),\\
        y_{n+1} = y_n + \tfrac{\Delta t}{6}(k_1 + 4k_2 + k_3).

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK3 step.

    Notes
    -----
    Third-order accurate with local truncation error :math:`\mathcal{O}(\Delta t^3)`.
    """
    dt = environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args)
    k3 = f(tree_map(lambda x, k1_val, k2_val: x - dt * k1_val + 2 * dt * k2_val, y, k1, k2), t + dt, *args)
    return tree_map(lambda x, _k1, _k2, _k3: x + dt / 6 * (_k1 + 4 * _k2 + _k3), y, k1, k2, k3)


def ode_rk4_step(
    f: ODE,
    y: PyTree,
    t: DT,
    *args
):
    r"""
    Classical fourth-order Runge–Kutta (RK4) step for ODEs.

    The standard RK4 scheme uses four stages:

    .. math::

        k_1 = f(y_n, t_n),\quad
        k_2 = f\big(y_n + \tfrac{\Delta t}{2}k_1,\ t_n + \tfrac{\Delta t}{2}\big),\\
        k_3 = f\big(y_n + \tfrac{\Delta t}{2}k_2,\ t_n + \tfrac{\Delta t}{2}\big),\\
        k_4 = f\big(y_n + \Delta t\,k_3,\ t_n + \Delta t\big),\\
        y_{n+1} = y_n + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4).

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK4 step.

    Notes
    -----
    Fourth-order accurate with local truncation error :math:`\mathcal{O}(\Delta t^4)`.
    """
    dt = environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args)
    k3 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k2), t + dt / 2, *args)
    k4 = f(tree_map(lambda x, k: x + dt * k, y, k3), t + dt, *args)
    return tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt / 6 * (_k1 + 2 * _k2 + 2 * _k3 + _k4),
        y, k1, k2, k3, k4
    )


def ode_expeuler_step(
    f: ODE,
    y: PyTree,
    t: DT,
    *args
):
    r"""
    One-step Exponential Euler method for solving ODEs.

    Examples
    --------

    >>> def fun(x, t):
    ...     return -x
    >>> x = 1.0
    >>> exp_euler_step(fun, x， 0.)

    If the variable ( $x$ ) has units of ( $[X]$ ), then the drift term ( $\text{drift_fn}(x)$ ) should
    have units of ( $[X]/[T]$ ), where ( $[T]$ ) is the unit of time.

    If the variable ( x ) has units of ( [X] ), then the diffusion term ( \text{diffusion_fn}(x) )
    should have units of ( [X]/\sqrt{[T]} ).

    Args:
        fun: Callable. The function to be solved.
        diffusion: Callable. The diffusion function.
        *args: The input arguments.
        drift: Callable. The drift function.

    Returns:
        The one-step solution of the ODE.
    """
    assert callable(f), 'The input function should be callable.'
    if u.math.get_dtype(y) not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
        raise ValueError(
            f'The input data type should be float64, float32, float16, or bfloat16 '
            f'when using Exponential Euler method. But we got {y.dtype}.'
        )
    dt = environ.get('dt')
    linear, derivative = vector_grad(f, argnums=0, return_value=True)(y, t, *args)
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    phi = u.math.exprel(dt * linear)
    x_next = y + dt * phi * derivative
    return x_next
