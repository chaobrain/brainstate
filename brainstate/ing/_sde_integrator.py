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

"""
Stochastic differential equation (SDE) one-step integrators.

This module provides compact steppers for integrating SDEs inside simulation
loops. The steppers operate on arbitrary JAX PyTrees, making them suitable for
state containers used across BrainState.

Available steppers
------------------
- ``sde_euler_step``: Euler–Maruyama (Ito) scheme; strong order 0.5.
- ``sde_milstein_step``: Milstein scheme (Ito/Stratonovich); strong order 1.0.
- ``sde_expeuler_step``: Exponential Euler using linearized drift plus diffusion.

Notes
-----
All steppers use the global time step ``dt`` from ``brainstate.environ`` and
draw Gaussian noise using ``brainstate.random``. Noise is applied per PyTree
leaf and scaled by ``sqrt(dt)``.
"""

from typing import Callable

import brainunit as u
import jax.numpy as jnp
from jax.tree_util import tree_map

from brainstate import environ, random
from brainstate.augment import vector_grad
from brainstate.typing import PyTree, ArrayLike

__all__ = [
    'sde_euler_step',
    'sde_milstein_step',
    'sde_expeuler_step',
]

DT = ArrayLike
DF = Callable[[PyTree, DT, ...], PyTree]
DG = Callable[[PyTree, DT, ...], PyTree]


def sde_euler_step(
    df: DF,
    dg: DG,
    y: PyTree,
    t: DT,
    *args,
    sde_type: str = 'ito',
):
    """One Euler–Maruyama step for Ito SDEs.

    This integrates an Ito SDE of the form

    .. math:: dy = f(y, t)\,dt + g(y, t)\,dW,

    where ``f`` is the drift and ``g`` is the diffusion, using
    ``y_{n+1} = y_n + f(y_n, t_n) dt + g(y_n, t_n) dW_n`` with
    ``dW_n ~ Normal(0, dt)`` applied per PyTree leaf.

    Parameters
    ----------
    df : Callable[[PyTree, ArrayLike, ...], PyTree]
        Drift function ``f(y, t, *args)`` returning a PyTree matching ``y``.
    dg : Callable[[PyTree, ArrayLike, ...], PyTree]
        Diffusion function ``g(y, t, *args)`` returning a PyTree matching ``y``.
    y : PyTree
        Current state.
    t : ArrayLike
        Current time (scalar or array broadcastable with ``y`` leaves).
    *args
        Extra arguments passed to ``df`` and ``dg``.
    sde_type : {'ito'}, optional
        Interpretation of the SDE. Only ``'ito'`` is supported, by default 'ito'.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}`` with the same tree structure as ``y``.

    Notes
    -----
    - Strong order 0.5, weak order 1.0.
    - Uses ``dt = brainstate.environ.get_dt()`` and Gaussian noise scaled by
      ``sqrt(dt)`` via ``brainstate.random.randn_like`` for each leaf of ``y``.
    """
    assert sde_type in ['ito']

    dt = environ.get_dt()
    dt_sqrt = jnp.sqrt(dt)
    y_bars = tree_map(
        lambda y0, drift, diffusion: y0 + drift * dt + diffusion * random.randn_like(y0) * dt_sqrt,
        y,
        df(y, t, *args),
        dg(y, t, *args),
    )
    return y_bars


def sde_milstein_step(
    df: DF,
    dg: DG,
    y: PyTree,
    t: DT,
    *args,
    sde_type: str = 'ito',
):
    """One Milstein step for Ito or Stratonovich SDEs.

    This integrates an SDE of the form

    .. math:: dy = f(y, t)\,dt + g(y, t)\,dW,

    using the Milstein scheme. In Ito form, the update is

    .. math::
        y_{n+1} = y_n + f_n dt + g_n dW_n + \tfrac{1}{2} g_n \partial_y g_n (dW_n^2 - dt),

    while for Stratonovich (``sde_type='stra'``) the last term uses ``dW_n^2``
    instead of ``(dW_n^2 - dt)``. The directional derivative ``\partial_y g`` is
    approximated here by a finite difference using an intermediate evaluation.

    Parameters
    ----------
    df : Callable[[PyTree, ArrayLike, ...], PyTree]
        Drift function ``f(y, t, *args)``.
    dg : Callable[[PyTree, ArrayLike, ...], PyTree]
        Diffusion function ``g(y, t, *args)``.
    y : PyTree
        Current state.
    t : ArrayLike
        Current time.
    *args
        Extra arguments forwarded to ``df`` and ``dg``.
    sde_type : {'ito', 'stra'}, optional
        Interpretation of the SDE: Ito or Stratonovich (``'stra'``).

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.

    Notes
    -----
    - Strong order 1.0 (Ito), offering higher accuracy than Euler–Maruyama.
    - The derivative term is realized via a finite-difference correction using an
      auxiliary diffusion evaluation at an intermediate state.
    """
    assert sde_type in ['ito', 'stra']

    dt = environ.get_dt()
    dt_sqrt = jnp.sqrt(dt)

    # drift values
    drifts = df(y, t, *args)

    # diffusion values
    diffusions = dg(y, t, *args)

    # intermediate results
    y_bars = tree_map(lambda y0, drift, diffusion: y0 + drift * dt + diffusion * dt_sqrt, y, drifts, diffusions)
    diffusion_bars = dg(y_bars, t, *args)

    # integral results
    def f_integral(y0, drift, diffusion, diffusion_bar):
        noise = random.randn_like(y0) * dt_sqrt
        noise_p2 = (noise ** 2 - dt) if sde_type == 'ito' else noise ** 2
        minus = (diffusion_bar - diffusion) / 2 / dt_sqrt
        return y0 + drift * dt + diffusion * noise + minus * noise_p2

    integrals = tree_map(f_integral, y, drifts, diffusions, diffusion_bars)
    return integrals


def sde_expeuler_step(
    df: DF,
    dg: DG,
    y: PyTree,
    t: DT,
    *args,
):
    r"""One Exponential Euler step for SDEs with linearized drift.

    The drift ``f`` is locally linearized and integrated exactly over one step
    via the exponential relative function, while the diffusion term from ``g``
    is added in Euler form with Gaussian noise scaled by ``sqrt(dt)``.

    Parameters
    ----------
    df : Callable[[PyTree, ArrayLike, ...], PyTree]
        Drift function ``f(y, t, *args)``. Its value and vector–Jacobian product
        are used internally for the exponential update.
    dg : Callable[[PyTree, ArrayLike, ...], PyTree]
        Diffusion function ``g(y, t, *args)``.
    y : PyTree
        Current state. Must have a floating dtype.
    t : ArrayLike
        Current time.
    *args
        Extra arguments forwarded to ``df`` and ``dg``. The first extra argument
        is used solely to determine the shape for ``random.randn_like`` when
        sampling the diffusion noise.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}`` with the same structure as ``y``.

    Notes
    -----
    - Uses ``dt = brainstate.environ.get('dt')`` for the step size.
    - Requires floating dtypes for ``y`` (float16/32/64 or bfloat16).
    - Unit consistency is validated using ``brainunit``; a mismatch between the
      drift update and diffusion units raises a ``ValueError``.
    """
    assert callable(df), 'The input function should be callable.'
    assert callable(dg), 'The input function should be callable.'
    if u.math.get_dtype(y) not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
        raise ValueError(
            f'The input data type should be float64, float32, float16, or bfloat16 '
            f'when using Exponential Euler method. But we got {y.dtype}.'
        )

    # drift
    dt = environ.get('dt')
    linear, derivative = vector_grad(df, argnums=0, return_value=True)(y, t, *args)
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    phi = u.math.exprel(dt * linear)
    x_next = y + dt * phi * derivative

    # diffusion
    diffusion_part = dg(y, t, *args) * u.math.sqrt(dt) * random.randn_like(args[0])
    if u.get_dim(x_next) != u.get_dim(diffusion_part):
        drift_unit = u.get_unit(x_next)
        time_unit = u.get_unit(dt)
        raise ValueError(
            f"Drift unit is {drift_unit}, "
            f"expected diffusion unit is {drift_unit / time_unit ** 0.5}, "
            f"but we got {u.get_unit(diffusion_part)}."
        )
    x_next += diffusion_part
    return x_next

