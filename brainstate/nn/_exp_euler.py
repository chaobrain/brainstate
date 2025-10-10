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


from typing import Callable

import brainunit as u
import jax.numpy as jnp

from brainstate import environ, random
from brainstate.transform import vector_grad

__all__ = [
    'exp_euler_step',
]


def exp_euler_step(
    fn: Callable, *args, **kwargs
):
    r"""
    One-step Exponential Euler method for solving ODEs and SDEs.

    The Exponential Euler method is a numerical integration scheme that provides improved
    stability for stiff differential equations by exactly integrating the linear part of
    the equation. For ODEs, it solves equations of the form:

    .. math::
        \frac{dx}{dt} = f(x, t)

    For SDEs, it handles equations of the form:

    .. math::
        dx = f(x, t)dt + g(x, t)dW

    where :math:`f(x, t)` is the drift term and :math:`g(x, t)` is the diffusion term.

    The method linearizes the drift function around the current state and uses the
    matrix exponential to integrate the linear part exactly, while treating the
    remainder with standard Euler stepping.

    Parameters
    ----------
    fn : Callable
        The drift function :math:`f(x, t)` to be integrated. This function should
        take the state variable as the first argument, followed by optional time
        and other arguments. It should return the derivative :math:`dx/dt`.
    *args
        Variable arguments. If the first argument is callable, it is treated as
        the diffusion function for SDE integration. Otherwise, arguments are
        passed to the drift function. The first non-callable argument should be
        the state variable :math:`x`.
    **kwargs
        Additional keyword arguments passed to the drift and diffusion functions.

    Returns
    -------
    x_next : ArrayLike
        The state variable after one integration step of size ``dt``, where ``dt``
        is obtained from the environment via ``environ.get('dt')``.

    Raises
    ------
    ValueError
        If the input state variable dtype is not float16, bfloat16, float32, or float64.
    ValueError
        If drift and diffusion terms have incompatible units.
    AssertionError
        If ``fn`` is not callable or if no state variable is provided in ``*args``.

    Notes
    -----
    **Unit Compatibility:**

    - If the state variable :math:`x` has units :math:`[X]`, the drift function
      :math:`f(x, t)` should return values with units :math:`[X]/[T]`, where
      :math:`[T]` is the unit of time.

    - If the state variable :math:`x` has units :math:`[X]`, the diffusion function
      :math:`g(x, t)` should return values with units :math:`[X]/\sqrt{[T]}`.

    **Algorithm:**

    The method computes the Jacobian :math:`J = \frac{\partial f}{\partial x}` and
    uses the exponential-related function :math:`\varphi(z) = (e^z - 1)/z` to update:

    .. math::
        x_{n+1} = x_n + dt \cdot \varphi(dt \cdot J) \cdot f(x_n, t_n)

    For SDEs, a stochastic term is added:

    .. math::
        x_{n+1} = x_{n+1} + g(x_n, t_n) \sqrt{dt} \cdot \mathcal{N}(0, I)

    Examples
    --------
    **ODE Integration:**

    Simple exponential decay equation :math:`\frac{dx}{dt} = -x`:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Set time step in environment
        >>> brainstate.environ.set(dt=0.01)
        >>>
        >>> # Define drift function
        >>> def drift(x, t):
        ...     return -x
        >>>
        >>> # Initial condition
        >>> x0 = jnp.array(1.0)
        >>>
        >>> # Single integration step
        >>> x1 = brainstate.nn.exp_euler_step(drift, x0, None)
        >>> print(x1)  # Should be close to exp(-0.01) ≈ 0.99

    **SDE Integration:**

    Ornstein-Uhlenbeck process :math:`dx = -\theta x dt + \sigma dW`:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Set time step
        >>> brainstate.environ.set(dt=0.01)
        >>>
        >>> # Define drift and diffusion
        >>> theta = 0.5
        >>> sigma = 0.3
        >>>
        >>> def drift(x, t):
        ...     return -theta * x
        >>>
        >>> def diffusion(x, t):
        ...     return jnp.full_like(x, sigma)
        >>>
        >>> # Initial condition
        >>> x0 = jnp.array(1.0)
        >>>
        >>> # Single SDE integration step
        >>> x1 = brainstate.nn.exp_euler_step(drift, diffusion, x0, None)

    **Multi-dimensional system:**

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> brainstate.environ.set(dt=0.01)
        >>>
        >>> # Coupled oscillator system
        >>> def drift(x, t):
        ...     x1, x2 = x[0], x[1]
        ...     return jnp.array([-x1 + x2, -x2 - x1])
        >>>
        >>> x0 = jnp.array([1.0, 0.0])
        >>> x1 = brainstate.nn.exp_euler_step(drift, x0, None)

    See Also
    --------
    brainstate.transform.vector_grad : Compute vector-Jacobian product used internally.
    brainstate.environ.get : Retrieve environment variables like ``dt``.

    References
    ----------
    .. [1] Hochbruck, M., & Ostermann, A. (2010). Exponential integrators.
           Acta Numerica, 19, 209-286.
    .. [2] Cox, S. M., & Matthews, P. C. (2002). Exponential time differencing
           for stiff systems. Journal of Computational Physics, 176(2), 430-455.
    """
    # Validate inputs
    assert callable(fn), 'The drift function should be callable.'
    assert len(args) > 0, 'The input arguments should not be empty.'

    # Parse arguments: check if first arg is diffusion function
    diffusion = None
    if callable(args[0]):
        diffusion = args[0]
        args = args[1:]
        assert len(args) > 0, 'State variable is required after diffusion function.'

    # Validate state variable dtype
    state = u.math.asarray(args[0])
    dtype = u.math.get_dtype(state)
    if dtype not in [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]:
        raise ValueError(
            f'State variable dtype must be float16, bfloat16, float32, or float64 '
            f'for Exponential Euler method, but got {dtype}.'
        )

    # Get time step from environment
    dt = environ.get_dt()

    # Compute drift term with Jacobian
    # vector_grad returns (Jacobian, function_value)
    jacobian, drift_value = vector_grad(fn, argnums=0, return_value=True)(*args, **kwargs)

    # Convert Jacobian to proper units: [derivative_unit / state_unit] = [1/T]
    jacobian_with_unit = u.Quantity(
        u.get_mantissa(jacobian),
        u.get_unit(drift_value) / u.get_unit(jacobian)
    )

    # Compute phi function: phi(z) = (exp(z) - 1) / z
    # This is the exponential-related function for stability
    phi = u.math.exprel(dt * jacobian_with_unit)

    # Update state using exponential Euler scheme
    x_next = state + dt * phi * drift_value

    # Add diffusion term for SDE if provided
    if diffusion is not None:
        # Compute diffusion coefficient
        diffusion_coef = diffusion(*args, **kwargs)

        # Generate random noise and scale by sqrt(dt)
        noise = random.randn_like(state)
        diffusion_term = diffusion_coef * u.math.sqrt(dt) * noise

        # Validate unit compatibility between drift and diffusion
        if u.get_dim(x_next) != u.get_dim(diffusion_term):
            drift_unit = u.get_unit(x_next)
            time_unit = u.get_unit(dt)
            expected_diffusion_unit = drift_unit / time_unit ** 0.5
            actual_diffusion_unit = u.get_unit(diffusion_term)
            raise ValueError(
                f"Unit mismatch between drift and diffusion terms. "
                f"State has unit {u.get_unit(state)}, "
                f"drift produces unit {drift_unit}, "
                f"expected diffusion unit {expected_diffusion_unit}, "
                f"but got {actual_diffusion_unit}."
            )

        x_next = x_next + diffusion_term

    return x_next
