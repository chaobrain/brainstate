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


from typing import Callable

import brainunit as u
import jax.numpy as jnp

from brainstate import environ, random
from brainstate.augment import vector_grad

__all__ = [
    'exp_euler_step',
]


def exp_euler_step(
    fn: Callable, *args, **kwargs
):
    r"""
    One-step Exponential Euler method for solving ODEs.

    Examples
    --------

    >>> def fun(x, t):
    ...     return -x
    >>> x = 1.0
    >>> exp_euler_step(fun, x, None)

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
    assert callable(fn), 'The input function should be callable.'
    assert len(args) > 0, 'The input arguments should not be empty.'
    if callable(args[0]):
        diffusion = args[0]
        args = args[1:]
    else:
        diffusion = None
    assert len(args) > 0, 'The input arguments should not be empty.'
    if u.math.get_dtype(args[0]) not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
        raise ValueError(
            f'The input data type should be float64, float32, float16, or bfloat16 '
            f'when using Exponential Euler method. But we got {args[0].dtype}.'
        )

    # drift
    dt = environ.get('dt')
    linear, derivative = vector_grad(fn, argnums=0, return_value=True)(*args, **kwargs)
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    phi = u.math.exprel(dt * linear)
    x_next = args[0] + dt * phi * derivative

    # diffusion
    if diffusion is not None:
        diffusion_part = diffusion(*args, **kwargs) * u.math.sqrt(dt) * random.randn_like(args[0])
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
