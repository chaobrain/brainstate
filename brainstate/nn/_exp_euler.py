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

from brainstate.ing import ode_expeuler_step, sde_expeuler_step

__all__ = [
    'exp_euler_step',
]


def exp_euler_step(fn: Callable, *args):
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
        return sde_expeuler_step(fn, args[0], args[1], 0., *args[2:])
    else:
        return ode_expeuler_step(fn, args[0], 0., *args[1:])
