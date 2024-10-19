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


from __future__ import annotations

from typing import Callable

import brainunit as u
import jax.numpy as jnp

from brainstate import environ
from brainstate.augment import vector_grad

__all__ = [
  'exp_euler_step',
]


def exp_euler_step(fun: Callable, *args, **kwargs):
  """
  Exponential Euler method for solving ODEs.

  Examples
  --------
  >>> def fun(x, t):
  ...     return -x
  >>> x = 1.0
  >>> exp_euler_step(fun, x, None)

  Args:
    fun: Callable. The function to be solved.

  Returns:
    The integral function.
  """
  assert len(args) > 0, 'The input arguments should not be empty.'
  if args[0].dtype not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
    raise ValueError(
      'The input data type should be float32, float64, float16, or bfloat16 '
      'when using Exponential Euler method.'
      f'But we got {args[0].dtype}.'
    )
  dt = environ.get('dt')
  linear, derivative = vector_grad(fun, argnums=0, return_value=True)(*args, **kwargs)
  phi = u.math.exprel(dt * linear)
  return args[0] + dt * phi * derivative
