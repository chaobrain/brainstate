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

# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from brainstate import init
from brainstate._state import ShortTermState
from brainstate.mixin import DelayedInit, AlignPost
from brainstate.nn._dynamics._dynamics_base import Dynamics
from brainstate.nn._exp_euler import exp_euler_step
from brainstate.typing import ArrayLike, Size

__all__ = [
  'Synapse', 'Expon', 'STP', 'STD',
]


class Synapse(Dynamics, AlignPost, DelayedInit):
  """
  Base class for synapse dynamics.
  """
  __module__ = 'brainstate.nn'


class Expon(Synapse):
  r"""Exponential decay synapse model.

  Args:
    tau: float. The time constant of decay. [ms]
    %s
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      tau: ArrayLike = 8.0,
  ):
    super().__init__(
      name=name,
      size=size,
      keep_size=keep_size
    )

    # parameters
    self.tau = init.param(tau, self.varshape)

  def init_state(self, batch_size: int = None, **kwargs):
    self.g = ShortTermState(init.param(init.Constant(0.), self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.g.value = init.param(init.Constant(0.), self.varshape, batch_size)

  def update(self, x=None):
    self.g.value = exp_euler_step(lambda g: -g / self.tau, self.g.value)
    if x is not None:
      self.align_post_input_add(x)
    return self.g.value

  def align_post_input_add(self, x):
    self.g.value += x


class STP(Synapse):
  r"""Synaptic output with short-term plasticity.

  %s

  Args:
    tau_f: float, ArrayType, Callable. The time constant of short-term facilitation.
    tau_d: float, ArrayType, Callable. The time constant of short-term depression.
    U: float, ArrayType, Callable. The fraction of resources used per action potential.
    %s
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      U: ArrayLike = 0.15,
      tau_f: ArrayLike = 1500.,
      tau_d: ArrayLike = 200.,
  ):
    super().__init__(name=name,
                     size=size,
                     keep_size=keep_size)

    # parameters
    self.tau_f = init.param(tau_f, self.varshape)
    self.tau_d = init.param(tau_d, self.varshape)
    self.U = init.param(U, self.varshape)

  def init_state(self, batch_size: int = None, **kwargs):
    self.x = ShortTermState(init.param(init.Constant(1.), self.varshape, batch_size))
    self.u = ShortTermState(init.param(init.Constant(self.U), self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.x.value = init.param(init.Constant(1.), self.varshape, batch_size)
    self.u.value = init.param(init.Constant(self.U), self.varshape, batch_size)

  def update(self, pre_spike):
    du = lambda u: self.U - u / self.tau_f
    dx = lambda x: (1 - x) / self.tau_d
    u = exp_euler_step(du, self.u.value)
    x = exp_euler_step(dx, self.x.value)

    # --- original code:
    #   if pre_spike.dtype == jax.numpy.bool_:
    #     u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
    #     x = bm.where(pre_spike, x - u * self.x, x)
    #   else:
    #     u = pre_spike * (u + self.U * (1 - self.u)) + (1 - pre_spike) * u
    #     x = pre_spike * (x - u * self.x) + (1 - pre_spike) * x

    # --- simplified code:
    u = u + pre_spike * self.U * (1 - self.u.value)
    x = x - pre_spike * u * self.x.value

    self.u.value = u
    self.x.value = x
    return u * x


class STD(Synapse):
  r"""Synaptic output with short-term depression.

  %s

  Args:
    tau: float, ArrayType, Callable. The time constant of recovery of the synaptic vesicles.
    U: float, ArrayType, Callable. The fraction of resources used per action potential.
    %s
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      # synapse parameters
      tau: ArrayLike = 200.,
      U: ArrayLike = 0.07,
  ):
    super().__init__(name=name,
                     size=size,
                     keep_size=keep_size)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.U = init.param(U, self.varshape)

  def init_state(self, batch_size: int = None, **kwargs):
    self.x = ShortTermState(init.param(init.Constant(1.), self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.x.value = init.param(init.Constant(1.), self.varshape, batch_size)

  def update(self, pre_spike):
    dx = lambda x: (1 - x) / self.tau
    x = exp_euler_step(dx, self.x.value)

    # --- original code:
    # self.x.value = bm.where(pre_spike, x - self.U * self.x, x)

    # --- simplified code:
    self.x.value = x - pre_spike * self.U * self.x.value

    return self.x.value
