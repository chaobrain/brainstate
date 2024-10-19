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

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from brainstate import init, surrogate
from brainstate._state import ShortTermState
from brainstate.mixin import DelayedInit
from brainstate.nn._dynamics._dynamics_base import Dynamics
from brainstate.nn._dynamics._state_delay import StateWithDelay
from brainstate.nn._exp_euler import exp_euler_step
from brainstate.typing import DTypeLike, ArrayLike, Size

__all__ = [
  'Neuron', 'IF', 'LIF', 'ALIF',
]


class Neuron(Dynamics, DelayedInit):
  """
  Base class for neuronal dynamics.

  Note here we use the ``ExplicitInOutSize`` mixin to explicitly specify the input and output shape.

  Moreover, all neuron models are differentiable since they use surrogate gradient functions to
  generate the spiking state.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      spk_fun: Callable = surrogate.InvSquareGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      detach_spk: bool = False,
      name: Optional[str] = None,
  ):
    super().__init__(in_size, keep_size=keep_size, name=name)
    self.spk_reset = spk_reset
    self.spk_dtype = spk_dtype
    self.spk_fun = spk_fun
    self.detach_spk = detach_spk

  def get_spike(self, *args, **kwargs):
    raise NotImplementedError


class IF(Neuron):
  """
  Integrate-and-fire neuron model.
  """

  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      tau: ArrayLike = 5.,
      V_th: ArrayLike = 1.,
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      name: str = None,
  ):
    super().__init__(in_size, keep_size=keep_size, name=name,
                     spk_fun=spk_fun, spk_dtype=spk_dtype, spk_reset=spk_reset)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.V_th = init.param(V_th, self.varshape)

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = StateWithDelay(init.param(jnp.zeros, self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.V.reset(init.param(jnp.zeros, self.varshape, batch_size))

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    v_scaled = (V - self.V_th) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x=0.):
    # reset
    last_V = self.V.value
    last_spike = self.get_spike(self.V.value)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
    V = last_V - V_th * last_spike
    # membrane potential
    dv = lambda v: (-v + self.sum_current_inputs(x, v)) / self.tau
    V = exp_euler_step(dv, V)
    V = self.sum_delta_inputs(V)
    self.V.value = V
    return self.get_spike(V)


class LIF(Neuron):
  """Leaky integrate-and-fire neuron model."""
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      tau: ArrayLike = 5.,
      V_th: ArrayLike = 1.,
      V_reset: ArrayLike = 0.,
      V_rest: ArrayLike = 0.,
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      name: str = None,
  ):
    super().__init__(in_size,
                     keep_size=keep_size,
                     name=name,
                     spk_fun=spk_fun,
                     spk_dtype=spk_dtype,
                     spk_reset=spk_reset)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.V_th = init.param(V_th, self.varshape)
    self.V_rest = init.param(V_rest, self.varshape)
    self.V_reset = init.param(V_reset, self.varshape)

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = StateWithDelay(init.param(init.Constant(self.V_reset), self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.V.reset(init.param(init.Constant(self.V_reset), self.varshape, batch_size))

  def get_spike(self, V: ArrayLike = None):
    V = self.V.value if V is None else V
    v_scaled = (V - self.V_th) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x=0.):
    last_v = self.V.value
    lst_spk = self.get_spike(last_v)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
    V = last_v - (V_th - self.V_reset) * lst_spk
    # membrane potential
    dv = lambda v: (-v + self.V_rest + self.sum_current_inputs(x, v)) / self.tau
    V = exp_euler_step(dv, V)
    V = self.sum_delta_inputs(V)
    self.V.value = V
    return self.get_spike(V)


class ALIF(Neuron):
  """Adaptive Leaky Integrate-and-Fire (LIF) neuron model."""
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      tau: ArrayLike = 5.,
      tau_a: ArrayLike = 100.,
      V_th: ArrayLike = 1.,
      beta: ArrayLike = 0.1,
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      name: str = None,
  ):
    super().__init__(in_size, keep_size=keep_size, name=name, spk_fun=spk_fun,
                     spk_dtype=spk_dtype, spk_reset=spk_reset)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.tau_a = init.param(tau_a, self.varshape)
    self.V_th = init.param(V_th, self.varshape)
    self.beta = init.param(beta, self.varshape)

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = StateWithDelay(init.param(init.Constant(0.), self.varshape, batch_size))
    self.a = ShortTermState(init.param(init.Constant(0.), self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.V.reset(init.param(init.Constant(0.), self.varshape, batch_size))
    self.a.value = init.param(init.Constant(0.), self.varshape, batch_size)

  def get_spike(self, V=None, a=None):
    V = self.V.value if V is None else V
    a = self.a.value if a is None else a
    v_scaled = (V - self.V_th - self.beta * a) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x=0.):
    last_v = self.V.value
    last_a = self.a.value
    lst_spk = self.get_spike(last_v, last_a)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
    V = last_v - V_th * lst_spk
    a = last_a + lst_spk
    # membrane potential
    dv = lambda v: (-v + self.sum_current_inputs(x, v)) / self.tau
    da = lambda a: -a / self.tau_a
    V = exp_euler_step(dv, V)
    a = exp_euler_step(da, a)
    self.V.value = self.sum_delta_inputs(V)
    self.a.value = a
    return self.get_spike(self.V.value, self.a.value)
