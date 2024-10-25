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

import brainunit as u
import jax

from brainstate import init, surrogate, environ
from brainstate._state import ShortTermState
from brainstate.mixin import ParamDesc
from brainstate.nn._dynamics._dynamics_base import Dynamics
from brainstate.nn._dynamics._state_delay import StateWithDelay
from brainstate.nn._exp_euler import exp_euler_step
from brainstate.typing import ArrayLike, Size

__all__ = [
  'Neuron', 'IF', 'LIF', 'LIFRef', 'ALIF',
]


class Neuron(Dynamics, ParamDesc):
  """
  Base class for neuronal dynamics.

  All neuron models are differentiable since they use surrogate gradient functions to
  generate the spiking state.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      spk_fun: Callable = surrogate.InvSquareGrad(),
      spk_reset: str = 'soft',
      name: Optional[str] = None,
  ):
    super().__init__(size, keep_size=keep_size, name=name)
    self.spk_reset = spk_reset
    self.spk_fun = spk_fun

  def get_spike(self, *args, **kwargs):
    raise NotImplementedError


class IF(Neuron):
  """
  Integrate-and-fire neuron model.
  """

  __module__ = 'brainstate.nn'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      R: ArrayLike = 1. * u.ohm,
      tau: ArrayLike = 5. * u.ms,
      V_th: ArrayLike = 1. * u.mV,  # should be positive
      V_initializer: Callable = init.Constant(0. * u.mV),
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      name: str = None,
  ):
    super().__init__(size, keep_size=keep_size, name=name,
                     spk_fun=spk_fun, spk_reset=spk_reset)

    # parameters
    self.R = init.param(R, self.varshape)
    self.tau = init.param(tau, self.varshape)
    self.V_th = init.param(V_th, self.varshape)
    self.V_initializer = V_initializer

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = StateWithDelay(init.param(self.V_initializer, self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.V.reset(init.param(self.V_initializer, self.varshape, batch_size))

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    v_scaled = (V - self.V_th) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x=0. * u.mA):
    # reset
    last_V = self.V.value
    last_spike = self.get_spike(self.V.value)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
    V = last_V - V_th * last_spike
    # membrane potential
    dv = lambda v: (-v + self.R * self.sum_current_inputs(x, v)) / self.tau
    V = exp_euler_step(dv, V)
    V = self.sum_delta_inputs(V)
    self.V.value = V
    return self.get_spike(V)


class LIF(Neuron):
  """
  Leaky integrate-and-fire neuron model.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      R: ArrayLike = 1. * u.ohm,
      tau: ArrayLike = 5. * u.ms,
      V_th: ArrayLike = 1. * u.mV,
      V_reset: ArrayLike = 0. * u.mV,
      V_rest: ArrayLike = 0. * u.mV,
      V_initializer: Callable = init.Constant(0. * u.mV),
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      name: str = None,
  ):
    super().__init__(size, keep_size=keep_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

    # parameters
    self.R = init.param(R, self.varshape)
    self.tau = init.param(tau, self.varshape)
    self.V_th = init.param(V_th, self.varshape)
    self.V_rest = init.param(V_rest, self.varshape)
    self.V_reset = init.param(V_reset, self.varshape)
    self.V_initializer = V_initializer

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = StateWithDelay(init.param(self.V_initializer, self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.V.reset(init.param(self.V_initializer, self.varshape, batch_size))

  def get_spike(self, V: ArrayLike = None):
    V = self.V.value if V is None else V
    v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
    return self.spk_fun(v_scaled)

  def update(self, x=0. * u.mA):
    last_v = self.V.value
    lst_spk = self.get_spike(last_v)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
    V = last_v - (V_th - self.V_reset) * lst_spk
    # membrane potential
    dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
    V = exp_euler_step(dv, V)
    V = self.sum_delta_inputs(V)
    self.V.value = V
    return self.get_spike(V)


class LIFRef(Neuron):
  """
  Leaky integrate-and-fire neuron model with refractory period.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      R: ArrayLike = 1. * u.ohm,
      tau: ArrayLike = 5. * u.ms,
      tau_ref: ArrayLike = 5. * u.ms,
      V_th: ArrayLike = 1. * u.mV,
      V_reset: ArrayLike = 0. * u.mV,
      V_rest: ArrayLike = 0. * u.mV,
      V_initializer: Callable = init.Constant(0. * u.mV),
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      name: str = None,
  ):
    super().__init__(size, keep_size=keep_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

    # parameters
    self.R = init.param(R, self.varshape)
    self.tau = init.param(tau, self.varshape)
    self.tau_ref = init.param(tau_ref, self.varshape)
    self.V_th = init.param(V_th, self.varshape)
    self.V_rest = init.param(V_rest, self.varshape)
    self.V_reset = init.param(V_reset, self.varshape)
    self.V_initializer = V_initializer

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = StateWithDelay(init.param(self.V_initializer, self.varshape, batch_size))
    self.last_spike_time = ShortTermState(init.param(init.Constant(-1e7 * u.ms), self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.V.reset(init.param(self.V_initializer, self.varshape, batch_size))
    self.last_spike_time.value = init.param(init.Constant(-1e7 * u.ms), self.varshape, batch_size)

  def get_spike(self, V: ArrayLike = None):
    V = self.V.value if V is None else V
    v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
    return self.spk_fun(v_scaled)

  def update(self, x=0. * u.mA):
    t = environ.get('t')
    last_v = self.V.value
    lst_spk = self.get_spike(last_v)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
    last_v = last_v - (V_th - self.V_reset) * lst_spk
    # membrane potential
    dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
    V = exp_euler_step(dv, last_v)
    V = self.sum_delta_inputs(V)
    self.V.value = u.math.where(t - self.last_spike_time.value < self.tau_ref, last_v, V)
    # spike time evaluation
    lst_spk_time = u.math.where(self.V.value >= self.V_th, environ.get('t'), self.last_spike_time.value)
    self.last_spike_time.value = jax.lax.stop_gradient(lst_spk_time)
    return self.get_spike()


class ALIF(Neuron):
  """
  Adaptive Leaky Integrate-and-Fire (LIF) neuron model.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      R: ArrayLike = 1. * u.ohm,
      tau: ArrayLike = 5. * u.ms,
      tau_a: ArrayLike = 100. * u.ms,
      V_th: ArrayLike = 1. * u.mV,
      V_reset: ArrayLike = 0. * u.mV,
      V_rest: ArrayLike = 0. * u.mV,
      beta: ArrayLike = 0.1 * u.mV,
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      V_initializer: Callable = init.Constant(0. * u.mV),
      a_initializer: Callable = init.Constant(0.),
      name: str = None,
  ):
    super().__init__(size, keep_size=keep_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

    # parameters
    self.R = init.param(R, self.varshape)
    self.tau = init.param(tau, self.varshape)
    self.tau_a = init.param(tau_a, self.varshape)
    self.V_th = init.param(V_th, self.varshape)
    self.V_reset = init.param(V_reset, self.varshape)
    self.V_rest = init.param(V_rest, self.varshape)
    self.beta = init.param(beta, self.varshape)

    # functions
    self.V_initializer = V_initializer
    self.a_initializer = a_initializer

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = StateWithDelay(init.param(self.V_initializer, self.varshape, batch_size))
    self.a = ShortTermState(init.param(self.a_initializer, self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.V.reset(init.param(self.V_initializer, self.varshape, batch_size))
    self.a.value = init.param(self.a_initializer, self.varshape, batch_size)

  def get_spike(self, V=None, a=None):
    V = self.V.value if V is None else V
    a = self.a.value if a is None else a
    v_scaled = (V - self.V_th - self.beta * a) / (self.V_th - self.V_reset)
    return self.spk_fun(v_scaled)

  def update(self, x=0. * u.mA):
    last_v = self.V.value
    last_a = self.a.value
    lst_spk = self.get_spike(last_v, last_a)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
    V = last_v - (V_th - self.V_reset) * lst_spk
    a = last_a + lst_spk
    # membrane potential
    dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
    da = lambda a: -a / self.tau_a
    V = exp_euler_step(dv, V)
    a = exp_euler_step(da, a)
    self.V.value = self.sum_delta_inputs(V)
    self.a.value = a
    return self.get_spike(self.V.value, self.a.value)
