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

import brainunit as u

from brainstate import init, environ
from brainstate._state import ShortTermState
from brainstate.mixin import ParamDesc, AlignPost
from brainstate.nn._dynamics._dynamics_base import Dynamics
from brainstate.nn._exp_euler import exp_euler_step
from brainstate.typing import ArrayLike, Size


__all__ = [
  'Synapse', 'Expon', 'STP', 'STD', 'AMPA', 'GABAa',
]


class Synapse(Dynamics, ParamDesc):
  """
  Base class for synapse dynamics.
  """
  __module__ = 'brainstate.nn'


class Expon(Synapse, AlignPost):
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
      tau: ArrayLike = 8.0 * u.ms,
      g_initializer: ArrayLike = init.ZeroInit(unit=u.mS),
  ):
    super().__init__(name=name, size=size, keep_size=keep_size)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.g_initializer = g_initializer

  def init_state(self, batch_size: int = None, **kwargs):
    self.g = ShortTermState(init.param(self.g_initializer, self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.g.value = init.param(self.g_initializer, self.varshape, batch_size)

  def update(self, x=None):
    g = exp_euler_step(lambda g: self.sum_current_inputs(-g) / self.tau, self.g.value)
    self.g.value = self.sum_delta_inputs(g)
    if x is not None: self.g.value += x
    return self.g.value


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
      tau_f: ArrayLike = 1500. * u.ms,
      tau_d: ArrayLike = 200. * u.ms,
  ):
    super().__init__(name=name, size=size, keep_size=keep_size)

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
      tau: ArrayLike = 200. * u.ms,
      U: ArrayLike = 0.07,
  ):
    super().__init__(name=name, size=size, keep_size=keep_size)

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


class AMPA(Synapse):
  r"""AMPA synapse model.

  **Model Descriptions**

  AMPA receptor is an ionotropic receptor, which is an ion channel.
  When it is bound by neurotransmitters, it will immediately open the
  ion channel, causing the change of membrane potential of postsynaptic neurons.

  A classical model is to use the Markov process to model ion channel switch.
  Here :math:`g` represents the probability of channel opening, :math:`1-g`
  represents the probability of ion channel closing, and :math:`\alpha` and
  :math:`\beta` are the transition probability. Because neurotransmitters can
  open ion channels, the transfer probability from :math:`1-g` to :math:`g`
  is affected by the concentration of neurotransmitters. We denote the concentration
  of neurotransmitters as :math:`[T]` and get the following Markov process.

  .. image:: ../../_static/synapse_markov.png
      :align: center

  We obtained the following formula when describing the process by a differential equation.

  .. math::

      \frac{ds}{dt} =\alpha[T](1-g)-\beta g

  where :math:`\alpha [T]` denotes the transition probability from state :math:`(1-g)`
  to state :math:`(g)`; and :math:`\beta` represents the transition probability of
  the other direction. :math:`\alpha` is the binding constant. :math:`\beta` is the
  unbinding constant. :math:`[T]` is the neurotransmitter concentration, and
  has the duration of 0.5 ms.

  Moreover, the post-synaptic current on the post-synaptic neuron is formulated as

  .. math::

      I_{syn} = g_{max} g (V-E)

  where :math:`g_{max}` is the maximum conductance, and `E` is the reverse potential.

  This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

  .. [1] Vijayan S, Kopell N J. Thalamic model of awake alpha oscillations
         and implications for stimulus processing[J]. Proceedings of the
         National Academy of Sciences, 2012, 109(45): 18553-18558.

  Args:
    alpha: float, ArrayType, Callable. Binding constant. [ms^-1 mM^-1]
    beta: float, ArrayType, Callable. Unbinding constant. [ms^-1 mM^-1]
    T: float, ArrayType, Callable. Transmitter concentration when synapse is triggered by
      a pre-synaptic spike. Default 0.5 [mM].
    T_dur: float, ArrayType, Callable. Transmitter concentration duration time after being triggered. Default 1 [ms]
  """

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      alpha: ArrayLike = 0.98 / (u.ms * u.mM),
      beta: ArrayLike = 0.18 / u.ms,
      T: ArrayLike = 0.5 * u.mM,
      T_dur: ArrayLike = 0.5 * u.ms,
      g_initializer: ArrayLike = init.ZeroInit(),
  ):
    super().__init__(name=name, size=size, keep_size=keep_size)

    # parameters
    self.alpha = init.param(alpha, self.varshape)
    self.beta = init.param(beta, self.varshape)
    self.T = init.param(T, self.varshape)
    self.T_duration = init.param(T_dur, self.varshape)
    self.g_initializer = g_initializer

  def init_state(self, batch_size=None):
    self.g = ShortTermState(init.param(self.g_initializer, self.varshape, batch_size))
    self.spike_arrival_time = ShortTermState(init.param(init.Constant(-1e7 * u.ms), self.varshape, batch_size))

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.g.value = init.param(self.g_initializer, self.varshape, batch_or_mode)
    self.spike_arrival_time.value = init.param(init.Constant(-1e7 * u.ms), self.varshape, batch_or_mode)

  def dg(self, g, t, TT):
    return self.alpha * TT * (1 - g) - self.beta * g

  def update(self, pre_spike):
    t = environ.get('t')
    self.spike_arrival_time.value = u.math.where(pre_spike, t, self.spike_arrival_time.value)
    TT = ((t - self.spike_arrival_time.value) < self.T_duration) * self.T
    self.g.value = exp_euler_step(self.dg, self.g.value, t, TT)
    return self.g.value


class GABAa(AMPA):
  r"""GABAa synapse model.

  **Model Descriptions**

  GABAa synapse model has the same equation with the `AMPA synapse <./brainmodels.synapses.AMPA.rst>`_,

  .. math::

      \frac{d g}{d t}&=\alpha[T](1-g) - \beta g \\
      I_{syn}&= - g_{max} g (V - E)

  but with the difference of:

  - Reversal potential of synapse :math:`E` is usually low, typically -80. mV
  - Activating rate constant :math:`\alpha=0.53`
  - De-activating rate constant :math:`\beta=0.18`
  - Transmitter concentration :math:`[T]=1\,\mu ho(\mu S)` when synapse is
    triggered by a pre-synaptic spike, with the duration of 1. ms.

  This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

  .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
         on the integrative properties of neocortical pyramidal neurons
         in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.

  Args:
    alpha: float, ArrayType, Callable. Binding constant. Default 0.062
    beta: float, ArrayType, Callable. Unbinding constant. Default 3.57
    T: float, ArrayType, Callable. Transmitter concentration when synapse is triggered by
      a pre-synaptic spike.. Default 1 [mM].
    T_dur: float, ArrayType, Callable. Transmitter concentration duration time
      after being triggered. Default 1 [ms]
  """

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      alpha: ArrayLike = 0.53 / (u.ms * u.mM),
      beta: ArrayLike = 0.18 / u.ms,
      T: ArrayLike = 1.0 * u.mM,
      T_dur: ArrayLike = 1.0 * u.ms,
  ):
    super().__init__(alpha=alpha, beta=beta, T=T,
                     T_dur=T_dur, name=name,
                     size=size, keep_size=keep_size)
