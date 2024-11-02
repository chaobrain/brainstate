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

#
# Implementation of the paper:
#
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007),
#   Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#

import brainunit as u
import dendritex as dx
import matplotlib.pyplot as plt

import brainstate as bst

num_exc = 3200
num_inh = 800

area = 20000 * u.um ** 2
area = area.in_unit(u.cm ** 2)
Cm = (1 * u.uF * u.cm ** -2) * area  # Membrane Capacitance [pF]

gl = (5. * u.nS * u.cm ** -2) * area  # Leak Conductance   [nS]
g_Na = (100. * u.mS * u.cm ** -2) * area  # Sodium Conductance [nS]
g_Kd = (30. * u.mS * u.cm ** -2) * area  # K Conductance      [nS]

El = -60. * u.mV  # Resting Potential [mV]
ENa = 50. * u.mV  # reversal potential (Sodium) [mV]
EK = -90. * u.mV  # reversal potential (Potassium) [mV]
VT = -63. * u.mV  # Threshold Potential [mV]
V_th = -20. * u.mV  # Spike Threshold [mV]

# Time constants
taue = 5. * u.ms  # Excitatory synaptic time constant [ms]
taui = 10. * u.ms  # Inhibitory synaptic time constant [ms]

# Reversal potentials
Ee = 0. * u.mV  # Excitatory reversal potential (mV)
Ei = -80. * u.mV  # Inhibitory reversal potential (Potassium) [mV]

# excitatory synaptic weight
we = 6. * u.nS  # excitatory synaptic conductance [nS]

# inhibitory synaptic weight
wi = 67. * u.nS  # inhibitory synaptic conductance [nS]


class HH(bst.nn.Dynamics):
  """
  Hodgkin-Huxley neuron model.
  """

  def __init__(self, in_size):
    super().__init__(in_size)

  def init_state(self, *args, **kwargs):
    # variables
    self.V = bst.HiddenState(El + (bst.random.randn(*self.varshape) * 5 - 5) * u.mV)
    self.m = bst.HiddenState(u.math.zeros(self.varshape))
    self.n = bst.HiddenState(u.math.zeros(self.varshape))
    self.h = bst.HiddenState(u.math.zeros(self.varshape))
    self.spike = bst.HiddenState(u.math.zeros(self.varshape, dtype=bool))

  def reset_state(self, *args, **kwargs):
    self.V.value = El + (bst.random.randn(self.varshape) * 5 - 5)
    self.m.value = u.math.zeros(self.varshape)
    self.n.value = u.math.zeros(self.varshape)
    self.h.value = u.math.zeros(self.varshape)
    self.spike.value = u.math.zeros(self.varshape, dtype=bool)

  def dV(self, V, t, m, h, n, Isyn):
    gna = g_Na * (m * m * m) * h
    gkd = g_Kd * (n * n * n * n)
    dVdt = (-gl * (V - El) - gna * (V - ENa) - gkd * (V - EK) + self.sum_current_inputs(Isyn, V)) / Cm
    return dVdt

  def dm(self, m, t, V, ):
    a = (- V + VT) / u.mV + 13
    b = (V - VT) / u.mV - 40
    m_alpha = 0.32 * a / (u.math.exp(a / 4) - 1.)
    m_beta = 0.28 * b / (u.math.exp(b / 5) - 1)
    dmdt = (m_alpha * (1 - m) - m_beta * m) / u.ms
    return dmdt

  def dh(self, h, t, V):
    c = (- V + VT) / u.mV + 17
    d = (V - VT) / u.mV - 40
    h_alpha = 0.128 * u.math.exp(c / 18)
    h_beta = 4. / (1 + u.math.exp(-d / 5))
    dhdt = (h_alpha * (1 - h) - h_beta * h) / u.ms
    return dhdt

  def dn(self, n, t, V):
    c = (- V + VT) / u.mV + 15
    d = (- V + VT) / u.mV + 10
    n_alpha = 0.032 * c / (u.math.exp(c / 5) - 1.)
    n_beta = .5 * u.math.exp(d / 40)
    dndt = (n_alpha * (1 - n) - n_beta * n) / u.ms
    return dndt

  def update(self, x=0. * u.mA):
    t = bst.environ.get('t')
    last_V = self.V.value
    V = bst.nn.exp_euler_step(self.dV, last_V, t, self.m.value, self.h.value, self.n.value, x)
    m = bst.nn.exp_euler_step(self.dm, self.m.value, t, last_V)
    h = bst.nn.exp_euler_step(self.dh, self.h.value, t, last_V)
    n = bst.nn.exp_euler_step(self.dn, self.n.value, t, last_V)
    self.spike.value = u.math.logical_and(last_V < V_th, V >= V_th)
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.V.value = V
    return self.spike.value


class EINet(bst.nn.DynamicsGroup):
  def __init__(self):
    super().__init__()
    self.n_exc = 3200
    self.n_inh = 800
    self.varshape = self.n_exc + self.n_inh
    self.N = HH(self.varshape)

    self.E = bst.nn.AlignPostProj(
      comm=bst.event.FixedProb(self.n_exc, self.varshape, prob=0.02, weight=we),
      syn=bst.nn.Expon(size=self.varshape, tau=taue),
      out=bst.nn.COBA(E=Ee),
      post=self.N
    )
    self.I = bst.nn.AlignPostProj(
      comm=bst.event.FixedProb(self.n_inh, self.varshape, prob=0.02, weight=wi),
      syn=bst.nn.Expon(size=self.varshape, tau=taui),
      out=bst.nn.COBA(E=Ei),
      post=self.N
    )

  def update(self, t):
    with bst.environ.context(t=t):
      spk = self.N.spike.value
      self.E(spk[:self.n_exc])
      self.I(spk[self.n_exc:])
      return self.N()


# network
net = EINet()
bst.nn.init_all_states(net, exclude=dx.IonChannel)

# simulation
with bst.environ.context(dt=0.1 * u.ms):
  times = u.math.arange(0. * u.ms, 1000. * u.ms, bst.environ.get_dt())
  spikes = bst.compile.for_loop(net.update, times, pbar=bst.compile.ProgressBar(10))

# visualization
times = times.to_decimal(u.ms)
t_indices, n_indices = u.math.where(spikes)
plt.plot(times[t_indices], n_indices, 'k.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
