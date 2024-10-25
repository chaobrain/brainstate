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
# Implementation of the EI network from Brunel (1996) with the brainstate package.
#
# - Van Vreeswijk, Carl, and Haim Sompolinsky. “Chaos in neuronal networks with balanced
#   excitatory and inhibitory activity.” Science 274.5293 (1996): 1724-1726.
#
# Dynamic of membrane potential is given as:
#
# $$ \tau \frac {dV_i}{dt} = -(V_i - V_{rest}) + I_i^{ext} + I_i^{net} (t) $$
#
# where $I_i^{net}(t)$ represents the synaptic current, which describes the sum of excitatory and inhibitory neurons.
#
# $$ I_i^{net} (t) = J_E \sum_{j=1}^{pN_e} \sum_{t_j^\alpha < t} f(t-t_j^\alpha ) - J_I \sum_{j=1}^{pN_i} \sum_{t_j^\alpha < t} f(t-t_j^\alpha )$$
#
# where
#
# $$ f(t) = \begin{cases} {\rm exp} (-\frac t {\tau_s} ), \quad t \geq 0 \\
# 0, \quad t < 0 \end{cases} $$
#
# Parameters: $J_E = \frac 1 {\sqrt {pN_e}}, J_I = \frac 1 {\sqrt {pN_i}}$
#


import brainunit as u
import brainstate as bst
import matplotlib.pyplot as plt


class EINet(bst.nn.DynamicsGroup):
  def __init__(self, n_exc, n_inh, prob, JE, JI):
    super().__init__()
    self.n_exc = n_exc
    self.n_inh = n_inh
    self.num = n_exc + n_inh

    # neurons
    self.N = bst.nn.LIF(n_exc + n_inh, V_rest=-52. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV, tau=10. * u.ms,
                        V_initializer=bst.init.Normal(-60., 10., unit=u.mV), spk_reset='soft')

    # synapses
    self.E = bst.nn.AlignPostProj(
      comm=bst.event.FixedProb(n_exc, self.num, prob, JE),
      syn=bst.nn.Expon.desc(self.num, tau=2. * u.ms),
      out=bst.nn.CUBA.desc(),
      post=self.N,
    )
    self.I = bst.nn.AlignPostProj(
      comm=bst.event.FixedProb(n_inh, self.num, prob, JI),
      syn=bst.nn.Expon.desc(self.num, tau=2. * u.ms),
      out=bst.nn.CUBA.desc(),
      post=self.N,
    )

  def update(self, inp):
    spks = self.N.get_spike() != 0.
    self.E(spks[:self.n_exc])
    self.I(spks[self.n_exc:])
    self.N(inp)
    return self.N.get_spike()


# connectivity
num_exc = 500
num_inh = 500
prob = 0.1
# external current
Ib = 3. * u.mA
# excitatory and inhibitory synaptic weights
JE = 1 / u.math.sqrt(prob * num_exc) * u.mS
JI = -1 / u.math.sqrt(prob * num_inh) * u.mS

# network
bst.environ.set(dt=0.1 * u.ms)
net = EINet(num_exc, num_inh, prob=prob, JE=JE, JI=JI)
bst.nn.init_all_states(net)

# simulation
times = u.math.arange(0. * u.ms, 1000. * u.ms, bst.environ.get_dt())
spikes = bst.compile.for_loop(lambda t: net.update(Ib), times, pbar=bst.compile.ProgressBar(10))

# visualization
times = times.to_decimal(u.ms)
t_indices, n_indices = u.math.where(spikes)
plt.plot(times[t_indices], n_indices, 'k.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()