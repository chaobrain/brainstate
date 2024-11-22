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

V_th = -20. * u.mV


class HH(dx.neurons.SingleCompartment):
    def __init__(self, in_size):
        super().__init__(in_size, C=0.0002 * u.uF)
        self.na = dx.ions.SodiumFixed(in_size)
        self.na.add_elem(INa=dx.channels.INa_TM1991(in_size, g_max=0.02 * u.mS, V_sh=-63. * u.mV))

        self.k = dx.ions.PotassiumFixed(in_size, E=-90 * u.mV)
        self.k.add_elem(IK=dx.channels.IK_TM1991(in_size, g_max=0.006 * u.mS, V_sh=-63. * u.mV))

        self.IL = dx.channels.IL(in_size, E=-60. * u.mV, g_max=0.001 * u.nS)

    def update(self):
        dx.rk4_step(self, bst.environ.get('t'), 0. * u.nA)


class EINet(bst.nn.DynamicsGroup):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.num = self.n_exc + self.n_inh
        self.N = HH(self.num)

        self.E = bst.nn.AlignPostProj(
            comm=bst.event.FixedProb(self.n_exc, self.num, prob=0.02, weight=6. * u.nS),
            syn=bst.nn.Expon(self.num, tau=5. * u.ms),
            out=bst.nn.COBA(E=0. * u.mV),
            post=self.N
        )
        self.I = bst.nn.AlignPostProj(
            comm=bst.event.FixedProb(self.n_inh, self.num, prob=0.02, weight=67. * u.nS),
            syn=bst.nn.Expon(self.num, tau=10. * u.ms),
            out=bst.nn.COBA(E=-80. * u.mV),
            post=self.N
        )

    def init_state(self, *args, **kwargs):
        self.last_v = bst.ShortTermState(self.N.V.value)

    def reset_state(self, *args, **kwargs):
        self.last_v.value = self.N.V.value

    def update(self, t):
        with bst.environ.context(t=t):
            spk = u.math.squeeze(u.math.logical_and(self.last_v.value < V_th, self.N.V.value >= V_th))
            self.last_v.value = self.N.V.value
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N()
            spk = u.math.squeeze(u.math.logical_and(self.last_v.value < V_th, self.N.V.value >= V_th))
            return spk, self.N.V.value[:10]


# network
net = EINet()
bst.nn.init_all_states(net, exclude=dx.IonChannel)

# simulation
with bst.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 1000. * u.ms, bst.environ.get_dt())
    spikes, vs = bst.compile.for_loop(net.update, times, pbar=bst.compile.ProgressBar(10))

# visualization
plt.plot(times, vs)
plt.show()

t_indices, n_indices = u.math.where(spikes)
plt.scatter(times[t_indices], n_indices, s=1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
