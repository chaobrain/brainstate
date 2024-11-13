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
# which is based on the balanced network proposed by:
#
# - Vogels, T. P. and Abbott, L. F. (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., J. Neurosci., 25, 46, 10786–95
#


import brainunit as u
import matplotlib.pyplot as plt

import brainstate as bst


class EINet(bst.nn.DynamicsGroup):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.num = self.n_exc + self.n_inh
        self.N = bst.nn.LIFRef(
            self.num, V_rest=-49. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=bst.init.Normal(-55., 2., unit=u.mV)
        )
        self.E = bst.nn.AlignPostProj(
            comm=bst.event.FixedProb(self.n_exc, self.num, prob=0.02, weight=1.62 * u.mS),
            syn=bst.nn.Expon.desc(self.num, tau=5. * u.ms),
            out=bst.nn.CUBA.desc(scale=u.volt),
            post=self.N
        )
        self.I = bst.nn.AlignPostProj(
            comm=bst.event.FixedProb(self.n_inh, self.num, prob=0.02, weight=-9.0 * u.mS),
            syn=bst.nn.Expon.desc(self.num, tau=10. * u.ms),
            out=bst.nn.CUBA.desc(scale=u.volt),
            post=self.N
        )

    def update(self, t, inp):
        with bst.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            return self.N.get_spike()


# network
net = EINet()
bst.nn.init_all_states(net)

# simulation
with bst.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 1000. * u.ms, bst.environ.get_dt())
    spikes = bst.compile.for_loop(lambda t: net.update(t, 20. * u.mA), times, pbar=bst.compile.ProgressBar(10))

# visualization
t_indices, n_indices = u.math.where(spikes)
plt.plot(times[t_indices], n_indices, 'k.', markersze=1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
