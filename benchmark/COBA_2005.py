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


import time

import brainunit as u
import jax

import braintools
import brainstate


class EINet(brainstate.nn.DynamicsGroup):
    def __init__(self, scale):
        super().__init__()
        self.n_exc = int(3200 * scale)
        self.n_inh = int(800 * scale)
        self.num = self.n_exc + self.n_inh
        self.N = brainstate.nn.LIFRef(
            self.num, V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=braintools.init.NormalInit(-55., 2., unit=u.mV)
        )
        self.E = brainstate.nn.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.num, conn_num=80 / self.num, conn_weight=0.6 * u.mS),
            syn=brainstate.nn.Expon.desc(self.num, tau=5. * u.ms),
            out=brainstate.nn.COBA.desc(E=0. * u.mV),
            post=self.N
        )
        self.I = brainstate.nn.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.num, conn_num=80 / self.num, conn_weight=6.7 * u.mS),
            syn=brainstate.nn.Expon.desc(self.num, tau=10. * u.ms),
            out=brainstate.nn.COBA.desc(E=-80. * u.mV),
            post=self.N
        )

    def init_state(self, *args, **kwargs):
        self.rate = brainstate.ShortTermState(u.math.zeros(self.num))

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            self.rate.value += self.N.get_spike()


@brainstate.compile.jit(static_argnums=0)
def run(scale: float):
    # network
    net = EINet(scale)
    brainstate.nn.init_all_states(net)

    duration = 1e4 * u.ms
    # simulation
    with brainstate.environ.context(dt=0.1 * u.ms):
        times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
        brainstate.compile.for_loop(lambda t: net.update(t, 20. * u.mA), times)

    return net.num, net.rate.value.sum() / net.num / duration.to_decimal(u.second)


for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
    jax.block_until_ready(run(s))

    t0 = time.time()
    n, rate = jax.block_until_ready(run(s))
    t1 = time.time()
    print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')

# A6000 NVIDIA GPU

# scale=1, size=4000, time = 2.659956455230713 s, firing rate = 50.62445068359375 Hz
# scale=2, size=8000, time = 2.7318649291992188 s, firing rate = 50.613040924072266 Hz
# scale=4, size=16000, time = 2.807222604751587 s, firing rate = 50.60573959350586 Hz
# scale=6, size=24000, time = 3.026782512664795 s, firing rate = 50.60918045043945 Hz
# scale=8, size=32000, time = 3.1258811950683594 s, firing rate = 50.607574462890625 Hz
# scale=10, size=40000, time = 3.172346353530884 s, firing rate = 50.60942840576172 Hz
# scale=20, size=80000, time = 3.751189947128296 s, firing rate = 50.612369537353516 Hz
# scale=40, size=160000, time = 5.0217814445495605 s, firing rate = 50.617958068847656 Hz
# scale=60, size=240000, time = 7.002646207809448 s, firing rate = 50.61948776245117 Hz
# scale=80, size=320000, time = 9.384576320648193 s, firing rate = 50.618499755859375 Hz
# scale=100, size=400000, time = 11.69654369354248 s, firing rate = 50.61605453491211 Hz


# AMD Ryzen 7 7840HS

# scale=1, size=4000, time = 4.436027526855469 s, firing rate = 50.6119270324707 Hz
# scale=2, size=8000, time = 8.349745273590088 s, firing rate = 50.612266540527344 Hz
# scale=4, size=16000, time = 16.39163303375244 s, firing rate = 50.61349105834961 Hz
# scale=6, size=24000, time = 15.725558042526245 s, firing rate = 50.6125602722168 Hz
# scale=8, size=32000, time = 21.31995177268982 s, firing rate = 50.61244583129883 Hz
# scale=10, size=40000, time = 27.811061143875122 s, firing rate = 50.61423873901367 Hz
# scale=20, size=80000, time = 45.54235219955444 s, firing rate = 50.61320877075195 Hz
# scale=40, size=160000, time = 82.22228026390076 s, firing rate = 50.61309814453125 Hz
# scale=60, size=240000, time = 125.44037556648254 s, firing rate = 50.613094329833984 Hz
# scale=80, size=320000, time = 171.20458459854126 s, firing rate = 50.613365173339844 Hz
# scale=100, size=400000, time = 215.4547393321991 s, firing rate = 50.6129150390625 Hz
