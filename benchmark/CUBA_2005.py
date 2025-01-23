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

import os
import sys

sys.path.append('../')
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import time

import brainunit as u

import brainstate as bst


class EINet(bst.nn.DynamicsGroup):
    def __init__(self, scale=1.0):
        super().__init__()
        self.n_exc = int(3200 * scale)
        self.n_inh = int(800 * scale)
        self.num = self.n_exc + self.n_inh
        self.N = bst.nn.LIFRef(
            self.num, V_rest=-49. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=bst.init.Normal(-55., 2., unit=u.mV)
        )
        self.E = bst.nn.AlignPostProj(
            comm=bst.event.FixedProb(self.n_exc, self.num, prob=80 / self.num, weight=1.62 * u.mS),
            syn=bst.nn.Expon.desc(self.num, tau=5. * u.ms),
            out=bst.nn.CUBA.desc(scale=u.volt),
            post=self.N
        )
        self.I = bst.nn.AlignPostProj(
            comm=bst.event.FixedProb(self.n_inh, self.num, prob=80 / self.num, weight=-9.0 * u.mS),
            syn=bst.nn.Expon.desc(self.num, tau=10. * u.ms),
            out=bst.nn.CUBA.desc(scale=u.volt),
            post=self.N
        )

    def init_state(self, *args, **kwargs):
        self.rate = bst.ShortTermState(u.math.zeros(self.num))

    def update(self, t, inp):
        with bst.environ.context(t=t):
            spk = self.N.get_spike()
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            self.rate.value += self.N.get_spike()


@bst.compile.jit(static_argnums=0)
def run(scale: float):
    # network
    net = EINet(scale)
    bst.nn.init_all_states(net)

    duration = 1e4 * u.ms
    # simulation
    with bst.environ.context(dt=0.1 * u.ms):
        times = u.math.arange(0. * u.ms, duration, bst.environ.get_dt())
        bst.compile.for_loop(lambda t: net.update(t, 20. * u.mA), times,
                             # pbar=bst.compile.ProgressBar(100)
                             )

    return net.num, net.rate.value.sum() / net.num / duration.to_decimal(u.second)


for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
    jax.block_until_ready(run(s))

    t0 = time.time()
    n, rate = jax.block_until_ready(run(s))
    t1 = time.time()
    print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')

# A6000 NVIDIA GPU

# scale=1, size=4000, time = 2.6354849338531494 s, firing rate = 24.982027053833008 Hz
# scale=2, size=8000, time = 2.6781561374664307 s, firing rate = 23.719463348388672 Hz
# scale=4, size=16000, time = 2.7448785305023193 s, firing rate = 24.592931747436523 Hz
# scale=6, size=24000, time = 2.8237478733062744 s, firing rate = 24.159996032714844 Hz
# scale=8, size=32000, time = 2.9344418048858643 s, firing rate = 24.956790924072266 Hz
# scale=10, size=40000, time = 3.042517900466919 s, firing rate = 23.644424438476562 Hz
# scale=20, size=80000, time = 3.6727631092071533 s, firing rate = 24.226743698120117 Hz
# scale=40, size=160000, time = 4.857396602630615 s, firing rate = 24.329742431640625 Hz
# scale=60, size=240000, time = 6.812030792236328 s, firing rate = 24.370006561279297 Hz
# scale=80, size=320000, time = 9.227966547012329 s, firing rate = 24.41067886352539 Hz
# scale=100, size=400000, time = 11.405697584152222 s, firing rate = 24.32524871826172 Hz


# AMD Ryzen 7 7840HS

# scale=1, size=4000, time = 1.1661601066589355 s, firing rate = 22.438201904296875 Hz
# scale=2, size=8000, time = 3.3255884647369385 s, firing rate = 23.868364334106445 Hz
# scale=4, size=16000, time = 6.950139999389648 s, firing rate = 24.21693229675293 Hz
# scale=6, size=24000, time = 10.011993169784546 s, firing rate = 24.240270614624023 Hz
# scale=8, size=32000, time = 13.027734518051147 s, firing rate = 24.753198623657227 Hz
# scale=10, size=40000, time = 16.449942350387573 s, firing rate = 24.7176570892334 Hz
# scale=20, size=80000, time = 30.754598140716553 s, firing rate = 24.119956970214844 Hz
# scale=40, size=160000, time = 63.6387836933136 s, firing rate = 24.72784996032715 Hz
# scale=60, size=240000, time = 78.58532166481018 s, firing rate = 24.402742385864258 Hz
# scale=80, size=320000, time = 102.4250214099884 s, firing rate = 24.59092140197754 Hz
# scale=100, size=400000, time = 145.35173273086548 s, firing rate = 24.33751106262207 Hz
