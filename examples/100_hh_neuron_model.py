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


import brainunit as u
import brainstate
import jax.numpy as jnp
import matplotlib.pyplot as plt


class HH(brainstate.nn.Dynamics):
    def __init__(
        self,
        in_size,
        ENa=50. * u.mV, gNa=120. * u.mS / u.cm ** 2,
        EK=-77. * u.mV, gK=36. * u.mS / u.cm ** 2,
        EL=-54.387 * u.mV, gL=0.03 * u.mS / u.cm ** 2,
        V_th=20. * u.mV,
        C=1.0 * u.uF / u.cm ** 2
    ):
        # initialization
        super().__init__(in_size)

        # parameters
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th

    # m channel
    m_alpha = lambda self, V: 1. / u.math.exprel(-(V / u.mV + 40) / 10)
    m_beta = lambda self, V: 4.0 * jnp.exp(-(V / u.mV + 65) / 18)
    m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))
    dm = lambda self, m, t, V: (self.m_alpha(V) * (1 - m) - self.m_beta(V) * m) / u.ms

    # h channel
    h_alpha = lambda self, V: 0.07 * jnp.exp(-(V / u.mV + 65) / 20.)
    h_beta = lambda self, V: 1 / (1 + jnp.exp(-(V / u.mV + 35) / 10))
    h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))
    dh = lambda self, h, t, V: (self.h_alpha(V) * (1 - h) - self.h_beta(V) * h) / u.ms

    # n channel
    n_alpha = lambda self, V: 0.1 / u.math.exprel(-(V / u.mV + 55) / 10)
    n_beta = lambda self, V: 0.125 * jnp.exp(-(V / u.mV + 65) / 80)
    n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))
    dn = lambda self, n, t, V: (self.n_alpha(V) * (1 - n) - self.n_beta(V) * n) / u.ms

    def init_state(self, batch_size=None):
        self.V = brainstate.HiddenState(jnp.ones(self.varshape, brainstate.environ.dftype()) * -65. * u.mV)
        self.m = brainstate.HiddenState(self.m_inf(self.V.value))
        self.h = brainstate.HiddenState(self.h_inf(self.V.value))
        self.n = brainstate.HiddenState(self.n_inf(self.V.value))

    def dV(self, V, t, m, h, n, I):
        I = self.sum_current_inputs(I, V)
        I_Na = (self.gNa * m * m * m * h) * (V - self.ENa)
        n2 = n * n
        I_K = (self.gK * n2 * n2) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        dVdt = (- I_Na - I_K - I_leak + I) / self.C
        return dVdt

    def update(self, x=0. * u.mA / u.cm ** 2):
        t = brainstate.environ.get('t')
        V = brainstate.nn.exp_euler_step(self.dV, self.V.value, t, self.m.value, self.h.value, self.n.value, x)
        m = brainstate.nn.exp_euler_step(self.dm, self.m.value, t, self.V.value)
        h = brainstate.nn.exp_euler_step(self.dh, self.h.value, t, self.V.value)
        n = brainstate.nn.exp_euler_step(self.dn, self.n.value, t, self.V.value)
        V = self.sum_delta_inputs(init=V)
        spike = jnp.logical_and(self.V.value < self.V_th, V >= self.V_th)
        self.V.value = V
        self.m.value = m
        self.h.value = h
        self.n.value = n
        return spike



hh = HH(10)
brainstate.nn.init_all_states(hh)
dt = 0.01 * u.ms


def run(t, inp):
    with brainstate.environ.context(t=t, dt=dt):
        hh(inp)
    return hh.V.value


times = u.math.arange(0. * u.ms, 100. * u.ms, dt)
vs = brainstate.compile.for_loop(
    run,
    # times, random inputs
    times, brainstate.random.uniform(1., 10., times.shape) * u.uA / u.cm ** 2,
    pbar=brainstate.compile.ProgressBar(count=100)
)

plt.plot(times.to_decimal(u.ms), vs.to_decimal(u.mV))
plt.show()
