# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainpy
import braintools
import brainunit as u

import brainstate

brainstate.environ.set(dt=0.1 * u.ms)


# Create a network with connections
class SimpleNet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = brainpy.state.LIFRef(5, V_rest=-65. * u.mV, V_th=-50. * u.mV,
                                        V_reset=-60. * u.mV, tau=20. * u.ms, tau_ref=5. * u.ms)
        self.post = brainpy.state.LIFRef(3, V_rest=-65. * u.mV, V_th=-50. * u.mV,
                                         V_reset=-60. * u.mV, tau=10. * u.ms, tau_ref=5. * u.ms)
        self.conn = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(5, 3, conn_num=0.2, conn_weight=1.0 * u.mS),
            syn=brainpy.state.Expon.desc(3, tau=5. * u.ms),
            out=brainpy.state.CUBA.desc(scale=u.volt),
            post=self.post
        )

    def update(self, t):
        with brainstate.environ.context(t=t):
            pre_spk = self.pre.get_spike() != 0.
            self.conn(pre_spk)
            self.pre(0. * u.mA)
            self.post(0. * u.mA)


class TwoPopNet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 100
        self.n_inh = 25

        # Excitatory population
        self.exc = brainpy.state.LIFRef(
            self.n_exc,
            V_rest=-65. * u.mV,
            V_th=-50. * u.mV,
            V_reset=-60. * u.mV,
            tau=20. * u.ms,
            tau_ref=5. * u.ms,
        )

        # Inhibitory population
        self.inh = brainpy.state.LIFRef(
            self.n_inh,
            V_rest=-65. * u.mV,
            V_th=-50. * u.mV,
            V_reset=-60. * u.mV,
            tau=10. * u.ms,
            tau_ref=5. * u.ms,
        )

        # Excitatory -> Inhibitory projection
        self.exc2inh = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                self.n_exc, self.n_inh,
                conn_num=0.1,
                conn_weight=1.0 * u.mS
            ),
            syn=brainpy.state.Expon.desc(self.n_inh, tau=5. * u.ms),
            out=brainpy.state.CUBA.desc(scale=u.volt),
            post=self.inh
        )

    def update(self, t, inp_exc, inp_inh):
        with brainstate.environ.context(t=t):
            exc_spk = self.exc.get_spike() != 0.
            self.exc2inh(exc_spk)
            self.exc(inp_exc)
            self.inh(inp_inh)
            return self.exc.get_spike(), self.inh.get_spike()


class Single_Pop_EI_COBA_Net(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.num = self.n_exc + self.n_inh
        self.N = brainpy.state.LIFRef(
            self.num, V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=braintools.init.Normal(-55., 2., unit=u.mV)
        )
        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.num, conn_num=0.02, conn_weight=0.6 * u.mS),
            syn=brainpy.state.Expon.desc(self.num, tau=5. * u.ms),
            out=brainpy.state.COBA.desc(E=0. * u.mV),
            post=self.N
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.num, conn_num=0.02, conn_weight=6.7 * u.mS),
            syn=brainpy.state.Expon.desc(self.num, tau=10. * u.ms),
            out=brainpy.state.COBA.desc(E=-80. * u.mV),
            post=self.N
        )

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            return self.N.get_spike()

    def step_run(self, t, inp):
        with brainstate.environ.context(t=t):
            return self.update(t, inp)


class Single_Pop_EI_CUBA_Net(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.num = self.n_exc + self.n_inh
        self.N = brainpy.state.LIFRef(
            self.num, V_rest=-49. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=braintools.init.Normal(-55. * u.mV, 2. * u.mV)
        )
        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.num, conn_num=0.02, conn_weight=1.62 * u.mS),
            syn=brainpy.state.Expon.desc(self.num, tau=5. * u.ms),
            out=brainpy.state.CUBA.desc(scale=u.volt),
            post=self.N
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.num, conn_num=0.02, conn_weight=-9.0 * u.mS),
            syn=brainpy.state.Expon.desc(self.num, tau=10. * u.ms),
            out=brainpy.state.CUBA.desc(scale=u.volt),
            post=self.N
        )

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            return self.N.get_spike()

    def step_run(self, t, inp):
        with brainstate.environ.context(t=t):
            return self.update(t, inp)


class HH(brainpy.state.Neuron):
    """
    Hodgkin-Huxley neuron model.
    """

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

    def __init__(self, in_size):
        super().__init__(in_size)

    def init_state(self, *args, **kwargs):
        # variables
        self.V = brainstate.HiddenState(self.El + (brainstate.random.randn(*self.varshape) * 5 - 5) * u.mV)
        self.m = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=brainstate.environ.dftype()))
        self.n = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=brainstate.environ.dftype()))
        self.h = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=brainstate.environ.dftype()))
        self.spike = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=bool))

    def reset_state(self, *args, **kwargs):
        self.V.value = self.El + (brainstate.random.randn(self.varshape) * 5 - 5)
        self.m.value = u.math.zeros(self.varshape)
        self.n.value = u.math.zeros(self.varshape)
        self.h.value = u.math.zeros(self.varshape)
        self.spike.value = u.math.zeros(self.varshape, dtype=bool)

    def dV(self, V, m, h, n, Isyn):
        gna = self.g_Na * (m * m * m) * h
        gkd = self.g_Kd * (n * n * n * n)
        dVdt = (-self.gl * (V - self.El) - gna * (V - self.ENa) -
                gkd * (V - self.EK) + self.sum_current_inputs(Isyn, V)) / self.Cm
        return dVdt

    def dm(self, m, V, ):
        a = (- V + self.VT) / u.mV + 13
        b = (V - self.VT) / u.mV - 40
        m_alpha = 0.32 * 4 / u.math.exprel(a / 4)
        m_beta = 0.28 * 5 / u.math.exprel(b / 5)
        dmdt = (m_alpha * (1 - m) - m_beta * m) / u.ms
        return dmdt

    def dh(self, h, V):
        c = (- V + self.VT) / u.mV + 17
        d = (V - self.VT) / u.mV - 40
        h_alpha = 0.128 * u.math.exp(c / 18)
        h_beta = 4. / (1 + u.math.exp(-d / 5))
        dhdt = (h_alpha * (1 - h) - h_beta * h) / u.ms
        return dhdt

    def dn(self, n, V):
        c = (- V + self.VT) / u.mV + 15
        d = (- V + self.VT) / u.mV + 10
        n_alpha = 0.032 * 5 / u.math.exprel(c / 5)
        n_beta = .5 * u.math.exp(d / 40)
        dndt = (n_alpha * (1 - n) - n_beta * n) / u.ms
        return dndt

    def update(self, x=0. * u.mA):
        last_V = self.V.value
        V = brainstate.nn.exp_euler_step(self.dV, last_V, self.m.value, self.h.value, self.n.value, x)
        m = brainstate.nn.exp_euler_step(self.dm, self.m.value, last_V)
        h = brainstate.nn.exp_euler_step(self.dh, self.h.value, last_V)
        n = brainstate.nn.exp_euler_step(self.dn, self.n.value, last_V)
        self.spike.value = u.math.logical_and(last_V < self.V_th, V >= self.V_th)
        self.m.value = m
        self.h.value = h
        self.n.value = n
        self.V.value = V
        return self.spike.value


class Single_Pop_HH_EI_Net(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.varshape = self.n_exc + self.n_inh
        self.N = HH(self.varshape)

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

        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.varshape, conn_num=0.02, conn_weight=we),
            syn=brainpy.state.Expon(self.varshape, tau=taue),
            out=brainpy.state.COBA(E=Ee),
            post=self.N
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.varshape, conn_num=0.02, conn_weight=wi),
            syn=brainpy.state.Expon(self.varshape, tau=taui),
            out=brainpy.state.COBA(E=Ei),
            post=self.N
        )

    def update(self, t):
        with brainstate.environ.context(t=t):
            spk = self.N.spike.value
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            r = self.N()
            return r

    def run(self):
        import matplotlib.pyplot as plt

        # network
        net = Single_Pop_HH_EI_Net()
        brainstate.nn.init_all_states(net)

        # simulation
        with brainstate.environ.context(dt=0.04 * u.ms):
            times = u.math.arange(0. * u.ms, 300. * u.ms, brainstate.environ.get_dt())
            times = u.math.asarray(times, dtype=brainstate.environ.dftype())
            spikes = brainstate.transform.for_loop(net.update, times, pbar=brainstate.transform.ProgressBar(100))

        # visualization
        t_indices, n_indices = u.math.where(spikes)
        plt.scatter(u.math.asarray(times[t_indices] / u.ms), n_indices, s=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
        plt.show()
