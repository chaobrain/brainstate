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

from typing import Optional

import braincell
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


class _HH(brainpy.state.Neuron):
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
        self.N = _HH(self.varshape)

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


class _HH_braincell(braincell.SingleCompartment):
    def __init__(self, in_size):
        area = 20000 * u.um ** 2
        area = area.in_unit(u.cm ** 2)
        Cm = (1 * u.uF * u.cm ** -2) * area  # Membrane Capacitance [pF]

        super().__init__(in_size, C=Cm, solver='ind_exp_euler')
        self.na = braincell.ion.SodiumFixed(in_size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.INa_TM1991(in_size, g_max=100. * u.mS / u.cm ** 2 * area, V_sh=-63. * u.mV))

        self.k = braincell.ion.PotassiumFixed(in_size, E=-90 * u.mV)
        self.k.add(IK=braincell.channel.IK_TM1991(in_size, g_max=30. * u.mS / u.cm ** 2 * area, V_sh=-63. * u.mV))

        self.IL = braincell.channel.IL(in_size, E=-60. * u.mV, g_max=5. * u.nS / u.cm ** 2 * area)


class Single_Pop_HH_braincell_EI_Net(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.num = self.n_exc + self.n_inh
        self.N = _HH(self.num)

        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.num, conn_num=0.02, conn_weight=6. * u.nS),
            syn=brainpy.state.Expon(self.num, tau=5. * u.ms),
            out=brainpy.state.COBA(E=0. * u.mV),
            post=self.N
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.num, conn_num=0.02, conn_weight=67. * u.nS),
            syn=brainpy.state.Expon(self.num, tau=10. * u.ms),
            out=brainpy.state.COBA(E=-80. * u.mV),
            post=self.N
        )

    def update(self, t):
        with brainstate.environ.context(t=t):
            spk = self.N.spike.value
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            spk = self.N(0. * u.nA)
            return spk


def single_pop_strial_network():
    class NaChannel(braincell.channel.INa_p3q_markov):
        def f_p_alpha(self, V):
            return 0.32 * 4. / u.math.exprel(-(V / u.mV + 54.) / 4.)

        def f_p_beta(self, V):
            return 0.28 * 5. / u.math.exprel((V / u.mV + 27.) / 5.)

        def f_q_alpha(self, V):
            return 0.128 * u.math.exp(-(V / u.mV + 50.) / 18.)

        def f_q_beta(self, V):
            return 4. / (1 + u.math.exp(-(V / u.mV + 27.) / 5.))

    class KChannel(braincell.channel.IK_p4_markov):
        def f_p_alpha(self, V):
            return 0.032 * 5. / u.math.exprel(-(V / u.mV + 52.) / 5.)

        def f_p_beta(self, V):
            return 0.5 * u.math.exp(-(V / u.mV + 57.) / 40.)

    class MChannel(braincell.channel.PotassiumChannel):
        def __init__(self, size, g_max=1.3 * (u.mS / u.cm ** 2), E=-95. * u.mV, T=u.celsius2kelvin(37)):
            super().__init__(size)
            self.g_max = g_max
            self.E = E
            self.T = T
            self.phi = 2.3 ** ((u.kelvin2celsius(T) - 23.) / 10)  # temperature scaling factor

        def f_p_alpha(self, V):
            return self.phi * 1e-4 * 9 / u.math.exprel(-(V / u.mV + 30.) / 9.)

        def f_p_beta(self, V):
            return self.phi * 1e-4 * 9 / u.math.exp((V / u.mV + 30.) / 9.)

        def current(self, V, K: braincell.IonInfo):
            return self.g_max * self.p.value * (K.E - V)

        def compute_derivative(self, V, K: braincell.IonInfo):
            # Update the channel state based on the membrane potential V and time step dt
            alpha = self.f_p_alpha(V)
            beta = self.f_p_beta(V)
            p_inf = alpha / (alpha + beta)
            p_tau = 1. / (alpha + beta) * u.ms
            self.p.derivative = (p_inf - self.p.value) / p_tau

        def init_state(self, V, K: braincell.IonInfo, *args, **kwargs):
            alpha = self.f_p_alpha(V)
            beta = self.f_p_beta(V)
            p_inf = alpha / (alpha + beta)
            self.p = braincell.DiffEqState(p_inf)

    class GABAa(brainpy.state.Synapse):
        def __init__(self, in_size, g_max=0.1 * (u.mS / u.cm ** 2), tau=13.0 * u.ms):
            super().__init__(in_size)
            self.g_max = g_max
            self.tau = tau

        def init_state(self, **kwargs):
            self.g = brainstate.HiddenState(u.math.zeros(self.out_size))

        def g_gaba(self, V):
            return 2. * (1. + u.math.tanh(V / (4.0 * u.mV))) / u.ms

        def update(self, pre_V):
            dg = lambda g: self.g_gaba(pre_V) * (1. - g) - g / self.tau
            self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
            return self.g.value

    class MSNCell(braincell.SingleCompartment):
        def __init__(self, size, solver='rk4', g_M=1.3 * (u.mS / u.cm ** 2)):
            super().__init__(size, solver=solver, C=1.0 * u.uF / u.cm ** 2)

            self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
            self.na.add(INa=NaChannel(size, g_max=100. * (u.mS / u.cm ** 2)))

            self.k = braincell.ion.PotassiumFixed(size, E=-100. * u.mV)
            self.k.add(IK=KChannel(size, g_max=80. * (u.mS / u.cm ** 2)))
            self.k.add(IM=MChannel(size, g_max=g_M))

            self.IL = braincell.channel.IL(size, E=-67. * u.mV, g_max=0.1 * (u.mS / u.cm ** 2))

    class StraitalNetwork(brainstate.nn.Module):
        def __init__(self, size, g_M=1.3 * (u.mS / u.cm ** 2)):
            super().__init__()

            self.pop = MSNCell(size, solver='ind_exp_euler', g_M=g_M)
            self.syn = GABAa(size, g_max=0.1 * (u.mS / u.cm ** 2), tau=13.0 * u.ms)
            self.conn = brainpy.state.CurrentProj(
                # comm=brainstate.nn.FixedNumConn(size, size, 0.3, 0.1 / (size * 0.3) * (u.mS / u.cm ** 2)),
                comm=brainstate.nn.AllToAll(size, size, w_init=0.1 / size * (u.mS / u.cm ** 2)),
                out=brainpy.state.COBA(E=-80. * u.mV),
                post=self.pop,
            )

        def update(self, x=0. * u.nA / u.cm ** 2):
            self.conn(self.syn.g.value)
            spk = self.pop(x)
            self.syn(self.pop.V.value)
            return spk

        def step_run(self, i):
            with brainstate.environ.context(i=i, t=brainstate.environ.get_dt() * i):
                inp = (0.12 + brainstate.random.randn() * 1.4) * (u.uA / u.cm ** 2)
                spk = self.update(inp)
                current = self.conn.out(self.pop.V.value)
                return spk, u.math.sum(current)

    return StraitalNetwork(size=1, g_M=1.3 * (u.mS / u.cm ** 2))


# Table 1: specific neuron model parameters
RS_par = dict(
    Vth=-40 * u.mV, delta=2. * u.mV, tau_ref=5. * u.ms, tau_w=500 * u.ms,
    a=4 * u.nS, b=20 * u.pA, C=150 * u.pF, gL=10 * u.nS, EL=-65 * u.mV, V_reset=-65 * u.mV,
    E_e=0. * u.mV, E_i=-80. * u.mV
)
FS_par = dict(
    Vth=-47.5 * u.mV, delta=0.5 * u.mV, tau_ref=5. * u.ms, tau_w=500 * u.ms,
    a=0 * u.nS, b=0 * u.pA, C=150 * u.pF, gL=10 * u.nS, EL=-65 * u.mV, V_reset=-65 * u.mV,
    E_e=0. * u.mV, E_i=-80. * u.mV
)
Ch_par = dict(
    Vth=-47.5 * u.mV, delta=0.5 * u.mV, tau_ref=1. * u.ms, tau_w=50 * u.ms,
    a=80 * u.nS, b=150 * u.pA, C=150 * u.pF, gL=10 * u.nS, EL=-58 * u.mV, V_reset=-65 * u.mV,
    E_e=0. * u.mV, E_i=-80. * u.mV,
)


class _AdEx(brainpy.state.Neuron):
    def __init__(
        self,
        in_size,
        # neuronal parameters
        Vth=-40 * u.mV, delta=2. * u.mV, tau_ref=5. * u.ms, tau_w=500 * u.ms,
        a=4 * u.nS, b=20 * u.pA, C=150 * u.pF,
        gL=10 * u.nS, EL=-65 * u.mV, V_reset=-65 * u.mV, V_sp_th=-40. * u.mV,
        # synaptic parameters
        tau_e=1.5 * u.ms, tau_i=7.5 * u.ms, E_e=0. * u.mV, E_i=-80. * u.mV,
        # other parameters
        V_initializer=braintools.init.Uniform(-65., -50., unit=u.mV),
        w_initializer=braintools.init.Constant(0. * u.pA),
        ge_initializer=braintools.init.Constant(0. * u.nS),
        gi_initializer=braintools.init.Constant(0. * u.nS),
    ):
        super().__init__(in_size=in_size)

        # neuronal parameters
        self.Vth = Vth
        self.delta = delta
        self.tau_ref = tau_ref
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.C = C
        self.gL = gL
        self.EL = EL
        self.V_reset = V_reset
        self.V_sp_th = V_sp_th

        # synaptic parameters
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.E_e = E_e
        self.E_i = E_i

        # other parameters
        self.V_initializer = V_initializer
        self.w_initializer = w_initializer
        self.ge_initializer = ge_initializer
        self.gi_initializer = gi_initializer

    def init_state(self):
        # neuronal variables
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape))
        self.w = brainstate.HiddenState(braintools.init.param(self.w_initializer, self.varshape))
        self.t_last_spike = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape)
        )
        self.spike = brainstate.HiddenState(braintools.init.param(lambda s: u.math.zeros(s, bool), self.varshape))

        # synaptic parameters
        self.ge = brainstate.HiddenState(braintools.init.param(self.ge_initializer, self.varshape))
        self.gi = brainstate.HiddenState(braintools.init.param(self.gi_initializer, self.varshape))

    def dV(self, V, w, ge, gi, Iext):
        I = ge * (self.E_e - V) + gi * (self.E_i - V)
        Iext = self.sum_current_inputs(Iext)
        dVdt = (self.gL * self.delta * u.math.exp((V - self.Vth) / self.delta)
                - w + self.gL * (self.EL - V) + I + Iext) / self.C
        return dVdt

    def dw(self, w, V):
        dwdt = (self.a * (V - self.EL) - w) / self.tau_w
        return dwdt

    def update(self, x=0. * u.pA):
        # numerical integration
        ge = brainstate.nn.exp_euler_step(lambda g: -g / self.tau_e, self.ge.value)
        ge = self.sum_delta_inputs(ge, label='ge')
        gi = brainstate.nn.exp_euler_step(lambda g: -g / self.tau_i, self.gi.value)
        gi = self.sum_delta_inputs(gi, label='gi')
        V = brainstate.nn.exp_euler_step(self.dV, self.V.value, self.w.value, self.ge.value, self.gi.value, x)
        V = self.sum_delta_inputs(V, label='V')
        w = brainstate.nn.exp_euler_step(self.dw, self.w.value, self.V.value)
        # spike detection
        t = brainstate.environ.get('t')
        refractory = (t - self.t_last_spike.value) <= self.tau_ref
        V = u.math.where(refractory, self.V.value, V)
        spike = V >= self.V_sp_th
        self.V.value = u.math.where(spike, self.V_reset, V)
        self.w.value = u.math.where(spike, w + self.b, w)
        self.ge.value = ge
        self.gi.value = gi
        self.spike.value = spike
        self.t_last_spike.value = u.math.where(spike, t, self.t_last_spike.value)
        return spike


class Two_Pop_one_Noise_AI_Net(brainstate.nn.Module):
    def __init__(self, delay: Optional[u.Quantity] = 1.5 * u.ms):
        super().__init__()

        self.num_exc = 20000
        self.num_inh = 5000
        self.exc_syn_tau = 5. * u.ms
        self.inh_syn_tau = 5. * u.ms
        self.exc_syn_weight = 1. * u.nS
        self.inh_syn_weight = 5. * u.nS
        self.delay = delay
        self.ext_weight = 1.0 * u.nS

        # neuronal populations
        RS_par_ = RS_par.copy()
        FS_par_ = FS_par.copy()
        RS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        FS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        self.fs_pop = _AdEx(self.num_inh, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **FS_par_)
        self.rs_pop = _AdEx(self.num_exc, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **RS_par_)
        self.ext_pop = brainpy.state.PoissonEncoder(self.num_exc)

        # Poisson inputs
        self.ext_to_FS = brainpy.state.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_exc, self.num_inh, 0.02, self.ext_weight),
            post=self.fs_pop,
            label='ge'
        )
        self.ext_to_RS = brainpy.state.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_exc, self.num_exc, 0.02, self.ext_weight),
            post=self.rs_pop,
            label='ge'
        )

        # synaptic projections
        self.RS_to_FS = brainpy.state.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay) if delay else self.rs_pop.prefetch('spike'),
            comm=brainstate.nn.EventFixedProb(self.num_exc, self.num_inh, 0.02, self.exc_syn_weight),
            post=self.fs_pop,
            label='ge'
        )
        self.RS_to_RS = brainpy.state.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay) if delay else self.rs_pop.prefetch('spike'),
            comm=brainstate.nn.EventFixedProb(self.num_exc, self.num_exc, 0.02, self.exc_syn_weight),
            post=self.rs_pop,
            label='ge'
        )
        self.FS_to_FS = brainpy.state.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay) if delay else self.fs_pop.prefetch('spike'),
            comm=brainstate.nn.EventFixedProb(self.num_inh, self.num_inh, 0.02, self.inh_syn_weight),
            post=self.fs_pop,
            label='gi'
        )
        self.FS_to_RS = brainpy.state.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay) if delay else self.fs_pop.prefetch('spike'),
            comm=brainstate.nn.EventFixedProb(self.num_inh, self.num_exc, 0.02, self.inh_syn_weight),
            post=self.rs_pop,
            label='gi'
        )

    def update(self, i, freq):
        with brainstate.environ.context(t=brainstate.environ.get_dt() * i, i=i):
            ext_spikes = self.ext_pop(freq)
            self.ext_to_FS(ext_spikes)
            self.ext_to_RS(ext_spikes)
            self.RS_to_RS()
            self.RS_to_FS()
            self.FS_to_FS()
            self.FS_to_RS()
            self.rs_pop()
            self.fs_pop()
            return {
                'FS.V0': self.fs_pop.V.value[0],
                'RS.V0': self.rs_pop.V.value[0],
                'FS.spike': self.fs_pop.spike.value,
                'RS.spike': self.rs_pop.spike.value
            }


def four_pop_HH_braincell_thalamus_net():
    pass
