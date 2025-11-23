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
