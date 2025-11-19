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
import matplotlib.pyplot as plt

import brainstate
from brainstate.experimental.neuron_ir import compile_fn

brainstate.environ.set(dt=0.1 * u.ms)


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


class TestDisplayer:
    def test_visualize(self):
        net = TwoPopNet()
        brainstate.nn.init_all_states(net)

        def update(t, inp_exc, inp_inh):
            return net.update(t, inp_exc, inp_inh)

        t = 0. * u.ms
        inp_exc = 5. * u.mA
        inp_inh = 3. * u.mA

        parser = compile_fn(update)
        compiled = parser(t, inp_exc, inp_inh)

        fig = compiled.graph.visualize(layout='tb')
        plt.show()
        plt.close()


class TestTextDisplayer:
    def test_text(self):
        net = TwoPopNet()
        brainstate.nn.init_all_states(net)

        def update(t, inp_exc, inp_inh):
            return net.update(t, inp_exc, inp_inh)

        t = 0. * u.ms
        inp_exc = 5. * u.mA
        inp_inh = 3. * u.mA

        parser = compile_fn(update)
        compiled = parser(t, inp_exc, inp_inh)

        # Test 1: Basic text display (default)
        print("=== Test 1: Basic Display ===")
        print(compiled.graph)
        print()

        # Test 2: Verbose display
        print("=== Test 2: Verbose Display ===")
        from brainstate.experimental.neuron_ir import TextDisplayer
        displayer = TextDisplayer(compiled.graph)
        print(displayer.display(verbose=True))
        print()

        # Test 3: With JAXPR (first 3 lines only to keep output manageable)
        print("=== Test 3: Display with JAXPR (showing first 50 lines) ===")
        full_output = displayer.display(verbose=False, show_jaxpr=True)
        print(full_output)
