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

"""
Test cases for the SNN compiler.
"""

import brainpy
import brainunit as u

import brainstate
from brainstate.experimental.gdiist_bpu import GdiistBPUParser


def test_simple_lif():
    """Test compiler with a simple LIF neuron."""
    print("\n" + "=" * 80)
    print("Test 1: Simple LIF Neuron")
    print("=" * 80)

    # Create a simple LIF neuron
    brainstate.environ.set(dt=0.1 * u.ms)

    lif = brainpy.state.LIFRef(
        10,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-60. * u.mV,
        tau=20. * u.ms,
        tau_ref=5. * u.ms,
    )
    brainstate.nn.init_all_states(lif)

    # Define update function
    def update(t, inp):
        with brainstate.environ.context(t=t):
            lif(inp)
            return lif.get_spike()

    # Parse using Parser
    parser = GdiistBPUParser(update)

    t = 0. * u.ms
    inp = 5. * u.mA

    out = parser.parse(t, inp, display='text', verbose=True)
    print(f"\n[PASS] Parsing successful!")
    print(f"  - Groups: {len(out.compiled.groups)}")
    print(f"  - Projections: {len(out.compiled.projections)}")


def test_two_populations():
    """Test compiler with two connected populations."""
    print("\n" + "=" * 80)
    print("Test 2: Two Connected Populations")
    print("=" * 80)

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

    net = TwoPopNet()
    brainstate.nn.init_all_states(net)

    def update(t, inp_exc, inp_inh):
        return net.update(t, inp_exc, inp_inh)

    parser = GdiistBPUParser(update)

    t = 0. * u.ms
    inp_exc = 5. * u.mA
    inp_inh = 3. * u.mA

    out = parser.parse(t, inp_exc, inp_inh, display='text', verbose=True)
    print(f"\n[PASS] Parsing successful!")
    print(f"  - Groups: {len(out.compiled.groups)}")
    print(f"  - Projections: {len(out.compiled.projections)}")
