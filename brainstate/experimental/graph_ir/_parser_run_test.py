# Test the ParsedOutput.run function

import sys

sys.path.insert(0, '../gdiist_bpu')

import brainstate as bst
import brainunit as u
import brainpy

from brainstate.experimental.graph_ir import parse
from brainstate.transform import StatefulFunction

bst.environ.set(dt=0.1 * u.ms)


def test_simple_lif_run():
    """Test run function with a simple LIF neuron."""
    print("\n" + "=" * 80)
    print("Test: ParsedOutput.run with Simple LIF")
    print("=" * 80)

    # Create LIF neuron
    lif = brainpy.state.LIFRef(
        10,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-60. * u.mV,
        tau=20. * u.ms,
        tau_ref=5. * u.ms,
    )
    bst.nn.init_all_states(lif)

    # Define update function
    def update(t, inp):
        with bst.environ.context(t=t):
            lif(inp)
            return lif.get_spike()

    # Parse
    stateful_fn = StatefulFunction(update, ir_optimizations='dce')
    t = 0. * u.ms
    inp = 5. * u.mA
    parse_output = parse(stateful_fn)(t, inp)

    result = parse_output.run(t, inp)
    print(result)


def test_two_populations_run():
    """Test run function with two connected populations."""
    print("\n" + "=" * 80)
    print("Test: ParsedOutput.run with Two Populations")
    print("=" * 80)

    class TwoPopNet(bst.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_exc = 50
            self.n_inh = 12

            self.exc = brainpy.state.LIFRef(
                self.n_exc,
                V_rest=-65. * u.mV,
                V_th=-50. * u.mV,
                V_reset=-60. * u.mV,
                tau=20. * u.ms,
                tau_ref=5. * u.ms,
            )

            self.inh = brainpy.state.LIFRef(
                self.n_inh,
                V_rest=-65. * u.mV,
                V_th=-50. * u.mV,
                V_reset=-60. * u.mV,
                tau=10. * u.ms,
                tau_ref=5. * u.ms,
            )

            self.exc2inh = brainpy.state.AlignPostProj(
                comm=bst.nn.EventFixedProb(
                    self.n_exc, self.n_inh,
                    conn_num=0.1,
                    conn_weight=1.0 * u.mS
                ),
                syn=brainpy.state.Expon.desc(self.n_inh, tau=5. * u.ms),
                out=brainpy.state.CUBA.desc(scale=u.volt),
                post=self.inh
            )

        def update(self, t, inp_exc, inp_inh):
            with bst.environ.context(t=t):
                exc_spk = self.exc.get_spike() != 0.
                self.exc2inh(exc_spk)
                self.exc(inp_exc)
                self.inh(inp_inh)
                return self.exc.get_spike(), self.inh.get_spike()

    net = TwoPopNet()
    bst.nn.init_all_states(net)

    def update(t, inp_exc, inp_inh):
        return net.update(t, inp_exc, inp_inh)

    # Parse
    stateful_fn = StatefulFunction(update, ir_optimizations='dce')
    t = 0. * u.ms
    inp_exc = 5. * u.mA
    inp_inh = 3. * u.mA

    parse_output = parse(stateful_fn)(t, inp_exc, inp_inh)
    result = parse_output.run(t, inp_exc, inp_inh)

    print(result)
