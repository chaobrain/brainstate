# Test the ParsedResults.run function

import brainpy
import braintools
import brainunit as u
import matplotlib.pyplot as plt

import brainstate
from brainstate.experimental.neuron_ir import compile_fn

brainstate.environ.set(dt=0.1 * u.ms)


def test_simple_lif_run():
    """Test run function with a simple LIF neuron."""
    print("\n" + "=" * 80)
    print("Test: ParsedResults.run with Simple LIF")
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
    brainstate.nn.init_all_states(lif)

    # Define update function
    def update(t, inp):
        with brainstate.environ.context(t=t):
            lif(inp)
            return lif.get_spike()

    # Parse
    t = 0. * u.ms
    inp = 5. * u.mA
    parse_output = compile_fn(update)(t, inp)

    result = parse_output.run(t, inp)
    print(result)


def test_two_populations_run():
    """Test run function with two connected populations."""
    print("\n" + "=" * 80)
    print("Test: ParsedResults.run with Two Populations")
    print("=" * 80)

    class TwoPopNet(brainstate.nn.Module):
        """ """

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
                V_initializer=braintools.init.Uniform(-70., -40., unit=u.mV)
            )

            self.inh = brainpy.state.LIFRef(
                self.n_inh,
                V_rest=-65. * u.mV,
                V_th=-50. * u.mV,
                V_reset=-60. * u.mV,
                tau=10. * u.ms,
                tau_ref=5. * u.ms,
                V_initializer=braintools.init.Uniform(-70., -40., unit=u.mV)
            )

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

    # Parse
    t = 0. * u.ms
    inp_exc = 5. * u.mA
    inp_inh = 3. * u.mA

    parse_output = compile_fn(update)(t, inp_exc, inp_inh)
    true_out, compiled_out = parse_output.run(t, inp_exc, inp_inh, mode='debug')

    print(true_out)
    print(compiled_out)


def test_simple_lif():
    lif = brainpy.state.LIFRef(
        2,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-60. * u.mV,
        tau=20. * u.ms,
        tau_ref=5. * u.ms,
        V_initializer=braintools.init.Constant(-65. * u.mV)
    )
    brainstate.nn.init_all_states(lif)

    # Define update function
    def update(t, inp):
        with brainstate.environ.context(t=t):
            lif(inp)
            return lif.get_spike(), lif.V.value

    t = 0. * u.ms
    inp = 5. * u.mA

    parser = compile_fn(update)
    compiled = parser(t, inp)

    print(compiled.groups)
    print(compiled.projections)
    print(compiled.inputs)
    print(compiled.outputs)

    print(f"  - Groups: {len(compiled.groups)}")
    print(f"  - Projections: {len(compiled.projections)}")
    print(f"  - Inputs: {len(compiled.inputs)}")
    print(f"  - Outputs: {len(compiled.outputs)}")

    r = compiled.run(t, inp, mode='debug')
    print(r[0])
    print(r[1])


def test_two_populations():
    class TwoPopNet(brainstate.nn.Module):
        """ """

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

    t = 0. * u.ms
    inp_exc = 5. * u.mA
    inp_inh = 3. * u.mA

    parser = compile_fn(update)
    out = parser(t, inp_exc, inp_inh)

    out.graph.visualize()
    plt.show()

    print(f"  - Groups: {len(out.groups)}")
    print(f"  - Projections: {len(out.projections)}")
    print(f"  - Inputs: {len(out.inputs)}")
    print(f"  - Outputs: {len(out.outputs)}")

    for input in out.inputs:
        print(input.jaxpr)
        print(input.group.name)
        print()
        print()

    run_results = out.run(t, inp_exc, inp_inh)
    print(run_results)
