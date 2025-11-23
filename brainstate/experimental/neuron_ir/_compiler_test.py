# Test the ParsedResults.run function

import brainpy
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt

import brainstate
from brainstate.experimental.neuron_ir import compile_fn
from brainstate.experimental.neuron_ir._compiler import (
    NeuronIRCompiler,
    _build_state_mapping,
    _extract_consts_for_vars,
    _build_var_dependencies,
)
from brainstate.experimental.neuron_ir._model_for_test import TwoPopNet, SimpleNet
from brainstate.transform._make_jaxpr import StatefulFunction

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

    result = parse_output.run_compiled(t, inp)
    print(result)


def test_two_populations_run():
    """Test run function with two connected populations."""
    print("\n" + "=" * 80)
    print("Test: ParsedResults.run with Two Populations")
    print("=" * 80)

    net = TwoPopNet()
    brainstate.nn.init_all_states(net)

    def update(t, inp_exc, inp_inh):
        return net.update(t, inp_exc, inp_inh)

    # Parse
    t = 0. * u.ms
    inp_exc = 5. * u.mA
    inp_inh = 3. * u.mA

    parse_output = compile_fn(update)(t, inp_exc, inp_inh)
    true_out, compiled_out = parse_output.debug_compare(t, inp_exc, inp_inh)

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

    r = compiled.debug_compare(t, inp)
    print(r[0])
    print(r[1])


def test_two_populations():
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

    run_results = out.run_compiled(t, inp_exc, inp_inh)
    print(run_results)


# ============================================================================
# Unit Tests for Compiler Steps and Helper Functions
# ============================================================================

def test_step1_analyze_state_dependencies():
    """Test step1: state dependency analysis."""

    # Create simple LIF neuron
    lif = brainpy.state.LIFRef(
        5,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-60. * u.mV,
        tau=20. * u.ms,
        tau_ref=5. * u.ms,
    )
    brainstate.nn.init_all_states(lif)

    def update(t, inp):
        with brainstate.environ.context(t=t):
            lif(inp)
            return lif.get_spike()

    # Get jaxpr
    stateful_fn = StatefulFunction(update, return_only_write=True, ir_optimizations='dce')
    t = 0. * u.ms
    inp = 5. * u.mA
    jaxpr = stateful_fn.get_jaxpr(t, inp)

    # Build state mapping
    in_states = stateful_fn.get_states(t, inp)
    out_states = stateful_fn.get_write_states(t, inp)
    state_mapping = _build_state_mapping(jaxpr, in_states, out_states)

    # Create compiler
    compiler = NeuronIRCompiler(
        closed_jaxpr=jaxpr,
        in_states=in_states,
        out_states=out_states,
        invar_to_state=state_mapping['invar_to_state'],
        outvar_to_state=state_mapping['outvar_to_state'],
        state_to_invars=state_mapping['state_to_invars'],
        state_to_outvars=state_mapping['state_to_outvars'],
    )

    # Test step1
    state_groups = compiler.step1_analyze_state_dependencies()
    assert isinstance(state_groups, list), "step1 should return a list"
    assert len(state_groups) > 0, "step1 should find at least one state group"
    print(f"✓ Step1: Found {len(state_groups)} state groups")


def test_step2_build_groups():
    """Test step2: group building."""

    lif = brainpy.state.LIFRef(
        5, V_rest=-65. * u.mV, V_th=-50. * u.mV,
        V_reset=-60. * u.mV, tau=20. * u.ms, tau_ref=5. * u.ms
    )
    brainstate.nn.init_all_states(lif)

    def update(t, inp):
        with brainstate.environ.context(t=t):
            lif(inp)
            return lif.get_spike()

    stateful_fn = StatefulFunction(update, return_only_write=True, ir_optimizations='dce')
    t = 0. * u.ms
    inp = 5. * u.mA
    jaxpr = stateful_fn.get_jaxpr(t, inp)
    in_states = stateful_fn.get_states(t, inp)
    out_states = stateful_fn.get_write_states(t, inp)
    state_mapping = _build_state_mapping(jaxpr, in_states, out_states)

    compiler = NeuronIRCompiler(
        closed_jaxpr=jaxpr, in_states=in_states, out_states=out_states,
        invar_to_state=state_mapping['invar_to_state'],
        outvar_to_state=state_mapping['outvar_to_state'],
        state_to_invars=state_mapping['state_to_invars'],
        state_to_outvars=state_mapping['state_to_outvars'],
    )

    state_groups = compiler.step1_analyze_state_dependencies()
    groups = compiler.step2_build_groups(state_groups)

    assert isinstance(groups, list), "step2 should return a list"
    assert len(groups) == len(state_groups), "step2 should create one group per state group"
    assert all(hasattr(g, 'hidden_states') for g in groups), "Groups should have hidden_states"
    print(f"✓ Step2: Built {len(groups)} groups")


def test_step3_extract_connections():
    """Test step3: connection extraction."""

    net = SimpleNet()
    brainstate.nn.init_all_states(net)

    def update(t):
        net.update(t)

    stateful_fn = StatefulFunction(update, return_only_write=True, ir_optimizations='dce')
    t = 0. * u.ms
    jaxpr = stateful_fn.get_jaxpr(t)
    from brainstate.transform._ir_inline import inline_jit
    from brainstate.experimental.neuron_ir._utils import _is_connection
    jaxpr = inline_jit(jaxpr, _is_connection)

    in_states = stateful_fn.get_states(t)
    out_states = stateful_fn.get_write_states(t)
    state_mapping = _build_state_mapping(jaxpr, in_states, out_states)

    compiler = NeuronIRCompiler(
        closed_jaxpr=jaxpr, in_states=in_states, out_states=out_states,
        invar_to_state=state_mapping['invar_to_state'],
        outvar_to_state=state_mapping['outvar_to_state'],
        state_to_invars=state_mapping['state_to_invars'],
        state_to_outvars=state_mapping['state_to_outvars'],
    )

    connections = compiler.step3_extract_connections()
    assert isinstance(connections, list), "step3 should return a list"
    print(f"✓ Step3: Extracted {len(connections)} connections")


def test_helper_extract_consts_for_vars():
    """Test helper function _extract_consts_for_vars."""

    # Create simple function with consts
    def f(x):
        return x + 5.0

    jaxpr_out = jax.make_jaxpr(f)(3.0)
    jaxpr = jaxpr_out.jaxpr
    consts = jaxpr_out.consts

    # Extract consts for all constvars
    if jaxpr.constvars:
        extracted = _extract_consts_for_vars(jaxpr.constvars, jaxpr, consts)
        assert len(extracted) == len(consts), "Should extract all consts"
        print(f"✓ Helper _extract_consts_for_vars: Extracted {len(extracted)} consts")
    else:
        print("✓ Helper _extract_consts_for_vars: No consts to extract")


def test_helper_build_var_dependencies():
    """Test helper function _build_var_dependencies."""

    def f(x, y):
        z = x + y
        w = z * 2
        return w

    jaxpr_out = jax.make_jaxpr(f)(1.0, 2.0)
    jaxpr = jaxpr_out.jaxpr

    deps = _build_var_dependencies(jaxpr)
    assert isinstance(deps, dict), "Should return a dict"
    assert len(deps) > 0, "Should have dependencies"
    print(f"✓ Helper _build_var_dependencies: Built dependencies for {len(deps)} vars")


def test_compiler_full_pipeline():
    """Test complete compilation pipeline."""
    lif = brainpy.state.LIFRef(
        3, V_rest=-65. * u.mV, V_th=-50. * u.mV,
        V_reset=-60. * u.mV, tau=20. * u.ms, tau_ref=5. * u.ms
    )
    brainstate.nn.init_all_states(lif)

    def update(t, inp):
        with brainstate.environ.context(t=t):
            lif(inp)
            return lif.get_spike()

    t = 0. * u.ms
    inp = 5. * u.mA

    compiled = compile_fn(update)(t, inp)

    # Check all components were created
    assert len(compiled.groups) > 0, "Should have at least one group"
    assert compiled.inputs is not None, "Should have inputs"
    assert compiled.outputs is not None, "Should have outputs"
    assert compiled.graph is not None, "Should have graph"

    # Test execution
    result = compiled.run_compiled(t, inp)
    assert result is not None, "Compilation should produce a result"

    # Test debug compare
    orig, comp = compiled.debug_compare(t, inp)
    assert orig is not None and comp is not None, "Debug compare should produce results"

    print(f"✓ Full pipeline: {len(compiled.groups)} groups, "
          f"{len(compiled.projections)} projections, "
          f"{len(compiled.inputs)} inputs, "
          f"{len(compiled.outputs)} outputs")
