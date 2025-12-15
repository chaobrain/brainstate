# Test the ParsedResults.run function

import brainpy
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

import brainstate
from brainstate.experimental.neuroir import compile_fn
from brainstate.experimental.neuroir._compiler import (
    NeuronIRCompiler,
    _build_state_mapping,
    _extract_consts_for_vars,
    _build_var_dependencies,
)
from brainstate.experimental.neuroir._model_to_test import (
    TwoPopNet,
    SimpleNet,
    Single_Pop_EI_COBA_Net,
    Single_Pop_EI_CUBA_Net,
    Single_Pop_HH_EI_Net,
    Single_Pop_HH_braincell_EI_Net,
    Two_Pop_one_Noise_AI_Net,
    single_pop_strial_network,
)
from brainstate.transform._make_jaxpr import StatefulFunction

brainstate.environ.set(dt=0.1 * u.ms)


import pytest

pytest.skip('Test is not implemented yet.', allow_module_level=True)

def allclose(r1, r2):
    return all(jax.tree.leaves(jax.tree.map(np.allclose, r1, r2)))


# ============================================================================
# Integration Tests for Compiled Results Execution
# ============================================================================

class TestCompiledResultsExecution:
    """Tests for executing compiled results with run_compiled and debug_compare."""

    def test_simple_lif_run(self):
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
        compiled = compile_fn(update)(t, inp)
        print(compiled.graph)

        result = compiled.run_compiled(t, inp)
        print(result)

    def test_two_populations_run(self):
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
        print(parse_output.graph)
        parse_output.graph.visualize()
        plt.show()

        true_out, compiled_out = parse_output.debug_compare(t, inp_exc, inp_inh)
        print(true_out)
        print(compiled_out)

    def test_single_population_ei_network(self):
        net = Single_Pop_EI_COBA_Net()

        brainstate.nn.init_all_states(net)

        def update(t, inp):
            with brainstate.environ.context(t=t):
                return net(t, inp)

        t = 0. * u.ms
        inp = 20. * u.mA
        compiled = compile_fn(update)(t, inp)
        print(compiled.graph)

        r1, r2 = compiled.debug_compare(t, inp)
        print(r1)
        print(r2)
        assert allclose(r1, r2)

    def test_Single_Pop_EI_CUBA_Net(self):
        net = Single_Pop_EI_CUBA_Net()

        brainstate.nn.init_all_states(net)

        def update(t, inp):
            with brainstate.environ.context(t=t):
                return net(t, inp)

        t = 0. * u.ms
        inp = 20. * u.mA
        compiled = compile_fn(update)(t, inp)

        r1, r2 = compiled.debug_compare(t, inp)
        print(r1)
        print(r2)
        assert allclose(r1, r2)

    def test_Single_Pop_HH_EI_Net(self):
        net = Single_Pop_HH_EI_Net()
        brainstate.nn.init_all_states(net)
        t = 0. * u.ms
        compiled = compile_fn(net.update)(t)
        r1, r2 = compiled.debug_compare(t)
        print(compiled)
        print(compiled.graph)
        print(r1)
        print(r2)
        assert allclose(r1, r2)

    def test_Single_Pop_HH_braincell_EI_Net(self):
        net = Single_Pop_HH_braincell_EI_Net()
        brainstate.nn.init_all_states(net)
        t = 0. * u.ms
        compiled = compile_fn(net.update)(t)
        r1, r2 = compiled.debug_compare(t)
        print(compiled)
        print(compiled.graph)
        print(compiled.graph.groups[0])
        print(compiled.graph.projections[0])
        print(r1)
        print(r2)
        assert allclose(r1, r2)

    def test_single_pop_strial_network(self):
        net = single_pop_strial_network()
        brainstate.nn.init_all_states(net)
        i = 0
        compiled = compile_fn(net.step_run)(i)
        print(compiled)
        print(compiled.graph)

        # r1, r2 = compiled.debug_compare(i)

    def test_Two_Pop_one_Noise_AI_Net(self):
        net = Two_Pop_one_Noise_AI_Net(delay=None)
        brainstate.nn.init_all_states(net)
        inputs = (0, 2. * u.Hz)
        compiled = compile_fn(net.update)(*inputs)
        # print(compiled)
        # print(compiled.graph)

        print(compiled.graph.text(verbose=True, show_jaxpr=True))

        r1, r2 = compiled.debug_compare(*inputs)
        print(r1)
        print(r2)


# ============================================================================
# Integration Tests for Compilation OutputPrim Structure
# ============================================================================

class TestCompilationStructure:
    """Tests for verifying the structure of compilation outputs."""

    def test_simple_lif(self):
        """Test compilation structure for simple LIF neuron."""
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

        print(compiled.graph.groups)
        print(compiled.graph.projections)
        print(compiled.graph.inputs)
        print(compiled.graph.outputs)

        print(f"  - Groups: {len(compiled.graph.groups)}")
        print(f"  - Projections: {len(compiled.graph.projections)}")
        print(f"  - Inputs: {len(compiled.graph.inputs)}")
        print(f"  - Outputs: {len(compiled.graph.outputs)}")

        r = compiled.debug_compare(t, inp)
        print(r[0])
        print(r[1])

    def test_two_populations(self):
        """Test compilation structure for two connected populations."""
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

        print(f"  - Groups: {len(out.graph.groups)}")
        print(f"  - Projections: {len(out.graph.projections)}")
        print(f"  - Inputs: {len(out.graph.inputs)}")
        print(f"  - Outputs: {len(out.graph.outputs)}")

        for input in out.graph.inputs:
            print(input.jaxpr)
            print(input.group.name)
            print()
            print()

        run_results = out.run_compiled(t, inp_exc, inp_inh)
        print(run_results)

    def test_self_connection_compilation(self):
        """Test compilation and execution of networks with self-connections.

        This test verifies that:
        1. Self-connections (pre_group == post_group) are properly detected
        2. The graph structure correctly represents GroupPrim -> ProjectionPrim -> GroupPrim loops
        3. Execution handles circular dependencies correctly
        4. Validation accepts valid self-connections
        """
        print("\n" + "=" * 80)
        print("Test: Self-ConnectionPrim Support (Single_Pop_EI_COBA_Net)")
        print("=" * 80)

        # Create network with self-connections
        net = Single_Pop_EI_COBA_Net()
        brainstate.nn.init_all_states(net)

        def update(t, inp):
            with brainstate.environ.context(t=t):
                return net.update(t, inp)

        t = 0. * u.ms
        inp = 20. * u.mA

        # Compile with full validation
        parser = compile_fn(update, validation='all')
        compiled = parser(t, inp)

        print(f"\n  Graph Structure:")
        print(f"  - Groups: {len(compiled.graph.groups)}")
        print(f"  - Projections: {len(compiled.graph.projections)}")
        print(f"  - Inputs: {len(compiled.graph.inputs)}")
        print(f"  - Outputs: {len(compiled.graph.outputs)}")

        # Verify self-connections are detected
        self_projections = [p for p in compiled.graph.projections if p.pre_group == p.post_group]
        print(f"\n  Self-Connections Found: {len(self_projections)}")

        assert len(self_projections) > 0, "Should detect at least one self-connection"

        for proj in self_projections:
            print(f"    - {proj.pre_group.name} -> {proj.post_group.name} ({len(proj.connections)} connections)")
            assert proj.pre_group == proj.post_group, "Self-projection pre and post should be same"

        # Verify graph structure
        print(f"\n  Graph Edges: {compiled.graph.edge_count()}")
        edges = list(compiled.graph.edges())
        self_edges = [(s, t) for s, t in edges if s == t]
        print(f"  Direct Self-Edges: {len(self_edges)}")

        # Check for GroupPrim -> ProjectionPrim -> GroupPrim cycles
        for proj in self_projections:
            # Verify edges exist: GroupPrim -> ProjectionPrim
            group_to_proj_edge = (proj.pre_group, proj) in edges
            assert group_to_proj_edge, f"Missing edge: {proj.pre_group.name} -> ProjectionPrim"

            # Verify edges exist: ProjectionPrim -> GroupPrim
            proj_to_group_edge = (proj, proj.post_group) in edges
            assert proj_to_group_edge, f"Missing edge: ProjectionPrim -> {proj.post_group.name}"

            print(f"    [OK] Cycle verified: {proj.pre_group.name} -> ProjectionPrim -> {proj.post_group.name}")

        # Test execution
        print(f"\n  Testing Execution...")
        result = compiled.run_compiled(t, inp)
        print(f"    [OK] Execution successful, output shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")

        assert result is not None, "Execution should return results"

        # Test text display shows self-connections
        print(f"\n  Testing Display...")
        from brainstate.experimental.neuroir import TextDisplayer
        displayer = TextDisplayer(compiled.graph)
        text_output = displayer.display(verbose=False)

        # Check if self-projection is shown (Group_X → Group_X pattern)
        for proj in self_projections:
            proj_pattern = f"{proj.pre_group.name} → {proj.post_group.name}"
            assert proj_pattern in text_output, \
                f"Text display should show self-projection pattern: {proj_pattern}"
        print(f"    [OK] Text display correctly shows self-connections")

        print(f"\n{'=' * 80}")
        print("[OK] All self-connection tests passed!")
        print(f"{'=' * 80}\n")


# ============================================================================
# Unit Tests for Compiler Steps
# ============================================================================

class TestCompilerSteps:
    """Tests for individual compiler steps."""

    def test_step1_analyze_state_dependencies(self):
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

    def test_step2_build_groups(self):
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

    def test_step3_extract_connections(self):
        """Test step3: connection extraction."""

        net = SimpleNet()
        brainstate.nn.init_all_states(net)

        def update(t):
            net.update(t)

        stateful_fn = StatefulFunction(update, return_only_write=True, ir_optimizations='dce')
        t = 0. * u.ms
        jaxpr = stateful_fn.get_jaxpr(t)
        from brainstate.transform._ir_inline import inline_jit
        from brainstate.experimental.neuroir._utils import _is_connection
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


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Tests for compiler helper functions."""

    def test_extract_consts_for_vars(self):
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

    def test_build_var_dependencies(self):
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


# ============================================================================
# Full Pipeline Tests
# ============================================================================

class TestFullPipeline:
    """Tests for complete compilation pipeline."""

    def test_compiler_full_pipeline(self):
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
        assert len(compiled.graph.groups) > 0, "Should have at least one group"
        assert compiled.graph.inputs is not None, "Should have inputs"
        assert compiled.graph.outputs is not None, "Should have outputs"
        assert compiled.graph is not None, "Should have graph"

        # Test execution
        result = compiled.run_compiled(t, inp)
        assert result is not None, "Compilation should produce a result"

        # Test debug compare
        orig, comp = compiled.debug_compare(t, inp)
        assert orig is not None and comp is not None, "Debug compare should produce results"

        print(f"✓ Full pipeline: {len(compiled.graph.groups)} groups, "
              f"{len(compiled.graph.projections)} projections, "
              f"{len(compiled.graph.inputs)} inputs, "
              f"{len(compiled.graph.outputs)} outputs")
