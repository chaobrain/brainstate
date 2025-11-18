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


from typing import NamedTuple, List, Union

from brainstate._compatible_import import Jaxpr, Var
from brainstate._state import State

__all__ = [
    'Group',
    'Connection',
    'Projection',
    'Output',
    'Input',
    'Spike',
    'CompiledGraph',
]


class Group(NamedTuple):
    jaxpr: Jaxpr
    hidden_states: List[State]
    in_states: List[State]
    out_states: List[State]
    input_vars: List[Var]
    name: str = "Group"  # Add name field with default value

    @property
    def eqns(self):
        return self.jaxpr.eqns

    def has_outvar(self, outvar: Var) -> bool:
        return outvar in self.jaxpr.outvars

    def has_invar(self, invar: Var) -> bool:
        return invar in self.jaxpr.invars

    def has_in_state(self, state: State) -> bool:
        state_id = id(state)
        return any(id(s) == state_id for s in self.in_states)

    def has_out_state(self, state: State) -> bool:
        state_id = id(state)
        return any(id(s) == state_id for s in self.out_states)

    def has_hidden_state(self, state: State) -> bool:
        state_id = id(state)
        return any(id(s) == state_id for s in self.hidden_states)


class Connection(NamedTuple):
    jaxpr: Jaxpr


class Projection(NamedTuple):
    hidden_states: List[State]
    in_states: List[State]
    jaxpr: Jaxpr
    connections: List[Connection]
    pre_group: Group
    post_group: Group

    @property
    def pre(self):
        """Alias for pre_group for backward compatibility with display functions."""
        return self.pre_group

    @property
    def post(self):
        """Alias for post_group for backward compatibility with display functions."""
        return self.post_group


class Output(NamedTuple):
    jaxpr: Jaxpr
    hidden_states: List[State]
    in_states: List[State]
    group: Group


class Input(NamedTuple):
    jaxpr: Jaxpr
    group: Group


class Spike(NamedTuple):
    # 包含
    #   heaviside_surrogate_gradient
    state: State
    jaxpr: Jaxpr


# Type alias for elements that can appear in call_orders
CallOrderElement = Union[Group, Projection, Input, Output]


class CompiledGraph(NamedTuple):
    """Compiled representation of a Spiking Neural Network.

    This structure represents the decomposed computation graph of an SNN,
    organized into groups, projections, inputs, and outputs with their
    execution order.

    Attributes:
        groups: All neuron groups in the network
        projections: All projections (connections between groups)
        inputs: All input injections (external inputs to groups)
        outputs: All output extractions (from groups to network outputs)
        call_orders: Execution order of components (references to actual objects)
    """
    groups: List[Group]
    projections: List[Projection]
    inputs: List[Input]
    outputs: List[Output]
    call_orders: List[CallOrderElement]

    def display(self, mode='text'):
        """
        Display the compiled graph structure.

        Args:
            mode: Display mode

                - 'text': Detailed text display
                - 'visualization' or 'viz': Graphviz visualization (requires graphviz)
                - 'ascii': Simple ASCII art visualization

        Returns:
            For 'text' and 'ascii' modes: None (prints to stdout)
            For 'visualization' mode: graphviz.Digraph object (or None if graphviz not available)
        """
        if mode == 'text':
            return self._text_display()
        elif mode == 'visualization' or mode == 'viz':
            return self._vis_display()
        elif mode == 'ascii':
            return self._ascii_display()
        else:
            raise ValueError(f"Unknown display mode: {mode}. Use 'text', 'ascii', or 'visualization'")

    def _text_display(self):
        """Display compiled graph as formatted text."""
        print("=" * 80)
        print("COMPILED GRAPH STRUCTURE")
        print("=" * 80)

        # Summary
        print(f"\nSummary:")
        print(f"  Groups:      {len(self.groups)}")
        print(f"  Projections: {len(self.projections)}")
        print(f"  Inputs:      {len(self.inputs)}")
        print(f"  Outputs:     {len(self.outputs)}")
        print(f"  Call orders: {len(self.call_orders)}")

        # Groups
        if self.groups:
            print("\n" + "=" * 80)
            print("GROUPS")
            print("=" * 80)
            for i, group in enumerate(self.groups):
                print(f"\n[{i}] {group.name}")
                print(f"  Hidden states: {len(group.hidden_states)}")
                if group.hidden_states:
                    for state in group.hidden_states:
                        print(f"    - {_format_state(state)}")
                print(f"  In states:     {len(group.in_states)}")
                if group.in_states:
                    for state in group.in_states:
                        print(f"    - {_format_state(state)}")
                print(f"  Out states:    {len(group.out_states)}")
                if group.out_states:
                    for state in group.out_states:
                        print(f"    - {_format_state(state)}")
                print(f"  Input vars:    {len(group.input_vars)}")
                print(f"  Equations:     {len(group.jaxpr.eqns)}")

        # Projections
        if self.projections:
            print("\n" + "=" * 80)
            print("PROJECTIONS")
            print("=" * 80)
            for i, proj in enumerate(self.projections):
                print(f"\n[{i}] {proj.pre_group.name} -> {proj.post_group.name}")
                print(f"  Hidden states: {len(proj.hidden_states)}")
                if proj.hidden_states:
                    for state in proj.hidden_states:
                        print(f"    - {_format_state(state)}")
                print(f"  In states:     {len(proj.in_states)}")
                if proj.in_states:
                    for state in proj.in_states:
                        print(f"    - {_format_state(state)}")
                print(f"  Connections:   {len(proj.connections)}")
                print(f"  Equations:     {len(proj.jaxpr.eqns)}")
                # Show connection primitive names
                if proj.connections:
                    print(f"  Connection ops:")
                    for conn in proj.connections:
                        if conn.jaxpr.eqns:
                            for eqn in conn.jaxpr.eqns:
                                print(f"    - {eqn.primitive.name}")

        # Inputs
        if self.inputs:
            print("\n" + "=" * 80)
            print("INPUTS")
            print("=" * 80)
            for i, inp in enumerate(self.inputs):
                print(f"\n[{i}] Input -> {inp.group.name}")
                print(f"  Input vars:    {len(inp.jaxpr.invars)}")
                print(f"  Equations:     {len(inp.jaxpr.eqns)}")

        # Outputs
        if self.outputs:
            print("\n" + "=" * 80)
            print("OUTPUTS")
            print("=" * 80)
            for i, out in enumerate(self.outputs):
                print(f"\n[{i}] {out.group.name} -> Output")
                print(f"  Hidden states: {len(out.hidden_states)}")
                if out.hidden_states:
                    for state in out.hidden_states:
                        print(f"    - {_format_state(state)}")
                print(f"  Equations:     {len(out.jaxpr.eqns)}")

        # Call orders
        print("\n" + "=" * 80)
        print("EXECUTION ORDER")
        print("=" * 80)
        print()
        for i, component in enumerate(self.call_orders):
            comp_type = type(component).__name__
            if isinstance(component, Group):
                print(f"{i:2d}. [Group]      {component.name}")
            elif isinstance(component, Projection):
                print(f"{i:2d}. [Projection] {component.pre_group.name} -> {component.post_group.name}")
            elif isinstance(component, Input):
                print(f"{i:2d}. [Input]      -> {component.group.name}")
            elif isinstance(component, Output):
                print(f"{i:2d}. [Output]     {component.group.name} ->")
            else:
                print(f"{i:2d}. [{comp_type}]")

        print("\n" + "=" * 80)

    def _vis_display(self):
        raise NotImplementedError("Visualization display is not implemented.")

    def _ascii_display(self):
        """Display compiled graph as ASCII art."""
        print("=" * 80)
        print("COMPILED GRAPH - ASCII VISUALIZATION")
        print("=" * 80)
        print()

        # Build component to ID mapping
        comp_to_id = {}
        for i, group in enumerate(self.groups):
            comp_to_id[id(group)] = (f'G{i}', group.name)
        for i, proj in enumerate(self.projections):
            comp_to_id[id(proj)] = (f'P{i}', f'{proj.pre_group.name}->{proj.post_group.name}')

        # Show legend
        print("Legend:")
        print("  [G#] = Group")
        print("  (P#) = Projection")
        print("  <I#> = Input")
        print("  {O#} = Output")
        print()

        # Show execution order
        print("Execution Order:")
        print()
        for i, component in enumerate(self.call_orders):
            indent = "  " * min(i, 3)  # Limit indentation
            if isinstance(component, Group):
                comp_id, comp_name = comp_to_id[id(component)]
                print(f"{indent}{i}. [{comp_id}] {comp_name}")
            elif isinstance(component, Projection):
                comp_id, comp_name = comp_to_id[id(component)]
                print(f"{indent}{i}. ({comp_id}) {comp_name}")
            elif isinstance(component, Input):
                print(f"{indent}{i}. <I> → {component.group.name}")
            elif isinstance(component, Output):
                print(f"{indent}{i}. {{O}} {component.group.name} →")

        print()
        print("-" * 80)
        print("Data Flow Diagram:")
        print()

        # Draw data flow
        # First, collect all inputs
        if self.inputs:
            for i, inp in enumerate(self.inputs):
                target = comp_to_id.get(id(inp.group), ('?', 'Unknown'))[0]
                print(f"  <I{i}>  ────inject────>  [{target}]")
            print()

        # Draw groups and their relationships
        drawn_groups = set()
        for proj in self.projections:
            pre_id, pre_name = comp_to_id[id(proj.pre_group)]
            post_id, post_name = comp_to_id[id(proj.post_group)]
            proj_id, proj_name = comp_to_id[id(proj)]

            if id(proj.pre_group) not in drawn_groups:
                print(f"  [{pre_id}] {pre_name}")
                print(f"    | ({len(proj.pre_group.hidden_states)} states, {len(proj.pre_group.jaxpr.eqns)} eqns)")
                drawn_groups.add(id(proj.pre_group))

            # Show projection
            conn_info = f"{len(proj.connections)} conn" if proj.connections else "no conn"
            print(f"    +---spikes---> ({proj_id}) {proj_name}")
            print(f"    |              ({conn_info})")
            print(f"    +---currents-->")
            print()

            if id(proj.post_group) not in drawn_groups:
                print(f"  [{post_id}] {post_name}")
                print(f"    | ({len(proj.post_group.hidden_states)} states, {len(proj.post_group.jaxpr.eqns)} eqns)")
                drawn_groups.add(id(proj.post_group))
            print()

        # Draw remaining groups (not involved in projections)
        for group in self.groups:
            if id(group) not in drawn_groups:
                group_id, group_name = comp_to_id[id(group)]
                print(f"  [{group_id}] {group_name}")
                print(f"    ({len(group.hidden_states)} states, {len(group.jaxpr.eqns)} eqns)")
                print()

        # Draw outputs
        if self.outputs:
            print()
            for i, out in enumerate(self.outputs):
                source = comp_to_id.get(id(out.group), ('?', 'Unknown'))[0]
                print(f"  [{source}]  ────extract────>  {{O{i}}}")

        print()
        print("=" * 80)


def _format_state(state: State) -> str:
    """Format a State object for display."""
    if hasattr(state, 'value'):
        val = state.value
        if hasattr(val, 'shape'):
            shape_str = f"shape={val.shape}"
        else:
            shape_str = f"value={val}"
        return f"{type(state).__name__}({shape_str})"
    return f"{type(state).__name__}()"

