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
        pass

    def _text_display(self):
        pass

    def _vis_display(self):
        pass



