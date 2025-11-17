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


from typing import NamedTuple, List

from brainstate._compatible_import import Jaxpr, Var
from brainstate._state import State

__all__ = [
    'Dynamics',
    'Connection',
    'Projection',
    'Output',
    'Spike',
]


class Dynamics(NamedTuple):
    jaxpr: Jaxpr
    in_states: List[State]
    out_states: List[State]

    @property
    def eqns(self):
        return self.jaxpr.eqns

    def has_out_var(self, outvar: Var):
        return outvar in self.jaxpr.outvars

    def has_in_var(self, invar: Var):
        return invar in self.jaxpr.invars

    def has_in_state(self, state: State):
        state_id = id(state)
        return any(id(s) == state_id for s in self.in_states)

    def has_out_state(self, state: State):
        state_id = id(state)
        return any(id(s) == state_id for s in self.out_states)


class Connection(NamedTuple):
    """
    """
    state: State
    jaxpr: Jaxpr


class Projection(NamedTuple):
    pre: Dynamics
    post: Dynamics
    jaxpr: Jaxpr


class Output(NamedTuple):
    jaxpr: Jaxpr
    states: List[State]


class Spike(NamedTuple):
    # 包含
    #   heaviside_surrogate_gradient
    state: State
    jaxpr: Jaxpr
