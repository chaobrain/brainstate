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

from brainstate._state import State
from brainstate._compatible_import import ClosedJaxpr, JaxprEqn, Jaxpr

__all__ = [
    'Node',
    'Connection',
    'Output',
]


class Node(NamedTuple):
    name: str
    jaxpr: Jaxpr
    in_states: List[State]
    out_states: List[State]

    @property
    def eqns(self):
        return self.jaxpr.eqns


class Connection(NamedTuple):
    pre: Node
    post: Node
    jaxpr: Jaxpr


class Output(NamedTuple):
    pass
