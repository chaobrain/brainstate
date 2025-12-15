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

from collections import defaultdict
from typing import Sequence, Dict, List, Set

from jax.extend.core.primitives import dot_general_p, conv_general_dilated_p

from brainstate._compatible_import import is_jit_primitive, JaxprEqn, Var
from brainstate._state import State


def get_hidden_name(hidden: State):
    name = getattr(hidden, 'name', None)
    if name:
        return name
    else:
        return f'State@{id(hidden):x}'


def _is_connection(eqn: JaxprEqn) -> bool:
    """Return ``True`` if ``eqn`` performs a connection-like data transfer.

    Parameters
    ----------
    eqn : JaxprEqn
        Equation emitted in the step JAXPR.

    Returns
    -------
    bool
        ``True`` when the primitive corresponds to a brainevent connection or a
        dense/conv primitive, ``False`` otherwise.
    """
    # Check if equation is a jit-wrapped brainevent operation that should be a connection
    if is_jit_primitive(eqn):
        # Check if the function name starts with 'brainevent'
        if 'name' in eqn.params:
            name = eqn.params['name']
            if isinstance(name, str) and name.startswith('brainevent'):
                return True

    # Check for brainevent primitive names (after inline_jit)
    # These are the actual brainevent connection primitives
    primitive_name = eqn.primitive.name
    if any(
        keyword in primitive_name
        for keyword in [
            'binary_fixed_num_mv',
            'event_ell_mv',
            'event_csr_matvec',
            'event_coo_matvec',
            'taichi_mv',
            'mv_prob',
            'mv_',  # General matrix-vector operations from brainevent
        ]
    ):
        return True

    # check for standard connection primitives
    if eqn.primitive in [
        dot_general_p,
        conv_general_dilated_p,
    ]:
        return True

    return False


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for grouping states."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        """Create a singleton set containing ``x`` if it does not exist."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        """Return the canonical representative for the set containing ``x``."""
        if x not in self.parent:
            self.make_set(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Merge the sets containing ``x`` and ``y`` using union-by-rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_groups(self) -> List[Set]:
        """Return the current disjoint sets as a list of ``set`` objects."""
        groups_dict = defaultdict(set)
        for x in self.parent.keys():
            root = self.find(x)
            groups_dict[root].add(x)
        return list(groups_dict.values())


def find_in_states(
    in_var_to_state: Dict[Var, State],
    in_vars: Sequence[Var]
):
    """Return the list of :class:`State` objects referenced by ``in_vars``."""
    in_states = []
    in_state_ids = set()
    for var in in_vars:
        if var in in_var_to_state:
            st = in_var_to_state[var]
            if id(st) not in in_state_ids:
                in_state_ids.add(id(st))
                in_states.append(st)
    return in_states


def find_out_states(
    out_var_to_state: Dict[Var, State],
    out_vars: Sequence[Var]
):
    """Return the list of :class:`State` objects referenced by ``out_vars``."""
    out_states = []
    out_state_ids = set()
    for var in out_vars:
        if var in out_var_to_state:
            st = out_var_to_state[var]
            if id(st) not in out_state_ids:
                out_state_ids.add(id(st))
                out_states.append(st)
    return out_states
