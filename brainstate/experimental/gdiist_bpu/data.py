# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
from typing import NamedTuple, List, Set, Dict, Any, Optional

from jax._src.core import Var, JaxprEqn


class Group(NamedTuple):
    """Represents a computational group with same inputs, outputs, and parameters."""
    id: str  # Unique group identifier
    input_vars: Set[Var]  # Input variables to this group
    output_vars: Set[Var]  # Output variables from this group
    equations: List[JaxprEqn]  # Equations within this group
    parameters: Dict[str, Any]  # Group parameters (shared across equations)
    primitive_name: Optional[str]  # Primary primitive type in this group

    def __str__(self):
        primitive = self.primitive_name or 'unknown'
        # Plain text format without colors
        lines = [
            f"Group<{self.id}>:",
            f"  Primitive: {primitive}",
            f"  Equations: {len(self.equations)}",
            f"  Input vars: {len(self.input_vars)}",
            f"  Output vars: {len(self.output_vars)}"
        ]
        return '\n'.join(lines)

    def __repr__(self):
        primitive = self.primitive_name or 'unknown'
        return (f"Group(id={self.id}, primitive={primitive}, "
                f"eqs={len(self.equations)}, ins={len(self.input_vars)}, outs={len(self.output_vars)})")


class Connection(NamedTuple):
    """Represents a connection between two groups."""
    pre: Group  # Source group
    post: Group  # Target group

    def __str__(self):
        return f"Connection:{self.pre.id} → {self.post.id}"

    def __repr__(self):
        return f"Connection({self.pre.id} → {self.post.id})"
