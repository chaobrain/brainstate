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

# Core transformation infrastructure
from ._make_jaxpr import (
    StatefulFunction, make_jaxpr,
)
from ._grad_transform import (
    GradientTransform,
)

# JIT compilation
from ._jit import (
    jit,
)
from ._jit_named_scope import (
    jit_named_scope,
)
from ._grad_checkpoint import (
    checkpoint, remat,
)

# Mapping transformations
from ._mapping1 import (
    vmap, vmap_new_states,
)
from ._mapping2 import (
    StatefulMapping, vmap2, pmap2, map, vmap2_new_states, pmap2_new_states,
)
from ._mapping3 import (
    StateAxes, model_vmap, model_pmap,
)

# Gradient transformations
from ._grad_grad import (
    vector_grad, grad, fwd_grad,
)
from ._grad_jacobian import (
    jacrev, jacfwd, jacobian,
)
from ._grad_hessian import (
    hessian,
)
from ._grad_sofo import (
    sofo_grad,
)

# Control flow
from ._conditions import (
    cond, switch, ifelse,
)
from ._loop_no_collection import (
    while_loop, bounded_while_loop,
)
from ._loop_collect_return import (
    scan, checkpointed_scan, for_loop, checkpointed_for_loop,
)

# Utilities
from ._error_if import (
    jit_error_if,
)
from ._find_state import (
    StateFinder,
)
from ._progress_bar import (
    ProgressBar,
)
from ._unvmap import (
    unvmap,
)
from ._eval_shape import (
    eval_shape,
)

# IR (Internal Representation) utilities
from ._ir_inline import (
    inline_jit,
)
from ._ir_optim import (
    constant_fold, dead_code_elimination, common_subexpression_elimination,
    copy_propagation, algebraic_simplification, optimize_jaxpr,
)
from ._ir_processing import (
    eqns_to_closed_jaxpr, eqns_to_jaxpr,
)
from ._ir_tocode import (
    fn_to_python_code, jaxpr_to_python_code,
)
from ._ir_visualize import (
    draw, view_pydot, draw_dot_graph,
)

__all__ = [
    # Core transformation infrastructure
    'StatefulFunction',
    'make_jaxpr',
    'GradientTransform',

    # JIT compilation
    'jit',
    'jit_named_scope',
    'checkpoint',
    'remat',

    # Mapping transformations
    'vmap',
    'vmap_new_states',
    'StatefulMapping',
    'vmap2',
    'vmap2_new_states',
    'pmap2',
    'pmap2_new_states',
    'map',
    'StateAxes',
    'model_vmap',
    'model_pmap',

    # Gradient transformations
    'vector_grad',
    'grad',
    'fwd_grad',
    'jacrev',
    'jacfwd',
    'jacobian',
    'hessian',
    'sofo_grad',

    # Control flow
    'cond',
    'switch',
    'ifelse',
    'while_loop',
    'bounded_while_loop',
    'scan',
    'checkpointed_scan',
    'for_loop',
    'checkpointed_for_loop',

    # Utilities
    'jit_error_if',
    'StateFinder',
    'ProgressBar',
    'unvmap',
    'eval_shape',

    # IR utilities
    'inline_jit',
    'constant_fold',
    'dead_code_elimination',
    'common_subexpression_elimination',
    'copy_propagation',
    'algebraic_simplification',
    'optimize_jaxpr',
    'eqns_to_closed_jaxpr',
    'eqns_to_jaxpr',
    'fn_to_python_code',
    'jaxpr_to_python_code',
    'draw',
    'view_pydot',
    'draw_dot_graph',
]
