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

"""
This module includes transformations for augmenting the functionalities of JAX code.
"""

from ._autograd import GradientTransform, grad, vector_grad, hessian, jacobian, jacrev, jacfwd
from ._eval_shape import abstract_init
from ._mapping import vmap, pmap, map, vmap_new_states
from ._random import restore_rngs

__all__ = [
    'GradientTransform', 'grad', 'vector_grad', 'hessian', 'jacobian', 'jacrev', 'jacfwd',
    'abstract_init',
    'vmap', 'pmap', 'map', 'vmap_new_states',
    'restore_rngs',
]
