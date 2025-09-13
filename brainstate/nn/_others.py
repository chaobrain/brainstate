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

from functools import partial

import jax
import jax.numpy as jnp

from brainstate.typing import PyTree

__all__ = [
    'clip_grad_norm',
]


def clip_grad_norm(
    grad: PyTree,
    max_norm: float | jax.Array,
    norm_type: int | str | None = None
):
    """
    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        grad (PyTree): an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients.
        norm_type (int, str, None): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """
    norm_fn = partial(jnp.linalg.norm, ord=norm_type)
    norm = norm_fn(jnp.asarray(jax.tree.leaves(jax.tree.map(norm_fn, grad))))
    return jax.tree.map(lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6)), grad)
