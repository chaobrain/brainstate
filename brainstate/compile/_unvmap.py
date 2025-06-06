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

import jax
import jax.core
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp

from brainstate._compatible_import import Primitive
from brainstate._utils import set_module_as

__all__ = [
    "unvmap",
]


@set_module_as('brainstate.augment')
def unvmap(x, op: str = 'any'):
    if op == 'all':
        return unvmap_all(x)
    elif op == 'any':
        return unvmap_any(x)
    elif op == 'none':
        return _without_vmap(x)
    elif op == 'max':
        return unvmap_max(x)
    else:
        raise ValueError(f'Do not support type: {op}')


# unvmap_all

unvmap_all_p = Primitive("unvmap_all")


def unvmap_all(x):
    """As `jnp.all`, but ignores batch dimensions."""
    return unvmap_all_p.bind(x)


def _unvmap_all_impl(x):
    return jnp.all(x)


def _unvmap_all_abstract_eval(x):
    return jax.core.ShapedArray(shape=(), dtype=jax.numpy.bool_.dtype)  # pyright: ignore


def _unvmap_all_batch(x, batch_axes):
    (x,) = x
    return unvmap_all(x), batching.not_mapped


unvmap_all_p.def_impl(_unvmap_all_impl)
unvmap_all_p.def_abstract_eval(_unvmap_all_abstract_eval)
batching.primitive_batchers[unvmap_all_p] = _unvmap_all_batch  # pyright: ignore
mlir.register_lowering(
    unvmap_all_p,
    mlir.lower_fun(_unvmap_all_impl, multiple_results=False),
)

# unvmap_any

unvmap_any_p = Primitive("unvmap_any")


def unvmap_any(x):
    """As `jnp.any`, but ignores batch dimensions."""
    return unvmap_any_p.bind(x)


def _unvmap_any_impl(x):
    return jnp.any(x)


def _unvmap_any_abstract_eval(x):
    return jax.core.ShapedArray(shape=(), dtype=jax.numpy.bool_.dtype)  # pyright: ignore


def _unvmap_any_batch(x, batch_axes):
    (x,) = x
    return unvmap_any(x), batching.not_mapped


unvmap_any_p.def_impl(_unvmap_any_impl)
unvmap_any_p.def_abstract_eval(_unvmap_any_abstract_eval)
batching.primitive_batchers[unvmap_any_p] = _unvmap_any_batch  # pyright: ignore
mlir.register_lowering(
    unvmap_any_p,
    mlir.lower_fun(_unvmap_any_impl, multiple_results=False),
)

# unvmap_max

unvmap_max_p = Primitive("unvmap_max")


def unvmap_max(x):
    """As `jnp.max`, but ignores batch dimensions."""
    return unvmap_max_p.bind(x)


def _unvmap_max_impl(x):
    return jnp.max(x)


def _unvmap_max_abstract_eval(x):
    return jax.core.ShapedArray(shape=(), dtype=x.dtype)


def _unvmap_max_batch(x, batch_axes):
    (x,) = x
    return unvmap_max(x), batching.not_mapped


unvmap_max_p.def_impl(_unvmap_max_impl)
unvmap_max_p.def_abstract_eval(_unvmap_max_abstract_eval)
batching.primitive_batchers[unvmap_max_p] = _unvmap_max_batch  # pyright: ignore
mlir.register_lowering(
    unvmap_max_p,
    mlir.lower_fun(_unvmap_max_impl, multiple_results=False),
)


def _without_vmap(x):
    return _no_vmap_prim.bind(x)


def _without_vmap_imp(x):
    return x


def _without_vmap_abs(x):
    return x


def _without_vmap_batch(x, batch_axes):
    (x,) = x
    return _without_vmap(x), batching.not_mapped


_no_vmap_prim = Primitive('no_vmap')
_no_vmap_prim.def_impl(_without_vmap_imp)
_no_vmap_prim.def_abstract_eval(_without_vmap_abs)
batching.primitive_batchers[_no_vmap_prim] = _without_vmap_batch
mlir.register_lowering(_no_vmap_prim, mlir.lower_fun(_without_vmap_imp, multiple_results=False))
