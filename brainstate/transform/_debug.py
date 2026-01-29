# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

__all__ = [
    'breakpoint_if',
]


def breakpoint(pred, **breakpoint_kwargs):
    """As `jax.debug.breakpoint`, but only triggers if `pred` is True.

    **Arguments:**

    - `pred`: the predicate for whether to trigger the breakpoint.
    - `**breakpoint_kwargs`: any other keyword arguments to forward to `jax.debug.breakpoint`.

    """

    # We can't just write `jax.debug.breakpoint` for the second branch. For some reason
    # it needs as lambda wrapper.

    from brainstate.transform._unvmap import unvmap_any
    token = breakpoint_kwargs.get("token", None)
    return jax.lax.cond(
        unvmap_any(pred),
        lambda: jax.debug.breakpoint(**breakpoint_kwargs),
        lambda: token,
    )


def breakpoint_if(
    *data,
    nan: bool = True,
    inf: bool = True,
    **breakpoint_kwargs,
):
    """As `jax.debug.breakpoint`, but only triggers if `pred` is True.

    **Arguments:**

    - `pred`: the predicate for whether to trigger the breakpoint.
    - `**breakpoint_kwargs`: any other keyword arguments to forward to `jax.debug.breakpoint`.

    """

    # We can't just write `jax.debug.breakpoint` for the second branch. For some reason
    # it needs as lambda wrapper.

    from brainstate.transform._unvmap import unvmap_any

    data = jax.tree.map(
        lambda x: unvmap_any(jax.numpy.any(x == jnp.nan)),
        jax.tree.leaves(data)
    )


    token = breakpoint_kwargs.get("token", None)
    return jax.lax.cond(
        unvmap_any(pred),
        lambda: jax.debug.breakpoint(**breakpoint_kwargs),
        lambda: token,
    )
