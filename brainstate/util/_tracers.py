# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import jax
import jax.core

from brainstate._compatible_import import get_opaque_trace_state
from ._pretty_repr import PrettyRepr, PrettyType, PrettyAttr

__all__ = [
    'StateJaxTracer',
]


def current_jax_trace():
    """Returns the Jax tracing state."""
    if jax.__version_info__ <= (0, 4, 33):  # pragma: no cover
        return jax.core.thread_local_state.trace_state.trace_stack.dynamic
    return get_opaque_trace_state(convention="nnx")


class StateJaxTracer(PrettyRepr):
    """
    Snapshot of the active JAX trace used to detect cross-trace State leakage.

    On construction this captures the JAX tracing state that is currently in
    effect (via :func:`current_jax_trace`). It can later be compared against the
    trace that is active at access time to verify that a :class:`~brainstate.State`
    is being used within the same trace it was created in. A mismatch indicates
    that the state has leaked across JAX trace boundaries (for example, a value
    captured inside one ``jit``/``grad``/``vmap`` trace being read from another),
    which would violate JAX's tracing semantics.

    See Also
    --------
    current_jax_trace : Return the currently active JAX tracing state.
    """

    __slots__ = ['_jax_trace']

    def __init__(self):
        self._jax_trace = current_jax_trace()

    @property
    def jax_trace(self):
        return self._jax_trace

    def is_valid(self) -> bool:
        return self._jax_trace == current_jax_trace()

    def __eq__(self, other):
        return isinstance(other, StateJaxTracer) and self._jax_trace == other._jax_trace

    def __hash__(self):
        # Defining ``__eq__`` resets ``__hash__`` to ``None`` (making instances
        # unhashable); restore it so tracers can live in sets/dict keys. The
        # captured JAX trace (``OpaqueTraceState``) supports ``==`` but is itself
        # unhashable, so we cannot derive the hash from it. Use a constant, type
        # based hash: this keeps the eq/hash invariant (equal tracers hash equal)
        # while accepting that distinct traces collide -- correctness over spread.
        return hash(StateJaxTracer)

    def __pretty_repr__(self):
        yield PrettyType(f'{type(self).__name__}')
        yield PrettyAttr('jax_trace', self._jax_trace)
