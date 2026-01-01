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

from ._pretty_repr import PrettyRepr, PrettyType, PrettyAttr

__all__ = [
    'StateJaxTracer',
]


def current_jax_trace():
    """Returns the Jax tracing state."""
    if jax.__version_info__ <= (0, 4, 33):
        return jax.core.thread_local_state.trace_state.trace_stack.dynamic
    return jax.core.get_opaque_trace_state(convention="nnx")


class StateJaxTracer(PrettyRepr):
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

    def __pretty_repr__(self):
        yield PrettyType(f'{type(self).__name__}')
        yield PrettyAttr('jax_trace', self._jax_trace)
