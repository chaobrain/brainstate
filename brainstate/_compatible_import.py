# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import importlib.util
from contextlib import contextmanager
from typing import Iterable, Hashable

import jax

__all__ = [
    'ClosedJaxpr',
    'Primitive',
    'extend_axis_env_nd',
    'jaxpr_as_fun',
    'get_aval',
    'Tracer',
    'to_concrete_aval',
    'brainevent',
]

brainevent_installed = importlib.util.find_spec('brainevent') is not None

from jax.core import get_aval, Tracer

if jax.__version_info__ < (0, 4, 38):
    from jax.core import ClosedJaxpr, extend_axis_env_nd, Primitive, jaxpr_as_fun
else:
    from jax.extend.core import ClosedJaxpr, Primitive, jaxpr_as_fun
    from jax.core import trace_ctx


    @contextmanager
    def extend_axis_env_nd(name_size_pairs: Iterable[tuple[Hashable, int]]):
        prev = trace_ctx.axis_env
        try:
            trace_ctx.set_axis_env(prev.extend_pure(name_size_pairs))
            yield
        finally:
            trace_ctx.set_axis_env(prev)


def to_concrete_aval(aval):
    aval = get_aval(aval)
    if isinstance(aval, Tracer):
        return aval.to_concrete_value()
    return aval


if brainevent_installed:
    import brainevent
else:

    class BrainEvent:
        def __getattr__(self, item):
            raise ImportError('brainevent is not installed, please install brainevent first.')


    brainevent = BrainEvent()
