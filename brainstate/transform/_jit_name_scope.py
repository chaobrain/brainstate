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

from typing import Tuple, Callable

from ._jit import jit

__all__ = [
    'jit_name_scope',
]


def jit_name_scope(
    scope_name: str,
    static_argnums: Tuple[int, ...] = (),
    static_argnames: Tuple[str, ...] = ()
):
    """
    Decorator that wraps a function with JAX's JIT compilation and sets its name.

    Args:
        scope_name: Name to set for the function.
        static_argnums: Tuple of positional argument indices to be treated as static.
        static_argnames: Tuple of keyword argument names to be treated as static.

    Returns:
        Decorated function with JAX JIT compilation applied.

    """

    assert isinstance(scope_name, str), f'scope_name must be a string, got {type(scope_name)}'

    def decorator(fn: Callable):
        return jit(
            fn,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            name=f"{scope_name}.{fn.__name__}"
        )

    return decorator
