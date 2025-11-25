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
import functools
from typing import Sequence, Union, Any

from ._jit import jit

__all__ = [
    'jit_named_scope',
]


def fn_to_call(
    fn,
    name,
    non_static_argnums,
    non_static_argnames,
    static_argnums,
    static_argnames,
    *args, **kwargs,
):
    assert not (non_static_argnums is None and non_static_argnames is None)
    if non_static_argnums is not None:
        assert static_argnums is None, "Cannot specify both static_argnums and non_static_argnums."
        non_static_argnums = (non_static_argnums,) if isinstance(non_static_argnums,
                                                                 int) else non_static_argnums
        assert isinstance(non_static_argnums, (tuple, list)), "non_static_argnums must be a tuple or list."
        non_static_argnums = [i + len(args) if i < 0 else i for i in non_static_argnums]
        static_argnums = sorted(tuple(set(range(len(args))) - set(non_static_argnums)))

    if non_static_argnames is not None:
        assert static_argnames is None, "Cannot specify both static_argnames and non_static_argnames."
        non_static_argnames = (non_static_argnames,) if isinstance(non_static_argnames,
                                                                   str) else non_static_argnames
        assert isinstance(non_static_argnames, (tuple, list)), "non_static_argnames must be a tuple or list."
        static_argnames = tuple(set(kwargs.keys()) - set(non_static_argnames))

    _jit_fn = jit(
        fn,
        name=name,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )
    return _jit_fn(*args, **kwargs)


class FunctionToCall:
    def __init__(
        self,
        name: str,
        static_argnums: Union[int, Sequence[int]] = None,
        non_static_argnums: Union[int, Sequence[int]] = None,
        static_argnames: Union[str, Sequence[str]] = None,
        non_static_argnames: Union[str, Sequence[str]] = None,
    ):
        # parameters
        self.name = name
        self.static_argnums = static_argnums
        self.static_argnames = static_argnames
        self.non_static_argnums = non_static_argnums
        self.non_static_argnames = non_static_argnames

        # function and jitted function
        self._fn = None
        self._jit_fn = None

    def __call__(self, fn) -> Union['FunctionToCall', Any]:
        if self.non_static_argnums is None and self.non_static_argnames is None:
            return jit(
                fn,
                name=self.name,
                static_argnums=self.static_argnums,
                static_argnames=self.static_argnames,
            )

        else:
            return functools.partial(
                fn_to_call,
                fn=fn,
                name=self.name,
                non_static_argnums=self.non_static_argnums,
                non_static_argnames=self.non_static_argnames,
                static_argnums=self.static_argnums,
                static_argnames=self.static_argnames,
            )


def jit_named_scope(
    name: str,
    static_argnums: Union[int, Sequence[int]] = None,
    non_static_argnums: Union[int, Sequence[int]] = None,
    static_argnames: Union[str, Sequence[str]] = None,
    non_static_argnames: Union[str, Sequence[str]] = None,
) -> FunctionToCall:
    """
    Decorator that wraps a function with JAX's JIT compilation and sets its name.

    Args:
        name: Name to set for the function.
        static_argnums: Tuple of positional argument indices to be treated as static.
            When ``non_static_argnums`` is specified, this argument is ignored.
        non_static_argnums: Tuple of positional argument indices to be treated as non-static.
            When ``static_argnums`` is specified, this argument is ignored.
        static_argnames: Tuple of keyword argument names to be treated as static.
            When ``non_static_argnames`` is specified, this argument is ignored.
        non_static_argnames: Tuple of keyword argument names to be treated as non-static.
            When ``static_argnames`` is specified, this argument is ignored.

    Returns:
        Decorated function with JAX JIT compilation applied.
    """
    return FunctionToCall(
        name=name,
        static_argnums=static_argnums,
        non_static_argnums=non_static_argnums,
        static_argnames=static_argnames,
        non_static_argnames=non_static_argnames,
    )
