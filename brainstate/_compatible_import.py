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


from contextlib import contextmanager
from functools import partial
from typing import Iterable, Hashable, TypeVar, Callable

import jax

__all__ = [
    'ClosedJaxpr',
    'Primitive',
    'extend_axis_env_nd',
    'jaxpr_as_fun',
    'get_aval',
    'Tracer',
    'to_concrete_aval',
    'safe_map',
    'safe_zip',
    'unzip2',
    'wraps',
    'Device',
    'wrap_init',
]

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

from saiunit._compatible_import import wrap_init

from jax.core import get_aval, Tracer

if jax.__version_info__ < (0, 5, 0):
    from jax.lib.xla_client import Device
else:
    from jax import Device

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

if jax.__version_info__ < (0, 6, 0):
    from jax.util import safe_map, safe_zip, unzip2, wraps

else:
    def safe_map(f, *args):
        args = list(map(list, args))
        n = len(args[0])
        for arg in args[1:]:
            assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
        return list(map(f, *args))


    def safe_zip(*args):
        args = list(map(list, args))
        n = len(args[0])
        for arg in args[1:]:
            assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
        return list(zip(*args))


    def unzip2(xys: Iterable[tuple[T1, T2]]) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
        """Unzip sequence of length-2 tuples into two tuples."""
        # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
        # is too permissive about inputs, and does not guarantee a length-2 output.
        xs: list[T1] = []
        ys: list[T2] = []
        for x, y in xys:
            xs.append(x)
            ys.append(y)
        return tuple(xs), tuple(ys)


    def fun_name(fun: Callable):
        name = getattr(fun, "__name__", None)
        if name is not None:
            return name
        if isinstance(fun, partial):
            return fun_name(fun.func)
        else:
            return "<unnamed function>"


    def wraps(
        wrapped: Callable,
        namestr: str | None = None,
        docstr: str | None = None,
        **kwargs,
    ) -> Callable[[T], T]:
        """
        Like functools.wraps, but with finer-grained control over the name and docstring
        of the resulting function.
        """

        def wrapper(fun: T) -> T:
            try:
                name = fun_name(wrapped)
                doc = getattr(wrapped, "__doc__", "") or ""
                fun.__dict__.update(getattr(wrapped, "__dict__", {}))
                fun.__annotations__ = getattr(wrapped, "__annotations__", {})
                fun.__name__ = name if namestr is None else namestr.format(fun=name)
                fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
                fun.__doc__ = (doc if docstr is None
                               else docstr.format(fun=name, doc=doc, **kwargs))
                fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)
                fun.__wrapped__ = wrapped
            except Exception:
                pass
            return fun

        return wrapper


def to_concrete_aval(aval):
    aval = get_aval(aval)
    if isinstance(aval, Tracer):
        return aval.to_concrete_value()
    return aval

