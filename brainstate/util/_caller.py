# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors.
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

import dataclasses
from typing import Any, TypeVar, Protocol, Generic

import jax

__all__ = [
    'DelayedAccessor',
    'CallableProxy',
    'ApplyCaller',
]

A = TypeVar('A', covariant=True)  # type: ignore[not-supported-yet]


def _identity(x):
    return x


@dataclasses.dataclass(frozen=True)
class GetItem:
    key: Any


@dataclasses.dataclass(frozen=True)
class GetAttr:
    name: str


@dataclasses.dataclass(frozen=True)
class DelayedAccessor:
    actions: tuple[GetItem | GetAttr, ...] = ()

    def __call__(self, x):
        for action in self.actions:
            if isinstance(action, GetItem):
                x = x[action.key]
            elif isinstance(action, GetAttr):
                x = getattr(x, action.name)
        return x

    def __getattr__(self, name):
        return DelayedAccessor(self.actions + (GetAttr(name),))

    def __getitem__(self, key):
        return DelayedAccessor(self.actions + (GetItem(key),))


jax.tree_util.register_static(DelayedAccessor)


class _AccessorCall(Protocol):
    def __call__(self, accessor: DelayedAccessor, /, *args, **kwargs) -> Any:
        ...


class CallableProxy:
    def __init__(
        self, fun: _AccessorCall, accessor: DelayedAccessor | None = None
    ):
        self._callable = fun
        self._accessor = DelayedAccessor() if accessor is None else accessor

    def __call__(self, *args, **kwargs):
        return self._callable(self._accessor, *args, **kwargs)

    def __getattr__(self, name) -> 'CallableProxy':
        return CallableProxy(self._callable, getattr(self._accessor, name))

    def __getitem__(self, key) -> 'CallableProxy':
        return CallableProxy(self._callable, self._accessor[key])


class ApplyCaller(Protocol, Generic[A]):
    def __getattr__(self, __name) -> 'ApplyCaller[A]':
        ...

    def __getitem__(self, __name) -> 'ApplyCaller[A]':
        ...

    def __call__(self, *args, **kwargs) -> tuple[Any, A]:
        ...
