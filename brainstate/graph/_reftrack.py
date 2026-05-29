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
# ==============================================================================

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import Any, TypeVar

from brainstate.util import MappingReprMixin

__all__ = ['RefMap']

A = TypeVar('A')
B = TypeVar('B')


class RefMap(MutableMapping[A, B], MappingReprMixin[A, B]):
    """A mutable mapping that keys entries by object **identity** (``id``).

    Two keys that compare equal but are distinct objects occupy distinct
    slots. This is the workhorse of graph flattening: it tracks which nodes and
    states have already been visited and the global integer index assigned to
    each, so that shared references deduplicate and cycles terminate.

    Parameters
    ----------
    mapping : Mapping or Iterable of tuple, optional
        Initial ``(key, value)`` pairs to populate the map with.

    Notes
    -----
    Lookups, insertions, and deletions are all keyed on ``id(key)``. The
    original key object is retained internally so that iteration yields the
    real keys (not their ids), and so keys are kept alive for the lifetime of
    the entry.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.graph import RefMap
        >>> a, b = [1, 2, 3], [1, 2, 3]     # equal but distinct objects
        >>> m = RefMap()
        >>> m[a] = 0
        >>> m[b] = 1
        >>> len(m)                          # not collapsed by ==
        2
        >>> m[a]
        0
    """

    __module__ = 'brainstate.graph'

    def __init__(self, mapping: Mapping[A, B] | Iterable[tuple[A, B]] = ()) -> None:
        self._mapping: dict[int, tuple[A, B]] = {}
        self.update(mapping)

    def __getitem__(self, key: A) -> B:
        return self._mapping[id(key)][1]

    def __contains__(self, key: Any) -> bool:
        return id(key) in self._mapping

    def __setitem__(self, key: A, value: B) -> None:
        self._mapping[id(key)] = (key, value)

    def __delitem__(self, key: A) -> None:
        del self._mapping[id(key)]

    def __iter__(self) -> Iterator[A]:
        return (key for key, _ in self._mapping.values())

    def __len__(self) -> int:
        return len(self._mapping)

    def __str__(self) -> str:
        return repr(self)
