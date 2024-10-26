# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors & 2024 BDP Ecosystem.
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

from __future__ import annotations

import builtins
import dataclasses
import typing
from typing import TYPE_CHECKING

from brainstate.typing import Filter, PathParts, Predicate, Key

if TYPE_CHECKING:
  ellipsis = builtins.ellipsis
else:
  ellipsis = typing.Any

__all__ = [
  'to_predicate',
]


def to_predicate(the_filter: Filter) -> Predicate:
  """
  Converts a Filter to a predicate function.
  """

  if isinstance(the_filter, str):
    return WithTagFilter(the_filter)
  elif isinstance(the_filter, type):
    return OfTypeFilter(the_filter)
  elif isinstance(the_filter, bool):
    if the_filter:
      return EverythingFilter()
    else:
      return NothingFilter()
  elif the_filter is Ellipsis:
    return EverythingFilter()
  elif the_filter is None:
    return NothingFilter()
  elif callable(the_filter):
    return the_filter
  elif isinstance(the_filter, (list, tuple)):
    return AnyFilter(*the_filter)
  else:
    raise TypeError(f'Invalid collection filter: {the_filter:!r}. ')


@dataclasses.dataclass(frozen=True)
class WithTagFilter:
  tag: str

  def __call__(self, path: PathParts, x: typing.Any):
    return hasattr(x, 'tag') and x.tag == self.tag

  def __repr__(self):
    return f'WithTag({self.tag!r})'


@dataclasses.dataclass(frozen=True)
class PathContainsFilter:
  key: Key

  def __call__(self, path: PathParts, x: typing.Any):
    return self.key in path

  def __repr__(self):
    return f'PathContains({self.key!r})'


@dataclasses.dataclass(frozen=True)
class OfTypeFilter:
  type: type

  def __call__(self, path: PathParts, x: typing.Any):
    return isinstance(x, self.type) or (
        hasattr(x, 'type') and issubclass(x.type, self.type)
    )

  def __repr__(self):
    return f'OfType({self.type!r})'


class AnyFilter:
  def __init__(self, *filters: Filter):
    self.predicates = tuple(
      to_predicate(collection_filter) for collection_filter in filters
    )

  def __call__(self, path: PathParts, x: typing.Any):
    return any(predicate(path, x) for predicate in self.predicates)

  def __repr__(self):
    return f'Any({", ".join(map(repr, self.predicates))})'

  def __eq__(self, other):
    return isinstance(other, AnyFilter) and self.predicates == other.predicates

  def __hash__(self):
    return hash(self.predicates)


class AllFilter:
  def __init__(self, *filters: Filter):
    self.predicates = tuple(
      to_predicate(collection_filter) for collection_filter in filters
    )

  def __call__(self, path: PathParts, x: typing.Any):
    return all(predicate(path, x) for predicate in self.predicates)

  def __repr__(self):
    return f'All({", ".join(map(repr, self.predicates))})'

  def __eq__(self, other):
    return isinstance(other, AllFilter) and self.predicates == other.predicates

  def __hash__(self):
    return hash(self.predicates)


class NotFilter:
  def __init__(self, collection_filter: Filter, /):
    self.predicate = to_predicate(collection_filter)

  def __call__(self, path: PathParts, x: typing.Any):
    return not self.predicate(path, x)

  def __repr__(self):
    return f'Not({self.predicate!r})'

  def __eq__(self, other):
    return isinstance(other, NotFilter) and self.predicate == other.predicate

  def __hash__(self):
    return hash(self.predicate)


class EverythingFilter:
  def __call__(self, path: PathParts, x: typing.Any):
    return True

  def __repr__(self):
    return 'Everything()'

  def __eq__(self, other):
    return isinstance(other, EverythingFilter)

  def __hash__(self):
    return hash(EverythingFilter)


class NothingFilter:
  def __call__(self, path: PathParts, x: typing.Any):
    return False

  def __repr__(self):
    return 'Nothing()'

  def __eq__(self, other):
    return isinstance(other, NothingFilter)

  def __hash__(self):
    return hash(NothingFilter)
