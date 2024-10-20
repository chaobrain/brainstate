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

from collections.abc import Callable, Mapping, MutableMapping
from typing import Iterator, TypeVar, Hashable, Union, TYPE_CHECKING, Iterable, Any, Optional, Tuple, Dict

import jax

from brainstate.typing import Filter, PathParts
from ._filter import to_predicate
from ._pretty_repr import PrettyRepr, PrettyType, PrettyAttr, pretty_repr_avoid_duplicate
from ._struct import dataclass

__all__ = [
  'NestedMapping', 'FlattedMapping', 'flat_mapping', 'nest_mapping',
]

A = TypeVar('A')
K = TypeVar('K', bound=Hashable)
V = TypeVar('V')

FlattedStateMapping = dict[PathParts, V]
ExtractValueFn = Callable[[Any], Any]
SetValueFn = Callable[[V, Any], V]


# the empty node is a struct.dataclass to be compatible with JAX.
@dataclass
class _EmptyNode:
  pass


_default_leaf = lambda *args: False
empty_node = _EmptyNode()
IsLeafCallable = Callable[[Tuple[Any, ...], Mapping[Any, Any]], bool]


def flat_mapping(
    xs: Mapping[Any, Any],
    /,
    *,
    keep_empty_nodes: bool = False,
    is_leaf: Optional[IsLeafCallable] = _default_leaf,
    sep: Optional[str] = None
) -> 'FlattedMapping':
  """Flatten a nested mapping.

  The nested keys are flattened to a tuple. See ``unflatten_mapping`` on how to
  restore the nested mapping.

  Example::

    >>> xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    >>> flat_xs = flat_mapping(xs)
    >>> flat_xs
    {('foo',): 1, ('bar', 'a'): 2}

  Note that empty mappings are ignored and will not be restored by
  ``unflatten_mapping``.

  Args:
    xs: A nested mapping.
    keep_empty_nodes: replaces empty mappings with ``empty_node``.
    is_leaf: An optional function that takes the next nested mapping and nested
             keys and returns True if the nested mapping is a leaf (i.e., should not be
             flattened further).
    sep: If specified, then the keys of the returned mapping will be
         ``sep``-joined strings (if ``None``, then keys will be tuples).

  Returns:
    The flattened mapping.
  """
  assert isinstance(xs, Mapping), f'expected Mapping; got {type(xs).__qualname__}'

  if sep is None:
    def _key(path: Tuple[Any, ...]) -> Tuple[Any, ...] | str:
      return path
  else:

    def _key(path: Tuple[Any, ...]) -> Tuple[Any, ...] | str:
      return sep.join(path)

  def _flatten(xs: Any, prefix: Tuple[Any, ...]) -> Dict[Any, Any]:
    if not isinstance(xs, Mapping) or is_leaf(prefix, xs):
      return {_key(prefix): xs}

    result = {}
    is_empty = True
    for key, value in xs.items():
      is_empty = False
      result.update(_flatten(value, prefix + (key,)))
    if keep_empty_nodes and is_empty:
      if prefix == ():  # when the whole input is empty
        return {}
      return {_key(prefix): empty_node}
    return result

  return FlattedMapping(_flatten(xs, ()))


def nest_mapping(
    xs: Any,
    /,
    *,
    sep: str | None = None
) -> 'NestedMapping':
  """Unflatten a mapping.

  See ``flatten_mapping``

  Example::

    >>> flat_xs = {
    ...   ('foo',): 1,
    ...   ('bar', 'a'): 2,
    ... }
    >>> xs = nest_mapping(flat_xs)
    >>> xs
    {'foo': 1, 'bar': {'a': 2}}

  Args:
    xs: a flattened mapping.
    sep: separator (same as used with ``flatten_mapping()``).

  Returns:
    The nested mapping.
  """
  assert isinstance(xs, Mapping), f'expected Mapping; got {type(xs).__qualname__}'
  result: Dict[Any, Any] = {}
  for path, value in xs.items():
    if sep is not None:
      path = path.split(sep)
    if value is empty_node:
      value = {}
    cursor = result
    for key in path[:-1]:
      if key not in cursor:
        cursor[key] = {}
      cursor = cursor[key]
    cursor[path[-1]] = value
  return NestedMapping(result)


def _default_compare(x, values):
  return id(x) in values


def _default_process(x):
  return id(x)


class NestedStateRepr(PrettyRepr):
  def __init__(self, state: StateMapping):
    self.state = state

  def __pretty_repr__(self):
    yield PrettyType('', value_sep=': ', start='{', end='}')

    for r in self.state.__pretty_repr__():
      if isinstance(r, PrettyType):
        continue
      yield r

  def __treescope_repr__(self, path, subtree_renderer):
    children = {}
    for k, v in self.state.items():
      if isinstance(v, StateMapping):
        v = NestedStateRepr(v)
      children[k] = v
    # Render as the dictionary itself at the same path.
    return subtree_renderer(children, path=path)


class StateMapping(MutableMapping[K, V], PrettyRepr):
  __module__ = 'brainstate.util'

  def __init__(
      self,
      mapping: Union[Mapping[K, Mapping | V], Iterator[Tuple[K, Mapping | V]]],
      /,
      *,
      _copy: bool = True,
  ):
    if _copy:
      _mapping = dict(mapping)
    else:
      _mapping = mapping
    if not isinstance(_mapping, dict):
      raise ValueError(f'Expected a dictionary when `_copy=False`, got {type(mapping)} instead.')

    if TYPE_CHECKING:
      self._mapping = _mapping
    else:
      super().__setattr__('_mapping', _mapping)

  @property
  def raw_mapping(self) -> Mapping[K, Mapping[K, Any] | V]:
    """
    The raw mapping of the ``NestedMapping``.

    Returns:
      The raw mapping.
    """
    return self._mapping  # type: ignore

  def __contains__(self, key) -> bool:
    return key in self._mapping

  def __getitem__(self, key: K) -> V:  # type: ignore
    raise NotImplementedError

  def __getattr__(self, key: K) -> NestedMapping | V:  # type: ignore[misc]
    if '_mapping' not in vars(self) or key not in self._mapping:
      raise AttributeError(f"No attribute '{key}' in NestedMapping")
    return self[key]

  def __setitem__(self, key: K, value: NestedMapping | V) -> None:
    if key == '__orig_class__':
      object.__setattr__(self, key, value)  # type: ignore
    elif isinstance(value, NestedMapping):
      self._mapping[key] = value._mapping
    else:
      self._mapping[key] = value

  __setattr__ = __setitem__  # type: ignore

  def __delitem__(self, key: K) -> None:
    del self._mapping[key]

  def __iter__(self) -> Iterator[K]:
    return iter(self._mapping)

  def __len__(self) -> int:
    return len(self._mapping)

  def __pretty_repr__(self):
    yield from pretty_repr_avoid_duplicate(self, _default_repr_object, _default_repr_attr)

  def split(self, *filters) -> Union[StateMapping[K, V], Tuple[StateMapping[K, V], ...]]:
    raise NotImplementedError

  def filter(self, *filters) -> Union[StateMapping[K, V], Tuple[StateMapping[K, V], ...]]:
    raise NotImplementedError

  def merge(self, *states) -> StateMapping[K, V]:
    raise NotImplementedError


def _default_repr_object(node: StateMapping):
  yield PrettyType(type(node), value_sep=': ', start='({', end='})')


def _default_repr_attr(node: StateMapping):
  for k, v in node.items():
    if isinstance(v, StateMapping):
      v = NestedStateRepr(v)
    yield PrettyAttr(repr(k), v)


class NestedMapping(StateMapping):
  """
  A pytree-like structure that contains a ``Mapping`` from strings or integers to leaves. 
  
  A valid leaf type is either :class:`State`, ``jax.Array``, ``numpy.ndarray`` or nested
  ``NestedMapping`` and ``FlattedMapping``.
  """
  __module__ = 'brainstate.util'

  def __getitem__(self, key: K) -> NestedMapping | V:
    value = self._mapping[key]
    if isinstance(value, dict):
      return NestedMapping(value, _copy=False)
    elif isinstance(value, Mapping):
      return NestedMapping(value)
    return value

  def __or__(self, other: NestedMapping[K, V]) -> NestedMapping[K, V]:
    if not other:
      return self
    assert isinstance(other, NestedMapping), f'expected NestedMapping; got {type(other).__qualname__}'
    return NestedMapping.merge(self, other)

  def __sub__(self, other: NestedMapping[K, V]) -> NestedMapping[K, V]:
    if not other:
      return self

    assert isinstance(other, NestedMapping), f'expected NestedMapping; got {type(other).__qualname__}'
    self_flat = self.to_flat()
    other_flat = other.to_flat()
    diff = {k: v for k, v in self_flat.items() if k not in other_flat}
    return NestedMapping.from_flat(diff)

  def to_flat(self) -> FlattedMapping:
    """
    Flatten the nested mapping into a flat mapping.

    Returns:
      The flattened mapping.
    """
    return flat_mapping(self._mapping)

  @classmethod
  def from_flat(
      cls,
      flat_mapping: Mapping[PathParts, V] | Iterable[tuple[PathParts, V]],
  ) -> NestedMapping:
    """
    Create a ``NestedMapping`` from a flat mapping.

    Args:
      flat_mapping: The flat mapping.

    Returns:
      The ``NestedMapping``.
    """
    if not isinstance(flat_mapping, Mapping):
      flat_mapping = dict(flat_mapping)
    nested_state = nest_mapping(flat_mapping)
    return cls(nested_state)

  def split(  # type: ignore[misc]
      self,
      first: Filter,
      /,
      *filters: Filter
  ) -> Union[NestedMapping[K, V], tuple[NestedMapping[K, V], ...]]:
    """
    Split a ``NestedMapping`` into one or more ``NestedMapping``'s. The
    user must pass at least one ``Filter`` (i.e. :class:`State`),
    and the filters must be exhaustive (i.e. they must cover all
    :class:`State` types in the ``NestedMapping``).

    Example usage::

      >>> import brainstate as bst

      >>> class Model(bst.nn.Module):
      ...   def __init__(self):
      ...     super().__init__()
      ...     self.batchnorm = bst.nn.BatchNorm1d([10, 3])
      ...     self.linear = bst.nn.Linear(2, 3)
      ...   def __call__(self, x):
      ...     return self.linear(self.batchnorm(x))

      >>> model = Model()
      >>> state_map = bst.graph.states_as_trees(model)
      >>> param, others = state_map.split(bst.ParamState, ...)

    Arguments:
      first: The first filter
      *filters: The optional, additional filters to group the state into mutually exclusive substates.

    Returns:
      One or more ``States`` equal to the number of filters passed.
    """
    filters = (first, *filters)
    *states_, rest = _split_nested_mapping(self, *filters)
    if rest:
      raise ValueError(f'Non-exhaustive filters, got a non-empty remainder: {rest}.\n'
                       f'Use `...` to match all remaining elements.')

    states: NestedMapping | Tuple[NestedMapping, ...]
    if len(states_) == 1:
      states = states_[0]
    else:
      states = tuple(states_)
    return states  # type: ignore[bad-return-type]

  def filter(
      self,
      first: Filter,
      /,
      *filters: Filter,
  ) -> Union[NestedMapping[K, V], Tuple[NestedMapping[K, V], ...]]:
    """
    Filter a ``NestedMapping`` into one or more ``NestedMapping``'s. The
    user must pass at least one ``Filter`` (i.e. :class:`State`).
    This method is similar to :meth:`split() <flax.nnx.NestedMapping.state.split>`,
    except the filters can be non-exhaustive.

    Arguments:
      first: The first filter
      *filters: The optional, additional filters to group the state into mutually exclusive substates.

    Returns:
      One or more ``States`` equal to the number of filters passed.
    """
    *states_, _rest = _split_nested_mapping(self, first, *filters)
    assert len(states_) == len(filters) + 1, f'Expected {len(filters) + 1} states, got {len(states_)}'
    if len(states_) == 1:
      states = states_[0]
    else:
      states = tuple(states_)
    return states  # type: ignore[bad-return-type]

  @staticmethod
  def merge(
      state: NestedMapping[K, V] | FlattedMapping[K, V],
      /,
      *states: NestedMapping[K, V] | FlattedMapping[K, V]
  ) -> NestedMapping[K, V]:
    """
    The inverse of :meth:`split()`.

    ``merge`` takes one or more ``StateMapping``'s and creates a new ``StateMapping``.

    Args:
      state: A ``StateMapping`` object.
      *states: Additional ``StateMapping`` objects.

    Returns:
      The merged ``StateMapping``.
    """
    if not states:
      return state
    states = (state, *states)
    new_state: FlattedMapping = dict()
    for state in states:
      if isinstance(state, NestedMapping):
        new_state.update(state.to_flat())  # type: ignore[attribute-error] # pytype is wrong here
      elif isinstance(state, FlattedMapping):
        new_state.update(state._mapping)
      else:
        raise TypeError(f'Expected Nested or Flatted Mapping, got {type(state)} instead.')
    return NestedMapping.from_flat(new_state)

  def to_pure_dict(
      self,
  ) -> Dict[str, Any]:
    flat_values = {k: x for k, x in self.to_flat().items()}
    return dict(nest_mapping(flat_values).raw_mapping)

  def replace_by_pure_dict(
      self,
      pure_dict: Dict[str, Any],
      replace_fn: Optional[SetValueFn] = None
  ):
    # Works for nnx.State and nnx.StateAsPyTree
    if replace_fn is None:
      replace_fn = lambda x, v: x.replace(v) if hasattr(x, 'replace') else v
    current_flat = self.to_flat()
    for kp, v in flat_mapping(pure_dict).items():
      if kp not in current_flat:
        raise ValueError(f'key in pure_dict not available in state: {kp}')
      current_flat[kp] = replace_fn(current_flat[kp], v)
    self.update(nest_mapping(current_flat))


class FlattedMapping(StateMapping):
  """
  A pytree-like structure that contains a ``Mapping`` from strings or integers to leaves.

  A valid leaf type is either :class:`State`, ``jax.Array``, ``numpy.ndarray`` or Python variables.

  A ``NestedMapping`` can be generated by either calling :func:`states()` or
  :func:`nodes()` on the :class:`Module`.

  Example usage::

    >>> import brainstate as bst
    >>> import jax.numpy as jnp
    >>>
    >>> class Model(bst.nn.Module):
    ...   def __init__(self):
    ...     super().__init__()
    ...     self.batchnorm = bst.nn.BatchNorm1d([10, 3])
    ...     self.linear = bst.nn.Linear(2, 3)
    ...   def __call__(self, x):
    ...     return self.linear(self.batchnorm(x))
    >>>
    >>> model = Model()

    >>> # retrieve the states of the model
    >>> model.states()  # with the same to the function of ``brainstate.graph.states()``
    FlattedMapping({
      ('batchnorm', 'running_mean'): LongTermState(
        value=Array([[0., 0., 0.]], dtype=float32)
      ),
      ('batchnorm', 'running_var'): LongTermState(
        value=Array([[1., 1., 1.]], dtype=float32)
      ),
      ('batchnorm', 'weight'): ParamState(
        value={'bias': Array([[0., 0., 0.]], dtype=float32), 'scale': Array([[1., 1., 1.]], dtype=float32)}
      ),
      ('linear', 'weight'): ParamState(
        value={'weight': Array([[-1.797768  ,  0.04032926, -0.9236347 ,  1.4785787 ],
               [ 0.46879035,  0.09185036,  0.3434357 ,  0.71752423],
               [ 1.606369  , -0.06655207, -0.38215336, -1.6266369 ]],      dtype=float32), 'bias': Array([0., 0., 0., 0.], dtype=float32)}
      )
    })

    >>> # retrieve the nodes of the model
    >>> model.nodes()  # with the same to the function of ``brainstate.graph.nodes()``
    FlattedMapping({
      ('batchnorm', 'bias_initializer'): Constant(value=0.0, dtype=<class 'numpy.float32'>),
      ('batchnorm', 'scale_initializer'): Constant(value=1.0, dtype=<class 'numpy.float32'>),
      ('batchnorm',): BatchNorm1d(
        in_size=(10, 3),
        out_size=(10, 3),
        affine=True,
        bias_initializer=Constant(value=0.0, dtype=<class 'numpy.float32'>),
        scale_initializer=Constant(value=1.0, dtype=<class 'numpy.float32'>),
        dtype=<class 'numpy.float32'>,
        track_running_stats=True,
        momentum='Array(shape=(), dtype=float32)',
        epsilon='Array(shape=(), dtype=float32)',
        feature_axis=(1,),
        axis_name=None,
        axis_index_groups=None,
        running_mean=LongTermState(
          value=Array([[0., 0., 0.]], dtype=float32),
        ),
        running_var=LongTermState(
          value=Array([[1., 1., 1.]], dtype=float32),
        ),
        weight=ParamState(
          value={'bias': Array([[0., 0., 0.]], dtype=float32), 'scale': Array([[1., 1., 1.]], dtype=float32)},
        )
      ),
      ('linear',): Linear(
        in_size=(10, 3),
        out_size=(10, 4),
        w_mask=None,
        weight=ParamState(
          value={'weight': Array([[ 0.46051726,  1.311077  ,  1.1818101 , -0.98705536],
                 [-0.55922306, -0.6097833 ,  0.9610266 , -0.9381395 ],
                 [ 1.1144904 , -0.88757664,  0.4731988 ,  0.3364881 ]],      dtype=float32), 'bias': Array([0., 0., 0., 0.], dtype=float32)},
        )
      ),
      (): Model(
        batchnorm=BatchNorm1d(...),
        linear=Linear(...)
      )
    })
  """
  __module__ = 'brainstate.util'

  def __getitem__(self, key: K) -> FlattedMapping | V:
    value = self._mapping[key]
    if isinstance(value, dict):
      return FlattedMapping(value, _copy=False)
    elif isinstance(value, Mapping):
      return FlattedMapping(value)
    return value

  def __or__(self, other: FlattedMapping[K, V]) -> FlattedMapping[K, V]:
    if not other:
      return self
    assert isinstance(other, FlattedMapping), f'expected NestedMapping; got {type(other).__qualname__}'
    return FlattedMapping.merge(self, other)

  def __sub__(self, other: FlattedMapping[K, V]) -> FlattedMapping[K, V]:
    if not other:
      return self
    assert isinstance(other, FlattedMapping), f'expected NestedMapping; got {type(other).__qualname__}'
    diff = {k: v for k, v in self.items() if k not in other}
    return FlattedMapping(diff)

  def to_nest(self) -> NestedMapping:
    """
    Unflatten the flat mapping into a nested mapping.

    Returns:
      The nested mapping.
    """
    return nest_mapping(dict(self._mapping))

  @classmethod
  def from_nest(
      cls, flat_mapping: Mapping[PathParts, V] | Iterable[tuple[PathParts, V]],
  ) -> FlattedMapping:
    """
    Create a ``NestedMapping`` from a flat mapping.

    Args:
      flat_mapping: The flat mapping.

    Returns:
      The ``NestedMapping``.
    """
    return flat_mapping(dict(flat_mapping))

  def split(  # type: ignore[misc]
      self,
      first: Filter,
      /,
      *filters: Filter
  ) -> Union[FlattedMapping[K, V], tuple[FlattedMapping[K, V], ...]]:
    """
    Split a ``FlattedMapping`` into one or more ``FlattedMapping``'s. The
    user must pass at least one ``Filter`` (i.e. :class:`State`),
    and the filters must be exhaustive (i.e. they must cover all
    :class:`State` types in the ``NestedMapping``).

    Arguments:
      first: The first filter
      *filters: The optional, additional filters to group the state into mutually exclusive substates.

    Returns:
      One or more ``States`` equal to the number of filters passed.
    """
    filters = (first, *filters)
    *states_, rest = _split_flatted_mapping(self, *filters)
    if rest:
      raise ValueError(f'Non-exhaustive filters, got a non-empty remainder: {rest}.\n'
                       f'Use `...` to match all remaining elements.')

    states: FlattedMapping | Tuple[FlattedMapping, ...]
    if len(states_) == 1:
      states = states_[0]
    else:
      states = tuple(states_)
    return states  # type: ignore[bad-return-type]

  def filter(
      self,
      first: Filter,
      /,
      *filters: Filter,
  ) -> Union[FlattedMapping[K, V], Tuple[FlattedMapping[K, V], ...]]:
    """
    Filter a ``FlattedMapping`` into one or more ``FlattedMapping``'s. The
    user must pass at least one ``Filter`` (i.e. :class:`State`).
    This method is similar to :meth:`split() <flax.nnx.NestedMapping.state.split>`,
    except the filters can be non-exhaustive.

    Arguments:
      first: The first filter
      *filters: The optional, additional filters to group the state into mutually exclusive substates.

    Returns:
      One or more ``States`` equal to the number of filters passed.
    """
    *states_, _rest = _split_flatted_mapping(self, first, *filters)
    assert len(states_) == len(filters) + 1, f'Expected {len(filters) + 1} states, got {len(states_)}'
    if len(states_) == 1:
      states = states_[0]
    else:
      states = tuple(states_)
    return states  # type: ignore[bad-return-type]

  @staticmethod
  def merge(
      state: FlattedMapping[K, V] | NestedMapping[K, V],
      /,
      *states: FlattedMapping[K, V] | NestedMapping[K, V]
  ) -> FlattedMapping[K, V]:
    """
    The inverse of :meth:`split()`.

    ``merge`` takes one or more ``FlattedMapping``'s and creates a new ``FlattedMapping``.

    Args:
      state: A ``StateMapping`` object.
      *states: Additional ``StateMapping`` objects.

    Returns:
      The merged ``StateMapping``.
    """
    if not states:
      return state
    states = (state, *states)
    new_state: FlattedStateMapping[V] = {}
    for state in states:
      if isinstance(state, NestedMapping):
        new_state.update(state.to_flat())  # type: ignore[attribute-error] # pytype is wrong here
      elif isinstance(state, FlattedMapping):
        new_state.update(state._mapping)
      else:
        raise TypeError(f'Expected Nested or Flatted Mapping, got {type(state)} instead.')
    return FlattedMapping(new_state)

  def remove_by_values(
      self,
      *values: Iterable[V],
      process_fn: Optional[Callable[[V], V]] = _default_process,
      contain_fn: Optional[Callable[[V, Tuple[V]], bool]] = _default_compare
  ) -> FlattedMapping[K, V]:
    values = jax.tree.map(process_fn, values)
    new_flat_state = {k: v for k, v in self._mapping.items() if contain_fn(v, values)}
    return FlattedMapping(new_flat_state)

  def to_pure_dict(self, ) -> Dict[str, Any]:
    return dict(self.raw_mapping)


def _split_nested_mapping(
    mapping: NestedMapping[K, V],
    *filters: Filter,
) -> Tuple[NestedMapping[K, V], ...]:
  # check if the filters are exhaustive
  for i, filter_ in enumerate(filters):
    if filter_ in (..., True) and i != len(filters) - 1:
      remaining_filters = filters[i + 1:]
      if not all(f in (..., True) for f in remaining_filters):
        raise ValueError('`...` or `True` can only be used as the last filters, '
                         f'got {filter_} it at index {i}.')

  # change the filters to predicates
  predicates = tuple(map(to_predicate, filters))

  # we have n + 1 state mappings, where n is the number of predicates
  # the last state mapping is for values that don't match any predicate
  flat_states: tuple[FlattedStateMapping[V], ...] = tuple({} for _ in range(len(predicates) + 1))

  assert isinstance(mapping, NestedMapping), f'expected NestedMapping; got {type(mapping).__qualname__}'
  flat_state = mapping.to_flat()
  for path, value in flat_state.items():
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i][path] = value  # type: ignore[index] # mypy is wrong here?
        break
    else:
      # if we didn't break, set leaf to last state
      flat_states[-1][path] = value  # type: ignore[index] # mypy is wrong here?

  return tuple(NestedMapping.from_flat(flat_state) for flat_state in flat_states)


def _split_flatted_mapping(
    mapping: FlattedMapping[K, V],
    *filters: Filter,
) -> Tuple[FlattedMapping[K, V], ...]:
  # check if the filters are exhaustive
  for i, filter_ in enumerate(filters):
    if filter_ in (..., True) and i != len(filters) - 1:
      remaining_filters = filters[i + 1:]
      if not all(f in (..., True) for f in remaining_filters):
        raise ValueError('`...` or `True` can only be used as the last filters, '
                         f'got {filter_} it at index {i}.')

  # change the filters to predicates
  predicates = tuple(map(to_predicate, filters))

  # we have n + 1 state mappings, where n is the number of predicates
  # the last state mapping is for values that don't match any predicate
  flat_states: tuple[FlattedStateMapping[V], ...] = tuple({} for _ in range(len(predicates) + 1))

  assert isinstance(mapping, FlattedMapping), f'expected FlattedMapping; got {type(mapping).__qualname__}'
  for path, value in mapping.items():
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i][path] = value  # type: ignore[index] # mypy is wrong here?
        break
    else:
      # if we didn't break, set leaf to last state
      flat_states[-1][path] = value  # type: ignore[index] # mypy is wrong here?

  return tuple(FlattedMapping(flat_state) for flat_state in flat_states)


# register ``NestedMapping`` as a pytree
def _nest_flatten_with_keys(x: NestedMapping):
  items = sorted(x._mapping.items())
  children = tuple((jax.tree_util.DictKey(key), value) for key, value in items)
  return children, tuple(key for key, _ in items)


def _nest_unflatten(
    static: Tuple[K, ...],
    leaves: Tuple[V, ...] | Tuple[Dict[K, V]],
):
  return NestedMapping(zip(static, leaves))


jax.tree_util.register_pytree_with_keys(NestedMapping,
                                        _nest_flatten_with_keys,
                                        _nest_unflatten)  # type: ignore[arg-type]


# register ``FlattedMapping`` as a pytree

def _flat_flatten_with_keys(x: FlattedMapping):
  items = sorted(x._mapping.items())
  children = tuple((jax.tree_util.DictKey(key), value) for key, value in items)
  return children, tuple(key for key, _ in items)


def _flat_unflatten(
    static: Tuple[K, ...],
    leaves: Tuple[V, ...] | Tuple[Dict[K, V]],
):
  return FlattedMapping(zip(static, leaves))


jax.tree_util.register_pytree_with_keys(FlattedMapping,
                                        _flat_flatten_with_keys,
                                        _flat_unflatten)  # type: ignore[arg-type]
