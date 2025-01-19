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

from __future__ import annotations

from collections import abc
from typing import TypeVar, Hashable, Union, Iterable, Any, Optional, Tuple, Dict

import jax

from brainstate.typing import Filter, PathParts
from ._filter import to_predicate
from ._pretty_repr import PrettyRepr, PrettyType, PrettyAttr, pretty_repr_avoid_duplicate, get_repr
from ._struct import dataclass

__all__ = [
    'NestedDict', 'FlattedDict', 'flat_mapping', 'nest_mapping',
]

A = TypeVar('A')
K = TypeVar('K', bound=Hashable)
V = TypeVar('V')

FlattedStateMapping = dict[PathParts, V]
ExtractValueFn = abc.Callable[[Any], Any]
SetValueFn = abc.Callable[[V, Any], V]


# the empty node is a struct.dataclass to be compatible with JAX.
@dataclass
class _EmptyNode:
    pass


_default_leaf = lambda *args: False
empty_node = _EmptyNode()
IsLeafCallable = abc.Callable[[Tuple[Any, ...], abc.Mapping[Any, Any]], bool]


def flat_mapping(
    xs: abc.Mapping[Any, Any],
    /,
    *,
    keep_empty_nodes: bool = False,
    is_leaf: Optional[IsLeafCallable] = _default_leaf,
    sep: Optional[str] = None
) -> 'FlattedDict':
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
    assert isinstance(xs, abc.Mapping), f'expected Mapping; got {type(xs).__qualname__}'

    if sep is None:
        def _key(path: Tuple[Any, ...]) -> Tuple[Any, ...] | str:
            return path
    else:

        def _key(path: Tuple[Any, ...]) -> Tuple[Any, ...] | str:
            return sep.join(path)

    def _flatten(xs: Any, prefix: Tuple[Any, ...]) -> Dict[Any, Any]:
        if not isinstance(xs, abc.Mapping) or is_leaf(prefix, xs):
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

    return FlattedDict(_flatten(xs, ()))


def nest_mapping(
    xs: Any,
    /,
    *,
    sep: str | None = None
) -> 'NestedDict':
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
    assert isinstance(xs, abc.Mapping), f'expected Mapping; got {type(xs).__qualname__}'
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
    return NestedDict(result)


def _default_compare(x, values):
    return id(x) in values


def _default_process(x):
    return id(x)


class NestedStateRepr(PrettyRepr):
    def __init__(self, state: PrettyDict):
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
            if isinstance(v, PrettyDict):
                v = NestedStateRepr(v)
            children[k] = v
        # Render as the dictionary itself at the same path.
        return subtree_renderer(children, path=path)


class PrettyDict(dict, PrettyRepr):
    __module__ = 'brainstate.util'

    def __getattr__(self, key: K) -> NestedMapping | V:  # type: ignore[misc]
        return self[key]

    def treefy_state(self):
        """
        Convert the ``State`` objects to a reference tree of the state.
        """
        from brainstate._state import State
        leaves, treedef = jax.tree.flatten(self)
        leaves = jax.tree.map(lambda x: x.to_state_ref() if isinstance(x, State) else x, leaves)
        return treedef.unflatten(leaves)

    def to_dict(self) -> Dict[K, Dict[K, Any] | V]:
        """
        Convert the ``PrettyDict`` to a dictionary.

        Returns:
          The dictionary.
        """
        return dict(self)  # type: ignore

    def __repr__(self) -> str:
        # repr the individual object with the pretty representation
        return get_repr(self)

    def __pretty_repr__(self):
        yield from pretty_repr_avoid_duplicate(self, _default_repr_object, _default_repr_attr)

    def split(self, *filters) -> Union[PrettyDict[K, V], Tuple[PrettyDict[K, V], ...]]:
        raise NotImplementedError

    def filter(self, *filters) -> Union[PrettyDict[K, V], Tuple[PrettyDict[K, V], ...]]:
        raise NotImplementedError

    def merge(self, *states) -> PrettyDict[K, V]:
        raise NotImplementedError

    def subset(self, *filters) -> Union[PrettyDict[K, V], Tuple[PrettyDict[K, V], ...]]:
        """
        Subset a ``PrettyDict`` into one or more ``PrettyDict``'s. The user must pass at least one
        ``Filter`` (i.e. :class:`State`), and the filters must be exhaustive (i.e. they must cover all
        :class:`State` types in the ``PrettyDict``).
        """
        return self.filter(*filters)


def _default_repr_object(node: PrettyDict):
    yield PrettyType(type(node), value_sep=': ', start='({', end='})')


def _default_repr_attr(node: PrettyDict):
    for k, v in node.items():
        if isinstance(v, dict):
            v = PrettyDict(v)
        if isinstance(v, PrettyDict):
            v = NestedStateRepr(v)
        yield PrettyAttr(repr(k), v)


class NestedDict(PrettyDict):
    """
    A pytree-like structure that contains a ``Mapping`` from strings or integers to leaves.

    A valid leaf type is either :class:`State`, ``jax.Array``, ``numpy.ndarray`` or nested
    ``NestedDict`` and ``FlattedDict``.
    """
    __module__ = 'brainstate.util'

    def __or__(self, other: NestedDict[K, V]) -> NestedDict[K, V]:
        if not other:
            return self
        assert isinstance(other, NestedDict), f'expected NestedDict; got {type(other).__qualname__}'
        return NestedDict.merge(self, other)

    def __sub__(self, other: NestedDict[K, V]) -> NestedDict[K, V]:
        if not other:
            return self

        assert isinstance(other, NestedDict), f'expected NestedDict; got {type(other).__qualname__}'
        self_flat = self.to_flat()
        other_flat = other.to_flat()
        diff = {k: v for k, v in self_flat.items() if k not in other_flat}
        return NestedDict.from_flat(diff)

    def to_flat(self) -> FlattedDict:
        """
        Flatten the nested mapping into a flat mapping.

        Returns:
          The flattened mapping.
        """
        return flat_mapping(self)

    @classmethod
    def from_flat(cls, flat_dict: abc.Mapping[PathParts, V] | Iterable[tuple[PathParts, V]]) -> NestedDict:
        """
        Create a ``NestedDict`` from a flat mapping.

        Args:
          flat_dict: The flat mapping.

        Returns:
          The ``NestedDict``.
        """
        nested_state = nest_mapping(dict(flat_dict))
        return cls(nested_state)

    def split(  # type: ignore[misc]
        self,
        first: Filter,
        /,
        *filters: Filter
    ) -> Union[NestedDict[K, V], Tuple[NestedDict[K, V], ...]]:
        """
        Split a ``NestedDict`` into one or more ``NestedDict``'s. The
        user must pass at least one ``Filter`` (i.e. :class:`State`),
        and the filters must be exhaustive (i.e. they must cover all
        :class:`State` types in the ``NestedDict``).

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
          >>> state_map = bst.graph.treefy_states(model)
          >>> param, others = state_map.treefy_split(bst.ParamState, ...)

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

        states: NestedDict | Tuple[NestedDict, ...]
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
    ) -> Union[NestedDict[K, V], Tuple[NestedDict[K, V], ...]]:
        """
        Filter a ``NestedDict`` into one or more ``NestedDict``'s. The
        user must pass at least one ``Filter`` (i.e. :class:`State`).
        This method is similar to :meth:`split() <flax.nnx.NestedDict.state.split>`,
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
        state: NestedDict[K, V] | FlattedDict[K, V],
        /,
        *states: NestedDict[K, V] | FlattedDict[K, V]
    ) -> NestedDict[K, V]:
        """
        The inverse of :meth:`split()`.

        ``merge`` takes one or more ``PrettyDict``'s and creates a new ``PrettyDict``.

        Args:
          state: A ``PrettyDict`` object.
          *states: Additional ``PrettyDict`` objects.

        Returns:
          The merged ``PrettyDict``.
        """
        if not states:
            return state
        states = (state, *states)
        new_state: FlattedDict = FlattedDict()
        for state in states:
            if isinstance(state, NestedDict):
                new_state.update(state.to_flat())  # type: ignore[attribute-error] # pytype is wrong here
            elif isinstance(state, FlattedDict):
                new_state.update(state)
            else:
                raise TypeError(f'Expected Nested or Flatted Mapping, got {type(state)} instead.')
        return NestedDict.from_flat(new_state)

    def to_pure_dict(self) -> Dict[str, Any]:
        flat_values = {k: x for k, x in self.to_flat().items()}
        return nest_mapping(flat_values).to_dict()

    def replace_by_pure_dict(
        self,
        pure_dict: Dict[str, Any],
        replace_fn: Optional[SetValueFn] = None
    ):
        if replace_fn is None:
            replace_fn = lambda x, v: x.replace(v) if hasattr(x, 'replace') else v
        current_flat = self.to_flat()
        for kp, v in flat_mapping(pure_dict).items():
            if kp not in current_flat:
                raise ValueError(f'key in pure_dict not available in state: {kp}')
            current_flat[kp] = replace_fn(current_flat[kp], v)
        self.update(nest_mapping(current_flat))


class FlattedDict(PrettyDict):
    """
    A pytree-like structure that contains a ``Mapping`` from strings or integers to leaves.

    A valid leaf type is either :class:`State`, ``jax.Array``, ``numpy.ndarray`` or Python variables.

    A ``NestedDict`` can be generated by either calling :func:`states()` or
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
      FlattedDict({
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
          value={'weight': Array([[-0.21467684,  0.7621282 , -0.50756454, -0.49047297],
                 [-0.90413696,  0.6711    , -0.1254792 ,  0.50412565],
                 [ 0.23975602,  0.47905368,  1.4851435 ,  0.16745673]],      dtype=float32), 'bias': Array([0., 0., 0., 0.], dtype=float32)}
        )
      })

      >>> # retrieve the nodes of the model
      >>> model.nodes()  # with the same to the function of ``brainstate.graph.nodes()``
      FlattedDict({
        ('batchnorm',): BatchNorm1d(
          in_size=(10, 3),
          out_size=(10, 3),
          affine=True,
          bias_initializer=Constant(value=0.0, dtype=<class 'numpy.float32'>),
          scale_initializer=Constant(value=1.0, dtype=<class 'numpy.float32'>),
          dtype=<class 'numpy.float32'>,
          track_running_stats=True,
          momentum=Array(shape=(), dtype=float32),
          epsilon=Array(shape=(), dtype=float32),
          feature_axis=(1,),
          axis_name=None,
          axis_index_groups=None,
          running_mean=LongTermState(
            value=Array(shape=(1, 3), dtype=float32)
          ),
          running_var=LongTermState(
            value=Array(shape=(1, 3), dtype=float32)
          ),
          weight=ParamState(
            value={'bias': Array(shape=(1, 3), dtype=float32), 'scale': Array(shape=(1, 3), dtype=float32)}
          )
        ),
        ('linear',): Linear(
          in_size=(10, 3),
          out_size=(10, 4),
          w_mask=None,
          weight=ParamState(
            value={'bias': Array(shape=(4,), dtype=float32), 'weight': Array(shape=(3, 4), dtype=float32)}
          )
        ),
        (): Model(
          batchnorm=BatchNorm1d(...),
          linear=Linear(...)
        )
      })
    """
    __module__ = 'brainstate.util'

    def __or__(self, other: FlattedDict[K, V]) -> FlattedDict[K, V]:
        if not other:
            return self
        assert isinstance(other, FlattedDict), f'expected NestedDict; got {type(other).__qualname__}'
        return FlattedDict.merge(self, other)

    def __sub__(self, other: FlattedDict[K, V]) -> FlattedDict[K, V]:
        if not other:
            return self
        assert isinstance(other, FlattedDict), f'expected NestedDict; got {type(other).__qualname__}'
        diff = {k: v for k, v in self.items() if k not in other}
        return FlattedDict(diff)

    def to_nest(self) -> NestedDict:
        """
        Unflatten the flat mapping into a nested mapping.

        Returns:
          The nested mapping.
        """
        return nest_mapping(self)

    @classmethod
    def from_nest(
        cls, nested_dict: abc.Mapping[PathParts, V] | Iterable[tuple[PathParts, V]],
    ) -> FlattedDict:
        """
        Create a ``NestedDict`` from a flat mapping.

        Args:
          nested_dict: The flat mapping.

        Returns:
          The ``NestedDict``.
        """
        return flat_mapping(nested_dict)

    def split(  # type: ignore[misc]
        self,
        first: Filter,
        /,
        *filters: Filter
    ) -> Union[FlattedDict[K, V], tuple[FlattedDict[K, V], ...]]:
        """
        Split a ``FlattedDict`` into one or more ``FlattedDict``'s. The
        user must pass at least one ``Filter`` (i.e. :class:`State`),
        and the filters must be exhaustive (i.e. they must cover all
        :class:`State` types in the ``NestedDict``).

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

        states: FlattedDict | Tuple[FlattedDict, ...]
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
    ) -> Union[FlattedDict[K, V], Tuple[FlattedDict[K, V], ...]]:
        """
        Filter a ``FlattedDict`` into one or more ``FlattedDict``'s. The
        user must pass at least one ``Filter`` (i.e. :class:`State`).
        This method is similar to :meth:`split() <flax.nnx.NestedDict.state.split>`,
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
        state: FlattedDict[K, V] | NestedDict[K, V],
        /,
        *states: FlattedDict[K, V] | NestedDict[K, V]
    ) -> FlattedDict[K, V]:
        """
        The inverse of :meth:`split()`.

        ``merge`` takes one or more ``FlattedDict``'s and creates a new ``FlattedDict``.

        Args:
          state: A ``PrettyDict`` object.
          *states: Additional ``PrettyDict`` objects.

        Returns:
          The merged ``PrettyDict``.
        """
        if not states:
            return state
        states = (state, *states)
        new_state: FlattedStateMapping[V] = {}
        for state in states:
            if isinstance(state, NestedDict):
                new_state.update(state.to_flat())  # type: ignore[attribute-error] # pytype is wrong here
            elif isinstance(state, FlattedDict):
                new_state.update(state)
            else:
                raise TypeError(f'Expected Nested or Flatted Mapping, got {type(state)} instead.')
        return FlattedDict(new_state)

    def to_dict_values(self):
        from brainstate._state import State
        return {k: v.value if isinstance(v, State) else v for k, v in self.items()}


def _split_nested_mapping(
    mapping: NestedDict[K, V],
    *filters: Filter,
) -> Tuple[NestedDict[K, V], ...]:
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

    assert isinstance(mapping, NestedDict), f'expected NestedDict; got {type(mapping).__qualname__}'
    flat_state = mapping.to_flat()
    for path, value in flat_state.items():
        for i, predicate in enumerate(predicates):
            if predicate(path, value):
                flat_states[i][path] = value  # type: ignore[index] # mypy is wrong here?
                break
        else:
            # if we didn't break, set leaf to last state
            flat_states[-1][path] = value  # type: ignore[index] # mypy is wrong here?

    return tuple(NestedDict.from_flat(flat_state) for flat_state in flat_states)


def _split_flatted_mapping(
    mapping: FlattedDict[K, V],
    *filters: Filter,
) -> Tuple[FlattedDict[K, V], ...]:
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

    assert isinstance(mapping, FlattedDict), f'expected FlattedDict; got {type(mapping).__qualname__}'
    for path, value in mapping.items():
        for i, predicate in enumerate(predicates):
            if predicate(path, value):
                flat_states[i][path] = value  # type: ignore[index] # mypy is wrong here?
                break
        else:
            # if we didn't break, set leaf to last state
            flat_states[-1][path] = value  # type: ignore[index] # mypy is wrong here?

    return tuple(FlattedDict(flat_state) for flat_state in flat_states)


# register ``NestedDict`` as a pytree
def _nest_flatten_with_keys(x: NestedDict):
    items = sorted(x.items())
    children = tuple((jax.tree_util.DictKey(key), value) for key, value in items)
    return children, tuple(key for key, _ in items)


def _nest_unflatten(
    static: Tuple[K, ...],
    leaves: Tuple[V, ...] | Tuple[Dict[K, V]],
):
    return NestedDict(zip(static, leaves))


jax.tree_util.register_pytree_with_keys(NestedDict,
                                        _nest_flatten_with_keys,
                                        _nest_unflatten)  # type: ignore[arg-type]


# register ``FlattedDict`` as a pytree

def _flat_unflatten(
    static: Tuple[K, ...],
    leaves: Tuple[V, ...] | Tuple[Dict[K, V]],
):
    return FlattedDict(zip(static, leaves))


jax.tree_util.register_pytree_with_keys(FlattedDict,
                                        _nest_flatten_with_keys,
                                        _flat_unflatten)  # type: ignore[arg-type]
