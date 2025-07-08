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

"""Utilities for defining custom classes that can be used with jax transformations."""

import collections
import dataclasses
from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Any, TypeVar

import jax
from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet

__all__ = [
    'dataclass',
    'field',
    'PyTreeNode',
    'FrozenDict',
]

K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')


def field(pytree_node=True, *, metadata=None, **kwargs):
    return dataclasses.field(metadata=(metadata or {}) | {'pytree_node': pytree_node}, **kwargs)


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def dataclass(clz: T, **kwargs) -> T:
    """
    Create a class which can be passed to functional transformations.

    .. note::
      Inherit from ``PyTreeNode`` instead to avoid type checking issues when
      using PyType.

    Jax transformations such as ``jax.jit`` and ``jax.grad`` require objects that are
    immutable and can be mapped over using the ``jax.tree_util`` methods.
    The ``dataclass`` decorator makes it easy to define custom classes that can be
    passed safely to Jax. For example::

      >>> import brainstate as brainstate
      >>> import jax
      >>> from typing import Any, Callable

      >>> @brainstate.util.dataclass
      ... class Model:
      ...   params: Any
      ...   # use pytree_node=False to indicate an attribute should not be touched
      ...   # by Jax transformations.
      ...   apply_fn: Callable = brainstate.util.field(pytree_node=False)

      ...   def __apply__(self, *args):
      ...     return self.apply_fn(*args)

      >>> params = {}
      >>> params_b = {}
      >>> apply_fn = lambda v, x: x
      >>> model = Model(params, apply_fn)

      >>> # model.params = params_b  # Model is immutable. This will raise an error.
      >>> model_b = model.replace(params=params_b)  # Use the replace method instead.

      >>> # This class can now be used safely in Jax to compute gradients w.r.t. the
      >>> # parameters.
      >>> model = Model(params, apply_fn)
      >>> loss_fn = lambda model: 3.
      >>> model_grad = jax.grad(loss_fn)(model)

    Note that dataclasses have an auto-generated ``__init__`` where
    the arguments of the constructor and the attributes of the created
    instance match 1:1. This correspondence is what makes these objects
    valid containers that work with JAX transformations and
    more generally the ``jax.tree_util`` library.

    Sometimes a "smart constructor" is desired, for example because
    some of the attributes can be (optionally) derived from others.
    The way to do this with Flax dataclasses is to make a static or
    class method that provides the smart constructor.
    This way the simple constructor used by ``jax.tree_util`` is
    preserved. Consider the following example::

      >>> @brainstate.util.dataclass
      ... class DirectionAndScaleKernel:
      ...   direction: jax.Array
      ...   scale: jax.Array

      ...   @classmethod
      ...   def create(cls, kernel):
      ...     scale = jax.numpy.linalg.norm(kernel, axis=0, keepdims=True)
      ...     direction = direction / scale
      ...     return cls(direction, scale)

    Args:
      clz: the class that will be transformed by the decorator.
    Returns:
      The new class.
    """
    # check if already a flax dataclass
    if '_brainstate_dataclass' in clz.__dict__:
        return clz

    if 'frozen' not in kwargs.keys():
        kwargs['frozen'] = True
    data_clz = dataclasses.dataclass(**kwargs)(clz)  # type: ignore
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(data_clz):
        is_pytree_node = field_info.metadata.get('pytree_node', True)
        if is_pytree_node:
            data_fields.append(field_info.name)
        else:
            meta_fields.append(field_info.name)

    def replace(self, **updates):
        """ "Returns a new object replacing the specified fields with new values."""
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    # Remove this guard once minimux JAX version is >0.4.26.
    try:
        if hasattr(jax.tree_util, 'register_dataclass'):
            jax.tree_util.register_dataclass(
                data_clz, data_fields, meta_fields
            )
        else:
            raise NotImplementedError
    except NotImplementedError:

        def iterate_clz(x):
            meta = tuple(getattr(x, name) for name in meta_fields)
            data = tuple(getattr(x, name) for name in data_fields)
            return data, meta

        def iterate_clz_with_keys(x):
            meta = tuple(getattr(x, name) for name in meta_fields)
            data = tuple(
                (jax.tree_util.GetAttrKey(name), getattr(x, name))
                for name in data_fields
            )
            return data, meta

        def clz_from_iterable(meta, data):
            meta_args = tuple(zip(meta_fields, meta))
            data_args = tuple(zip(data_fields, data))
            kwargs = dict(meta_args + data_args)
            return data_clz(**kwargs)

        jax.tree_util.register_pytree_with_keys(
            data_clz,
            iterate_clz_with_keys,
            clz_from_iterable,
            iterate_clz,
        )

    # add a _brainstate_dataclass flag to distinguish from regular dataclasses
    data_clz._brainstate_dataclass = True  # type: ignore[attr-defined]

    return data_clz  # type: ignore


TNode = TypeVar('TNode', bound='PyTreeNode')


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
class PyTreeNode:
    """Base class for dataclasses that should act like a JAX pytree node.

    See ``flax.struct.dataclass`` for the ``jax.tree_util`` behavior.
    This base class additionally avoids type checking errors when using PyType.

    Example::

      >>> import brainstate as brainstate
      >>> import jax
      >>> from typing import Any, Callable

      >>> class Model(brainstate.util.PyTreeNode):
      ...   params: Any
      ...   # use pytree_node=False to indicate an attribute should not be touched
      ...   # by Jax transformations.
      ...   apply_fn: Callable = brainstate.util.field(pytree_node=False)

      ...   def __apply__(self, *args):
      ...     return self.apply_fn(*args)

      >>> params = {}
      >>> params_b = {}
      >>> apply_fn = lambda v, x: x
      >>> model = Model(params, apply_fn)

      >>> # model.params = params_b  # Model is immutable. This will raise an error.
      >>> model_b = model.replace(params=params_b)  # Use the replace method instead.

      >>> # This class can now be used safely in Jax to compute gradients w.r.t. the
      >>> # parameters.
      >>> model = Model(params, apply_fn)
      >>> loss_fn = lambda model: 3.
      >>> model_grad = jax.grad(loss_fn)(model)
    """

    def __init_subclass__(cls, **kwargs):
        dataclass(cls, **kwargs)  # pytype: disable=wrong-arg-types

    def __init__(self, *args, **kwargs):
        # stub for pytype
        raise NotImplementedError

    def replace(self: TNode, **overrides) -> TNode:
        # stub for pytype
        raise NotImplementedError


def _indent(x, num_spaces):
    indent_str = ' ' * num_spaces
    lines = x.split('\n')
    assert not lines[-1]
    # skip the final line because it's empty and should not be indented.
    return '\n'.join(indent_str + line for line in lines[:-1]) + '\n'


@jax.tree_util.register_pytree_with_keys_class
class FrozenDict(Mapping[K, V]):
    """
    An immutable variant of the Python dict.
    """

    __slots__ = ('_dict', '_hash')

    def __init__(self, *args, __unsafe_skip_copy__=False, **kwargs):  # pylint: disable=invalid-name
        # make sure the dict is as
        xs = dict(*args, **kwargs)
        if __unsafe_skip_copy__:
            self._dict = xs
        else:
            self._dict = _prepare_freeze(xs)

        self._hash = None

    def __getitem__(self, key):
        v = self._dict[key]
        if isinstance(v, dict):
            return FrozenDict(v)
        return v

    def __setitem__(self, key, value):
        raise ValueError('FrozenDict is immutable.')

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return self.pretty_repr()

    def __reduce__(self):
        return FrozenDict, (self.unfreeze(),)

    def pretty_repr(self, num_spaces=4):
        """Returns an indented representation of the nested dictionary."""

        def pretty_dict(x):
            if not isinstance(x, dict):
                return repr(x)
            rep = ''
            for key, val in x.items():
                rep += f'{key}: {pretty_dict(val)},\n'
            if rep:
                return '{\n' + _indent(rep, num_spaces) + '}'
            else:
                return '{}'

        return f'FrozenDict({pretty_dict(self._dict)})'

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    def copy(
        self,
        add_or_replace: Mapping[K, V] = MappingProxyType({})
    ) -> 'FrozenDict[K, V]':
        """Create a new FrozenDict with additional or replaced entries."""
        return type(self)({**self, **unfreeze(add_or_replace)})  # type: ignore[arg-type]

    def keys(self):
        return FrozenKeysView(self)

    def values(self):
        return FrozenValuesView(self)

    def items(self):
        for key in self._dict:
            yield (key, self[key])

    def pop(self, key: K) -> tuple['FrozenDict[K, V]', V]:
        """Create a new FrozenDict where one entry is removed.

        Example::

          >>> variables = FrozenDict({'params': {...}, 'batch_stats': {...}})
          >>> new_variables, params = variables.pop('params')

        Args:
          key: the key to remove from the dict
        Returns:
          A pair with the new FrozenDict and the removed value.
        """
        value = self[key]
        new_dict = dict(self._dict)
        new_dict.pop(key)
        new_self = type(self)(new_dict)
        return new_self, value

    def unfreeze(self) -> dict[K, V]:
        """Unfreeze this FrozenDict.

        Returns:
          An unfrozen version of this FrozenDict instance.
        """
        return unfreeze(self)

    def tree_flatten_with_keys(self) -> tuple[tuple[Any, ...], Hashable]:
        """Flattens this FrozenDict.

        Returns:
          A flattened version of this FrozenDict instance.
        """
        sorted_keys = sorted(self._dict)
        return tuple(
            [(jax.tree_util.DictKey(k), self._dict[k]) for k in sorted_keys]
        ), tuple(sorted_keys)

    @classmethod
    def tree_unflatten(cls, keys, values):
        # data is already deep copied due to tree map mechanism
        # we can skip the deep copy in the constructor
        return cls({k: v for k, v in zip(keys, values)}, __unsafe_skip_copy__=True)


def _prepare_freeze(xs: Any) -> Any:
    """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
    if isinstance(xs, FrozenDict):
        # we can safely ref share the internal state of a FrozenDict
        # because it is immutable.
        return xs._dict  # pylint: disable=protected-access
    if not isinstance(xs, dict):
        # return a leaf as is.
        return xs
    # recursively copy dictionary to avoid ref sharing
    return {key: _prepare_freeze(val) for key, val in xs.items()}


def freeze(xs: Mapping[Any, Any]) -> FrozenDict[Any, Any]:
    """Freeze a nested dict.

    Makes a nested ``dict`` immutable by transforming it into ``FrozenDict``.

    Args:
      xs: Dictionary to freeze (a regualr Python dict).
    Returns:
      The frozen dictionary.
    """
    return FrozenDict(xs)


def unfreeze(x: FrozenDict | dict[str, Any]) -> dict[Any, Any]:
    """Unfreeze a FrozenDict.

    Makes a mutable copy of a ``FrozenDict`` mutable by transforming
    it into (nested) dict.

    Args:
      x: Frozen dictionary to unfreeze.
    Returns:
      The unfrozen dictionary (a regular Python dict).
    """
    if isinstance(x, FrozenDict):
        # deep copy internal state of a FrozenDict
        # the dict branch would also work here but
        # it is much less performant because jax.tree_util.tree_map
        # uses an optimized C implementation.
        return jax.tree_util.tree_map(lambda y: y, x._dict)  # type: ignore
    elif isinstance(x, dict):
        ys = {}
        for key, value in x.items():
            ys[key] = unfreeze(value)
        return ys
    else:
        return x


def copy(
    x: FrozenDict | dict[str, Any],
    add_or_replace: FrozenDict[str, Any] | dict[str, Any] = FrozenDict({}),
) -> FrozenDict | dict[str, Any]:
    """Create a new dict with additional and/or replaced entries. This is a utility
    function that can act on either a FrozenDict or regular dict and mimics the
    behavior of ``FrozenDict.copy``.

    Example::

      >>> variables = FrozenDict({'params': {...}, 'batch_stats': {...}})
      >>> new_variables = copy(variables, {'additional_entries': 1})

    Args:
      x: the dictionary to be copied and updated
      add_or_replace: dictionary of key-value pairs to add or replace in the dict x
    Returns:
      A new dict with the additional and/or replaced entries.
    """

    if isinstance(x, FrozenDict):
        return x.copy(add_or_replace)
    elif isinstance(x, dict):
        new_dict = jax.tree_util.tree_map(
            lambda x: x, x
        )  # make a deep copy of dict x
        new_dict.update(add_or_replace)
        return new_dict
    raise TypeError(f'Expected FrozenDict or dict, got {type(x)}')


def pop(
    x: FrozenDict | dict[str, Any], key: str
) -> tuple[FrozenDict | dict[str, Any], Any]:
    """Create a new dict where one entry is removed. This is a utility
    function that can act on either a FrozenDict or regular dict and
    mimics the behavior of ``FrozenDict.pop``.

    Example::

      >>> variables = FrozenDict({'params': {...}, 'batch_stats': {...}})
      >>> new_variables, params = pop(variables, 'params')

    Args:
      x: the dictionary to remove the entry from
      key: the key to remove from the dict
    Returns:
      A pair with the new dict and the removed value.
    """

    if isinstance(x, FrozenDict):
        return x.pop(key)
    elif isinstance(x, dict):
        new_dict = jax.tree_util.tree_map(
            lambda x: x, x
        )  # make a deep copy of dict x
        value = new_dict.pop(key)
        return new_dict, value
    raise TypeError(f'Expected FrozenDict or dict, got {type(x)}')


def pretty_repr(x: Any, num_spaces: int = 4) -> str:
    """Returns an indented representation of the nested dictionary.
    This is a utility function that can act on either a FrozenDict or
    regular dict and mimics the behavior of ``FrozenDict.pretty_repr``.
    If x is any other dtype, this function will return ``repr(x)``.

    Args:
      x: the dictionary to be represented
      num_spaces: the number of space characters in each indentation level
    Returns:
      An indented string representation of the nested dictionary.
    """

    if isinstance(x, FrozenDict):
        return x.pretty_repr()
    else:

        def pretty_dict(x):
            if not isinstance(x, dict):
                return repr(x)
            rep = ''
            for key, val in x.items():
                rep += f'{key}: {pretty_dict(val)},\n'
            if rep:
                return '{\n' + _indent(rep, num_spaces) + '}'
            else:
                return '{}'

        return pretty_dict(x)


class FrozenKeysView(collections.abc.KeysView):
    """A wrapper for a more useful repr of the keys in a frozen dict."""

    def __repr__(self):
        return f'frozen_dict_keys({list(self)})'


class FrozenValuesView(collections.abc.ValuesView):
    """A wrapper for a more useful repr of the values in a frozen dict."""

    def __repr__(self):
        return f'frozen_dict_values({list(self)})'
