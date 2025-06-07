# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import builtins
import functools as ft
import importlib
import inspect
from typing import (
    Any, Callable, Hashable, List, Protocol, Tuple, TypeVar, Union,
    runtime_checkable, TYPE_CHECKING, Generic, Sequence
)

import brainunit as u
import jax
import numpy as np

tp = importlib.import_module("typing")

__all__ = [
    'PathParts',
    'Predicate',
    'Filter',
    'PyTree',
    'Size',
    'Shape',
    'Axes',
    'SeedOrKey',
    'ArrayLike',
    'DType',
    'DTypeLike',
    'Missing',
]

K = TypeVar('K')


@runtime_checkable
class Key(Hashable, Protocol):
    def __lt__(self: K, value: K, /) -> bool:
        ...


Ellipsis = builtins.ellipsis if TYPE_CHECKING else Any

PathParts = Tuple[Key, ...]
Predicate = Callable[[PathParts, Any], bool]
FilterLiteral = Union[type, str, Predicate, bool, Ellipsis, None]
Filter = Union[FilterLiteral, Tuple['Filter', ...], List['Filter']]

_T = TypeVar("_T")

_Annotation = TypeVar("_Annotation")


class _Array(Generic[_Annotation]):
    pass


_Array.__module__ = "builtins"


def _item_to_str(item: Union[str, type, slice]) -> str:
    if isinstance(item, slice):
        if item.step is not None:
            raise NotImplementedError
        return _item_to_str(item.start) + ": " + _item_to_str(item.stop)
    elif item is ...:
        return "..."
    elif inspect.isclass(item):
        return item.__name__
    else:
        return repr(item)


def _maybe_tuple_to_str(
    item: Union[str, type, slice, Tuple[Union[str, type, slice], ...]]
) -> str:
    if isinstance(item, tuple):
        if len(item) == 0:
            # Explicit brackets
            return "()"
        else:
            # No brackets
            return ", ".join([_item_to_str(i) for i in item])
    else:
        return _item_to_str(item)


class Array:
    def __class_getitem__(cls, item):
        class X:
            pass

        X.__module__ = "builtins"
        X.__qualname__ = _maybe_tuple_to_str(item)
        return _Array[X]


# Same __module__ trick here again. (So that we get the correct display when
# doing `def f(x: Array)` as well as `def f(x: Array["dim"])`.
#
# Don't need to set __qualname__ as that's already correct.
Array.__module__ = "builtins"


class _FakePyTree(Generic[_T]):
    pass


_FakePyTree.__name__ = "PyTree"
_FakePyTree.__qualname__ = "PyTree"
_FakePyTree.__module__ = "builtins"


class _MetaPyTree(type):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("PyTree cannot be instantiated")

    # Can't return a generic (e.g. _FakePyTree[item]) because generic aliases don't do
    # the custom __instancecheck__ that we want.
    # We can't add that __instancecheck__  via subclassing, e.g.
    # type("PyTree", (Generic[_T],), {}), because dynamic subclassing of typeforms
    # isn't allowed.
    # Likewise we can't do types.new_class("PyTree", (Generic[_T],), {}) because that
    # has __module__ "types", e.g. we get types.PyTree[int].
    @ft.lru_cache(maxsize=None)
    def __getitem__(cls, item):
        if isinstance(item, tuple):
            if len(item) == 2:

                class X(PyTree):
                    leaftype = item[0]
                    structure = item[1].strip()

                if not isinstance(X.structure, str):
                    raise ValueError(
                        "The structure annotation `struct` in "
                        "`brainstate.typing.PyTree[leaftype, struct]` must be be a string, "
                        f"e.g. `brainstate.typing.PyTree[leaftype, 'T']`. Got '{X.structure}'."
                    )
                pieces = X.structure.split()
                if len(pieces) == 0:
                    raise ValueError(
                        "The string `struct` in `brainstate.typing.PyTree[leaftype, struct]` "
                        "cannot be the empty string."
                    )
                for piece_index, piece in enumerate(pieces):
                    if (piece_index == 0) or (piece_index == len(pieces) - 1):
                        if piece == "...":
                            continue
                    if not piece.isidentifier():
                        raise ValueError(
                            "The string `struct` in "
                            "`brainstate.typing.PyTree[leaftype, struct]` must be be a "
                            "whitespace-separated sequence of identifiers, e.g. "
                            "`brainstate.typing.PyTree[leaftype, 'T']` or "
                            "`brainstate.typing.PyTree[leaftype, 'foo bar']`.\n"
                            "(Here, 'identifier' is used in the same sense as in "
                            "regular Python, i.e. a valid variable name.)\n"
                            f"Got piece '{piece}' in overall structure '{X.structure}'."
                        )
                name = str(_FakePyTree[item[0]])[:-1] + ', "' + item[1].strip() + '"]'
            else:
                raise ValueError(
                    "The subscript `foo` in `brainstate.typing.PyTree[foo]` must either be a "
                    "leaf type, e.g. `PyTree[int]`, or a 2-tuple of leaf and "
                    "structure, e.g. `PyTree[int, 'T']`. Received a tuple of length "
                    f"{len(item)}."
                )
        else:
            name = str(_FakePyTree[item])

            class X(PyTree):
                leaftype = item
                structure = None

        X.__name__ = name
        X.__qualname__ = name
        if getattr(tp, "GENERATING_DOCUMENTATION", False):
            X.__module__ = "builtins"
        else:
            X.__module__ = "brainstate.typing"
        return X


# Can't do `class PyTree(Generic[_T]): ...` because we need to override the
# instancecheck for PyTree[foo], but subclassing
# `type(Generic[int])`, i.e. `typing._GenericAlias` is disallowed.
PyTree = _MetaPyTree("PyTree", (), {})
if getattr(tp, "GENERATING_DOCUMENTATION", False):
    PyTree.__module__ = "builtins"
else:
    PyTree.__module__ = "brainstate.typing"
PyTree.__doc__ = """Represents a PyTree.

Annotations of the following sorts are supported:
```python
a: PyTree
b: PyTree[LeafType]
c: PyTree[LeafType, "T"]
d: PyTree[LeafType, "S T"]
e: PyTree[LeafType, "... T"]
f: PyTree[LeafType, "T ..."]
```

These correspond to:

a. A plain `PyTree` can be used an annotation, in which case `PyTree` is simply a
    suggestively-named alternative to `Any`.
    ([By definition all types are PyTrees.](https://jax.readthedocs.io/en/latest/pytrees.html))

b. `PyTree[LeafType]` denotes a PyTree all of whose leaves match `LeafType`. For
    example, `PyTree[int]` or `PyTree[Union[str, Float32[Array, "b c"]]]`.

c. A structure name can also be passed. In this case
    `jax.tree_util.tree_structure(...)` will be called, and bound to the structure name.
    This can be used to mark that multiple PyTrees all have the same structure:
    ```python
    def f(x: PyTree[int, "T"], y: PyTree[int, "T"]):
        ...
    ```

d. A composite structure can be declared. In this case the variable must have a PyTree
    structure each to the composition of multiple previously-bound PyTree structures.
    For example:
    ```python
    def f(x: PyTree[int, "T"], y: PyTree[int, "S"], z: PyTree[int, "S T"]):
        ...

    x = (1, 2)
    y = {"key": 3}
    z = {"key": (4, 5)}  # structure is the composition of the structures of `y` and `z`
    f(x, y, z)
    ```
    When performing runtime type-checking, all the individual pieces must have already
    been bound to structures, otherwise the composite structure check will throw an error.

e. A structure can begin with a `...`, to denote that the lower levels of the PyTree
    must match the declared structure, but the upper levels can be arbitrary. As in the
    previous case, all named pieces must already have been seen and their structures
    bound.

f. A structure can end with a `...`, to denote that the PyTree must be a prefix of the
    declared structure, but the lower levels can be arbitrary. As in the previous two
    cases, all named pieces must already have been seen and their structures bound.
"""  # noqa: E501

Size = Union[int, Sequence[int], np.integer, Sequence[np.integer]]
Axes = Union[int, Sequence[int]]
SeedOrKey = Union[int, jax.Array, np.ndarray]
Shape = Sequence[int]

# --- Array --- #

# ArrayLike is a Union of all objects that can be implicitly converted to a
# standard JAX array (i.e. not including future non-standard array types like
# KeyArray and BInt). It's different than np.typing.ArrayLike in that it doesn't
# accept arbitrary sequences, nor does it accept string data.
ArrayLike = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_, np.number,  # NumPy scalar types
    bool, int, float, complex,  # Python scalar types
    u.Quantity,  # Quantity
]

# --- Dtype --- #


DType = np.dtype


class SupportsDType(Protocol):
    @property
    def dtype(self) -> DType: ...


# DTypeLike is meant to annotate inputs to np.dtype that return
# a valid JAX dtype. It's different than numpy.typing.DTypeLike
# because JAX doesn't support objects or structured dtypes.
# Unlike np.typing.DTypeLike, we exclude None, and instead require
# explicit annotations when None is acceptable.
DTypeLike = Union[
    str,  # like 'float32', 'int32'
    type[Any],  # like np.float32, np.int32, float, int
    np.dtype,  # like np.dtype('float32'), np.dtype('int32')
    SupportsDType,  # like jnp.float32, jnp.int32
]


class Missing:
    pass
