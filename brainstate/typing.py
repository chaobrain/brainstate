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

"""
Comprehensive type annotations for BrainState.

This module provides a collection of type aliases, protocols, and generic types
specifically designed for scientific computing, neural network modeling, and
array operations within the BrainState ecosystem.

The type system is designed to be compatible with JAX, NumPy, and BrainUnit,
providing comprehensive type hints for arrays, shapes, seeds, and PyTree structures.

Examples
--------
Basic usage with array types:

.. code-block:: python

    >>> import brainstate
    >>> from brainstate.typing import ArrayLike, Shape, DTypeLike
    >>>
    >>> def process_array(data: ArrayLike, shape: Shape, dtype: DTypeLike) -> brainstate.Array:
    ...     return brainstate.asarray(data, dtype=dtype).reshape(shape)

Using PyTree annotations:

.. code-block:: python

    >>> from brainstate.typing import PyTree
    >>>
    >>> def tree_function(tree: PyTree[float, "T"]) -> PyTree[float, "T"]:
    ...     return brainstate.tree_map(lambda x: x * 2, tree)
"""

import builtins
import functools
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
    # Path and filter types
    'PathParts',
    'Predicate',
    'Filter',
    'FilterLiteral',

    # Array and shape types
    'Array',
    'ArrayLike',
    'Shape',
    'Size',
    'Axes',
    'DType',
    'DTypeLike',
    'SupportsDType',

    # PyTree types
    'PyTree',

    # Random number generation
    'SeedOrKey',

    # Utility types
    'Key',
    'Missing',

    # Type variables
    'K',
    '_T',
    '_Annotation',
]

# ============================================================================
# Type Variables
# ============================================================================

K = TypeVar('K', bound='Key')
"""Type variable for keys that must be comparable and hashable."""

_T = TypeVar("_T")
"""Generic type variable for any type."""

_Annotation = TypeVar("_Annotation")
"""Type variable for array annotations."""


# ============================================================================
# Key and Path Types
# ============================================================================

@runtime_checkable
class Key(Hashable, Protocol):
    """Protocol for keys that can be used in PyTree paths.

    A Key must be both hashable and comparable, making it suitable
    for use as dictionary keys and for ordering operations.

    Examples
    --------
    Valid key types include:

    .. code-block:: python

        >>> # String keys
        >>> key1: Key = "layer1"
        >>>
        >>> # Integer keys
        >>> key2: Key = 42
        >>>
        >>> # Custom hashable objects
        >>> class CustomKey:
        ...     def __init__(self, name: str):
        ...         self.name = name
        ...
        ...     def __hash__(self) -> int:
        ...         return hash(self.name)
        ...
        ...     def __eq__(self, other) -> bool:
        ...         return isinstance(other, CustomKey) and self.name == other.name
        ...
        ...     def __lt__(self, other) -> bool:
        ...         return isinstance(other, CustomKey) and self.name < other.name
    """

    def __lt__(self: K, value: K, /) -> bool:
        """Less than comparison for ordering keys.

        Parameters
        ----------
        value : Key
            The key to compare against.

        Returns
        -------
        bool
            True if this key is less than the other key.
        """
        ...


Ellipsis = builtins.ellipsis if TYPE_CHECKING else Any
"""Type alias for ellipsis, used in filter expressions."""

PathParts = Tuple[Key, ...]
"""Tuple of keys representing a path through a PyTree structure.

Examples
--------
.. code-block:: python

    >>> # Path to a nested value in a PyTree
    >>> path: PathParts = ("model", "layers", 0, "weights")
    >>>
    >>> # Empty path representing the root
    >>> root_path: PathParts = ()
"""

Predicate = Callable[[PathParts, Any], bool]
"""Function that takes a path and value, returning whether it matches some condition.

Parameters
----------
path : PathParts
    The path to the value in the PyTree.
value : Any
    The value at that path.

Returns
-------
bool
    True if the path/value combination matches the predicate.

Examples
--------
.. code-block:: python

    >>> def is_weight_matrix(path: PathParts, value: Any) -> bool:
    ...     '''Check if a value is a weight matrix (2D array).'''
    ...     return len(path) > 0 and "weight" in str(path[-1]) and hasattr(value, 'ndim') and value.ndim == 2
    >>>
    >>> def is_bias_vector(path: PathParts, value: Any) -> bool:
    ...     '''Check if a value is a bias vector (1D array).'''
    ...     return len(path) > 0 and "bias" in str(path[-1]) and hasattr(value, 'ndim') and value.ndim == 1
"""

FilterLiteral = Union[type, str, Predicate, bool, Ellipsis, None]
"""Basic filter types that can be used to select parts of a PyTree.

Components
----------
type
    Filter by type, e.g., `float`, `jax.Array`.
str
    Filter by string matching in path keys.
Predicate
    Custom function for complex filtering logic.
bool
    Simple True/False filter.
Ellipsis
    Wildcard filter that matches anything.
None
    Filter that matches None values.

Examples
--------
.. code-block:: python

    >>> # Filter by type
    >>> float_filter: FilterLiteral = float
    >>>
    >>> # Filter by string pattern
    >>> weight_filter: FilterLiteral = "weight"
    >>>
    >>> # Custom predicate filter
    >>> matrix_filter: FilterLiteral = lambda path, x: hasattr(x, 'ndim') and x.ndim == 2
"""

Filter = Union[FilterLiteral, Tuple['Filter', ...], List['Filter']]
"""Flexible filter type that can be a single filter or combination of filters.

This allows for complex filtering patterns by combining multiple filter criteria.

Examples
--------
.. code-block:: python

    >>> # Single filter
    >>> simple_filter: Filter = "weight"
    >>>
    >>> # Tuple of filters (all must match)
    >>> combined_filter: Filter = (float, "weight")
    >>>
    >>> # List of filters (any can match)
    >>> alternative_filter: Filter = [int, float, "bias"]
    >>>
    >>> # Nested combinations
    >>> complex_filter: Filter = [
    ...     ("weight", lambda p, x: x.ndim == 2),  # 2D weight matrices
    ...     ("bias", lambda p, x: x.ndim == 1),    # 1D bias vectors
    ... ]
"""


# ============================================================================
# Array Annotation Types
# ============================================================================

class _Array(Generic[_Annotation]):
    """Internal generic array type for creating custom array annotations."""
    pass


_Array.__module__ = "builtins"


def _item_to_str(item: Union[str, type, slice]) -> str:
    """Convert an array annotation item to its string representation.

    Parameters
    ----------
    item : Union[str, type, slice]
        The item to convert to string.

    Returns
    -------
    str
        String representation of the item.

    Raises
    ------
    NotImplementedError
        If slice has a step component.
    """
    if isinstance(item, slice):
        if item.step is not None:
            raise NotImplementedError("Slice steps are not supported in array annotations")
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
    """Convert array annotation items (potentially in tuple) to string representation.

    Parameters
    ----------
    item : Union[str, type, slice, Tuple[...]]
        Single item or tuple of items to convert.

    Returns
    -------
    str
        String representation of the item(s).
    """
    if isinstance(item, tuple):
        if len(item) == 0:
            # Explicit brackets for empty tuple
            return "()"
        else:
            # No brackets for non-empty tuple
            return ", ".join([_item_to_str(i) for i in item])
    else:
        return _item_to_str(item)


class Array:
    """Flexible array type annotation supporting shape and dtype specifications.

    This class provides a convenient way to annotate arrays with shape information,
    making code more self-documenting and enabling better static analysis.

    Examples
    --------
    Basic array annotations:

    .. code-block:: python

        >>> from brainstate.typing import Array
        >>>
        >>> # Any array
        >>> def process_array(x: Array) -> Array:
        ...     return x * 2
        >>>
        >>> # Array with specific shape annotation
        >>> def matrix_multiply(a: Array["m, n"], b: Array["n, k"]) -> Array["m, k"]:
        ...     return a @ b
        >>>
        >>> # Array with dtype and shape
        >>> def normalize_weights(weights: Array["batch, features"]) -> Array["batch, features"]:
        ...     return weights / weights.sum(axis=-1, keepdims=True)

    Advanced shape annotations:

    .. code-block:: python

        >>> # Using ellipsis for flexible dimensions
        >>> def flatten_batch(x: Array["batch, ..."]) -> Array["batch, -1"]:
        ...     return x.reshape(x.shape[0], -1)
        >>>
        >>> # Multiple shape constraints
        >>> def attention(
        ...     query: Array["batch, seq_len, d_model"],
        ...     key: Array["batch, seq_len, d_model"],
        ...     value: Array["batch, seq_len, d_model"]
        ... ) -> Array["batch, seq_len, d_model"]:
        ...     # Attention computation
        ...     pass
    """

    def __class_getitem__(cls, item):
        """Create a specialized Array type with shape/dtype annotations.

        Parameters
        ----------
        item : str, type, slice, or tuple
            Shape specification, dtype, or combination thereof.

        Returns
        -------
        _Array
            Specialized array type with the given annotation.
        """

        class X:
            pass

        X.__module__ = "builtins"
        X.__qualname__ = _maybe_tuple_to_str(item)
        return _Array[X]


# Set module for proper display in type hints
Array.__module__ = "builtins"


# ============================================================================
# PyTree Types
# ============================================================================

class _FakePyTree(Generic[_T]):
    """Internal generic PyTree type for creating specialized PyTree annotations."""
    pass


_FakePyTree.__name__ = "PyTree"
_FakePyTree.__qualname__ = "PyTree"
_FakePyTree.__module__ = "builtins"


class _MetaPyTree(type):
    """Metaclass for PyTree type that prevents instantiation and handles subscripting."""

    def __call__(self, *args, **kwargs):
        """Prevent direct instantiation of PyTree type.

        Raises
        ------
        RuntimeError
            Always raised since PyTree is a type annotation only.
        """
        raise RuntimeError("PyTree cannot be instantiated")

    # Can't return a generic (e.g. _FakePyTree[item]) because generic aliases don't do
    # the custom __instancecheck__ that we want.
    # We can't add that __instancecheck__  via subclassing, e.g.
    # type("PyTree", (Generic[_T],), {}), because dynamic subclassing of typeforms
    # isn't allowed.
    # Likewise we can't do types.new_class("PyTree", (Generic[_T],), {}) because that
    # has __module__ "types", e.g. we get types.PyTree[int].
    @functools.lru_cache(maxsize=None)
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

.. code-block:: python

    >>> a: PyTree
    >>> b: PyTree[LeafType]
    >>> c: PyTree[LeafType, "T"]
    >>> d: PyTree[LeafType, "S T"]
    >>> e: PyTree[LeafType, "... T"]
    >>> f: PyTree[LeafType, "T ..."]

These correspond to:

a. A plain `PyTree` can be used an annotation, in which case `PyTree` is simply a
    suggestively-named alternative to `Any`.
    ([By definition all types are PyTrees.](https://jax.readthedocs.io/en/latest/pytrees.html))

b. `PyTree[LeafType]` denotes a PyTree all of whose leaves match `LeafType`. For
    example, `PyTree[int]` or `PyTree[Union[str, Float32[Array, "b c"]]]`.

c. A structure name can also be passed. In this case
    `jax.tree_util.tree_structure(...)` will be called, and bound to the structure name.
    This can be used to mark that multiple PyTrees all have the same structure:
    
    .. code-block:: python
        
        >>> def f(x: PyTree[int, "T"], y: PyTree[int, "T"]):
        ...     ...

d. A composite structure can be declared. In this case the variable must have a PyTree
    structure each to the composition of multiple previously-bound PyTree structures.
    For example:
    
    .. code-block:: python
        
        >>> def f(x: PyTree[int, "T"], y: PyTree[int, "S"], z: PyTree[int, "S T"]):
        ...     ...
        >>>
        >>> x = (1, 2)
        >>> y = {"key": 3}
        >>> z = {"key": (4, 5)}  # structure is the composition of the structures of `y` and `z`
        >>> f(x, y, z)
    
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

# ============================================================================
# Shape and Size Types
# ============================================================================

Size = Union[int, Sequence[int], np.integer, Sequence[np.integer]]
"""Type for specifying array sizes and dimensions.

Can be a single integer for 1D sizes, or a sequence of integers for multi-dimensional shapes.
Supports both Python integers and NumPy integer types for compatibility.

Examples
--------
.. code-block:: python

    >>> # Single dimension
    >>> size1: Size = 10
    >>>
    >>> # Multiple dimensions
    >>> size2: Size = (3, 4, 5)
    >>>
    >>> # Using NumPy integers
    >>> size3: Size = np.int32(8)
    >>>
    >>> # Mixed sequence
    >>> size4: Size = [np.int64(2), 3, np.int32(4)]
"""

Shape = Sequence[int]
"""Type for array shapes as sequences of integers.

Represents the shape of an array as a sequence of dimension sizes.
More restrictive than Size as it requires a sequence.

Examples
--------
.. code-block:: python

    >>> # 2D array shape
    >>> matrix_shape: Shape = (10, 20)
    >>>
    >>> # 3D array shape
    >>> tensor_shape: Shape = (5, 10, 15)
    >>>
    >>> # 1D array shape (note: still needs to be a sequence)
    >>> vector_shape: Shape = (100,)
"""

Axes = Union[int, Sequence[int]]
"""Type for specifying axes along which operations should be performed.

Can be a single axis (integer) or multiple axes (sequence of integers).
Used in reduction operations, reshaping, and other array manipulations.

Examples
--------
.. code-block:: python

    >>> # Single axis
    >>> axis1: Axes = 0
    >>>
    >>> # Multiple axes
    >>> axis2: Axes = (0, 2)
    >>>
    >>> # All axes for global operations
    >>> axis3: Axes = tuple(range(ndim))
    >>>
    >>> def sum_along_axes(array: ArrayLike, axes: Axes) -> ArrayLike:
    ...     return jnp.sum(array, axis=axes)
"""

# ============================================================================
# Array Types
# ============================================================================

ArrayLike = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_, np.number,  # NumPy scalar types
    bool, int, float, complex,  # Python scalar types
    u.Quantity,  # BrainUnit quantity type
]
"""Union of all objects that can be implicitly converted to a JAX array.

This type is designed for JAX compatibility and excludes arbitrary sequences
and string data that numpy.typing.ArrayLike would include. It represents
data that can be safely converted to arrays without ambiguity.

Components
----------
jax.Array
    Native JAX arrays.
np.ndarray
    NumPy arrays that can be converted to JAX arrays.
np.bool_, np.number
    NumPy scalar types (bool, int8, float32, etc.).
bool, int, float, complex
    Python built-in scalar types.
u.Quantity
    BrainUnit quantities with physical units.

Examples
--------
.. code-block:: python

    >>> def process_data(data: ArrayLike) -> jax.Array:
    ...     '''Convert input to JAX array and process it.'''
    ...     array = jnp.asarray(data)
    ...     return array * 2
    >>>
    >>> # Valid inputs
    >>> process_data(jnp.array([1, 2, 3]))      # JAX array
    >>> process_data(np.array([1, 2, 3]))       # NumPy array
    >>> process_data([1, 2, 3])                 # Python list (via numpy)
    >>> process_data(42)                        # Python scalar
    >>> process_data(np.float32(3.14))          # NumPy scalar
    >>> process_data(1.5 * u.second)            # Quantity with units
"""

# ============================================================================
# Data Type Annotations
# ============================================================================

DType = np.dtype
"""Alias for NumPy's dtype type.

Used to represent data types of arrays in a clear and consistent manner.

Examples
--------
.. code-block:: python

    >>> def create_array(shape: Shape, dtype: DType) -> jax.Array:
    ...     return jnp.zeros(shape, dtype=dtype)
    >>>
    >>> # Usage
    >>> arr = create_array((3, 4), np.float32)
"""


class SupportsDType(Protocol):
    """Protocol for objects that have a dtype property.

    This protocol defines the interface for any object that exposes
    a dtype attribute, allowing for flexible type checking.

    Examples
    --------
    .. code-block:: python

        >>> def get_dtype(obj: SupportsDType) -> DType:
        ...     return obj.dtype
        >>>
        >>> # Works with arrays
        >>> arr = jnp.array([1.0, 2.0])
        >>> dtype = get_dtype(arr)  # float32
    """

    @property
    def dtype(self) -> DType:
        """Return the data type of the object.

        Returns
        -------
        DType
            The NumPy dtype of the object.
        """
        ...


DTypeLike = Union[
    str,  # String representations like 'float32', 'int32'
    type[Any],  # Type objects like np.float32, np.int32, float, int
    np.dtype,  # NumPy dtype objects
    SupportsDType,  # Objects with a dtype property
]
"""Union of types that can be converted to a valid JAX dtype.

This is more restrictive than numpy.typing.DTypeLike as JAX doesn't support
object arrays or structured dtypes. It excludes None to require explicit
handling of optional dtypes.

Components
----------
str
    String representations like 'float32', 'int32', 'bool'.
type[Any]
    Type objects like np.float32, float, int, bool.
np.dtype
    NumPy dtype objects created with np.dtype().
SupportsDType
    Any object with a .dtype property.

Examples
--------
.. code-block:: python

    >>> def cast_array(array: ArrayLike, dtype: DTypeLike) -> jax.Array:
    ...     '''Cast array to specified dtype.'''
    ...     return jnp.asarray(array, dtype=dtype)
    >>>
    >>> # Valid dtype specifications
    >>> cast_array(data, 'float32')           # String
    >>> cast_array(data, np.float32)          # NumPy type
    >>> cast_array(data, float)               # Python type
    >>> cast_array(data, np.dtype('int32'))   # NumPy dtype object
    >>> cast_array(data, other_array)         # Object with dtype property
"""

# ============================================================================
# Random Number Generation
# ============================================================================

SeedOrKey = Union[int, jax.Array, np.ndarray]
"""Type for random number generator seeds or keys.

Represents values that can be used to seed random number generators
or serve as PRNG keys in JAX's random number generation system.

Components
----------
int
    Integer seeds for random number generators.
jax.Array
    JAX PRNG keys (typically created with jax.random.PRNGKey).
np.ndarray
    NumPy arrays that can serve as random keys.

Examples
--------
.. code-block:: python

    >>> def generate_random(key: SeedOrKey, shape: Shape) -> jax.Array:
    ...     '''Generate random numbers using the provided seed or key.'''
    ...     if isinstance(key, int):
    ...         key = jax.random.PRNGKey(key)
    ...     return jax.random.normal(key, shape)
    >>>
    >>> # Valid seeds/keys
    >>> generate_random(42, (3, 4))                    # Integer seed
    >>> generate_random(jax.random.PRNGKey(123), (5,)) # JAX PRNG key
    >>> generate_random(np.array([1, 2], dtype=np.uint32), (2, 2))  # NumPy array
"""


# ============================================================================
# Utility Types
# ============================================================================

class Missing:
    """Sentinel class to represent missing or unspecified values.

    This class is used as a default value when None has semantic meaning
    and you need to distinguish between "None was passed" and "nothing was passed".

    Examples
    --------
    .. code-block:: python

        >>> _MISSING = Missing()
        >>>
        >>> def function_with_optional_param(value: Union[int, None, Missing] = _MISSING):
        ...     if value is _MISSING:
        ...         print("No value provided")
        ...     elif value is None:
        ...         print("None was explicitly provided")
        ...     else:
        ...         print(f"Value: {value}")
        >>>
        >>> function_with_optional_param()        # "No value provided"
        >>> function_with_optional_param(None)    # "None was explicitly provided"
        >>> function_with_optional_param(42)      # "Value: 42"
    """
    pass
