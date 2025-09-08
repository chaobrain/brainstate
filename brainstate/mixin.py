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

# -*- coding: utf-8 -*-

import operator
from typing import (
    Union, Sequence, Optional, TypeVar, _SpecialForm,
    _type_check, _remove_dups_flatten, _UnionGenericAlias
)

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar('T')

__all__ = [
    'Mixin',
    'ParamDesc',
    'ParamDescriber',
    'AlignPost',
    'BindCondData',

    # types
    'JointTypes',
    'OneOfTypes',

    # behavior modes
    'Mode',
    'JointMode',
    'Batching',
    'Training',
    'ArrayImpl',
]


def hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False


class Mixin(object):
    """Base Mixin object.

    The key for a :py:class:`~.Mixin` is that: no initialization function, only behavioral functions.
    """
    pass


class ParamDesc(Mixin):
    """
    :py:class:`~.Mixin` indicates the function for describing initialization parameters.

    This mixin enables the subclass has a classmethod ``desc``, which
    produces an instance of :py:class:`~.ParamDescriber`.

    Note this Mixin can be applied in any Python object.
    """

    non_hashable_params: Optional[Sequence[str]] = None

    @classmethod
    def desc(cls, *args, **kwargs) -> 'ParamDescriber':
        return ParamDescriber(cls, *args, **kwargs)


class HashableDict(dict):
    def __init__(self, the_dict: dict):
        out = dict()
        for k, v in the_dict.items():
            if not hashable(v):
                v = str(v)  # convert to string if not hashable
            out[k] = v
        super().__init__(out)

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class NoSubclassMeta(type):
    def __new__(cls, name, bases, classdict):
        for b in bases:
            if isinstance(b, NoSubclassMeta):
                raise TypeError("type '{0}' is not an acceptable base type".format(b.__name__))
        return type.__new__(cls, name, bases, dict(classdict))


class ParamDescriber(metaclass=NoSubclassMeta):
    """
    ParamDesc initialization for parameter describers.
    """

    def __init__(self, cls: T, *desc_tuple, **desc_dict):
        self.cls: type = cls

        # arguments
        self.args = desc_tuple
        self.kwargs = desc_dict

        # identifier
        self._identifier = (cls, tuple(desc_tuple), HashableDict(desc_dict))

    def __call__(self, *args, **kwargs) -> T:
        return self.cls(*self.args, *args, **self.kwargs, **kwargs)

    def init(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __instancecheck__(self, instance):
        if not isinstance(instance, ParamDescriber):
            return False
        if not issubclass(instance.cls, self.cls):
            return False
        return True

    @classmethod
    def __class_getitem__(cls, item: type):
        return ParamDescriber(item)

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        raise AttributeError('Cannot set the identifier.')


class AlignPost(Mixin):
    """
    Align post MixIn.

    This class provides a ``align_post_input_add()`` function for
    add external currents.
    """

    def align_post_input_add(self, *args, **kwargs):
        raise NotImplementedError


class BindCondData(Mixin):
    """Bind temporary conductance data.


    """
    _conductance: Optional

    def bind_cond(self, conductance):
        self._conductance = conductance

    def unbind_cond(self):
        self._conductance = None


def not_implemented(func):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f'{func.__name__} is not implemented.')

    wrapper.not_implemented = True
    return wrapper


class _MetaUnionType(type):
    def __new__(cls, name, bases, dct):
        if isinstance(bases, type):
            bases = (bases,)
        elif isinstance(bases, (list, tuple)):
            bases = tuple(bases)
            for base in bases:
                assert isinstance(base, type), f'Must be type. But got {base}'
        else:
            raise TypeError(f'Must be type. But got {bases}')
        return super().__new__(cls, name, bases, dct)

    def __instancecheck__(self, other):
        cls_of_other = other.__class__
        return all([issubclass(cls_of_other, cls) for cls in self.__bases__])

    def __subclasscheck__(self, subclass):
        return all([issubclass(subclass, cls) for cls in self.__bases__])


class _JointGenericAlias(_UnionGenericAlias, _root=True):
    def __subclasscheck__(self, subclass):
        return all([issubclass(subclass, cls) for cls in set(self.__args__)])


@_SpecialForm
def JointTypes(self, parameters):
    """Joint types; JointTypes[X, Y] means both X and Y.

    To define a union, use e.g. Union[int, str].

    Details:
    - The arguments must be types and there must be at least one.
    - None as an argument is a special case and is replaced by `type(None)`.
    - Unions of unions are flattened, e.g.::

        JointTypes[JointTypes[int, str], float] == JointTypes[int, str, float]

    - Unions of a single argument vanish, e.g.::

        JointTypes[int] == int  # The constructor actually returns int

    - Redundant arguments are skipped, e.g.::

        JointTypes[int, str, int] == JointTypes[int, str]

    - When comparing unions, the argument order is ignored, e.g.::

        JointTypes[int, str] == JointTypes[str, int]

    - You cannot subclass or instantiate a JointTypes.
    - You can use Optional[X] as a shorthand for JointTypes[X, None].
    """
    if parameters == ():
        raise TypeError("Cannot take a Joint of no types.")
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    msg = "JointTypes[arg, ...]: each arg must be a type."
    parameters = tuple(_type_check(p, msg) for p in parameters)
    parameters = _remove_dups_flatten(parameters)
    if len(parameters) == 1:
        return parameters[0]
    if len(parameters) == 2 and type(None) in parameters:
        return _UnionGenericAlias(self, parameters, name="Optional")
    return _JointGenericAlias(self, parameters)


@_SpecialForm
def OneOfTypes(self, parameters):
    """Sole type; OneOfTypes[X, Y] means either X or Y.

    To define a union, use e.g. OneOfTypes[int, str]. Details:
    - The arguments must be types and there must be at least one.
    - None as an argument is a special case and is replaced by
      type(None).
    - Unions of unions are flattened, e.g.::

        assert OneOfTypes[OneOfTypes[int, str], float] == OneOfTypes[int, str, float]

    - Unions of a single argument vanish, e.g.::

        assert OneOfTypes[int] == int  # The constructor actually returns int

    - Redundant arguments are skipped, e.g.::

        assert OneOfTypes[int, str, int] == OneOfTypes[int, str]

    - When comparing unions, the argument order is ignored, e.g.::

        assert OneOfTypes[int, str] == OneOfTypes[str, int]

    - You cannot subclass or instantiate a union.
    - You can use Optional[X] as a shorthand for OneOfTypes[X, None].
    """
    if parameters == ():
        raise TypeError("Cannot take a Sole of no types.")
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    msg = "OneOfTypes[arg, ...]: each arg must be a type."
    parameters = tuple(_type_check(p, msg) for p in parameters)
    parameters = _remove_dups_flatten(parameters)
    if len(parameters) == 1:
        return parameters[0]
    if len(parameters) == 2 and type(None) in parameters:
        return _UnionGenericAlias(self, parameters, name="Optional")
    return _UnionGenericAlias(self, parameters)


class Mode(Mixin):
    """
    Base class for computation behaviors.
    """

    def __repr__(self):
        return self.__class__.__name__

    def __eq__(self, other: 'Mode'):
        assert isinstance(other, Mode)
        return other.__class__ == self.__class__

    def is_a(self, mode: type):
        """
        Check whether the mode is exactly the desired mode.
        """
        assert isinstance(mode, type), 'Must be a type.'
        return self.__class__ == mode

    def has(self, mode: type):
        """
        Check whether the mode is included in the desired mode.
        """
        assert isinstance(mode, type), 'Must be a type.'
        return isinstance(self, mode)


class JointMode(Mode):
    """
    Joint mode.
    """

    def __init__(self, *modes: Mode):
        for m_ in modes:
            if not isinstance(m_, Mode):
                raise TypeError(f'The supported type must be a tuple/list of Mode. But we got {m_}')
        self.modes = tuple(modes)
        self.types = set([m.__class__ for m in modes])

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([repr(m) for m in self.modes])})'

    def has(self, mode: type):
        """
        Check whether the mode is included in the desired mode.
        """
        assert isinstance(mode, type), 'Must be a type.'
        return any([issubclass(cls, mode) for cls in self.types])

    def is_a(self, cls: type):
        """
        Check whether the mode is exactly the desired mode.
        """
        return JointTypes[tuple(self.types)] == cls

    def __getattr__(self, item):
        """
        Get the attribute from the mode.

        If the attribute is not found in the mode, then it will be searched in the base class.
        """
        if item in ['modes', 'types']:
            return super().__getattribute__(item)
        for m in self.modes:
            if hasattr(m, item):
                return getattr(m, item)
        return super().__getattribute__(item)


class Batching(Mode):
    """Batching mode."""

    def __init__(self, batch_size: int = 1, batch_axis: int = 0):
        self.batch_size = batch_size
        self.batch_axis = batch_axis

    def __repr__(self):
        return f'{self.__class__.__name__}(in_size={self.batch_size}, axis={self.batch_axis})'


class Training(Mode):
    """Training mode."""
    pass


class ArrayImpl(Mixin):
    value: jax.typing.ArrayLike

    def __hash__(self):
        return hash(self.value)

    @property
    def dtype(self):
        """Variable dtype."""
        return u.math.get_dtype(self.value)

    @property
    def shape(self):
        """Variable shape."""
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def imag(self):
        return self.value.image

    @property
    def real(self):
        return self.value.real

    @property
    def size(self):
        return self.value.size

    @property
    def T(self):
        return self.value.T

    def __format__(self, format_spec: str) -> str:
        return format(self.value)

    def __iter__(self):
        """Solve the issue of DeviceArray.__iter__.

        Details please see JAX issues:

        - https://github.com/google/jax/issues/7713
        - https://github.com/google/jax/pull/3821
        """
        for i in range(self.value.shape[0]):
            yield self.value[i]

    def __getitem__(self, index):
        if isinstance(index, slice) and (index == slice(None)):
            return self.value
        return self.value[index]

    def __setitem__(self, index, value):
        if isinstance(value, np.ndarray):
            value = u.math.asarray(value)

        # update
        self_value = self.value
        self.value = self_value.at[index].set(value)

    # ---------- #
    # operations #
    # ---------- #

    def __len__(self) -> int:
        return len(self.value)

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __eq__(self, oc):
        return self.value == oc

    def __ne__(self, oc):
        return self.value != oc

    def __lt__(self, oc):
        return self.value < oc

    def __le__(self, oc):
        return self.value <= oc

    def __gt__(self, oc):
        return self.value > oc

    def __ge__(self, oc):
        return self.value >= oc

    def __add__(self, oc):
        return self.value + oc

    def __radd__(self, oc):
        return self.value + oc

    def __iadd__(self, oc):
        # a += b
        self.value = self.value + oc
        return self

    def __sub__(self, oc):
        return self.value - oc

    def __rsub__(self, oc):
        return oc - self.value

    def __isub__(self, oc):
        # a -= b
        self.value = self.value - oc
        return self

    def __mul__(self, oc):
        return self.value * oc

    def __rmul__(self, oc):
        return oc * self.value

    def __imul__(self, oc):
        # a *= b
        self.value = self.value * oc
        return self

    def __rdiv__(self, oc):
        return oc / self.value

    def __truediv__(self, oc):
        return self.value / oc

    def __rtruediv__(self, oc):
        return oc / self.value

    def __itruediv__(self, oc):
        # a /= b
        self.value = self.value / oc
        return self

    def __floordiv__(self, oc):
        return self.value // oc

    def __rfloordiv__(self, oc):
        return oc // self.value

    def __ifloordiv__(self, oc):
        # a //= b
        self.value = self.value // oc
        return self

    def __divmod__(self, oc):
        return self.value.__divmod__(oc)

    def __rdivmod__(self, oc):
        return self.value.__rdivmod__(oc)

    def __mod__(self, oc):
        return self.value % oc

    def __rmod__(self, oc):
        return oc % self.value

    def __imod__(self, oc):
        # a %= b
        self.value = self.value % oc
        return self

    def __pow__(self, oc):
        return self.value ** oc

    def __rpow__(self, oc):
        return oc ** self.value

    def __ipow__(self, oc):
        # a **= b
        self.value = self.value ** oc
        return self

    def __matmul__(self, oc):
        return self.value @ oc

    def __rmatmul__(self, oc):
        return oc @ self.value

    def __imatmul__(self, oc):
        # a @= b
        self.value = self.value @ oc
        return self

    def __and__(self, oc):
        return self.value & oc

    def __rand__(self, oc):
        return oc & self.value

    def __iand__(self, oc):
        # a &= b
        self.value = self.value & oc
        return self

    def __or__(self, oc):
        return self.value | oc

    def __ror__(self, oc):
        return oc | self.value

    def __ior__(self, oc):
        # a |= b
        self.value = self.value | oc
        return self

    def __xor__(self, oc):
        return self.value ^ oc

    def __rxor__(self, oc):
        return oc ^ self.value

    def __ixor__(self, oc):
        # a ^= b
        self.value = self.value ^ oc
        return self

    def __lshift__(self, oc):
        return self.value << oc

    def __rlshift__(self, oc):
        return oc << self.value

    def __ilshift__(self, oc):
        # a <<= b
        self.value = self.value << oc
        return self

    def __rshift__(self, oc):
        return self.value >> oc

    def __rrshift__(self, oc):
        return oc >> self.value

    def __irshift__(self, oc):
        # a >>= b
        self.value = self.value >> oc
        return self

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)

    # ----------------------- #
    #      NumPy methods      #
    # ----------------------- #

    def all(self, axis=None, keepdims=False):
        """Returns True if all elements evaluate to True."""
        r = self.value.all(axis=axis, keepdims=keepdims)
        return r

    def any(self, axis=None, keepdims=False):
        """Returns True if any of the elements of a evaluate to True."""
        r = self.value.any(axis=axis, keepdims=keepdims)
        return r

    def argmax(self, axis=None):
        """Return indices of the maximum values along the given axis."""
        return self.value.argmax(axis=axis)

    def argmin(self, axis=None):
        """Return indices of the minimum values along the given axis."""
        return self.value.argmin(axis=axis)

    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        """Returns the indices that would partition this array."""
        return self.value.argpartition(kth=kth, axis=axis, kind=kind, order=order)

    def argsort(self, axis=-1, kind=None, order=None):
        """Returns the indices that would sort this array."""
        return self.value.argsort(axis=axis, kind=kind, order=order)

    def astype(self, dtype):
        """Copy of the array, cast to a specified type.

        Parameters::

        dtype: str, dtype
          Typecode or data-type to which the array is cast.
        """
        if dtype is None:
            return self.value
        else:
            return self.value.astype(dtype)

    def byteswap(self, inplace=False):
        """Swap the bytes of the array elements

        Toggle between low-endian and big-endian data representation by
        returning a byteswapped array, optionally swapped in-place.
        Arrays of byte-strings are not swapped. The real and imaginary
        parts of a complex number are swapped individually."""
        return self.value.byteswap(inplace=inplace)

    def choose(self, choices, mode='raise'):
        """Use an index array to construct a new array from a set of choices."""
        return self.value.choose(choices=choices, mode=mode)

    def clip(self, min=None, max=None):
        """Return an array whose values are limited to [min, max]. One of max or min must be given."""
        r = self.value.clip(min=min, max=max)
        return r

    def compress(self, condition, axis=None):
        """Return selected slices of this array along given axis."""
        return self.value.compress(condition=condition, axis=axis)

    def conj(self):
        """Complex-conjugate all elements."""
        return self.value.conj()

    def conjugate(self):
        """Return the complex conjugate, element-wise."""
        return self.value.conjugate()

    def copy(self):
        """Return a copy of the array."""
        return self.value.copy()

    def cumprod(self, axis=None, dtype=None):
        """Return the cumulative product of the elements along the given axis."""
        return self.value.cumprod(axis=axis, dtype=dtype)

    def cumsum(self, axis=None, dtype=None):
        """Return the cumulative sum of the elements along the given axis."""
        return self.value.cumsum(axis=axis, dtype=dtype)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Return specified diagonals."""
        return self.value.diagonal(offset=offset, axis1=axis1, axis2=axis2)

    def dot(self, b):
        """Dot product of two arrays."""
        return self.value.dot(b)

    def fill(self, value):
        """Fill the array with a scalar value."""
        self.value = u.math.ones_like(self.value) * value

    def flatten(self):
        return self.value.flatten()

    def item(self, *args):
        """Copy an element of an array to a standard Python scalar and return it."""
        return self.value.item(*args)

    def max(self, axis=None, keepdims=False, *args, **kwargs):
        """Return the maximum along a given axis."""
        res = self.value.max(axis=axis, keepdims=keepdims, *args, **kwargs)
        return res

    def mean(self, axis=None, dtype=None, keepdims=False, *args, **kwargs):
        """Returns the average of the array elements along given axis."""
        res = self.value.mean(axis=axis, dtype=dtype, keepdims=keepdims, *args, **kwargs)
        return res

    def min(self, axis=None, keepdims=False, *args, **kwargs):
        """Return the minimum along a given axis."""
        res = self.value.min(axis=axis, keepdims=keepdims, *args, **kwargs)
        return res

    def nonzero(self):
        """Return the indices of the elements that are non-zero."""
        return tuple(a for a in self.value.nonzero())

    def prod(self, axis=None, dtype=None, keepdims=False, initial=1, where=True):
        """Return the product of the array elements over the given axis."""
        res = self.value.prod(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
        return res

    def ptp(self, axis=None, keepdims=False):
        """Peak to peak (maximum - minimum) value along a given axis."""
        r = self.value.ptp(axis=axis, keepdims=keepdims)
        return r

    def put(self, indices, values):
        """Replaces specified elements of an array with given values.

        Parameters::

        indices: array_like
          Target indices, interpreted as integers.
        values: array_like
          Values to place in the array at target indices.
        """
        self.__setitem__(indices, values)

    def ravel(self, order=None):
        """Return a flattened array."""
        return self.value.ravel(order=order)

    def repeat(self, repeats, axis=None):
        """Repeat elements of an array."""
        return self.value.repeat(repeats=repeats, axis=axis)

    def reshape(self, *shape, order='C'):
        """Returns an array containing the same data with a new shape."""
        return self.value.reshape(*shape, order=order)

    def resize(self, new_shape):
        """Change shape and size of array in-place."""
        self.value = self.value.reshape(new_shape)

    def round(self, decimals=0):
        """Return ``a`` with each element rounded to the given number of decimals."""
        return self.value.round(decimals=decimals)

    def searchsorted(self, v, side='left', sorter=None):
        return self.value.searchsorted(v=v, side=side, sorter=sorter)

    def sort(self, axis=-1, stable=True, order=None):
        """Sort an array in-place.

        Parameters::

        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the
            last axis.
        stable : bool, optional
            Whether to use a stable sorting algorithm. The default is True.
        order : str or list of str, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  A single field can
            be specified as a string, and not all fields need be specified,
            but unspecified fields will still be used, in the order in which
            they come up in the dtype, to break ties.
        """
        self.value = self.value.sort(axis=axis, stable=stable, order=order)

    def squeeze(self, axis=None):
        """Remove axes of length one from ``a``."""
        return self.value.squeeze(axis=axis)

    def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
        r = self.value.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return r

    def sum(self, axis=None, dtype=None, keepdims=False, initial=0, where=True):
        """Return the sum of the array elements over the given axis."""
        res = self.value.sum(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
        return res

    def swapaxes(self, axis1, axis2):
        """Return a view of the array with `axis1` and `axis2` interchanged."""
        return self.value.swapaxes(axis1, axis2)

    def split(self, indices_or_sections, axis=0):
        return [a for a in u.math.split(self.value, indices_or_sections, axis=axis)]

    def take(self, indices, axis=None, mode=None):
        """Return an array formed from the elements of a at the given indices."""
        return self.value.take(indices=indices, axis=axis, mode=mode)

    def tobytes(self):
        return self.value.tobytes()

    def tolist(self):
        return self.value.tolist()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        """Return the sum along diagonals of the array."""
        return self.value.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    def transpose(self, *axes):
        return self.value.transpose(*axes)

    def tile(self, reps):
        return self.value.tile(reps)

    def var(self, axis=None, dtype=None, ddof=0, keepdims=False):
        """Returns the variance of the array elements, along given axis."""
        r = self.value.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return r

    def view(self, *args, dtype=None):
        if len(args) == 0:
            if dtype is None:
                raise ValueError('Provide dtype or shape.')
            else:
                return self.value.view(dtype)
        else:
            if isinstance(args[0], int):  # shape
                if dtype is not None:
                    raise ValueError('Provide one of dtype or shape. Not both.')
                return self.value.reshape(*args)
            else:  # dtype
                assert not isinstance(args[0], int)
                assert dtype is None
                return self.value.view(args[0])

    # ------------------
    # NumPy support
    # ------------------

    def numpy(self, dtype=None):
        """Convert to numpy.ndarray."""
        # warnings.warn('Deprecated since 2.1.12. Please use ".to_numpy()" instead.', DeprecationWarning)
        return np.asarray(self.value, dtype=dtype)

    def to_numpy(self, dtype=None):
        """Convert to numpy.ndarray."""
        return np.asarray(self.value, dtype=dtype)

    def to_jax(self, dtype=None):
        """Convert to jax.numpy.ndarray."""
        if dtype is None:
            return self.value
        else:
            return u.math.asarray(self.value, dtype=dtype)

    def __array__(self, dtype=None):
        """Support ``numpy.array()`` and ``numpy.asarray()`` functions."""
        return np.asarray(self.value, dtype=dtype)

    def __jax_array__(self):
        return self.value

    def __bool__(self) -> bool:
        return self.value.__bool__()

    def __float__(self):
        return self.value.__float__()

    def __int__(self):
        return self.value.__int__()

    def __complex__(self):
        return self.value.__complex__()

    def __hex__(self):
        assert self.ndim == 0, 'hex only works on scalar values'
        return hex(self.value)  # type: ignore

    def __oct__(self):
        assert self.ndim == 0, 'oct only works on scalar values'
        return oct(self.value)  # type: ignore

    def __index__(self):
        return operator.index(self.value)

    # ----------------------
    # PyTorch compatibility
    # ----------------------

    def unsqueeze(self, dim: int) -> 'Array':
        """
        Array.unsqueeze(dim) -> Array, or so called Tensor
        equals
        Array.expand_dims(dim)

        See :func:`brainpy.math.unsqueeze`
        """
        return u.math.expand_dims(self.value, dim)

    def expand_dims(self, axis: Union[int, Sequence[int]]) -> 'Array':
        return u.math.expand_dims(self.value, axis)

    def expand_as(self, array: Union['Array', jax.Array, np.ndarray]) -> 'Array':
        return u.math.broadcast_to(self.value, array)

    def pow(self, index: int):
        return self.value ** index

    def addr(
        self,
        vec1: Union['Array', jax.Array, np.ndarray],
        vec2: Union['Array', jax.Array, np.ndarray],
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
    ) -> Optional['Array']:
        r = alpha * u.math.outer(vec1, vec2) + beta * self.value
        return r

    def addr_(
        self,
        vec1: Union['Array', jax.Array, np.ndarray],
        vec2: Union['Array', jax.Array, np.ndarray],
        *,
        beta: float = 1.0,
        alpha: float = 1.0
    ):
        r = alpha * u.math.outer(vec1, vec2) + beta * self.value
        self.value = r
        return self

    def outer(self, other: Union['Array', jax.Array, np.ndarray]) -> 'Array':
        return u.math.outer(self.value, other.value)

    def abs(self) -> Optional['Array']:
        r = u.math.abs(self.value)
        return r

    def abs_(self):
        """
        in-place version of Array.abs()
        """
        self.value = u.math.abs(self.value)
        return self

    def add_(self, value):
        self.value += value
        return self

    def absolute(self) -> Optional['Array']:
        """
        alias of Array.abs
        """
        return self.abs()

    def absolute_(self):
        """
        alias of Array.abs_()
        """
        return self.abs_()

    def mul(self, value):
        return self.value * value

    def mul_(self, value):
        """
        In-place version of :meth:`~Array.mul`.
        """
        self.value *= value
        return self

    def multiply(self, value):  # real signature unknown; restored from __doc__
        """
        multiply(value) -> Tensor

        See :func:`torch.multiply`.
        """
        return self.value * value

    def multiply_(self, value):  # real signature unknown; restored from __doc__
        """
        multiply_(value) -> Tensor

        In-place version of :meth:`~Tensor.multiply`.
        """
        self.value *= value
        return self

    def sin(self) -> Optional['Array']:
        r = u.math.sin(self.value)
        return r

    def sin_(self):
        self.value = u.math.sin(self.value)
        return self

    def cos_(self):
        self.value = u.math.cos(self.value)
        return self

    def cos(self) -> Optional['Array']:
        r = u.math.cos(self.value)
        return r

    def tan_(self):
        self.value = u.math.tan(self.value)
        return self

    def tan(self) -> Optional['Array']:
        r = u.math.tan(self.value)
        return r

    def sinh_(self):
        self.value = u.math.sinh(self.value)
        return self

    def sinh(self) -> Optional['Array']:
        r = u.math.sinh(self.value)
        return r

    def cosh_(self):
        self.value = u.math.cosh(self.value)
        return self

    def cosh(self) -> Optional['Array']:
        r = u.math.cosh(self.value)
        return r

    def tanh_(self):
        self.value = u.math.tanh(self.value)
        return self

    def tanh(self) -> Optional['Array']:
        r = u.math.tanh(self.value)
        return r

    def arcsin_(self):
        self.value = u.math.arcsin(self.value)
        return self

    def arcsin(self) -> Optional['Array']:
        r = u.math.arcsin(self.value)
        return r

    def arccos_(self):
        self.value = u.math.arccos(self.value)
        return self

    def arccos(self) -> Optional['Array']:
        r = u.math.arccos(self.value)
        return r

    def arctan_(self):
        self.value = u.math.arctan(self.value)
        return self

    def arctan(self) -> Optional['Array']:
        r = u.math.arctan(self.value)
        return r

    def clamp(
        self,
        min_value: Optional[Union['Array', jax.Array, np.ndarray]] = None,
        max_value: Optional[Union['Array', jax.Array, np.ndarray]] = None,
    ) -> Optional['Array']:
        """
        return the value between min_value and max_value,
        if min_value is None, then no lower bound,
        if max_value is None, then no upper bound.
        """
        r = u.math.clip(self.value, min_value, max_value)
        return r

    def clamp_(
        self,
        min_value: Optional[Union['Array', jax.Array, np.ndarray]] = None,
        max_value: Optional[Union['Array', jax.Array, np.ndarray]] = None
    ):
        """
        return the value between min_value and max_value,
        if min_value is None, then no lower bound,
        if max_value is None, then no upper bound.
        """
        self.clamp(min_value, max_value)
        return self

    def clip_(
        self,
        min_value: Optional[Union['Array', jax.Array, np.ndarray]] = None,
        max_value: Optional[Union['Array', jax.Array, np.ndarray]] = None
    ):
        """
        alias for clamp_
        """
        self.value = self.clip(min_value, max_value)
        return self

    def clone(self) -> 'Array':
        return self.value.copy()

    def expand(self, *sizes) -> 'Array':
        """
        Expand an array to a new shape.

        Parameters::

        sizes : tuple or int
            The shape of the desired array. A single integer ``i`` is interpreted
            as ``(i,)``.

        Returns::

        expanded : Array
            A readonly view on the original array with the given shape. It is
            typically not contiguous. Furthermore, more than one element of a
            expanded array may refer to a single memory location.
        """
        l_ori = len(self.shape)
        l_tar = len(sizes)
        base = l_tar - l_ori
        sizes_list = list(sizes)
        if base < 0:
            raise ValueError(f'the number of sizes provided ({len(sizes)}) must be greater or equal to the number of '
                             f'dimensions in the tensor ({len(self.shape)})')
        for i, v in enumerate(sizes[:base]):
            if v < 0:
                raise ValueError(
                    f'The expanded size of the tensor ({v}) isn\'t allowed in a leading, non-existing dimension {i + 1}')
        for i, v in enumerate(self.shape):
            sizes_list[base + i] = v if sizes_list[base + i] == -1 else sizes_list[base + i]
            if v != 1 and sizes_list[base + i] != v:
                raise ValueError(
                    f'The expanded size of the tensor ({sizes_list[base + i]}) must match the existing size ({v}) at non-singleton '
                    f'dimension {i}.  Target sizes: {sizes}.  Tensor sizes: {self.shape}')
        return u.math.broadcast_to(self.value, tuple(sizes_list))

    def zero_(self):
        self.value = u.math.zeros_like(self.value)
        return self

    def fill_(self, value):
        self.fill(value)
        return self

    def bool(self):
        return u.math.asarray(self.value, dtype=jnp.bool_)

    def int(self):
        return u.math.asarray(self.value, dtype=jnp.int32)

    def long(self):
        return u.math.asarray(self.value, dtype=jnp.int64)

    def half(self):
        return u.math.asarray(self.value, dtype=jnp.float16)

    def float(self):
        return u.math.asarray(self.value, dtype=jnp.float32)

    def double(self):
        return u.math.asarray(self.value, dtype=jnp.float64)
