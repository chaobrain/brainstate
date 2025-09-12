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

from typing import (
    Sequence, Optional, TypeVar, _SpecialForm, _type_check, _remove_dups_flatten, _UnionGenericAlias
)

import jax

T = TypeVar('T')
ArrayLike = jax.typing.ArrayLike

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
    def identifier(self, value: ArrayLike):
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
