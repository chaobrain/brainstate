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

"""
Pretty representation utilities for creating human-readable string representations.

This module provides utilities for creating customizable pretty representations of
objects, with support for nested structures and circular reference detection.
"""

import contextlib
import dataclasses
import math
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Iterator, Mapping, TypeVar, Union, Callable, Optional

import brainunit as u
import jax
import numpy as np

from brainstate._compatible_import import Tracer

__all__ = [
    'yield_unique_pretty_repr_items',
    'PrettyType',
    'PrettyAttr',
    'PrettyRepr',
    'PrettyMapping',
    'MappingReprMixin',
    'pretty_repr_object',
    'pretty_format',
    'ReprOptions',
    'ReprSummary',
    'get_repr_options',
    'set_repr_options',
    'repr_options',
    'format_array_like',
    'format_bytes',
]

A = TypeVar('A')
B = TypeVar('B')


@dataclasses.dataclass
class PrettyType:
    """
    Configuration for pretty representation of objects.

    Attributes
    ----------
    type : Union[str, type]
        The type name or type object to display.
    start : str, default='('
        The opening delimiter for the representation.
    end : str, default=')'
        The closing delimiter for the representation.
    value_sep : str, default='='
        The separator between keys and values.
    elem_indent : str, default='  '
        The indentation for nested elements.
    empty_repr : str, default=''
        The representation for empty objects.
    ref : str or None, default=None
        When set, the object is rendered as this literal marker (for example
        ``'<shared: .a>'``) instead of being expanded.
    """
    type: Union[str, type]
    start: str = '('
    end: str = ')'
    value_sep: str = '='
    elem_indent: str = '  '
    empty_repr: str = ''
    ref: Optional[str] = None


@dataclasses.dataclass
class PrettyAttr:
    """
    Configuration for pretty representation of attributes.

    Attributes
    ----------
    key : str
        The attribute name or key. An empty key renders the value as a
        positional item (without ``key<value_sep>``).
    value : Union[str, Any]
        The attribute value. A ``str`` value is inserted verbatim; any other
        value is rendered recursively.
    start : str, default=''
        Optional prefix for the attribute.
    end : str, default=''
        Optional suffix for the attribute.
    comment : str or None, default=None
        Optional trailing ``# comment``, shown only when the enclosing group is
        broken over several lines.
    """
    key: str
    value: Union[str, Any]
    start: str = ''
    end: str = ''
    comment: Optional[str] = None


class PrettyRepr(ABC):
    """
    Interface for pretty representation of objects.

    This abstract base class provides a framework for creating custom
    pretty representations of objects by yielding PrettyType and PrettyAttr
    instances.

    Examples
    --------
    .. code-block:: python

        >>> class MyObject(PrettyRepr):
        ...     def __init__(self, key, value):
        ...         self.key = key
        ...         self.value = value
        ...
        ...     def __pretty_repr__(self):
        ...         yield PrettyType(type='MyObject', start='{', end='}')
        ...         yield PrettyAttr('key', self.key)
        ...         yield PrettyAttr('value', self.value)
        ...
        >>> obj = MyObject('foo', 42)
        >>> print(obj)
        MyObject{key=foo, value=42}
    """
    __slots__ = ()

    @abstractmethod
    def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
        """
        Generate the pretty representation of the object.

        Yields
        ------
        Union[PrettyType, PrettyAttr]
            First yield should be PrettyType, followed by PrettyAttr instances.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Generate string representation using pretty representation.

        Returns
        -------
        str
            The formatted string representation of the object.
        """
        return pretty_repr_object(self)

    def __str__(self) -> str:
        """
        Generate the display string used by ``print``.

        Identical to ``repr`` except that ANSI colors are applied when enabled
        through :func:`repr_options` / :func:`set_repr_options` (``color=True``)
        and the ``NO_COLOR`` environment variable is not set. A subclass that
        overrides ``__repr__`` with its own format gets that format unchanged.

        Returns
        -------
        str
            The formatted, possibly colored, string representation.
        """
        if not getattr(type(self).__repr__, '_uses_pretty_repr', False):
            return repr(self)
        return pretty_repr_object(self, color=_color_enabled())

    __repr__._uses_pretty_repr = True


def pretty_repr_elem(obj: PrettyType, elem: Any) -> str:
    """
    Constructs a string representation of a single element within a pretty representation.

    This function takes a `PrettyType` object and an element, which must be an instance
    of `PrettyAttr`, and generates a formatted string that represents the element. The
    formatting is based on the configuration provided by the `PrettyType` object.

    Parameters
    ----------
    obj : PrettyType
        The configuration object that defines how the element should be formatted.
        It includes details such as indentation, separators, and surrounding characters.
    elem : Any
        The element to be represented. It must be an instance of `PrettyAttr`, which
        contains the key and value to be formatted.

    Returns
    -------
    str
        A string that represents the element in a formatted manner, adhering to the
        configuration specified by the `PrettyType` object.

    Raises
    ------
    TypeError
        If the provided element is not an instance of `PrettyAttr`.
    """
    if not isinstance(elem, PrettyAttr):
        raise TypeError(f'Item must be Elem, got {type(elem).__name__}')

    value = elem.value if isinstance(elem.value, str) else repr(elem.value)
    value = value.replace('\n', '\n' + obj.elem_indent)

    return f'{obj.elem_indent}{elem.start}{elem.key}{obj.value_sep}{value}{elem.end}'


def pretty_repr_object(obj: PrettyRepr, *, color: bool = False) -> str:
    """
    Render an object implementing the :class:`PrettyRepr` interface.

    The items yielded by ``obj.__pretty_repr__()`` are turned into a document
    tree which is then laid out to fit the configured line width (see
    :class:`ReprOptions`): a group is printed on a single line when it fits,
    and is otherwise broken one item per line with trailing commas. Repeated
    and cyclic references are rendered as ``<shared: path>`` /
    ``<cycle: path>`` markers.

    Parameters
    ----------
    obj : PrettyRepr
        The object to render.
    color : bool, default=False
        Whether to apply ANSI colors to the output.

    Returns
    -------
    str
        The rendered representation.

    Raises
    ------
    TypeError
        If ``obj`` does not implement :class:`PrettyRepr`, or if the first item
        yielded by its ``__pretty_repr__`` method is not a :class:`PrettyType`.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import PrettyMapping, pretty_repr_object, repr_options
        >>> m = PrettyMapping({'a': 1, 'b': [1, 2]}, type_name='Cfg')
        >>> print(pretty_repr_object(m))
        Cfg{'a': 1, 'b': [1, 2]}
        >>>
        >>> with repr_options(width=16):
        ...     print(pretty_repr_object(m))
        Cfg{
          'a': 1,
          'b': [1, 2],
        }
    """
    if not isinstance(obj, PrettyRepr):
        raise TypeError(f'Object {obj!r} is not representable')

    return _render(obj, color)


def pretty_format(value: Any, *, color: bool = False) -> str:
    """
    Format any value with the pretty-repr engine.

    Unlike :func:`pretty_repr_object`, ``value`` does not need to implement
    :class:`PrettyRepr`: lists, tuples, dicts, dataclasses and array-likes are
    laid out and summarised the same way as inside a pretty ``repr``.

    Parameters
    ----------
    value : Any
        The value to format.
    color : bool, default=False
        Whether to apply ANSI colors to the output.

    Returns
    -------
    str
        The formatted value.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainstate.util import pretty_format
        >>> pretty_format({'w': np.zeros((2, 3), dtype=np.float32), 'n': 3})
        "{'w': f32[2,3], 'n': 3}"
    """
    return _render(value, color)


def _render(value: Any, color: bool) -> str:
    options = get_repr_options()
    is_root = CONTEXT.render_depth == 0
    CONTEXT.render_depth += 1
    try:
        doc = _DocBuilder(options).build(value, '', 0)
        text = _Layout(options, color).render(doc, 0, options.width)
        if is_root and options.summary and isinstance(doc, _Group):
            summary = _get_summary(value)
            footer = '' if summary is None else summary.footer()
            if footer:
                text += _style(f'  # {footer}', 'comment', color)
        return text
    finally:
        CONTEXT.render_depth -= 1


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReprOptions:
    """
    Options controlling how pretty representations are rendered.

    Parameters
    ----------
    width : int, default=88
        Target line width. Groups that fit within the remaining width are
        printed on one line; others are broken one item per line.
    max_depth : int or None, default=None
        Number of nested object levels expanded below the root; deeper objects
        are shown as ``Type(...)``. ``0`` collapses all children of the root.
        ``None`` means unlimited.
    max_items : int or None, default=64
        Maximum number of items shown per group; the rest are summarised as
        ``... (+N more)``. ``None`` means unlimited.
    summary : bool, default=True
        Whether to show parameter / state summaries for graph nodes.
    color : bool, default=False
        Whether ``str()`` / ``print()`` apply ANSI colors. ``repr()`` is never
        colored. Ignored when the ``NO_COLOR`` environment variable is set.
    short_dtypes : bool, default=True
        Whether arrays are shown with short dtype names (``f32[3]``) instead of
        the full names (``float32[3]``).

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import ReprOptions
        >>> ReprOptions(width=60).width
        60
    """
    width: int = 88
    max_depth: Optional[int] = None
    max_items: Optional[int] = 64
    summary: bool = True
    color: bool = False
    short_dtypes: bool = True

    def __post_init__(self):
        if not isinstance(self.width, int) or self.width < 1:
            raise ValueError(f'width must be a positive int, got {self.width!r}')
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 0):
            raise ValueError(f'max_depth must be None or a non-negative int, got {self.max_depth!r}')
        if self.max_items is not None and (not isinstance(self.max_items, int) or self.max_items < 1):
            raise ValueError(f'max_items must be None or a positive int, got {self.max_items!r}')


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


_GLOBAL_OPTIONS = ReprOptions(color=_env_flag('BRAINSTATE_REPR_COLOR'))


def get_repr_options() -> ReprOptions:
    """
    Return the pretty-repr options active in the current thread.

    Returns
    -------
    ReprOptions
        The innermost :func:`repr_options` override, or the process-wide
        default set by :func:`set_repr_options`.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import get_repr_options, repr_options
        >>> with repr_options(width=40):
        ...     print(get_repr_options().width)
        40
    """
    options = CONTEXT.options
    return _GLOBAL_OPTIONS if options is None else options


def set_repr_options(**kwargs) -> ReprOptions:
    """
    Update the process-wide default pretty-repr options.

    Parameters
    ----------
    **kwargs
        Fields of :class:`ReprOptions` to change.

    Returns
    -------
    ReprOptions
        The previous default options, which can be restored with
        ``set_repr_options(**dataclasses.asdict(previous))``.

    Raises
    ------
    TypeError
        If an unknown option name is given.
    ValueError
        If an option value is invalid.

    Examples
    --------
    .. code-block:: python

        >>> import dataclasses
        >>> from brainstate.util import set_repr_options, get_repr_options
        >>> previous = set_repr_options(width=100)
        >>> get_repr_options().width
        100
        >>> _ = set_repr_options(**dataclasses.asdict(previous))
    """
    global _GLOBAL_OPTIONS
    previous = _GLOBAL_OPTIONS
    _GLOBAL_OPTIONS = dataclasses.replace(previous, **kwargs)
    return previous


@contextlib.contextmanager
def repr_options(**kwargs) -> Iterator[ReprOptions]:
    """
    Temporarily override pretty-repr options in the current thread.

    Overrides nest, and the previous options are restored on exit, including
    when an exception is raised.

    Parameters
    ----------
    **kwargs
        Fields of :class:`ReprOptions` to change.

    Yields
    ------
    ReprOptions
        The options in effect inside the ``with`` block.

    Raises
    ------
    TypeError
        If an unknown option name is given.
    ValueError
        If an option value is invalid.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import PrettyMapping, repr_options
        >>> m = PrettyMapping({'a': 1, 'b': 2})
        >>> with repr_options(width=8):
        ...     print(repr(m))
        {
          'a': 1,
          'b': 2,
        }
    """
    previous = CONTEXT.options
    new = dataclasses.replace(get_repr_options(), **kwargs)
    CONTEXT.options = new
    try:
        yield new
    finally:
        CONTEXT.options = previous


def _color_enabled() -> bool:
    return get_repr_options().color and 'NO_COLOR' not in os.environ


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReprSummary:
    """
    Size summary of a graph node, shown as a ``# ...`` comment in its repr.

    Parameters
    ----------
    n_params : int
        Number of parameter elements (leaves of ``ParamState`` values).
    param_bytes : int
        Memory footprint of the parameters in bytes.
    n_state_values : int
        Number of elements held by non-parameter states.
    state_bytes : int
        Memory footprint of the non-parameter states in bytes.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import ReprSummary
        >>> s = ReprSummary(n_params=26, param_bytes=104)
        >>> s.footer()
        '26 params (104 B)'
        >>> s.comment()
        '26 params'
    """
    n_params: int = 0
    param_bytes: int = 0
    n_state_values: int = 0
    state_bytes: int = 0

    def footer(self) -> str:
        """Return the long form shown after the root object (empty if all zero)."""
        parts = []
        if self.n_params:
            parts.append(f'{_count(self.n_params, "param")} ({format_bytes(self.param_bytes)})')
        if self.n_state_values:
            parts.append(f'{_count(self.n_state_values, "state value")} ({format_bytes(self.state_bytes)})')
        return ', '.join(parts)

    def comment(self) -> str:
        """Return the short form shown after nested objects (empty if all zero)."""
        if self.n_params:
            return _count(self.n_params, 'param')
        if self.n_state_values:
            return _count(self.n_state_values, 'state value')
        return ''


def _count(n: int, noun: str) -> str:
    return f'{n:,} {noun}' if n == 1 else f'{n:,} {noun}s'


def format_bytes(n: int) -> str:
    """
    Format a byte count with binary units.

    Parameters
    ----------
    n : int
        Number of bytes.

    Returns
    -------
    str
        Human-readable size such as ``'104 B'`` or ``'1.5 KiB'``.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import format_bytes
        >>> format_bytes(104)
        '104 B'
        >>> format_bytes(1536)
        '1.5 KiB'
        >>> format_bytes(3 * 1024 ** 3)
        '3.0 GiB'
    """
    if n < 1024:
        return f'{int(n)} B'
    size = float(n)
    for unit in ('KiB', 'MiB', 'GiB'):
        size /= 1024
        if size < 1024 or unit == 'GiB':
            break
    return f'{size:.1f} {unit}'


def _get_summary(obj: Any) -> Optional[ReprSummary]:
    fn = getattr(obj, '__pretty_repr_summary__', None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        # A summary is decoration; never let it break a repr.
        return None


# ---------------------------------------------------------------------------
# Array formatting
# ---------------------------------------------------------------------------

_DTYPE_PREFIX = (
    ('bfloat', 'bf'),
    ('float', 'f'),
    ('uint', 'u'),
    ('int', 'i'),
    ('complex', 'c'),
)


def _short_dtype(name: str) -> str:
    for prefix, short in _DTYPE_PREFIX:
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return short + name[len(prefix):]
    return name


def _is_array_like(x: Any) -> bool:
    if isinstance(x, (np.ndarray, np.generic, jax.Array, jax.ShapeDtypeStruct)):
        return True
    # Abstract values such as ``ShapedArray``; duck-typed to stay robust
    # across JAX versions.
    return type(x).__name__ == 'ShapedArray' and hasattr(x, 'shape') and hasattr(x, 'dtype')


def _is_concrete(x: Any) -> bool:
    if isinstance(x, (np.ndarray, np.generic)):
        return True
    return isinstance(x, jax.Array) and not isinstance(x, Tracer)


def format_array_like(x: Any, short_dtypes: Optional[bool] = None) -> Optional[str]:
    """
    Summarise an array-like value as ``dtype[shape]``.

    Parameters
    ----------
    x : Any
        A ``jax.Array`` (including tracers), ``numpy.ndarray``, NumPy scalar,
        ``jax.ShapeDtypeStruct``, ``ShapedArray``, or a ``brainunit.Quantity``
        wrapping one of them.
    short_dtypes : bool or None, default=None
        Whether to use short dtype names. ``None`` uses the active
        :class:`ReprOptions`.

    Returns
    -------
    str or None
        The summary (``'f32[3,4]'``, ``'f32(0.5)'``, ``'f32[3] mV'``), or
        ``None`` when ``x`` is not array-like.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from brainstate.util import format_array_like
        >>> format_array_like(np.zeros((3, 4), dtype=np.float32))
        'f32[3,4]'
        >>> format_array_like(np.zeros((3, 4), dtype=np.float32), short_dtypes=False)
        'float32[3,4]'
        >>> format_array_like(np.float32(0.5))
        'f32(0.5)'
        >>> format_array_like(np.ones(3, dtype=np.float32) * u.mV)
        'f32[3] mV'
        >>> format_array_like([1, 2]) is None
        True
    """
    if short_dtypes is None:
        short_dtypes = get_repr_options().short_dtypes
    if isinstance(x, u.Quantity):
        mantissa = x.mantissa
        if isinstance(mantissa, (int, float, complex)) and not isinstance(mantissa, bool):
            return str(x)
        if not _is_array_like(mantissa):
            return None
        if x.is_unitless:
            return format_array_like(mantissa, short_dtypes)
        if np.ndim(mantissa) == 0 and _is_concrete(mantissa):
            return str(x)
        return f'{format_array_like(mantissa, short_dtypes)} {x.unit}'
    if not _is_array_like(x):
        return None
    try:
        dtype = np.dtype(x.dtype).name
        numpy_dtype = True
    except TypeError:
        # extended dtypes such as typed PRNG keys (``key<fry>``)
        dtype = str(x.dtype)
        numpy_dtype = False
    if short_dtypes:
        dtype = _short_dtype(dtype)
    shape = tuple(x.shape)
    if not shape and numpy_dtype and _is_concrete(x):
        # NumPy scalars print the shortest round-tripping form for their own
        # precision (``0.99`` rather than ``0.9900000095367432`` for float32).
        return f'{dtype}({str(np.asarray(x)[()])})'
    return f'{dtype}[{",".join(str(d) for d in shape)}]'


# ---------------------------------------------------------------------------
# Document tree and layout
# ---------------------------------------------------------------------------

_ANSI = {
    'type': '\x1b[1;36m',
    'key': '\x1b[34m',
    'array': '\x1b[32m',
    'ref': '\x1b[2;35m',
    'comment': '\x1b[2m',
}
_ANSI_RESET = '\x1b[0m'


def _style(text: str, kind: Optional[str], color: bool) -> str:
    if not color or kind is None or not text:
        return text
    return f'{_ANSI[kind]}{text}{_ANSI_RESET}'


@dataclasses.dataclass(eq=False)
class _Leaf:
    text: str
    kind: Optional[str] = None


@dataclasses.dataclass(eq=False)
class _Item:
    key: str  # attribute name / dict key; '' for positional items
    sep: str  # separator between key and value
    value: Any  # _Leaf | _Group
    start: str = ''
    end: str = ''
    label: str = ''  # e.g. '[0] ', shown only when the parent is broken
    comment: Union[None, str, Callable[[], str]] = None

    def prefix_width(self) -> int:
        return len(self.start) + (len(self.key) + len(self.sep) if self.key else 0)


@dataclasses.dataclass(eq=False)
class _Group:
    head: str
    open: str
    close: str
    items: list
    empty: str = ''
    indent: int = 2
    single_comma: bool = False  # one-element tuple: ``(x,)`` when flat


def _is_marker(item: _Item) -> bool:
    return isinstance(item.value, _Leaf) and item.value.kind == 'comment'


def _has_generated_repr(value: Any) -> bool:
    """Whether ``value`` is a dataclass instance using the generated ``__repr__``."""
    if isinstance(value, type) or not dataclasses.is_dataclass(value):
        return False
    # ``dataclasses`` builds ``__repr__`` from source compiled as ``<string>``
    # and wraps it with ``reprlib.recursive_repr`` (which sets ``__wrapped__``);
    # a hand-written ``__repr__`` has neither property.
    wrapped = getattr(type(value).__repr__, '__wrapped__', None)
    code = getattr(wrapped, '__code__', None)
    return code is not None and code.co_filename == '<string>'


def _close(iterator: Iterator) -> None:
    """Close a partially consumed ``__pretty_repr__`` generator (runs its cleanup)."""
    close = getattr(iterator, 'close', None)
    if close is not None:
        close()


def _child_path(path: str, key: str, sep: str, index: int) -> str:
    if not key:
        return f'{path}[{index}]'
    if sep.strip() == '=' and key.isidentifier():
        return f'{path}.{key}'
    return f'{path}[{key}]'


class _DocBuilder:
    """Turns objects into a document tree, tracking shared/cyclic references."""

    def __init__(self, options: ReprOptions):
        self.options = options
        # PrettyRepr objects already rendered: id -> (object, path). The object
        # is kept alive so that ids of temporaries cannot be reused.
        self.seen: dict[int, tuple[Any, str]] = {}
        # objects currently being rendered (ancestors): id -> path
        self.active: dict[int, str] = {}

    def build(self, value: Any, path: str, depth: int) -> Union[_Leaf, _Group]:
        if isinstance(value, PrettyRepr):
            return self._build_pretty(value, path, depth)
        if isinstance(value, type):
            return _Leaf(value.__name__)
        if type(value) in (list, tuple, dict) or _has_generated_repr(value):
            key = id(value)
            if key in self.active:
                return _Leaf(f'<cycle: {self.active[key] or "."}>', 'ref')
            self.active[key] = path
            try:
                return self._build_container(value, path, depth)
            finally:
                del self.active[key]
        text = format_array_like(value, self.options.short_dtypes)
        if text is not None:
            return _Leaf(text, 'array')
        return _Leaf(repr(value))

    def _build_container(self, value: Any, path: str, depth: int) -> _Group:
        if type(value) is list:
            return self._build_items(_Group('', '[', ']', []), [('', '', v) for v in value], path, depth)
        if type(value) is tuple:
            group = _Group('', '(', ')', [], single_comma=len(value) == 1)
            return self._build_items(group, [('', '', v) for v in value], path, depth)
        if type(value) is dict:
            entries = [(repr(k), ': ', v) for k, v in value.items()]
            return self._build_items(_Group('', '{', '}', []), entries, path, depth)
        entries = [  # dataclass with the generated ``__repr__``
            PrettyAttr(f.name, getattr(value, f.name))
            for f in dataclasses.fields(value) if f.repr
        ]
        group = _Group(type(value).__name__, '(', ')', [])
        return self._build_items(group, entries, path, depth, '=', quote=True)

    def _build_items(
        self, group: _Group, entries, path: str, depth: int, attr_sep: str = '', quote: bool = False
    ) -> _Group:
        """Build ``entries`` (``PrettyAttr``\\ s or ``(key, sep, value)`` triples) into ``group``.

        ``PrettyAttr`` string values are inserted verbatim unless ``quote``.
        """
        limit = self.options.max_items
        n_extra = 0
        for index, entry in enumerate(entries):
            if limit is not None and index >= limit:
                n_extra += 1
                continue
            if isinstance(entry, PrettyAttr):
                sep = attr_sep if entry.key else ''
                item = _Item(entry.key, sep, None, entry.start, entry.end, comment=entry.comment)
                raw = entry.value
                if isinstance(raw, str) and not quote:
                    # protocol: string values are inserted verbatim
                    item.value = _Leaf(raw)
                    group.items.append(item)
                    continue
            else:
                key, sep, raw = entry
                item = _Item(key, sep, None)
            item.value = self.build(raw, _child_path(path, item.key, item.sep, index), depth)
            if (
                item.comment is None
                and self.options.summary
                and isinstance(item.value, _Group)
                and hasattr(raw, '__pretty_repr_summary__')
            ):
                item.comment = self._summary_comment(raw)
            group.items.append(item)
        if n_extra:
            group.items.append(_Item('', '', _Leaf(f'... (+{n_extra} more)', 'comment')))
        self._label(group)
        return group

    @staticmethod
    def _summary_comment(raw: Any) -> Callable[[], str]:
        def comment() -> str:
            summary = _get_summary(raw)
            return '' if summary is None else summary.comment()

        return comment

    @staticmethod
    def _label(group: _Group):
        """Add ``[i]`` labels to broken sequences of nested objects."""
        if group.open not in ('[', '(') or any(it.key for it in group.items):
            return
        if not any(
            isinstance(it.value, _Group) and it.value.head
            or isinstance(it.value, _Leaf) and it.value.kind == 'ref'
            for it in group.items
        ):
            return
        for i, it in enumerate(group.items):
            if not _is_marker(it):
                it.label = f'[{i}] '

    def _build_pretty(self, obj: PrettyRepr, path: str, depth: int) -> Union[_Leaf, _Group]:
        key = id(obj)
        if key in self.active:
            return _Leaf(f'<cycle: {self.active[key] or "."}>', 'ref')
        if key in self.seen:
            return _Leaf(f'<shared: {self.seen[key][1] or "."}>', 'ref')

        iterator = iter(obj.__pretty_repr__())
        header = next(iterator, None)
        if not isinstance(header, PrettyType):
            raise TypeError(f'First item must be PrettyType, got {type(header).__name__}')
        if header.ref is not None:
            _close(iterator)
            return _Leaf(header.ref, 'ref')
        type_name = header.type if isinstance(header.type, str) else header.type.__name__
        # Only named objects count as a nesting level; anonymous wrappers such
        # as ``PrettyList`` / ``PrettyDict`` render like plain containers.
        if type_name:
            depth += 1
        max_depth = self.options.max_depth
        if max_depth is not None and depth > max_depth + 1:
            _close(iterator)
            return _Leaf(f'{type_name}{header.start}...{header.end}')

        self.seen[key] = (obj, path)
        self.active[key] = path
        try:
            group = _Group(type_name, header.start, header.end, [], header.empty_repr, len(header.elem_indent))
            attrs = []
            for elem in iterator:
                if not isinstance(elem, PrettyAttr):
                    raise TypeError(f'Item must be Elem, got {type(elem).__name__}')
                attrs.append(elem)
            return self._build_items(group, attrs, path, depth, header.value_sep)
        finally:
            del self.active[key]


class _Layout:
    """Lays out a document tree within the target width."""

    def __init__(self, options: ReprOptions, color: bool):
        self.options = options
        self.color = color
        self._widths: dict[int, float] = {}

    def flat_width(self, doc) -> float:
        """Width of ``doc`` printed on one line (``inf`` if it cannot be)."""
        key = id(doc)
        width = self._widths.get(key)
        if width is not None:
            return width
        if isinstance(doc, _Leaf):
            width = math.inf if '\n' in doc.text else len(doc.text)
        elif '\n' in doc.open or '\n' in doc.close:
            width = math.inf
        else:
            width = len(doc.head) + len(doc.open) + len(doc.close)
            if doc.items:
                width += 2 * (len(doc.items) - 1)
                for item in doc.items:
                    width += item.prefix_width() + self.flat_width(item.value) + len(item.end)
                width += doc.single_comma
            else:
                width += len(doc.empty)
        self._widths[key] = width
        return width

    def _prefix(self, item: _Item) -> str:
        if not item.key:
            return item.start
        return item.start + _style(item.key, 'key', self.color) + item.sep

    def flat(self, doc) -> str:
        if isinstance(doc, _Leaf):
            return _style(doc.text, doc.kind, self.color)
        head = _style(doc.head, 'type', self.color)
        if not doc.items:
            return f'{head}{doc.open}{doc.empty}{doc.close}'
        body = ', '.join(self._prefix(it) + self.flat(it.value) + it.end for it in doc.items)
        if doc.single_comma:
            body += ','
        return f'{head}{doc.open}{body}{doc.close}'

    def render(self, doc, indent: int, avail: float) -> str:
        """Render ``doc`` at ``indent`` with ``avail`` columns left on its first line."""
        if isinstance(doc, _Leaf):
            return _style(doc.text, doc.kind, self.color).replace('\n', '\n' + ' ' * indent)
        if not doc.items or self.flat_width(doc) <= avail:
            return self.flat(doc)

        inner = indent + doc.indent
        pad = ' ' * inner
        lines = [_style(doc.head, 'type', self.color) + doc.open]
        for item in doc.items:
            used = inner + len(item.label) + item.prefix_width() + len(item.end) + 1
            value = self.render(item.value, inner, self.options.width - used)
            line = f'{pad}{item.label}{self._prefix(item)}{value}{item.end},'
            comment = item.comment() if callable(item.comment) else item.comment
            if comment:
                line += _style(f'  # {comment}', 'comment', self.color)
            lines.append(line)
        lines.append(' ' * indent + doc.close)
        return '\n'.join(lines)


class MappingReprMixin(Mapping[A, B]):
    """
    Mapping mixin for pretty representation.

    This mixin only supplies a ``__pretty_repr__`` implementation (the hook
    consumed by :func:`pretty_repr_object`); it does **not** override ``__repr__``
    or ``__str__``. In particular, mixing it into ``dict`` does not change how the
    object prints, because ``dict.__repr__`` still wins via the MRO. To obtain the
    multi-line pretty string you must route through :func:`pretty_repr_object`,
    which also requires the object to be a :class:`PrettyRepr` instance.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util._pretty_repr import (
        ...     MappingReprMixin, PrettyRepr, pretty_repr_object
        ... )
        >>>
        >>> # Combine with ``PrettyRepr`` and expose the pretty output through an
        >>> # explicit ``__repr__`` (``dict.__repr__`` would otherwise shadow it).
        >>> class PrettyDict(dict, MappingReprMixin, PrettyRepr):
        ...     def __repr__(self):
        ...         return pretty_repr_object(self)
        ...
        >>> print(PrettyDict({'a': 1, 'b': 2}))
        {'a': 1, 'b': 2}
    """

    def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
        """
        Generate pretty representation for mapping.

        Yields
        ------
        Union[PrettyType, PrettyAttr]
            PrettyType followed by PrettyAttr for each key-value pair.
        """
        yield PrettyType(type='', value_sep=': ', start='{', end='}')

        for key, value in self.items():
            yield PrettyAttr(repr(key), value)


@dataclasses.dataclass(repr=False)
class PrettyMapping(PrettyRepr):
    """
    Pretty representation of a mapping.

    Attributes
    ----------
    mapping : Mapping
        The mapping to represent.
    type_name : str, default=''
        Optional type name to display.

    Examples
    --------
    .. code-block:: python

        >>> m = PrettyMapping({'a': 1, 'b': 2}, type_name='MyDict')
        >>> print(m)
        MyDict{'a': 1, 'b': 2}
    """
    mapping: Mapping
    type_name: str = ''

    def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
        """
        Generate pretty representation for the mapping.

        Yields
        ------
        Union[PrettyType, PrettyAttr]
            PrettyType followed by PrettyAttr for each key-value pair.
        """
        yield PrettyType(type=self.type_name, value_sep=': ', start='{', end='}')

        for key, value in self.mapping.items():
            yield PrettyAttr(repr(key), value)


@dataclasses.dataclass
class PrettyReprContext(threading.local):
    """
    A thread-local context for managing the state of pretty representation.

    This class is used to keep track of objects that have been seen during
    the generation of pretty representations, preventing infinite recursion
    in cases of circular references.

    Attributes
    ----------
    seen_modules_repr : dict[int, Any] | None
        A dictionary mapping object IDs to objects that have been seen
        during the pretty representation process. This is used to avoid
        representing the same object multiple times.
    """
    seen_modules_repr: dict[int, Any] | None = None
    options: Optional['ReprOptions'] = None
    render_depth: int = 0


CONTEXT = PrettyReprContext()


def _default_repr_object(node: Any) -> Iterator[PrettyType]:
    """
    Generate a default pretty representation for an object.

    This function yields a `PrettyType` instance that represents the type
    of the given object. It is used as a default method for representing
    objects when no custom representation function is provided.

    Parameters
    ----------
    node : Any
        The object for which the pretty representation is to be generated.

    Yields
    ------
    PrettyType
        An instance of `PrettyType` that contains the type information of
        the object.
    """
    yield PrettyType(type=type(node))


def _default_repr_attr(node: Any) -> Iterator[PrettyAttr]:
    """
    Generate a default pretty representation for the attributes of an object.

    This function iterates over the attributes of the given object and yields
    a `PrettyAttr` instance for each attribute that does not start with an
    underscore. The `PrettyAttr` instances contain the attribute name and its
    string representation.

    Parameters
    ----------
    node : Any
        The object whose attributes are to be represented.

    Yields
    ------
    PrettyAttr
        An instance of `PrettyAttr` for each non-private attribute of the object,
        containing the attribute name and its string representation.
    """
    for name, value in vars(node).items():
        if name.startswith('_'):
            continue
        yield PrettyAttr(name, repr(value))


def yield_unique_pretty_repr_items(
    node: Any,
    repr_object: Optional[Callable] = None,
    repr_attr: Optional[Callable] = None
) -> Iterator[Union[PrettyType, PrettyAttr]]:
    """
    Generate a pretty representation of an object while avoiding duplicate representations.

    This function yields a structured representation of an object, using custom or default
    methods for representing the object itself and its attributes. It ensures that each
    object is only represented once to prevent infinite recursion in cases of circular
    references.

    Parameters
    ----------
    node : Any
        The object to be represented.
    repr_object : Optional[Callable], optional
        A callable that yields the representation of the object itself.
        If not provided, a default representation function is used.
    repr_attr : Optional[Callable], optional
        A callable that yields the representation of the object's attributes.
        If not provided, a default attribute representation function is used.

    Yields
    ------
    Union[PrettyType, PrettyAttr]
        The pretty representation of the object and its attributes,
        avoiding duplicates by tracking seen objects.

    Examples
    --------
    .. code-block:: python

        >>> class Node:
        ...     def __init__(self, value, next=None):
        ...         self.value = value
        ...         self.next = next
        ...
        >>> # Create circular reference
        >>> node1 = Node(1)
        >>> node2 = Node(2, node1)
        >>> node1.next = node2
        ...
        >>> # This will handle circular reference gracefully
        >>> items = list(yield_unique_pretty_repr_items(node1))
        >>> [type(item).__name__ for item in items]
        ['PrettyType', 'PrettyAttr', 'PrettyAttr']
    """
    if repr_object is None:
        repr_object = _default_repr_object
    if repr_attr is None:
        repr_attr = _default_repr_attr

    if CONTEXT.seen_modules_repr is None:
        # CONTEXT.seen_modules_repr = set()
        CONTEXT.seen_modules_repr = dict()
        clear_seen = True
    else:
        clear_seen = False

    # Avoid infinite recursion
    if id(node) in CONTEXT.seen_modules_repr:
        yield PrettyType(type=type(node), empty_repr='...')
        return

    # repr object
    yield from repr_object(node)

    # Add to seen modules
    # CONTEXT.seen_modules_repr.add(id(node))
    CONTEXT.seen_modules_repr[id(node)] = node

    try:
        # repr attributes
        yield from repr_attr(node)
    finally:
        if clear_seen:
            CONTEXT.seen_modules_repr = None
