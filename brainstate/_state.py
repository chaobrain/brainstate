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

from __future__ import annotations

import contextlib
import dataclasses
import threading
from functools import wraps, partial
from typing import (Any, Union, Callable, Generic, Mapping,
                    TypeVar, Optional, TYPE_CHECKING, Tuple, Dict, List, Sequence)

import jax
import numpy as np
from jax.api_util import shaped_abstractify
from jax.extend import source_info_util

from brainstate.typing import ArrayLike, PyTree, Missing
from brainstate.util import DictManager, PrettyRepr, PrettyType, PrettyAttr, TraceContextError
from brainstate.util._tracers import StateJaxTracer

__all__ = [
    'State', 'ShortTermState', 'LongTermState', 'HiddenState', 'ParamState', 'TreefyState',

    'StateDictManager', 'StateTraceStack', 'check_state_value_tree', 'check_state_jax_tracer', 'catch_new_states',
]

A = TypeVar('A')
B = TypeVar('B')
F = TypeVar('F', bound=Callable[..., Any])

max_int = np.iinfo(np.int32)


# The global state of the state stack is accessed by a thread-local object.
# This allows concurrent tracing in separate threads; passing traced objects
# between threads is forbidden.
class ThreadLocalStack(threading.local):
    def __init__(self):
        self.state_stack: List[StateTraceStack] = []
        self.tree_check: List[bool] = [False]
        self.jax_tracer_check: List[bool] = [False]
        self.new_state_catcher: List[Catcher] = []


TRACE_CONTEXT = ThreadLocalStack()


@contextlib.contextmanager
def check_state_value_tree(val: bool = True) -> None:
    """
    The contex manager to check weather the tree structure of the state value keeps consistently.

    Once a :py:class:`~.State` is created, the tree structure of the value is fixed. In default,
    the tree structure of the value is not checked to avoid off the repeated evaluation.
    If you want to check the tree structure of the value once the new value is assigned,
    you can use this context manager.

    Example::

      >>> import brainstate as bst
      >>> import jax.numpy as jnp
      >>> state = bst.ShortTermState(jnp.zeros((2, 3)))
      >>> with bst.check_state_value_tree():
      >>>   # The line below will not raise an error.
      >>>   state.value = jnp.zeros((2, 3))
      ...
      >>>   # The following code will raise an error, since it changes the tree structure.
      >>>   state.value = (jnp.zeros((2, 3)), jnp.zeros((2, 3)))

    """
    try:
        TRACE_CONTEXT.tree_check.append(val)
        yield
    finally:
        TRACE_CONTEXT.tree_check.pop()


@contextlib.contextmanager
def catch_new_states(tag: str = None) -> List:
    try:
        catcher = Catcher(tag)
        TRACE_CONTEXT.new_state_catcher.append(catcher)
        yield catcher
    finally:
        TRACE_CONTEXT.new_state_catcher.pop()


class Catcher:
    def __init__(self, tag: str):
        self.tag = tag
        self.state_ids = set()
        self.states = []

    def append(self, state: State):
        if id(state) not in self.state_ids:
            self.state_ids.add(id(state))
            self.states.append(state)
            state.tag = self.tag


@contextlib.contextmanager
def check_state_jax_tracer(val: bool = True) -> None:
    """
    The context manager to check whether the state is valid to trace.

    Example::

      >>> import jax
      >>> import brainstate as bst
      >>> import jax.numpy as jnp
      >>>
      >>> a = bst.ShortTermState(jnp.zeros((2, 3)))
      >>>
      >>> @jax.jit
      >>> def run_state(b):
      >>>   a.value = b
      >>>   return a.value
      >>>
      >>>  # The following code will not raise an error, since the state is valid to trace.
      >>> run_state(jnp.ones((2, 3)))
      >>>
      >>> with check_state_jax_tracer():
      >>>   # The line below will not raise an error.
      >>>   run_state(jnp.ones((2, 4)))
    """
    try:
        TRACE_CONTEXT.jax_tracer_check.append(val)
        yield
    finally:
        TRACE_CONTEXT.jax_tracer_check.pop()


@dataclasses.dataclass
class StateMetadata(Generic[A]):
    """
    The state metadata.

    Args:
      raw_value: The raw value.
      metadata: The metadata.
    """
    raw_value: A
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def with_metadata(initializer: F, **metadata: Any) -> F:
    """
    A decorator to add metadata to the state.
    """

    @wraps(initializer)
    def wrapper(*args):
        return StateMetadata(initializer(*args), metadata=metadata)

    return wrapper  # type: ignore


def _get_trace_stack_level() -> int:
    return len(TRACE_CONTEXT.state_stack)


class State(Generic[A], PrettyRepr):
    """
    The pointer to specify the dynamical data.

    To implement a new subclass of :py:class:`~.State`, you only need to inherent this class:

    Example::

      >>> class MyState(State):
      >>>   pass

    The typical examples of :py:class:`~.State` subclass are:

    - :py:class:`~.ShortTermState`: The short-term state, which is used to store the short-term data in the program.
    - :py:class:`~.LongTermState`: The long-term state, which is used to store the long-term data in the program.
    - :py:class:`~.ParamState`: The parameter state, which is used to store the parameters in the program.
    - :py:class:`~.RandomState`: The random generator state, which is used to store the random key in the program.

    Args:
      value: PyTree. It can be anything as a pyTree.
    """
    __module__ = 'brainstate'
    _trace_state: StateJaxTracer
    _level: int
    _source_info: source_info_util.SourceInfo
    _name: Optional[str]
    _value: PyTree
    tag: Optional[str]

    def __init__(
        self,
        value: Union[PyTree[ArrayLike], StateMetadata[PyTree[ArrayLike]]],
        name: Optional[str] = None,
        **metadata: Any
    ):
        tag = metadata.pop('tag', None)

        # avoid using self._setattr to avoid the check
        vars(self)['_trace_state'] = StateJaxTracer()

        # set the value and metadata
        if isinstance(value, StateMetadata):
            metadata.update(dict(value.metadata))
            value = value.raw_value
        if isinstance(value, State):
            value = value.value

        # update metadata
        metadata.update(_value=value,
                        _level=_get_trace_stack_level(),
                        _source_info=source_info_util.current(),
                        _name=name,
                        tag=tag)

        # avoid using self._setattr to avoid the check
        vars(self).update(metadata)

        record_state_init(self)

    if not TYPE_CHECKING:
        def __setattr__(self, name: str, value: Any) -> None:
            return self._setattr(name, value)

    def _setattr(self, name: str, value: Any):
        """
        Check if the state is valid to mutate.
        """
        if TRACE_CONTEXT.jax_tracer_check[-1]:
            self.check_valid_trace(lambda: f'Cannot mutate {type(self).__name__} from a different trace level')
        object.__setattr__(self, name, value)

    def _setattr_no_check(self, name: str, value: Any):
        """
        Set the attribute without checking the trace level.
        """
        vars(self)[name] = value

    def check_valid_trace(self, error_msg: Callable[[], str]):
        """
        Check if the state is valid to trace.
        """
        if not self._trace_state.is_valid():
            raise TraceContextError(error_msg())

    @property
    def name(self) -> Optional[str]:
        """
        The name of the state.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        Set the name of the state.
        """
        self._setattr_no_check('_name', name)

    @property
    def value(self) -> PyTree[ArrayLike]:
        """
        The data and its value.
        """
        self.check_if_deleted()
        record_state_value_read(self)
        return self._value

    @value.setter
    def value(self, v) -> None:
        """
        Set the value of the state.

        Args:
          v: The value.
        """
        self.write_value(v)

    def write_value(self, v) -> None:
        # value checking
        if isinstance(v, State):
            raise ValueError('Cannot set value to a State, ' 'use `copy_from` method instead')
        self._check_value_tree(v)
        # write the value by the stack (>= level)
        record_state_value_write(self)
        # set the value
        self._value = v

    def restore_value(self, v) -> None:
        """
        Restore the value of the state.

        Args:
          v: The value.
        """
        # value checking
        if isinstance(v, State):
            raise ValueError('Cannot set value to a State, ' 'use `copy_from` method instead')
        with check_state_value_tree():
            self._check_value_tree(v)
        # record the value by the stack (>= level)
        record_state_value_restore(self)
        # set the value
        self._value = v

    def value_call(self, func: Callable[..., Any]) -> Any:
        """
        Call the function with the value of the state.
        """
        return jax.tree.map(func, self.value)

    def _check_value_tree(self, v):
        """
        Check if the value tree structure is consistent.
        """
        if TRACE_CONTEXT.tree_check[-1]:
            in_tree = jax.tree.structure(v)
            self_tree = jax.tree.structure(self._value)
            if in_tree != self_tree:
                self._raise_error_with_source_info(
                    ValueError(f'The given value {in_tree} does not match with the origin tree structure {self_tree}.')
                )

    def _raise_error_with_source_info(self, error: Exception):
        """
        Raise an error with the source information for easy debugging.
        """
        name_stack = source_info_util.current_name_stack() + self.source_info.name_stack
        with source_info_util.user_context(self.source_info.traceback, name_stack=name_stack):
            raise error

    def check_if_deleted(self):
        pass

    @property
    def source_info(self) -> source_info_util.SourceInfo:
        """
        The source information of the state, can be useful to identify
        the source code where the definition of the state.

        Returns:
          The source information.
        """
        return self._source_info

    def copy_from(self, other: State[A]) -> None:
        """
        Copy the state from another state.
        """
        if type(self) is not type(other):
            raise ValueError(f'Cannot copy from incompatible container, '
                             f'expected {type(self).__name__}, got {type(other).__name__}')
        if self is other:
            return

        # keep the trace state and stack level
        trace_state = self._trace_state
        level = self._level
        source_info = self._source_info

        # copy other metadata
        other_vars = vars(other).copy()
        del other_vars['_trace_state']
        del other_vars['_level']
        del other_vars['_source_info']

        # update the metadata
        vars_dict = vars(self)
        vars_dict.clear()
        vars_dict.update(other_vars, _trace_state=trace_state, _level=level, _source_info=source_info)

    def update_from_ref(self, state_ref: TreefyState[A]) -> None:
        """
        Update the state from the state reference :py:class:`TreefyState`.

        Args:
          state_ref: The state reference.
        """
        variable_vars = vars(self)
        variable_vars.update(**state_ref.get_metadata())
        self.value = state_ref.value

    def replace(self, value: Any = Missing, **kwargs) -> State[Any]:
        """
        Replace the attribute of the state.
        """
        if value is not Missing:
            kwargs['_value'] = value

        # return `value` if it is a State
        if '_value' in kwargs and isinstance(value := kwargs['_value'], State):
            # remove value from kwargs
            kwargs.pop('_value')
            if type(self) is not type(value):
                raise ValueError('Cannot replace value from incompatible container, '
                                 f'expected {type(self).__name__}, got {type(value).__name__}')
            # if kwargs aren't empty, recursively call replace
            # else return variable value
            if kwargs:
                return value.replace(**kwargs)
            else:
                return value

        # get and update attributes
        attributes = vars(self).copy()
        attributes.update(**kwargs)
        # return new instance with updated attributes
        obj = object.__new__(type(self))
        vars(obj).update(attributes)
        return obj

    def copy(self: State[A]) -> State[A]:
        """
        Copy the state.
        """
        obj = object.__new__(type(self))
        attributes = vars(self).copy()
        # keep its own trace state and stack level
        attributes['_trace_state'] = StateJaxTracer()
        attributes['_level'] = _get_trace_stack_level()
        attributes['_source_info'] = source_info_util.current()
        # update the metadata
        vars(obj).update(attributes)
        return obj

    def to_state_ref(self: State[A]) -> TreefyState[A]:
        metadata = vars(self).copy()
        del metadata['_value']
        del metadata['_trace_state']
        del metadata['_level']
        return TreefyState(type(self), self._value, **metadata)

    def __pretty_repr__(self):
        yield PrettyType(type=type(self))
        for name, value in vars(self).items():
            if name == '_value':
                name = 'value'
            if name == '_name':
                if value is None:
                    continue
                else:
                    name = 'name'
            if name == 'tag' and value is None:
                continue
            if name in ['_trace_state', '_level', '_source_info']:
                continue
            yield PrettyAttr(name, repr(value))

    def __treescope_repr__(self, path, subtree_renderer):
        children = {}
        for name, value in vars(self).items():
            if name == '_value':
                name = 'value'
            if name == '_name':
                if value is None:
                    continue
                else:
                    name = 'name'
            if name == 'tag' and value is None:
                continue
            if name in ['_trace_state', '_level', '_source_info']:
                continue
            children[name] = value

        import treescope  # type: ignore[import-not-found,import-untyped]
        return treescope.repr_lib.render_object_constructor(
            object_type=type(self),
            attributes=children,
            path=path,
            subtree_renderer=subtree_renderer,
        )

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(other) == vars(self)


def record_state_init(st: State[A]):
    trace: Catcher
    for trace in TRACE_CONTEXT.new_state_catcher:
        trace.append(st)


def record_state_value_read(st: State[A]):
    trace: StateTraceStack
    for trace in TRACE_CONTEXT.state_stack[st._level:]:
        trace.read_its_value(st)


def record_state_value_write(st: State[A]):
    trace: StateTraceStack
    for trace in TRACE_CONTEXT.state_stack[st._level:]:
        trace.write_its_value(st)


def record_state_value_restore(st: State[A]):
    record_state_value_read(st)


class ShortTermState(State):
    """
    The short-term state, which is used to store the short-term data in the program.

    For example, in a training process, the gradients of the model are short-term states.
    """

    __module__ = 'brainstate'


class LongTermState(State):
    """
    The long-term state, which is used to store the long-term data in the program.

    For example, in a training process, the weights of the model are long-term states.
    """

    __module__ = 'brainstate'


class HiddenState(ShortTermState):
    """
    The hidden state, which is used to store the hidden data in a dynamic model.
    """

    __module__ = 'brainstate'


class ParamState(LongTermState):
    """
    The parameter state, which is used to store the trainable parameters in the model.
    """

    __module__ = 'brainstate'


class StateDictManager(DictManager):
    """
    State stack, for collecting all :py:class:`~.State` used in the program.

    :py:class:`~.StateDictManager` supports all features of python dict.
    """

    __module__ = 'brainstate'

    def assign_values(self, *args: Dict) -> None:
        """
        Assign the value for each element according to the given ``data``.
        """
        for arg in args:
            assert isinstance(arg, dict), 'Must be an instance of dict.'
            for k, v in arg.items():
                self._set_elem(k, v)

    def split_values(self, *filters: type) -> Tuple[Dict, ...]:
        """
        Split the values into several subsets of stack by the given types.
        """
        results = tuple(DictManager() for _ in range(len(filters) + 1))
        for k, v in self.items():
            for i, filt in enumerate(filters):
                if isinstance(v, filt):
                    results[i][k] = v.value
                    break
            else:
                results[-1][k] = v.value
        return results

    def collect_values(self) -> Dict:
        """
        Collect the values by the given types.
        """
        results = DictManager()
        for k, v in self.items():
            results[k] = v.value
        return results

    def split(self, first: type, *others: type) -> Tuple['StateDictManager', ...]:
        return super().split(first, *others)

    def to_dict_values(self) -> Dict:
        """
        Convert the values into a dict.
        """
        return {k: v.value for k, v in self.items()}

    def _check_elem(self, elem):
        assert isinstance(elem, State), f'must be instance of {State}'

    def _set_elem(self, key: Any, value: Any) -> None:
        self[key].value = value


class StateTraceStack(Generic[A]):
    """
    The state trace stack, which is used to trace the states automatically.
    """

    def __init__(self, new_arg: Callable = None):
        self.states: List[State] = []
        self.been_writen: List[bool] = []  # False: read, True: write
        self._state_id_index = dict()
        self._original_state_values = []
        self._jax_trace_new_arg: Callable = new_arg

    @property
    def original_state_values(self) -> Tuple[PyTree, ...]:
        """
        The original values of the states.
        """
        return tuple(self._original_state_values)

    def set_new_arg(self, new_arg: Callable) -> None:
        self._jax_trace_new_arg = new_arg

    def new_arg(self, state: State) -> None:
        if self._jax_trace_new_arg is not None:
            # internal use
            state._value = jax.tree.map(lambda x: self._jax_trace_new_arg(shaped_abstractify(x)), state._value)

    def __enter__(self) -> 'StateTraceStack':
        TRACE_CONTEXT.state_stack.append(self)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        TRACE_CONTEXT.state_stack.pop()

    def read_its_value(self, state: State) -> None:
        """
        Read the value of the state.

        Args:
          state: The state.
        """
        id_ = id(state)
        if id_ not in self._state_id_index:
            self._state_id_index[id_] = len(self.states)
            self.states.append(state)
            self.been_writen.append(False)
            self._original_state_values.append(state._value)  # internal use
            self.new_arg(state)

    def write_its_value(self, state: State) -> None:
        """
        Write the value of the state.

        Args:
          state: The state.
        """
        id_ = id(state)
        if id_ not in self._state_id_index:
            self.read_its_value(state)
        index = self._state_id_index[id_]
        self.been_writen[index] = True

    def get_state_values(self, separate: bool = False, replace: bool = False
                         ) -> Sequence[PyTree] | Tuple[Sequence[PyTree], Sequence[PyTree]]:
        """
        Get the values of the states.
        """
        if separate:
            if replace:
                writes, reads = [], []
                for st, been_writen in zip(self.states, self.been_writen):
                    if been_writen:
                        writes.append(st.value)
                        reads.append(None)
                    else:
                        reads.append(st.value)
                        writes.append(None)
                return tuple(writes), tuple(reads)
            else:
                writes, reads = [], []
                for st, been_writen in zip(self.states, self.been_writen):
                    if been_writen:
                        writes.append(st.value)
                    else:
                        reads.append(st.value)
                return tuple(writes), tuple(reads)
        else:
            return tuple([st.value for st in self.states])

    def recovery_original_values(self) -> None:
        """
        Recovery the original values.
        """
        for st, val in zip(self.states, self._original_state_values):
            # internal use
            st._value = val

    def merge(self, *traces) -> 'StateTraceStack':
        """
        Merge other state traces.
        """
        trace: StateTraceStack
        for trace in traces:
            for st, been_writen, org_val in zip(trace.states, trace.been_writen, trace._original_state_values):
                if id(st) not in self._state_id_index:  # read the value
                    self._state_id_index[id(st)] = len(self.states)
                    self._original_state_values.append(org_val)  # add the original value
                    self.states.append(st)  # append the state
                    self.been_writen.append(False)
                if been_writen:
                    self.write_its_value(st)
        return self

    def get_read_states(self, replace_writen: bool = False) -> Tuple[State, ...]:
        """
        Read the states that are read by the function.

        Returns:
          The states that are read by the function.
        """
        if replace_writen:
            return tuple([st for st, been_writen in zip(self.states, self.been_writen) if not been_writen])
        else:
            return tuple([st if not been_writen else None
                          for st, been_writen in zip(self.states, self.been_writen)])

    def get_read_state_values(self, replace_writen: bool = False) -> Tuple[PyTree, ...]:
        """
        Read the states that are read by the function.

        Returns:
          The states that are read by the function.
        """
        if replace_writen:
            return tuple(
                [st.value if not been_writen else None for st, been_writen in zip(self.states, self.been_writen)])
        else:
            return tuple([st.value for st, been_writen in zip(self.states, self.been_writen) if not been_writen])

    def get_write_states(self, replace_read: bool = False) -> Tuple[State, ...]:
        """
        Read the states that are written by the function.

        Returns:
          The states that are written by the function.
        """
        if replace_read:
            return tuple([st if been_writen else None
                          for st, been_writen in zip(self.states, self.been_writen)])
        else:
            return tuple([st for st, been_writen in zip(self.states, self.been_writen) if been_writen])

    def get_write_state_values(self, replace_read: bool = False) -> Tuple[PyTree, ...]:
        """
        Read the states that are written by the function.

        Returns:
          The states that are written by the function.
        """
        if replace_read:
            return tuple([st.value if been_writen else None for st, been_writen in zip(self.states, self.been_writen)])
        else:
            return tuple([st.value for st, been_writen in zip(self.states, self.been_writen) if been_writen])

    def __add__(self, other: 'StateTraceStack') -> 'StateTraceStack':
        """
        Support the syntax of `+` to merge the state traces.
        """
        return StateTraceStack().merge(self, other)


class TreefyState(Generic[A], PrettyRepr):
    """
    The state as a pytree.
    """

    def __init__(
        self,
        type: type[State[Any]],
        value: A,
        **metadata
    ):
        self.type = type
        self.value = value
        vars(self).update(metadata)

    if TYPE_CHECKING:
        def __getattr__(self, name: str) -> None: ...

        def __setattr__(self, name: str, value: Any) -> None: ...

        def __delattr__(self, name: str) -> None: ...

    def __pretty_repr__(self):
        yield PrettyType(type=type(self))
        yield PrettyAttr('type', self.type.__name__)
        for name, value in vars(self).items():
            if name == '_value':
                name = 'value'
            if name == '_name':
                if value is None:
                    continue
                else:
                    name = 'name'
            if name in ['_trace_state', '_level', '_source_info', 'type']:
                continue
            yield PrettyAttr(name, repr(value))

    def __treescope_repr__(self, path, subtree_renderer):
        children = {'type': self.type}
        for name, value in vars(self).items():
            if name == 'type':
                continue
            children[name] = value

        import treescope  # type: ignore[import-not-found,import-untyped]
        return treescope.repr_lib.render_object_constructor(
            object_type=type(self),
            attributes=children,
            path=path,
            subtree_renderer=subtree_renderer,
        )

    def replace(self, value: B) -> TreefyState[B]:
        """
        Replace the value of the state reference.
        """
        return TreefyState(self.type, value, **self.get_metadata())

    def to_state(self) -> State[A]:
        """
        Convert the state reference to the state.
        """
        # we use object.__new__ to avoid calling __init__ and bypass the
        # __init__ logic which should not be called twice
        metadata = self.get_metadata()
        state = object.__new__(self.type)
        vars(state).update(metadata, _value=self.value, _trace_state=StateJaxTracer(), _level=_get_trace_stack_level())
        return state

    def copy(self: TreefyState[A]) -> TreefyState[A]:
        """
        Copy the state reference.
        """
        return jax.tree.map(lambda x: x, self)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata of the state reference
        """
        metadata = vars(self).copy()
        del metadata['type']
        del metadata['value']
        return metadata


def _state_ref_flatten(x: TreefyState[Any], *, with_keys: bool):
    metadata = tuple(x.get_metadata().items())
    if with_keys:
        node = (jax.tree_util.GetAttrKey('value'), x.value)
    else:
        node = x.value
    return (node,), (x.type, metadata)


def _state_ref_unflatten(
    static: Tuple[type[State[A]], Tuple[Tuple[str, Any], ...]],
    children: Tuple[A],
) -> TreefyState[A]:
    return TreefyState(type=static[0], value=children[0], **dict(static[1]))


jax.tree_util.register_pytree_with_keys(
    TreefyState,
    partial(_state_ref_flatten, with_keys=True),  # type: ignore
    _state_ref_unflatten,  # type: ignore
    flatten_func=partial(_state_ref_flatten, with_keys=False),  # type: ignore
)
