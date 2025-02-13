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
from typing import (
    Any,
    Union,
    Callable,
    Generic,
    Mapping,
    TypeVar,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Dict,
    List,
    Sequence,
    Generator,
)

import jax
import numpy as np
from jax.api_util import shaped_abstractify
from jax.extend import source_info_util

from brainstate.typing import ArrayLike, PyTree, Missing, Filter
from brainstate.util import DictManager, PrettyObject
from brainstate.util.filter import Nothing

__all__ = [
    'State', 'ShortTermState', 'LongTermState', 'HiddenState', 'ParamState', 'TreefyState',
    'FakedState',

    'StateDictManager',
    'StateTraceStack',
    'check_state_value_tree',
    'check_state_jax_tracer',
    'catch_new_states',
    'maybe_state',
]

A = TypeVar('A')
B = TypeVar('B')
F = TypeVar('F', bound=Callable[..., Any])

max_int = np.iinfo(np.int32)


# The global state of the state stack is accessed by a thread-local object.
# This allows concurrent tracing in separate threads; passing traced objects
# between threads is forbidden.
class ThreadLocalStack(threading.local):
    """
    A thread-local storage class for managing state-related information.

    This class provides thread-local storage for various state management components,
    ensuring that each thread has its own isolated set of state-related data structures.

    Attributes:
        state_stack (List[StateTraceStack]): A list to store StateTraceStack objects for the current thread.
        tree_check (List[bool]): A list of boolean flags for tree structure checking, initialized with [False].
        jax_tracer_check (List[bool]): A list of boolean flags for JAX tracer checking, initialized with [False].
        new_state_catcher (List[Catcher]): A list to store Catcher objects for capturing new states in the current thread.
    """

    def __init__(self):
        """
        Initialize the ThreadLocalStack with empty data structures.

        This constructor sets up the initial state for each thread-local instance,
        creating empty lists for state stack, tree checking, JAX tracer checking,
        and new state catching.
        """
        self.state_stack: List[StateTraceStack] = []
        self.tree_check: List[bool] = [False]
        self.jax_tracer_check: List[bool] = [False]
        self.new_state_catcher: List[Catcher] = []


TRACE_CONTEXT = ThreadLocalStack()


@contextlib.contextmanager
def check_state_value_tree(val: bool = True) -> Generator[None, None, None]:
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


class Catcher(PrettyObject):
    """
    The catcher to catch and manage new states.

    This class provides functionality to collect and tag new State objects.
    It ensures that each state is only added once and assigns a tag to each state.

    Attributes:
        tag (str): A string identifier used to tag the caught states.
        state_ids (set): A set of state IDs to ensure uniqueness.
        states (list): A list to store the caught State objects.
    """

    def __init__(
        self,
        tag: str,
        state_to_exclude: Filter = Nothing()
    ):
        """
        Initialize a new Catcher instance.

        Args:
            tag (str): The tag to be assigned to caught states.
            state_to_exclude (Filter, optional): A filter to exclude states from being caught.
        """
        if state_to_exclude is None:
            state_to_exclude = Nothing()
        self.state_to_exclude = state_to_exclude
        self.tag = tag
        self.state_ids = set()
        self.states = []

    def get_state_values(self) -> List[PyTree]:
        """
        Get the values of the caught states.

        Returns:
            list: A list of values of the caught states.
        """
        return [state.value for state in self.states]

    def get_states(self) -> List[State]:
        """
        Get the caught states.

        Returns:
            list: A list of the caught states.
        """
        return self.states

    def append(self, state: State):
        """
        Add a new state to the catcher if it hasn't been added before.

        This method adds the state to the internal list, records its ID,
        and assigns the catcher's tag to the state.

        Args:
            state (State): The State object to be added.
        """
        if self.state_to_exclude((), state):
            return
        if id(state) not in self.state_ids:
            self.state_ids.add(id(state))
            self.states.append(state)
            state.tag = self.tag

    def __iter__(self):
        """
        Allow iteration over the caught states.

        Returns:
            iterator: An iterator over the list of caught states.
        """
        return iter(self.states)

    def __len__(self):
        """
        Return the number of caught states.

        Returns:
            int: The number of caught states.
        """
        return len(self.states)

    def __getitem__(self, index):
        """
        Get a state by index.

        Args:
            index (int): The index of the state to retrieve.

        Returns:
            State: The state at the specified index.
        """
        return self.states[index]

    def clear(self):
        """
        Clear all caught states.
        """
        self.state_ids.clear()
        self.states.clear()

    def get_by_tag(self, tag: str):
        """
        Get all states with a specific tag.

        Args:
            tag (str): The tag to filter by.

        Returns:
            list: A list of states with the specified tag.
        """
        return [state for state in self.states if state.tag == tag]

    def remove(self, state: State):
        """
        Remove a specific state from the catcher.

        Args:
            state (State): The state to remove.
        """
        if id(state) in self.state_ids:
            self.state_ids.remove(id(state))
            self.states.remove(state)

    def __contains__(self, state: State):
        """
        Check if a state is in the catcher.

        Args:
            state (State): The state to check for.

        Returns:
            bool: True if the state is in the catcher, False otherwise.
        """
        return id(state) in self.state_ids


@contextlib.contextmanager
def catch_new_states(
    tag: str = None,
    state_to_exclude: Filter = Nothing()
) -> Generator[Catcher, None, None]:
    """
    A context manager that catches and tracks new states created within its scope.

    This function creates a new Catcher object and adds it to the TRACE_CONTEXT's
    new_state_catcher list. It allows for tracking and managing new states created
    within the context.

    Args:
        tag (str, optional): A string tag to associate with the caught states.
            Defaults to None.

    Yields:
        Catcher: A Catcher object that can be used to access and manage the
            newly created states within the context.

    Example::
    
        with catch_new_states("my_tag") as catcher:
            # Create new states here
            # They will be caught and tagged with "my_tag"
        # Access caught states through catcher object
    """
    try:
        catcher = Catcher(tag=tag, state_to_exclude=state_to_exclude)
        TRACE_CONTEXT.new_state_catcher.append(catcher)
        yield catcher
    finally:
        TRACE_CONTEXT.new_state_catcher.pop()


def maybe_state(val: Any) -> Any:
    """
    Extracts the value from a State object if given, otherwise returns the input value.

    This function is useful for handling both State objects and raw values uniformly.
    If the input is a State object, it returns the value stored in that State.
    If the input is not a State object, it returns the input as is.

    Args:
        val (Any): The input value, which can be either a State object or any other type.

    Returns:
        Any: The value stored in the State if the input is a State object,
             otherwise the input value itself.
    """
    if isinstance(val, State):
        return val.value
    else:
        return val


@contextlib.contextmanager
def check_state_jax_tracer(val: bool = True) -> Generator[None, None, None]:
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


class State(Generic[A], PrettyObject):
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
    _level: int
    _source_info: source_info_util.SourceInfo
    _name: Optional[str]
    _value: PyTree
    _been_writen: bool  # useful in `unflatten` and `flatten` graph processing
    tag: Optional[str]

    def __init__(
        self,
        value: Union[PyTree[ArrayLike], StateMetadata[PyTree[ArrayLike]]],
        name: Optional[str] = None,
        **metadata: Any
    ):
        tag = metadata.pop('tag', None)

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
                        tag=tag,
                        _been_writen=False)

        # avoid using self._setattr to avoid the check
        vars(self).update(metadata)

        # record the state initialization
        record_state_init(self)

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
        self._name = name

    @property
    def value(self) -> PyTree[ArrayLike]:
        """
        The data and its value.
        """
        record_state_value_read(self)
        return self._read_value()

    @value.setter
    def value(self, v) -> None:
        """
        Set the value of the state.

        Args:
          v: The value.
        """
        # NOTE: the following order is important

        if isinstance(v, State):  # value checking
            raise ValueError('Cannot set value to a State, ' 'use `copy_from` method instead')
        self._check_value_tree(v)  # check the tree structure
        record_state_value_write(self)  # record the value by the stack (>= level)
        self._been_writen = True  # set the flag
        self._write_value(v)  # write the value

    @property
    def stack_level(self):
        """
        The stack level of the state.

        Returns:
            The stack level.
        """
        return self._level

    @stack_level.setter
    def stack_level(self, level: int):
        """
        Set the stack level of the state.

        Args:
            level: The stack level.
        """
        self._level = level

    def _read_value(self) -> PyTree[ArrayLike]:
        """
        The interface to customize the value reading.
        """
        self.check_if_deleted()
        return self._value

    def _write_value(self, v) -> None:
        """
        The interface to customize the value writing.
        """
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
                self.raise_error_with_source_info(
                    ValueError(f'The given value {in_tree} does not match with the origin tree structure {self_tree}.')
                )

    def raise_error_with_source_info(self, error: Exception):
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

    def update_from_ref(self, state_ref: TreefyState[A]) -> None:
        """
        Update the state from the state reference :py:class:`TreefyState`.

        Args:
          state_ref: The state reference.
        """
        metadata = state_ref.get_metadata()
        variable_vars = vars(self)
        variable_vars.update(**metadata)
        if metadata.pop('_been_writen', True):
            self.value = state_ref.value
        else:
            self.restore_value(state_ref.value)

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
        attributes['_level'] = _get_trace_stack_level()
        attributes['_source_info'] = source_info_util.current()
        attributes.pop('_been_writen', None)
        # update the metadata
        vars(obj).update(attributes)
        return obj

    def to_state_ref(self: State[A]) -> TreefyState[A]:
        metadata = vars(self).copy()
        del metadata['_value']
        return TreefyState(type(self), self._value, **metadata)

    def __pretty_repr_item__(self, k, v):
        if k in ['_level', '_source_info', '_been_writen']:
            return None, None
        if k == '_value':
            return 'value', v

        if k == '_name':
            if self.name is None:
                return None, None
            else:
                return 'name', v

        if k == 'tag':
            if self.tag is None:
                return None, None
            else:
                return 'tag', v

        return k, v

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(other) == vars(self)

    def __hash__(self):
        """
        Make the state hashable.
        """
        return hash(id(self))

    def numel(self) -> int:
        """
        Calculate the total number of elements in the state value.

        This method traverses the state's value, which may be a nested structure (PyTree),
        and computes the sum of sizes of all leaf nodes.

        Returns:
            int: The total number of elements across all arrays in the state value.
                For scalar values, this will be 1. For arrays or nested structures,
                it will be the sum of the sizes of all contained arrays.

        Note:
            This method uses jax.tree.leaves to flatten any nested structure in the state value,
            and jax.numpy.size to compute the size of each leaf node.
        """
        sizes = [jax.numpy.size(val) for val in jax.tree.leaves(self._value)]
        return sum(sizes)


def record_state_init(st: State[A]):
    trace: Catcher
    for trace in TRACE_CONTEXT.new_state_catcher:
        trace.append(st)


def record_state_value_read(st: State[A]):
    trace: StateTraceStack
    for trace in TRACE_CONTEXT.state_stack[st.stack_level:]:
        trace.read_its_value(st)


def record_state_value_write(st: State[A]):
    trace: StateTraceStack
    for trace in TRACE_CONTEXT.state_stack[st.stack_level:]:
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


class BatchState(LongTermState):
    """
    The batch state, which is used to store the batch data in the program.
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


class FakedState:
    """
    The faked state, which is used to store the faked data in the program.
    """

    __module__ = 'brainstate'

    def __init__(self, value: Any, name: Optional[str] = None):
        self._value = value
        self._name = name

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, v) -> None:
        self._value = v

    def __repr__(self) -> str:
        return f'FakedState(value={self._value})'

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name


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
            st.restore_value(val)

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
            return tuple([st if not been_writen else None
                          for st, been_writen in zip(self.states, self.been_writen)])
        else:
            return tuple([st for st, been_writen in zip(self.states, self.been_writen) if not been_writen])

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


class TreefyState(Generic[A], PrettyObject):
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

    def __pretty_repr_item__(self, k, v):
        if k in ['_level', '_source_info', '_been_writen']:
            return None, None
        if k == '_value':
            return 'value', v

        if k == '_name':
            return (None, None) if v is None else ('name', v)
        return k, v

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
        vars(state).update(metadata, _value=self.value, _level=_get_trace_stack_level())
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
