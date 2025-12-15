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

from __future__ import annotations

import contextlib
import threading
from functools import partial
from typing import (
    Any,
    Union,
    Callable,
    Generic,
    TypeVar,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Dict,
    List,
    Sequence,
    Generator,
    Protocol
)

import brainunit as u
import jax
import numpy as np
from jax.api_util import shaped_abstractify
from jax.extend import source_info_util

from brainstate.typing import ArrayLike, PyTree, Missing, Filter
from brainstate.util import DictManager, PrettyObject
from brainstate.util.filter import Nothing

__all__ = [
    'State',
    'ShortTermState',
    'LongTermState',
    'HiddenState',
    'HiddenGroupState',
    'HiddenTreeState',
    'ParamState',
    'BatchState',
    'TreefyState',
    'FakeState',

    'StateDictManager',
    'StateTraceStack',
    'check_state_value_tree',
    'check_state_jax_tracer',
    'catch_new_states',
    'maybe_state',

    'DelayState',
]

A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T')
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
        new_state_catcher (List[StateCatcher]): A list to store Catcher objects for capturing new states in the current thread.
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
        self.new_state_catcher: List[StateCatcher] = []

    def get_trace_stack_level(self) -> int:
        return len(self.state_stack)


TRACE_CONTEXT = ThreadLocalStack()


@contextlib.contextmanager
def check_state_value_tree(val: bool = True) -> Generator[None, None, None]:
    """
    The contex manager to check weather the tree structure of the state value keeps consistently.

    Once a :py:class:`~.State` is created, the tree structure of the value is fixed. In default,
    the tree structure of the value is not checked to avoid off the repeated evaluation.
    If you want to check the tree structure of the value once the new value is assigned,
    you can use this context manager.

    Examples
    --------

    .. code-block:: python

      >>> import brainstate
      >>> import jax.numpy as jnp
      >>> state = brainstate.ShortTermState(jnp.zeros((2, 3)))
      >>> with brainstate.check_state_value_tree():
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

    Example
    -------

    .. code-block:: python

      >>> import jax
      >>> import brainstate
      >>> import jax.numpy as jnp
      >>>
      >>> a = brainstate.ShortTermState(jnp.zeros((2, 3)))
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



class State(Generic[A], PrettyObject):
    """
    A generic class representing a dynamic data pointer in the BrainState framework.

    The State class serves as a base for various types of state objects used to
    manage and track dynamic data within a program. It provides mechanisms for
    value storage, metadata management, and integration with the BrainState
    tracing system.

    Type Parameters:
        A: The type of the value stored in the state.

    Attributes:
        name (Optional[str]): An optional name for the state.
        value (PyTree): The actual value stored in the state.
        tag (Optional[str]): An optional tag for categorizing or grouping states.

    Args:
        value (Union[PyTree[ArrayLike], StateMetadata[PyTree[ArrayLike]]]):
            The initial value for the state. Can be a PyTree of array-like objects
            or a StateMetadata object.
        name (Optional[str]): An optional name for the state.
        **metadata: Additional metadata to be stored with the state.

    Example
    -------

    .. code-block:: python

        >>> class MyState(State):
        ...     pass
        >>> state = MyState(jnp.zeros((3, 3)), name="my_matrix")
        >>> print(state.value)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]

    Note:
        - Subclasses of :class:`State` (e.g., ShortTermState, LongTermState, ParamState,
          RandomState) are typically used for specific purposes in a program.
        - The class integrates with BrainState's tracing system to track state
          creation and modifications.

        The typical examples of :py:class:`~.State` subclass are:

        - :py:class:`ShortTermState`: The short-term state, which is used to store the short-term data in the program.
        - :py:class:`LongTermState`: The long-term state, which is used to store the long-term data in the program.
        - :py:class:`ParamState`: The parameter state, which is used to store the parameters in the program.
        - :py:class:`RandomState`: The random generator state, which is used to store the random key in the program.

    Args:
        value: PyTree. It can be anything as a pyTree.
        name: Optional[str]. The name of the state.
        tag: Optional[str]. The tag of the state.
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
        value: PyTree[ArrayLike],
        name: Optional[str] = None,
        **metadata: Any
    ):
        """
        Initialize a new HiddenState instance.

        This constructor sets up the initial state for a hidden state in a dynamic model,
        handling various input types and metadata.

        Args:
            value (Union[PyTree[ArrayLike], StateMetadata[PyTree[ArrayLike]]]):
                The initial value for the hidden state. Can be a PyTree of array-like objects
                or a StateMetadata object containing both value and metadata.
            name (Optional[str], optional): A name for the hidden state. Defaults to None.
            **metadata: Additional metadata to be stored with the hidden state, including:
                - tag (Optional[str]): A tag for categorizing or grouping states.
                - Any other custom metadata fields.

        Note:
            This method initializes the hidden state, processes the input value and metadata,
            sets up internal attributes, and records the state initialization.
        """
        tag = metadata.pop('tag', None)

        # set the value and metadata
        if isinstance(value, State):
            value = value.value

        # update metadata
        metadata.update(
            _value=value,
            _level=TRACE_CONTEXT.get_trace_stack_level(),
            _source_info=source_info_util.current(),
            _name=name,
            _been_writen=False,
            tag=tag,
        )

        # avoid using self._setattr to avoid the check
        vars(self).update(metadata)

        # record the state initialization
        record_state_init(self)

    def decrease_stack_level(self):
        """
        Decrease the stack level of the state by one, ensuring it doesn't go below zero.

        This method is used to adjust the stack level of the state, typically when
        exiting a nested context or scope. It ensures that the level never becomes
        negative.
        """
        self._level = max(self._level - 1, 0)

    def increase_stack_level(self):
        """
        Increase the stack level of the state by one.

        This method is used to adjust the stack level of the state, typically when
        entering a nested context or scope. It increments the internal level counter
        by one.
        """
        self._level = self._level + 1

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
        attributes['_level'] = TRACE_CONTEXT.get_trace_stack_level()
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
            return None
        if k == '_value':
            return 'value', jax.tree.map(shaped_abstractify, v)

        if k == '_name':
            if self.name is None:
                return None
            else:
                return 'name', v

        if k == 'tag':
            if self.tag is None:
                return None
            else:
                return 'tag', v

        return k, v

    # def __eq__(self, other: object) -> bool:
    #     return type(self) is type(other) and vars(other) == vars(self)

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
    """
    Record the initialization of a new :class:`State` object.

    This function iterates through all registered state catchers in the current
    trace context and appends the newly initialized state to each catcher.

    Args:
        st (State[A]): The newly initialized :class:`State` object to be recorded.

    Note:
        This function is typically called internally when a new :class:`State` object
        is created to ensure proper tracking and management of states within
        the current execution context.
    """
    trace: StateCatcher
    for trace in TRACE_CONTEXT.new_state_catcher:
        trace.append(st)


def record_state_value_read(st: State[A]):
    """
    Record that a state's value has been read in all relevant trace stacks.

    This function iterates through all state trace stacks at or above the
    state's stack level in the current trace context, and records that
    the given state's value has been read.

    Args:
        st (State[A]): The state object whose value read is being recorded.
                       'A' is a generic type parameter representing the
                       type of the state's value.

    Note:
        This function modifies the state trace stacks in the current
        trace context but does not return any value.
    """
    trace: StateTraceStack
    for trace in TRACE_CONTEXT.state_stack[st.stack_level:]:
        trace.read_its_value(st)


def record_state_value_write(st: State[A]):
    """
    Record that a state's value has been written in all relevant trace stacks.

    This function iterates through all state trace stacks at or above the
    state's stack level in the current trace context, and records that
    the given state's value has been written.

    Args:
        st (State[A]): The state object whose value write is being recorded.
                       'A' is a generic type parameter representing the
                       type of the state's value.

    Note:
        This function modifies the state trace stacks in the current
        trace context but does not return any value.
    """
    trace: StateTraceStack
    for trace in TRACE_CONTEXT.state_stack[st.stack_level:]:
        trace.write_its_value(st)


def record_state_value_restore(st: State[A]):
    """
    Record that a state's value has been restored.

    This function is used to indicate that a state's value has been restored
    to a previous value. It internally calls the record_state_value_read
    function to mark the state as having been accessed.

    Args:
        st (State[A]): The state object whose value restoration is being recorded.
                       'A' is a generic type parameter representing the
                       type of the state's value.

    See Also:
        record_state_value_read: Record that a state's value has been read.

    Note:
        This function does not actually restore the state's value; it only
        records that a restoration has occurred.
    """
    record_state_value_read(st)


class ShortTermState(State):
    """
    A class representing short-term state in a program.

    :class:`ShortTermState` is used to store temporary or transient data that is only relevant
    for a short duration within the program's execution. This class extends the base
    State class, inheriting its properties and methods while specifically denoting
    the short-term nature of the stored data.

    For example, in a machine learning training process, the gradients of the model
    would typically be represented as :class:`ShortTermState`, as they are computed and used
    within each iteration but not necessarily preserved across iterations.

    Attributes:
        Inherits all attributes from the base State class.

    Note:
        This class does not introduce new methods or attributes beyond those
        inherited from the State class. Its primary purpose is to semantically
        distinguish short-term states from other types of states in the program.

    Example:
        >>> gradient = ShortTermState(np.zeros(100), name="model_gradient")
        >>> intermediate_result = ShortTermState({}, name="layer_activations")
    """

    __module__ = 'brainstate'


class LongTermState(State):
    """
    The long-term state, which is used to store the long-term data in the program.

    This class extends the base :class:`State` class and is specifically designed to represent
    and manage long-term data within a program. Long-term states are typically used
    for data that persists across multiple iterations or epochs of a process.

    For example, in a training process, the weights of the model are considered
    long-term states as they are updated and maintained throughout the entire
    training procedure.

    Attributes:
        Inherits all attributes from the base :class:`State` class.

    Note:
        This class does not introduce new methods or attributes beyond those
        inherited from the :class:`State` class. Its primary purpose is to semantically
        distinguish long-term states from other types of states in the program.

    Example:
        >>> model_weights = LongTermState(np.random.randn(100, 100), name="model_weights")
        >>> optimizer_state = LongTermState({}, name="optimizer_state")
    """

    __module__ = 'brainstate'


class BatchState(LongTermState):
    """
    The batch state, which is used to store the batch data in the program.

    This class extends :class:`LongTermState` and is specifically designed to represent
    and manage batch data within a program. It provides a way to encapsulate
    batch-related information and associated metadata, facilitating operations
    like batch processing in machine learning or data analysis tasks.

    Attributes:
        Inherits all attributes from :class:`LongTermState`.

    Note:
        This class does not introduce new methods or attributes beyond those
        inherited from :class:`LongTermState`. Its primary purpose is to semantically
        distinguish batch states from other types of long-term states
        in the program.

    Example:
        >>> batch_data = BatchState(np.array([1, 2, 3, 4, 5]), name="current_batch")
        >>> batch_labels = BatchState(np.array([0, 1, 0, 1, 1]), name="batch_labels")
    """

    __module__ = 'brainstate'


class HiddenState(ShortTermState):
    """
    Represents hidden state variables in neurons or synapses.

    This class extends :class:`ShortTermState` and is specifically designed to represent
    and manage hidden states within dynamic models, such as recurrent neural networks.
    It provides a way to encapsulate hidden state values and associated metadata,
    facilitating operations like state updates during model execution.

    Note:
        :class:`HiddenState` and :class:`ParamState` are two most important state types
        in brainstate. The former is used to store the hidden states in neurons, synapses,
        or networks. The latter is used to store the trainable parameters in the model,
        such as synaptic weights.

    Note:
        From version 0.2.2, :class:`HiddenState` only supports value of numpy.ndarray,
        jax.Array or brainunit.Quantity. Moreover, it is equivalent to :class:`brainstate.HiddenState`.
        Dynamics models defined with :class:`HiddenState` can be seamlessly integrated with
        BrainScale online learning.

    Example:
        >>> lstm_hidden = HiddenState(np.zeros(128), name="lstm_hidden_state")
        >>> gru_hidden = HiddenState(np.zeros(64), name="gru_hidden_state")
    """

    __module__ = 'brainstate'

    value: ArrayLike

    def __init__(self, value: ArrayLike, name: Optional[str] = None):
        self._check_value(value)
        super().__init__(value, name=name)

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        Get the shape of the hidden state variable.

        This property returns the shape of the hidden state variable stored in the instance.
        It provides the dimensions of the array representing the hidden state.

        Returns:
            Tuple[int, ...]: A tuple representing the shape of the hidden state variable.
        """
        return self.value.shape

    @property
    def num_state(self) -> int:
        """
        Get the number of hidden states.

        This property returns the number of hidden states represented by the instance.
        For the `ETraceState` class, this is always 1, as it represents a single hidden state.

        Returns:
            int: The number of hidden states, which is 1 for this class.
        """
        return 1

    def _check_value(self, value: ArrayLike):
        if not isinstance(value, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(
                f'Currently, {HiddenState.__name__} only supports '
                f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                f'But we got {type(value)}.'
            )


class HiddenGroupState(HiddenState):
    """
    A group of multiple hidden states for eligibility trace-based learning.

    This class is used to define multiple hidden states within a single instance
    of :py:class:`ETraceState`. Normally, you should define multiple instances
    of :py:class:`ETraceState` to represent multiple hidden states. But
    :py:class:`HiddenGroupState` let your define multiple hidden states within
    a single instance.

    The following is the way to initialize the hidden states.

    .. code-block:: python

        import brainunit as u
        value = np.random.randn(10, 10, 5) * u.mV
        state = HiddenGroupState(value)

    Then, you can retrieve the hidden state value with the following method.

    .. code-block:: python

        state.get_value(0)  # get the first hidden state
        # or
        state.get_value('0')  # get the hidden state with the name '0'

    You can write the hidden state value with the following method.

    .. code-block:: python

        state.set_value({0: np.random.randn(10, 10) * u.mV})  # set the first hidden state
        # or
        state.set_value({'0': np.random.randn(10, 10) * u.mV})  # set the hidden state with the name '0'
        # or
        state.value = np.random.randn(10, 10, 5) * u.mV  # set all hidden state value

    Args:
        value: The values of the hidden states. It can be a sequence of hidden states,
            or a single hidden state with the last dimension as the number of hidden states,
            or a dictionary of hidden states.
    """

    __module__ = 'brainstate'
    value: ArrayLike
    name2index: Dict[str, int]

    def __init__(self, value: ArrayLike):
        value, name2index = self._check_value(value)
        self.name2index = name2index
        ShortTermState.__init__(self, value)

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        Get the shape of each hidden state variable.

        This property returns the shape of the hidden state variables, excluding
        the last dimension which represents the number of hidden states.

        Returns:
            Tuple[int, ...]: A tuple representing the shape of each hidden state variable.
        """
        return self.value.shape[:-1]

    @property
    def num_state(self) -> int:
        """
        Get the number of hidden states.

        This property returns the number of hidden states represented by the last dimension
        of the value array.

        Returns:
            int: The number of hidden states.
        """
        return self.value.shape[-1]

    def _check_value(self, value) -> Tuple[ArrayLike, Dict[str, int]]:
        """
        Validates the input value for hidden states and returns a tuple containing
        the processed value and a dictionary mapping state names to indices.

        This function ensures that the input value is of a supported type and has
        the required dimensionality for hidden states. It also constructs a mapping
        from string representations of indices to their integer counterparts.

        Parameters
        ----------
        value (ArrayLike): The input value representing hidden states.
            It must be an instance of numpy.ndarray, jax.Array, or brainunit.Quantity
            with at least two dimensions.

        Returns
        -------
        Tuple[ArrayLike, Dict[str, int]]: A tuple containing:
            - The validated and possibly modified input value.
            - A dictionary mapping string representations of indices to integer indices.

        Raises
        ------
        TypeError: If the input value is not of a supported type.
        ValueError: If the input value does not have the required number of dimensions.
        """
        if not isinstance(value, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(
                f'Currently, {self.__class__.__name__} only supports '
                f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                f'But we got {type(value)}.'
            )
        if value.ndim < 2:
            raise ValueError(
                f'Currently, {self.__class__.__name__} only supports '
                f'hidden states with more than 2 dimensions, where the last '
                f'dimension is the number of state size and the other dimensions '
                f'are the hidden shape. '
                f'But we got {value.ndim} dimensions.'
            )
        name2index = {str(i): i for i in range(value.shape[-1])}
        return value, name2index

    def get_value(self, item: int | str) -> ArrayLike:
        """
        Get the value of the hidden state with the item.

        Args:
            item: int or str. The index of the hidden state.
                - If int, the index of the hidden state.
                - If str, the name of the hidden state.
        Returns:
            The value of the hidden state.
        """
        if isinstance(item, int):
            assert item < self.value.shape[-1], (f'Index {item} out of range. '
                                                 f'The maximum index is {self.value.shape[-1] - 1}.')
            return self.value[..., item]
        elif isinstance(item, str):
            assert item in self.name2index, (f'Hidden state name {item} not found. '
                                             f'Please check the hidden state names.')
            index = self.name2index[item]
            return self.value[..., index]
        else:
            raise TypeError(
                f'Currently, {self.__class__.__name__} only supports '
                f'int or str for getting the hidden state. '
                f'But we got {type(item)}.'
            )

    def set_value(
        self,
        val: Dict[int | str, ArrayLike] | Sequence[ArrayLike]
    ) -> None:
        """
        Set the value of the hidden state with the specified item.

        This method updates the hidden state values based on the provided dictionary or sequence.
        The values are set according to the indices or names specified in the input.

        Parameters
        ----------
        val (Dict[int | str, ArrayLike] | Sequence[ArrayLike]):
            A dictionary or sequence containing the new values for the hidden states.
            - If a dictionary, keys can be integers (indices) or strings (names) of the hidden states.
            - If a sequence, it is converted to a dictionary with indices as keys.

        Returns
        -------
        None: This method does not return any value. It updates the hidden state values in place.
        """
        if isinstance(val, (tuple, list)):
            val = {i: v for i, v in enumerate(val)}
        assert isinstance(val, dict), (
            f'Currently, {self.__class__.__name__}.set_value() only supports '
            f'dictionary of hidden states. But we got {type(val)}.'
        )
        indices = []
        values = []
        for k, v in val.items():
            if isinstance(k, str):
                k = self.name2index[k]
            assert isinstance(k, int), (
                f'Key {k} should be int or str. '
                f'But we got {type(k)}.'
            )
            assert v.shape == self.varshape, (
                f'The shape of the hidden state should be {self.varshape}. '
                f'But we got {v.shape}.'
            )
            indices.append(k)
            values.append(v)
        values = u.math.stack(values, axis=-1)
        self.value = self.value.at[..., indices].set(values)


class HiddenTreeState(HiddenGroupState):
    """
    A pytree of multiple hidden states for eligibility trace-based learning.

    .. note::

        The value in this state class behaves likes a dictionary/sequence of hidden states.
        However, the state is actually stored as a single dimensionless array.

    There are two ways to define the hidden states.

    1. The first is to define a sequence of hidden states.

    .. code-block:: python

        import brainunit as u
        value = [np.random.randn(10, 10) * u.mV,
                 np.random.randn(10, 10) * u.mA,
                 np.random.randn(10, 10) * u.mS]
        state = HiddenTreeState(value)

    Then, you can retrieve the hidden state value with the following method.

    .. code-block:: python

        state.get_value(0)  # get the first hidden state
        # or
        state.get_value('0')  # get the hidden state with the name '0'

    You can write the hidden state value with the following method.

    .. code-block:: python

        state.set_value({0: np.random.randn(10, 10) * u.mV})  # set the first hidden state
        # or
        state.set_value({'1': np.random.randn(10, 10) * u.mA})  # set the hidden state with the name '1'
        # or
        state.set_value([np.random.randn(10, 10) * u.mV,
                         np.random.randn(10, 10) * u.mA,
                         np.random.randn(10, 10) * u.mS])  # set all hidden state value
        # or
        state.set_value({
            0: np.random.randn(10, 10) * u.mV,
            1: np.random.randn(10, 10) * u.mA,
            2: np.random.randn(10, 10) * u.mS
        })  # set all hidden state value

    2. The second is to define a dictionary of hidden states.

    .. code-block:: python

        import brainunit as u
        value = {'v': np.random.randn(10, 10) * u.mV,
                 'i': np.random.randn(10, 10) * u.mA,
                 'g': np.random.randn(10, 10) * u.mS}
        state = HiddenTreeState(value)

    Then, you can retrieve the hidden state value with the following method.

    .. code-block:: python

        state.get_value('v')  # get the hidden state with the name 'v'
        # or
        state.get_value('i')  # get the hidden state with the name 'i'

    You can write the hidden state value with the following method.

    .. code-block:: python

        state.set_value({'v': np.random.randn(10, 10) * u.mV})  # set the hidden state with the name 'v'
        # or
        state.set_value({'i': np.random.randn(10, 10) * u.mA})  # set the hidden state with the name 'i'
        # or
        state.set_value([np.random.randn(10, 10) * u.mV,
                         np.random.randn(10, 10) * u.mA,
                         np.random.randn(10, 10) * u.mS])  # set all hidden state value
        # or
        state.set_value({
            'v': np.random.randn(10, 10) * u.mV,
            'g': np.random.randn(10, 10) * u.mA,
            'i': np.random.randn(10, 10) * u.mS
        })  # set all hidden state value

    .. note::

        Avoid using ``HiddenTreeState.value`` to get the state value, or
        ``HiddenTreeState.value =`` to assign the state value.

        Instead, use ``HiddenTreeState.get_value()`` and ``HiddenTreeState.set_value()``.
        This is because ``.value`` loss hidden state units and other information,
        and it is only dimensionless data.

        This design aims to ensure that any etrace hidden state has only one array.


    Args:
        value: The values of the hidden states.
    """

    __module__ = 'brainstate'
    value: ArrayLike

    def __init__(
        self,
        value: Dict[str, ArrayLike] | Sequence[ArrayLike],
    ):
        value, name2unit, name2index = self._check_value(value)
        self.name2unit: Dict[str, u.Unit] = name2unit
        self.name2index: Dict[str, int] = name2index
        self.index2unit: Dict[int, u.Unit] = {i: v for i, v in enumerate(name2unit.values())}
        self.index2name: Dict[int, str] = {v: k for k, v in name2index.items()}
        ShortTermState.__init__(self, value)

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        The shape of each hidden state variable.
        """
        return self.value.shape[:-1]

    @property
    def num_state(self) -> int:
        """
        The number of hidden states.
        """
        assert self.value.shape[-1] == len(self.name2index), (
            f'The number of hidden states '
            f'is not equal to the number of hidden state names.'
        )
        return self.value.shape[-1]

    def _check_value(
        self,
        value: dict | Sequence
    ) -> Tuple[ArrayLike, Dict[str, u.Unit], Dict[str, int]]:
        """
        Validates and processes the input value to ensure it conforms to the expected format
        and structure for hidden states.

        This function checks if the input value is a dictionary or sequence of hidden states,
        verifies that all hidden states have the same shape, and extracts units and indices
        for each hidden state.

        Args:
            value (dict | Sequence): A dictionary or sequence representing hidden states.
                - If a sequence, it is converted to a dictionary with string indices as keys.
                - Each hidden state should be a numpy.ndarray, jax.Array, or brainunit.Quantity.

        Returns:
            Tuple[ArrayLike, Dict[str, u.Unit], Dict[str, int]]:
                - A stacked array of hidden state magnitudes.
                - A dictionary mapping hidden state names to their units.
                - A dictionary mapping hidden state names to their indices.

        Raises:
            TypeError: If any hidden state is not a numpy.ndarray, jax.Array, or brainunit.Quantity.
            ValueError: If hidden states do not have the same shape.
        """
        if isinstance(value, (tuple, list)):
            value = {str(i): v for i, v in enumerate(value)}
        assert isinstance(value, dict), (
            f'Currently, {self.__class__.__name__} only supports '
            f'dictionary/sequence of hidden states. But we got {type(value)}.'
        )
        shapes = []
        for k, v in value.items():
            if not isinstance(v, (np.ndarray, jax.Array, u.Quantity)):
                raise TypeError(
                    f'Currently, {self.__class__.__name__} only supports '
                    f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                    f'But we got {type(v)} for key {k}.'
                )
            shapes.append(v.shape)
        if len(set(shapes)) > 1:
            info = {k: v.shape for k, v in value.items()}
            raise ValueError(
                f'Currently, {self.__class__.__name__} only supports '
                f'hidden states with the same shape. '
                f'But we got {info}.'
            )
        name2unit = {k: u.get_unit(v) for k, v in value.items()}
        name2index = {k: i for i, k in enumerate(value.keys())}
        value = u.math.stack([u.get_magnitude(v) for v in value.values()], axis=-1)
        return value, name2unit, name2index

    def get_value(self, item: str | int) -> ArrayLike:
        """
        Get the value of the hidden state with the key.

        Args:
            item: The key of the hidden state.
                - If int, the index of the hidden state.
                - If str, the name of the hidden state.
        """
        if isinstance(item, int):
            assert item < self.value.shape[-1], (f'Index {item} out of range. '
                                                 f'The maximum index is {self.value.shape[-1] - 1}.')
            val = self.value[..., item]
        elif isinstance(item, str):
            assert item in self.name2index, (f'Hidden state name {item} not found. '
                                             f'Please check the hidden state names.')
            item = self.name2index[item]
            val = self.value[..., item]
        else:
            raise TypeError(
                f'Currently, {self.__class__.__name__} only supports '
                f'int or str for getting the hidden state. '
                f'But we got {type(item)}.'
            )
        if self.index2unit[item].dim.is_dimensionless:
            return val
        else:
            return val * self.index2unit[item]

    def set_value(
        self,
        val: Dict[int | str, ArrayLike] | Sequence[ArrayLike]
    ) -> None:
        """
        Set the value of the hidden state with the specified item.

        This method updates the hidden state values based on the provided dictionary or sequence.
        The values are set according to the indices or names specified in the input.

        Parameters
        ----------
        val (Dict[int | str, ArrayLike] | Sequence[ArrayLike]):
            A dictionary or sequence containing the new values for the hidden states.
            - If a dictionary, keys can be integers (indices) or strings (names) of the hidden states.
            - If a sequence, it is converted to a dictionary with indices as keys.

        Returns
        -------
        None: This method does not return any value. It updates the hidden state values in place.
        """
        if isinstance(val, (tuple, list)):
            val = {i: v for i, v in enumerate(val)}
        assert isinstance(val, dict), (f'Currently, {self.__class__.__name__}.set_value() only supports '
                                       f'dictionary of hidden states. But we got {type(val)}.')
        indices = []
        values = []
        for index, v in val.items():
            if isinstance(index, str):
                index = self.name2index[index]
            assert isinstance(index, int), (f'Key {index} should be int or str. '
                                            f'But we got {type(index)}.')
            assert v.shape == self.varshape, (f'The shape of the hidden state should be {self.varshape}. '
                                              f'But we got {v.shape}.')
            indices.append(index)
            values.append(u.Quantity(v).to(self.index2unit[index]).mantissa)
        if len(indices) == 0:
            raise ValueError(
                f'No hidden state is set. Please check the hidden state names or indices.'
            )
        if len(indices) == 1:
            indices = indices[0]
            values = values[0]
        else:
            indices = np.asarray(indices)
            values = u.math.stack(values, axis=-1)
        self.value = self.value.at[..., indices].set(values)


class ParamState(LongTermState):
    """
    The parameter state, which is used to store the trainable parameters in the model.

    This class extends :class:`LongTermState` and is specifically designed to represent
    and manage trainable parameters within a neural network or machine learning model.
    It provides a way to encapsulate parameter values and associated metadata,
    facilitating operations like parameter updates during training.

    Note:
        :class:`HiddenState` and :class:`ParamState` are two most important state types
        in brainstate. The former is used to store the hidden states in neurons, synapses,
        or networks. The latter is used to store the trainable parameters in the model,
        such as synaptic weights.

    Example:
        >>> weight = ParamState(np.random.randn(10, 10), name="layer1_weights")
        >>> bias = ParamState(np.zeros(10), name="layer1_bias")
    """

    __module__ = 'brainstate'


class FakeState:
    """
    The faked state, which is used to store the faked data in the program.
    """

    __module__ = 'brainstate'

    def __init__(self, value: Any, name: Optional[str] = None):
        """
        Initialize a FakeState instance.

        Args:
            value (Any): The value to be stored in the fake state.
            name (Optional[str], optional): The name of the fake state. Defaults to None.
        """
        self._value = value
        self._name = name

    @property
    def value(self) -> Any:
        """
        Get the value stored in the fake state.

        Returns:
            Any: The value stored in the fake state.
        """
        return self._value

    @value.setter
    def value(self, v) -> None:
        """
        Set the value of the fake state.

        Args:
            v (Any): The new value to be stored in the fake state.
        """
        self._value = v

    def __repr__(self) -> str:
        """
        Return a string representation of the FakeState instance.

        Returns:
            str: A string representation of the FakeState instance.
        """
        return f'FakedState(value={self._value})'

    @property
    def name(self) -> Optional[str]:
        """
        Get the name of the fake state.

        Returns:
            Optional[str]: The name of the fake state, or None if not set.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        Set the name of the fake state.

        Args:
            name (str): The new name for the fake state.
        """
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
    A stack for tracing and managing states during program execution.

    ``StateTraceStack`` is used to automatically trace and manage State objects,
    keeping track of which states are read from or written to during the
    execution of a function or block of code. It provides methods for
    recording state accesses, retrieving state values, and managing the
    lifecycle of states within a tracing context.

    The class is generic over type A, allowing for type-safe usage with
    different types of State objects.

    The ``StateTraceStack`` is a crucial component in implementing state-based
    computations and is particularly useful in scenarios involving automatic
    differentiation or other forms of program transformation.
    """

    def __init__(
        self,
        new_arg: Callable = None,
        name: Optional[str] = None,
    ):
        self.name = name
        self.states: List[State] = []
        self.been_writen: List[bool] = []  # False: read, True: write
        self._state_id_index = dict()
        self._original_state_values = []
        self._jax_trace_new_arg: Callable = new_arg
        self._stack_level = None

    def __str__(self) -> str:
        _stack_level = self.name if self._stack_level is None else self._stack_level
        if _stack_level is None:
            _stack_level = ''
        return f"{self.__class__.__name__}({_stack_level})"

    @property
    def original_state_values(self) -> Tuple[PyTree, ...]:
        """
        Get the original values of all states in the StateTraceStack.

        This property provides access to the initial values of all states
        that were captured when they were first added to the stack. It's
        useful for comparing current state values with their original values
        or for reverting states to their initial condition.

        Returns:
            Tuple[PyTree, ...]: A tuple containing the original values of all
            states in the order they were added to the stack. Each element
            is a PyTree representing the structure and values of a state.
        """
        return tuple(self._original_state_values)

    def set_new_arg(self, new_arg: Callable) -> None:
        self._jax_trace_new_arg = new_arg

    def new_arg(self, state: State) -> None:
        """
        Apply a transformation to the value of a given state using a predefined function.

        This method is used internally to transform the value of a state during tracing.
        If a transformation function (``_jax_trace_new_arg``) is defined, it applies this
        function to each element of the state's value using JAX's tree mapping.

        Args:
            state (State): The State object whose value needs to be transformed.

        Returns:
            None: This function modifies the state in-place and doesn't return anything.

        Note:
            This method is intended for internal use and relies on the presence of
            a ``_jax_trace_new_arg`` function, which should be set separately.
        """
        if self._jax_trace_new_arg is not None:
            # internal use
            state._value = self._jax_trace_new_arg(state)

    def __enter__(self) -> 'StateTraceStack':
        TRACE_CONTEXT.state_stack.append(self)
        self._stack_level = ' / '.join([st.name for st in TRACE_CONTEXT.state_stack if st.name is not None])
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        TRACE_CONTEXT.state_stack.pop()

    def read_its_value(self, state: State) -> None:
        """
        Record that a state's value has been read during tracing.

        This method marks the given state as having been read in the current
        tracing context. If the state hasn't been encountered before, it adds
        it to the internal tracking structures and applies any necessary
        transformations via the new_arg method.

        Args:
            state (State): The State object whose value is being read.

        Returns:
            None

        Note:
            This method updates the internal tracking of state accesses.
            It doesn't actually read or return the state's value.
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
        Record that a state's value has been written to during tracing.

        This method marks the given state as having been written to in the current
        tracing context. If the state hasn't been encountered before, it first
        records it as being read before marking it as written.

        Args:
            state (State): The State object whose value is being written to.

        Returns:
            None

        Note:
            This method updates the internal tracking of state modifications.
            It doesn't actually modify the state's value.
        """
        id_ = id(state)
        if id_ not in self._state_id_index:
            self.read_its_value(state)
        index = self._state_id_index[id_]
        self.been_writen[index] = True

    def get_state_values(
        self,
        separate: bool = False,
        replace: bool = False
    ) -> Sequence[PyTree] | Tuple[Sequence[PyTree], Sequence[PyTree]]:
        """
        Retrieve the values of all states in the StateTraceStack.

        This method returns the values of all states, optionally separating them
        into written and read states, and optionally replacing values with None
        for states that weren't accessed in a particular way.

        Args:
            separate (bool, optional): If True, separate the values into written
                and read states. If False, return all values in a single sequence.
                Defaults to False.
            replace (bool, optional): If True and separate is True, replace values
                with None for states that weren't written/read. If False, only
                include values for states that were written/read. Defaults to False.

        Returns:
            Sequence[PyTree] | Tuple[Sequence[PyTree], Sequence[PyTree]]:
                If separate is False:
                    A sequence of all state values.
                If separate is True:
                    A tuple containing two sequences:
                    - The first sequence contains values of written states.
                    - The second sequence contains values of read states.
                    If replace is True, these sequences will have None for
                    states that weren't written/read respectively.

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
        Restore the original values of all states in the StateTraceStack.

        This method iterates through all states in the stack and restores
        their values to the original ones that were captured when the states
        were first added to the stack. This is useful for reverting changes
        made during tracing or for resetting the states to their initial condition.

        Note:
            This method modifies the states in-place.

        Returns:
            None
        """
        for st, val in zip(self.states, self._original_state_values):
            # internal use
            st.restore_value(val)

    def merge(self, *traces) -> 'StateTraceStack':
        """
        Merge other state traces into the current ``StateTraceStack``.

        This method combines the states, their write status, and original values from
        other ``StateTraceStack`` instances into the current one. If a state from another
        trace is not present in the current trace, it is added. If a state is already
        present, its write status is updated if necessary.

        Args:
            *traces: Variable number of ``StateTraceStack`` instances to be merged into
                     the current instance.

        Returns:
            StateTraceStack: The current ``StateTraceStack`` instance with merged traces.

        Note:
            This method modifies the current ``StateTraceStack`` in-place and also returns it.
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
        Retrieve the states that were read during the function execution.

        This method returns the states that were accessed (read from) during
        the traced function's execution. It can optionally replace written
        states with None.

        Args:
            replace_writen (bool, optional): If True, replace written states with None
                in the returned tuple. If False, exclude written states entirely from
                the result. Defaults to False.

        Returns:
            Tuple[State, ...]: A tuple containing the read states.
                If replace_writen is True, the tuple will have the same length as the
                total number of states, with None for written states.
                If replace_writen is False, the tuple will only contain read-only states.
        """
        if replace_writen:
            return tuple([st if not been_writen else None
                          for st, been_writen in zip(self.states, self.been_writen)])
        else:
            return tuple([st for st, been_writen in zip(self.states, self.been_writen) if not been_writen])

    def get_read_state_values(self, replace_writen: bool = False) -> Tuple[PyTree, ...]:
        """
        Retrieve the values of states that were read during the function execution.

        This method returns the values of states that were accessed (read from) during
        the traced function's execution. It can optionally replace written states with None.

        Args:
            replace_writen (bool, optional): If True, replace the values of written
                states with None in the returned tuple. If False, exclude written
                states entirely from the result. Defaults to False.

        Returns:
            Tuple[PyTree, ...]: A tuple containing the values of read states.
                If replace_writen is True, the tuple will have the same length as the
                total number of states, with None for written states.
                If replace_writen is False, the tuple will only contain values of
                read-only states.
        """
        if replace_writen:
            return tuple(
                [st.value if not been_writen else None
                 for st, been_writen in zip(self.states, self.been_writen)]
            )
        else:
            return tuple([st.value for st, been_writen in zip(self.states, self.been_writen) if not been_writen])

    def get_write_states(self, replace_read: bool = False) -> Tuple[State, ...]:
        """
        Retrieve the states that were written during the function execution.

        This method returns the states that were modified (written to) during
        the traced function's execution. It can optionally replace unwritten (read-only)
        states with None.

        Args:
            replace_read (bool, optional): If True, replace read-only states with None
                in the returned tuple. If False, exclude read-only states entirely from
                the result. Defaults to False.

        Returns:
            Tuple[State, ...]: A tuple containing the written states.
                If replace_read is True, the tuple will have the same length as the
                total number of states, with None for read-only states.
                If replace_read is False, the tuple will only contain written states.
        """
        if replace_read:
            return tuple([st if been_writen else None
                          for st, been_writen in zip(self.states, self.been_writen)])
        else:
            return tuple([st for st, been_writen in zip(self.states, self.been_writen) if been_writen])

    def get_write_state_values(self, replace_read: bool = False) -> Tuple[PyTree, ...]:
        """
        Retrieve the values of states that were written during the function execution.

        This method returns the values of states that were modified (written to) during
        the traced function's execution. It can optionally replace unwritten (read-only)
        states with None.

        Args:
            replace_read (bool, optional): If True, replace the values of read-only
                states with None in the returned tuple. If False, exclude read-only
                states entirely from the result. Defaults to False.

        Returns:
            Tuple[PyTree, ...]: A tuple containing the values of written states.
                If replace_read is True, the tuple will have the same length as the
                total number of states, with None for read-only states.
                If replace_read is False, the tuple will only contain values of
                written states.

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

    def state_subset(self, state_type: type) -> List:
        """
        Get a subset of states of a specific type from the ``StateTraceStack``.

        This method filters the states in the ``StateTraceStack`` and returns only
        those that match the specified state type.

        Args:
            state_type (type): The type of state to filter by. This should be
                a subclass of State or State itself.

        Returns:
            List[State]: A list containing all states in the ``StateTraceStack``
            that are instances of the specified state_type.

        Example:
            >>> stack = StateTraceStack()
            >>> # Assume stack has been populated with various state types
            >>> short_term_states = stack.state_subset(ShortTermState)
        """
        return [st for st in self.states if isinstance(st, state_type)]

    def assign_state_vals(self, state_vals: Sequence[PyTree]) -> None:
        """
        Assign new values to the states tracked by this ``StateTraceStack``.

        This method updates the values of the states based on whether they were
        written to or only read during the tracing process. For states that were
        written to, it directly assigns the new value. For states that were only
        read, it restores the value using the state's restore_value method.

        Args:
            state_vals (Sequence[PyTree]): A sequence of new state values to be
                assigned. Each element in this sequence corresponds to a state
                in the ``StateTraceStack``'s states list.

        Raises:
            ValueError: If the length of state_vals doesn't match the number of
                states in the ``StateTraceStack``.

        Returns:
            None

        Note:
            The order of state_vals should match the order of states in the
            ``StateTraceStack``'s states list.
        """
        if len(state_vals) != len(self.states):
            raise ValueError(
                'The length of the state values must be equal to the states. '
                f'Bug got {len(state_vals)} and {len(self.states)}'
            )
        for st, written, val in zip(self.states, self.been_writen, state_vals):
            if written:
                st.value = val
            else:
                st.restore_value(val)

    def assign_state_vals_v2(
        self: StateTraceStack,
        read_state_vals: Sequence[PyTree],
        write_state_vals: Sequence[PyTree],
    ):
        """
        Write back state values to their corresponding states after computation.

        This function updates the state values based on whether they were written to
        during the computation. If a state was written to, it gets the new written value.
        If not, it restores its original read value.

        Parameters
        ----------
        read_state_vals : sequence of PyTree
            The original state values that were read at the beginning.
        write_state_vals : sequence of PyTree
            The new state values that were written during computation.

        Examples
        --------
        Basic usage in a compilation context:

        .. code-block:: python

            >>> import brainstate
            >>> import jax.numpy as jnp
            >>>
            >>> # Create states
            >>> state1 = brainstate.State(jnp.array([1.0, 2.0]))
            >>> state2 = brainstate.State(jnp.array([3.0, 4.0]))
            >>>
            >>> def f(x):
            ...     state1.value += x  # This state will be written
            ...     return state1.value + state2.value  # state2 is only read
            >>>
            >>> # During compilation, state values are collected and managed
            >>> # write_back_state_values ensures proper state management
        """
        if len(self.states) != len(self.been_writen):
            raise ValueError('The length of the state values must be equal to the states. ')
        if len(read_state_vals) != len(self.states):
            raise ValueError('The length of the read state values must be equal to the states. ')
        if len(write_state_vals) != len(self.states):
            raise ValueError('The length of the write state values must be equal to the states. ')
        for st, write, val_r, val_w in zip(
            self.states, self.been_writen, read_state_vals, write_state_vals
        ):
            if write:
                st.value = val_w
            else:
                st.restore_value(val_r)


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
            return None
        if k == '_value':
            return 'value', v

        if k == '_name':
            return None if v is None else ('name', v)
        return k, v

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
        metadata.pop('_value', None)
        metadata.pop('_level', None)
        vars(state).update(**metadata, _value=self.value, _level=TRACE_CONTEXT.get_trace_stack_level())
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


class StateCatcher(PrettyObject):
    """
    The catcher to catch and manage new states.

    This class provides functionality to collect and tag new State objects.
    It ensures that each state is only added once and assigns a tag to each state.

    Attributes:
        state_tag (str): A string identifier used to tag the caught states.
        state_ids (set): A set of state IDs to ensure uniqueness.
        states (list): A list to store the caught State objects.
    """

    def __init__(
        self,
        state_tag: str,
        state_to_exclude: Filter = Nothing()
    ):
        """
        Initialize a new Catcher instance.

        Args:
            state_tag (str): The tag to be assigned to caught states.
            state_to_exclude (Filter, optional): A filter to exclude states from being caught.
        """
        if state_to_exclude is None:
            state_to_exclude = Nothing()
        self.state_to_exclude = state_to_exclude
        self.state_tag = state_tag
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
            state.tag = self.state_tag

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
    state_tag: str = None,
    state_to_exclude: Filter = Nothing()
) -> Generator[StateCatcher, None, None]:
    """
    A context manager that catches and tracks new states created within its scope.

    This function creates a new Catcher object and adds it to the TRACE_CONTEXT's
    new_state_catcher list. It allows for tracking and managing new states created
    within the context.

    Args:
        state_tag (str, optional): A string tag to associate with the caught states.
            Defaults to None.
        state_to_exclude (Filter, optional): A filter object to specify which states
            should be excluded from catching. Defaults to Nothing(), which excludes no states.

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
        catcher = StateCatcher(state_tag=state_tag, state_to_exclude=state_to_exclude)
        TRACE_CONTEXT.new_state_catcher.append(catcher)
        yield catcher
    finally:
        TRACE_CONTEXT.new_state_catcher.pop()


class DelayState(ShortTermState):
    """
    Short-term state for storing delay data.
    """
    pass
