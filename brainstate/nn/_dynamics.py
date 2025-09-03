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


"""
All the basic dynamics class for the ``brainstate``.

For handling dynamical systems:

- ``DynamicsGroup``: The class for a group of modules, which update ``Projection`` first,
                   then ``Dynamics``, finally others.
- ``Projection``: The class for the synaptic projection.
- ``Dynamics``: The class for the dynamical system.

For handling the delays:

- ``Delay``: The class for all delays.
- ``DelayAccess``: The class for the delay access.

"""

from typing import Any, Dict, Callable, Hashable, Optional, Union, TypeVar, TYPE_CHECKING, Tuple

import jax
import numpy as np

from brainstate import environ
from brainstate._state import State
from brainstate.graph import Node
from brainstate.mixin import ParamDescriber
from brainstate.typing import Size, ArrayLike
from ._delay import StateWithDelay, Delay
from ._module import Module

__all__ = [
    'DynamicsGroup', 'Projection', 'Dynamics',
    'Prefetch', 'PrefetchDelay', 'PrefetchDelayAt', 'OutputDelayAt',
]

T = TypeVar('T')
_max_order = 10


class Projection(Module):
    """
    Base class for synaptic projection modules in neural network modeling.

    This class defines the interface for modules that handle projections between
    neural populations. Projections process input signals and transform them
    before they reach the target neurons, implementing the connectivity patterns
    in neural networks.

    In the BrainState execution order, Projection modules are updated before
    Dynamics modules, following the natural information flow in neural systems:
    1. Projections process inputs (synaptic transmission)
    2. Dynamics update neuron states (neural integration)

    The Projection class does not implement the update logic directly but delegates
    to its child nodes. If no child nodes exist, it raises a ValueError.

    Parameters
    ----------
    *args : Any
        Arguments passed to the parent Module class.
    **kwargs : Any
        Keyword arguments passed to the parent Module class.

    Raises
    ------
    ValueError
        If the update() method is called but no child nodes are defined.

    Notes
    -----
    Derived classes should implement specific projection behaviors, such as
    dense connectivity, sparse connectivity, or specific weight update rules.
    """
    __module__ = 'brainstate.nn'

    def update(self, *args, **kwargs):
        sub_nodes = tuple(self.nodes(allowed_hierarchy=(1, 1)).values())
        if len(sub_nodes):
            for node in sub_nodes:
                node(*args, **kwargs)
        else:
            raise ValueError('Do not implement the update() function.')


class Dynamics(Module):
    """
    Base class for implementing neural dynamics models in BrainState.

    Dynamics classes represent the core computational units in neural simulations,
    implementing the differential equations or update rules that govern neural activity.
    This class provides infrastructure for managing neural populations, handling inputs,
    and coordinating updates within the simulation framework.

    The Dynamics class serves several key purposes:
    1. Managing neuron population geometry and size information
    2. Handling current and delta (instantaneous change) inputs to neurons
    3. Supporting before/after update hooks for computational dependencies
    4. Providing access to delayed state variables through the prefetch mechanism
    5. Establishing the execution order in neural network simulations

    Parameters
    ----------
    in_size : Size
        The geometry of the neuron population. Can be an integer (e.g., 10) for
        1D neuron arrays, or a tuple (e.g., (10, 10)) for multi-dimensional populations.
    name : Optional[str], default=None
        Optional name identifier for this dynamics module.

    Attributes
    ----------
    in_size : tuple
        The shape/geometry of the neuron population.
    out_size : tuple
        The output shape, typically matches in_size.
    current_inputs : Optional[Dict[str, Union[Callable, ArrayLike]]]
        Dictionary of registered current input functions or arrays.
    delta_inputs : Optional[Dict[str, Union[Callable, ArrayLike]]]
        Dictionary of registered delta input functions or arrays.
    before_updates : Optional[Dict[Hashable, Callable]]
        Dictionary of functions to call before the main update.
    after_updates : Optional[Dict[Hashable, Callable]]
        Dictionary of functions to call after the main update.

    Notes
    -----
    In the BrainState execution sequence, Dynamics modules are updated after
    Projection modules and before other module types, reflecting the natural
    flow of information in neural systems.

    There are several essential attributes:

    - ``size``: the geometry of the neuron group. For example, `(10, )` denotes a line of
      neurons, `(10, 10)` denotes a neuron group aligned in a 2D space, `(10, 15, 4)` denotes
      a 3-dimensional neuron group.
    - ``num``: the flattened number of neurons in the group. For example, `size=(10, )` => \
      `num=10`, `size=(10, 10)` => `num=100`, `size=(10, 15, 4)` => `num=600`.


    See Also
    --------
    Module : Parent class providing base module functionality
    Projection : Class for handling synaptic projections between neural populations
    DynamicsGroup : Container for organizing multiple dynamics modules
    """

    __module__ = 'brainstate.nn'

    graph_invisible_attrs = ('_before_updates', '_after_updates', '_current_inputs', '_delta_inputs')

    # before updates
    _before_updates: Optional[Dict[Hashable, Callable]]

    # after updates
    _after_updates: Optional[Dict[Hashable, Callable]]

    # current inputs
    _current_inputs: Optional[Dict[str, ArrayLike | Callable]]

    # delta inputs
    _delta_inputs: Optional[Dict[str, ArrayLike | Callable]]

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
    ):
        # initialize
        super().__init__(name=name)

        # geometry size of neuron population
        if isinstance(in_size, (list, tuple)):
            if len(in_size) <= 0:
                raise ValueError(f'"in_size" must be int, or a tuple/list of int. But we got {type(in_size)}')
            if not isinstance(in_size[0], (int, np.integer)):
                raise ValueError(f'"in_size" must be int, or a tuple/list of int. But we got {type(in_size)}')
            in_size = tuple(in_size)
        elif isinstance(in_size, (int, np.integer)):
            in_size = (in_size,)
        else:
            raise ValueError(f'"in_size" must be int, or a tuple/list of int. But we got {type(in_size)}')
        self.in_size = in_size

        # current inputs
        self._current_inputs = None

        # delta inputs
        self._delta_inputs = None

        # before updates
        self._before_updates = None

        # after updates
        self._after_updates = None

        # in-/out- size of neuron population
        self.out_size = self.in_size

    # def __pretty_repr_item__(self, name, value):
    #     if name in [
    #         '_before_updates', '_after_updates', '_current_inputs', '_delta_inputs',
    #         '_in_size', '_out_size', '_name', '_mode',
    #     ]:
    #         return (name, value) if value is None else (name[1:], value)  # skip the first `_`
    #     return super().__pretty_repr_item__(name, value)

    @property
    def varshape(self):
        """
        Get the shape of variables in the neuron group.

        This property provides access to the geometry (shape) of the neuron population,
        which determines how variables and states are structured.

        Returns
        -------
        tuple
            A tuple representing the dimensional shape of the neuron group,
            matching the in_size parameter provided during initialization.

        See Also
        --------
        in_size : The input geometry specification for the neuron group
        """
        return self.in_size

    @property
    def current_inputs(self):
        """
        Get the dictionary of current inputs registered with this dynamics model.

        Current inputs represent direct input currents that flow into the model.

        Returns
        -------
        dict or None
            A dictionary mapping keys to current input functions or values,
            or None if no current inputs have been registered.

        See Also
        --------
        add_current_input : Register a new current input
        sum_current_inputs : Apply and sum all current inputs
        delta_inputs : Dictionary of instantaneous change inputs
        """
        return self._current_inputs

    @property
    def delta_inputs(self):
        """
        Get the dictionary of delta inputs registered with this dynamics model.

        Delta inputs represent instantaneous changes to state variables (dX/dt).

        Returns
        -------
        dict or None
            A dictionary mapping keys to delta input functions or values,
            or None if no delta inputs have been registered.

        See Also
        --------
        add_delta_input : Register a new delta input
        sum_delta_inputs : Apply and sum all delta inputs
        current_inputs : Dictionary of direct current inputs
        """
        return self._delta_inputs

    def add_current_input(
        self,
        key: str,
        inp: Union[Callable, ArrayLike],
        label: Optional[str] = None
    ):
        """
        Add a current input function or array to the dynamics model.

        Current inputs represent direct input currents that can be accessed during
        model updates through the `sum_current_inputs()` method.

        Parameters
        ----------
        key : str
            Unique identifier for this current input. Used to retrieve or reference
            the input later.
        inp : Union[Callable, ArrayLike]
            The input data or function that generates input data.
            - If callable: Will be called during updates with arguments passed to `sum_current_inputs()`
            - If array-like: Will be applied once and then automatically removed from available inputs
        label : Optional[str], default=None
            Optional grouping label for the input. When provided, allows selective
            processing of inputs by label in `sum_current_inputs()`.

        Raises
        ------
        ValueError
            If the key has already been used for a different current input.

        Notes
        -----
        - Inputs with the same label can be processed together using the `label`
          parameter in `sum_current_inputs()`.
        - Non-callable inputs are consumed when used (removed after first use).
        - Callable inputs persist and can be called repeatedly.

        See Also
        --------
        sum_current_inputs : Sum all current inputs matching a given label
        add_delta_input : Add a delta input function or array
        """
        key = _input_label_repr(key, label)
        if self._current_inputs is None:
            self._current_inputs = dict()
        if key in self._current_inputs:
            if id(self._current_inputs[key]) != id(inp):
                raise ValueError(f'Key "{key}" has been defined and used in the current inputs of {self}.')
        self._current_inputs[key] = inp

    def add_delta_input(
        self,
        key: str,
        inp: Union[Callable, ArrayLike],
        label: Optional[str] = None
    ):
        """
        Add a delta input function or array to the dynamics model.

        Delta inputs represent instantaneous changes to the model state (i.e., dX/dt contributions).
        This method registers a function or array that provides delta inputs which will be
        accessible during model updates through the `sum_delta_inputs()` method.

        Parameters
        ----------
        key : str
            Unique identifier for this delta input. Used to retrieve or reference
            the input later.
        inp : Union[Callable, ArrayLike]
            The input data or function that generates input data.
            - If callable: Will be called during updates with arguments passed to `sum_delta_inputs()`
            - If array-like: Will be applied once and then automatically removed from available inputs
        label : Optional[str], default=None
            Optional grouping label for the input. When provided, allows selective
            processing of inputs by label in `sum_delta_inputs()`.

        Raises
        ------
        ValueError
            If the key has already been used for a different delta input.

        Notes
        -----
        - Inputs with the same label can be processed together using the `label`
          parameter in `sum_delta_inputs()`.
        - Non-callable inputs are consumed when used (removed after first use).
        - Callable inputs persist and can be called repeatedly.

        See Also
        --------
        sum_delta_inputs : Sum all delta inputs matching a given label
        add_current_input : Add a current input function or array
        """
        key = _input_label_repr(key, label)
        if self._delta_inputs is None:
            self._delta_inputs = dict()
        if key in self._delta_inputs:
            if id(self._delta_inputs[key]) != id(inp):
                raise ValueError(f'Key "{key}" has been defined and used.')
        self._delta_inputs[key] = inp

    def get_input(self, key: str):
        """
        Get a registered input function by its key.

        Retrieves either a current input or a delta input function that was previously
        registered with the given key. This method checks both current_inputs and
        delta_inputs dictionaries for the specified key.

        Parameters
        ----------
        key : str
            The unique identifier used when the input function was registered.

        Returns
        -------
        Callable or ArrayLike
            The input function or array associated with the given key.

        Raises
        ------
        ValueError
            If no input function is found with the specified key in either
            current_inputs or delta_inputs.

        See Also
        --------
        add_current_input : Register a current input function
        add_delta_input : Register a delta input function

        Examples
        --------
        >>> model = Dynamics(10)
        >>> model.add_current_input('stimulus', lambda t: np.sin(t))
        >>> input_func = model.get_input('stimulus')
        >>> input_func(0.5)  # Returns sin(0.5)
        """
        if self._current_inputs is not None and key in self._current_inputs:
            return self._current_inputs[key]
        elif self._delta_inputs is not None and key in self._delta_inputs:
            return self._delta_inputs[key]
        else:
            raise ValueError(f'Input key {key} is not in current/delta inputs of the module {self}.')

    def sum_current_inputs(
        self,
        init: Any,
        *args,
        label: Optional[str] = None,
        **kwargs
    ):
        """
        Summarize all current inputs by applying and summing all registered current input functions.

        This method iterates through all registered current input functions (from `.current_inputs`)
        and applies them to calculate the total input current for the dynamics model. It adds all results
        to the initial value provided.

        Parameters
        ----------
        init : Any
            The initial value to which all current inputs will be added.
        *args : tuple
            Variable length argument list passed to each current input function.
        label : Optional[str], default=None
            If provided, only process current inputs with this label prefix.
            When None, process all current inputs regardless of label.
        **kwargs : dict
            Arbitrary keyword arguments passed to each current input function.

        Returns
        -------
        Any
            The initial value plus all applicable current inputs summed together.

        Notes
        -----
        - Non-callable current inputs are applied once and then automatically removed from
          the current_inputs dictionary.
        - Callable current inputs remain registered for subsequent calls.
        - When a label is provided, only current inputs with keys starting with that label
          are applied.
        """
        if self._current_inputs is None:
            return init
        if label is None:
            filter_fn = lambda k: True
        else:
            label_repr = _input_label_start(label)
            filter_fn = lambda k: k.startswith(label_repr)
        for key in tuple(self._current_inputs.keys()):
            if filter_fn(key):
                out = self._current_inputs[key]
                if callable(out):
                    try:
                        init = init + out(*args, **kwargs)
                    except Exception as e:
                        raise ValueError(
                            f'Error in delta input value {key}: {out}\n'
                            f'Error: {e}'
                        ) from e
                else:
                    try:
                        init = init + out
                    except Exception as e:
                        raise ValueError(
                            f'Error in delta input value {key}: {out}\n'
                            f'Error: {e}'
                        ) from e
                    self._current_inputs.pop(key)
        return init

    def sum_delta_inputs(
        self,
        init: Any,
        *args,
        label: Optional[str] = None,
        **kwargs
    ):
        """
        Summarize all delta inputs by applying and summing all registered delta input functions.

        This method iterates through all registered delta input functions (from `.delta_inputs`)
        and applies them to calculate instantaneous changes to model states. It adds all results
        to the initial value provided.

        Parameters
        ----------
        init : Any
            The initial value to which all delta inputs will be added.
        *args : tuple
            Variable length argument list passed to each delta input function.
        label : Optional[str], default=None
            If provided, only process delta inputs with this label prefix.
            When None, process all delta inputs regardless of label.
        **kwargs : dict
            Arbitrary keyword arguments passed to each delta input function.

        Returns
        -------
        Any
            The initial value plus all applicable delta inputs summed together.

        Notes
        -----
        - Non-callable delta inputs are applied once and then automatically removed from
          the delta_inputs dictionary.
        - Callable delta inputs remain registered for subsequent calls.
        - When a label is provided, only delta inputs with keys starting with that label
          are applied.
        """
        if self._delta_inputs is None:
            return init
        if label is None:
            filter_fn = lambda k: True
        else:
            label_repr = _input_label_start(label)
            filter_fn = lambda k: k.startswith(label_repr)
        for key in tuple(self._delta_inputs.keys()):
            if filter_fn(key):
                out = self._delta_inputs[key]
                if callable(out):
                    try:
                        init = init + out(*args, **kwargs)
                    except Exception as e:
                        raise ValueError(
                            f'Error in delta input function {key}: {out}\n'
                            f'Error: {e}'
                        ) from e
                else:
                    try:
                        init = init + out
                    except Exception as e:
                        raise ValueError(
                            f'Error in delta input value {key}: {out}\n'
                            f'Error: {e}'
                        ) from e
                    self._delta_inputs.pop(key)
        return init

    @property
    def before_updates(self):
        """
        Get the dictionary of functions to execute before the module's update.

        Returns
        -------
        dict or None
            Dictionary mapping keys to callable functions that will be executed
            before the main update, or None if no before updates are registered.

        Notes
        -----
        Before updates are executed in the order they were registered whenever
        the module is called via __call__.
        """
        return self._before_updates

    @property
    def after_updates(self):
        """
        Get the dictionary of functions to execute after the module's update.

        Returns
        -------
        dict or None
            Dictionary mapping keys to callable functions that will be executed
            after the main update, or None if no after updates are registered.

        Notes
        -----
        After updates are executed in the order they were registered whenever
        the module is called via __call__, and may optionally receive the return
        value from the update method.
        """
        return self._after_updates

    def _add_before_update(self, key: Any, fun: Callable):
        """
        Register a function to be executed before the module's update.

        Parameters
        ----------
        key : Any
            A unique identifier for the update function.
        fun : Callable
            The function to execute before the module's update.

        Raises
        ------
        KeyError
            If the key is already registered in before_updates.

        Notes
        -----
        Internal method used by the module system to register dependencies.
        """
        if self._before_updates is None:
            self._before_updates = dict()
        if key in self.before_updates:
            raise KeyError(f'{key} has been registered in before_updates of {self}')
        self.before_updates[key] = fun

    def _add_after_update(self, key: Any, fun: Callable):
        """
        Register a function to be executed after the module's update.

        Parameters
        ----------
        key : Any
            A unique identifier for the update function.
        fun : Callable
            The function to execute after the module's update.

        Raises
        ------
        KeyError
            If the key is already registered in after_updates.

        Notes
        -----
        Internal method used by the module system to register dependencies.
        """
        if self._after_updates is None:
            self._after_updates = dict()
        if key in self.after_updates:
            raise KeyError(f'{key} has been registered in after_updates of {self}')
        self.after_updates[key] = fun

    def _get_before_update(self, key: Any):
        """
        Retrieve a registered before-update function by its key.

        Parameters
        ----------
        key : Any
            The identifier of the before-update function to retrieve.

        Returns
        -------
        Callable
            The registered before-update function.

        Raises
        ------
        KeyError
            If the key is not registered in before_updates or if before_updates is None.
        """
        if self._before_updates is None:
            raise KeyError(f'{key} is not registered in before_updates of {self}')
        if key not in self.before_updates:
            raise KeyError(f'{key} is not registered in before_updates of {self}')
        return self.before_updates.get(key)

    def _get_after_update(self, key: Any):
        """
        Retrieve a registered after-update function by its key.

        Parameters
        ----------
        key : Any
            The identifier of the after-update function to retrieve.

        Returns
        -------
        Callable
            The registered after-update function.

        Raises
        ------
        KeyError
            If the key is not registered in after_updates or if after_updates is None.
        """
        if self._after_updates is None:
            raise KeyError(f'{key} is not registered in after_updates of {self}')
        if key not in self.after_updates:
            raise KeyError(f'{key} is not registered in after_updates of {self}')
        return self.after_updates.get(key)

    def _has_before_update(self, key: Any):
        """
        Check if a before-update function is registered with the given key.

        Parameters
        ----------
        key : Any
            The identifier to check for in the before_updates dictionary.

        Returns
        -------
        bool
            True if the key is registered in before_updates, False otherwise.
        """
        if self._before_updates is None:
            return False
        return key in self.before_updates

    def _has_after_update(self, key: Any):
        """
        Check if an after-update function is registered with the given key.

        Parameters
        ----------
        key : Any
            The identifier to check for in the after_updates dictionary.

        Returns
        -------
        bool
            True if the key is registered in after_updates, False otherwise.
        """
        if self._after_updates is None:
            return False
        return key in self.after_updates

    def __call__(self, *args, **kwargs):
        """
        The shortcut to call ``update`` methods.
        """

        # ``before_updates``
        if self.before_updates is not None:
            for model in self.before_updates.values():
                if hasattr(model, '_receive_update_input'):
                    model(*args, **kwargs)
                else:
                    model()

        # update the model self
        ret = self.update(*args, **kwargs)

        # ``after_updates``
        if self.after_updates is not None:
            for model in self.after_updates.values():
                if hasattr(model, '_not_receive_update_output'):
                    model()
                else:
                    model(ret)
        return ret

    def prefetch(self, item: str) -> 'Prefetch':
        """
        Create a reference to a state or variable that may not be initialized yet.

        This method allows accessing module attributes or states before they are
        fully defined, acting as a placeholder that will be resolved when called.
        Particularly useful for creating references to variables that will be defined
        during initialization or runtime.

        Parameters
        ----------
        item : str
            The name of the attribute or state to reference.

        Returns
        -------
        Prefetch
            A Prefetch object that provides access to the referenced item.

        Examples
        --------
        >>> import brainstate
        >>> import brainunit as u
        >>> neuron = brainstate.nn.LIF(...)
        >>> v_ref = neuron.prefetch('V')  # Reference to voltage
        >>> v_value = v_ref()  # Get current value
        >>> delayed_v = v_ref.delay.at(5.0 * u.ms)  # Get delayed value
        """
        return Prefetch(self, item)

    def align_pre(self, dyn: Union[ParamDescriber[T], T]) -> T:
        """
        Registers a dynamics module to execute after this module.

        This method establishes a sequential execution relationship where the specified
        dynamics module will be called after this module completes its update. This
        creates a feed-forward connection in the computational graph.

        Parameters
        ----------
        dyn : Union[ParamDescriber[T], T]
            The dynamics module to be executed after this module. Can be either:
            - An instance of Dynamics
            - A ParamDescriber that can instantiate a Dynamics object

        Returns
        -------
        T
            The dynamics module that was registered, allowing for method chaining.

        Raises
        ------
        TypeError
            If the input is not a Dynamics instance or a ParamDescriber that creates
            a Dynamics instance.

        Examples
        --------
        >>> import brainstate
        >>> n1 = brainstate.nn.LIF(10)
        >>> n1.align_pre(brainstate.nn.Expon.desc(n1.varshape))  # n2 will run after n1
        """
        if isinstance(dyn, Dynamics):
            self._add_after_update(id(dyn), dyn)
            return dyn
        elif isinstance(dyn, ParamDescriber):
            if not issubclass(dyn.cls, Dynamics):
                raise TypeError(f'The input {dyn} should be an instance of {Dynamics}.')
            if not self._has_after_update(dyn.identifier):
                self._add_after_update(
                    dyn.identifier,
                    dyn() if ('in_size' in dyn.kwargs or len(dyn.args) > 0) else dyn(in_size=self.varshape)
                )
            return self._get_after_update(dyn.identifier)
        else:
            raise TypeError(f'The input {dyn} should be an instance of {Dynamics} or a delayed initializer.')

    def prefetch_delay(self, state: str, delay_time, init: Callable = None) -> 'PrefetchDelayAt':
        """
        Create a reference to a delayed state or variable in the module.

        This method simplifies the process of accessing a delayed version of a state or variable
        within the module. It first creates a prefetch reference to the specified state,
        then specifies the delay time for accessing this state.

        Args:
            state (str): The name of the state or variable to reference.
            delay_time (ArrayLike): The amount of time to delay the variable access,
                typically in time units (e.g., milliseconds).
            init (Callable, optional): An optional initialization function to provide
                a default value if the delayed state is not yet available.

        Returns:
            PrefetchDelayAt: An object that provides access to the variable at the specified delay time.
        """
        return PrefetchDelayAt(self, state, delay_time, init=init)

    def output_delay(self, *delay_time) -> 'OutputDelayAt':
        """
        Create a reference to the delayed output of the module.

        This method simplifies the process of accessing a delayed version of the module's output.
        It instantiates an `OutputDelayAt` object, which can be used to retrieve the output value
        at the specified delay time.

        Args:
            delay (Optional[ArrayLike]): The amount of time to delay the output access,
                typically in time units (e.g., milliseconds). Defaults to None.

        Returns:
            OutputDelayAt: An object that provides access to the module's output at the specified delay time.
        """
        return OutputDelayAt(self, delay_time)


class Prefetch(Node):
    """
    Prefetch a state or variable in a module before it is initialized.


    This class provides a mechanism to reference a module's state or attribute
    that may not have been initialized yet. It acts as a placeholder or reference
    that will be resolved when called.

    Use cases:
    - Access variables within dynamics modules that will be defined later
    - Create references to states across module boundaries
    - Enable access to delayed states through the `.delay` property

    Parameters
    ----------
    module : Module
        The module that contains or will contain the referenced item.
    item : str
        The attribute name of the state or variable to prefetch.

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> neuron = brainstate.nn.LIF(...)
    >>> v_reference = neuron.prefetch('V')  # Reference to voltage before initialization
    >>> v_value = v_reference()  # Get the current value
    >>> delay_ref = v_reference.delay.at(5.0 * u.ms)  # Reference voltage delayed by 5ms

    Notes
    -----
    When called, this class retrieves the current value of the referenced item.
    Use the `.delay` property to access delayed versions of the state.

    """

    def __init__(self, module: Dynamics, item: str):
        """
        Initialize a Prefetch object.

        Parameters
        ----------
        module : Module
            The module that contains or will contain the referenced item.
        item : str
            The attribute name of the state or variable to prefetch.
        """
        super().__init__()
        self.module = module
        self.item = item

    @property
    def delay(self):
        """
        Access delayed versions of the prefetched item.

        Returns
        -------
        PrefetchDelay
            An object that provides access to delayed versions of the prefetched item.
        """
        return PrefetchDelay(self.module, self.item)
        # return PrefetchDelayAt(self.module, self.item, time)

    def __call__(self, *args, **kwargs):
        """
        Get the current value of the prefetched item.

        Returns
        -------
        Any
            The current value of the referenced item. If the item is a State object,
            returns its value attribute, otherwise returns the item itself.
        """
        item = _get_prefetch_item(self)
        return item.value if isinstance(item, State) else item

    def get_item_value(self):
        """
        Get the current value of the prefetched item.

        Similar to __call__, but explicitly named for clarity.

        Returns
        -------
        Any
            The current value of the referenced item. If the item is a State object,
            returns its value attribute, otherwise returns the item itself.
        """
        item = _get_prefetch_item(self)
        return item.value if isinstance(item, State) else item

    def get_item(self):
        """
        Get the referenced item object itself, not its value.

        Returns
        -------
        Any
            The actual referenced item from the module, which could be a State
            object or any other attribute.
        """
        return _get_prefetch_item(self)


class PrefetchDelay(Node):
    """
    Provides access to delayed versions of a prefetched state or variable.

    This class acts as an intermediary for accessing delayed values of module variables.
    It doesn't retrieve values directly but provides methods to specify the delay time
    via the `at()` method.

    Parameters
    ----------
    module : Dynamics
        The dynamics module that contains the referenced state or variable.
    item : str
        The name of the state or variable to access with delay.

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> neuron = brainstate.nn.LIF(10)
    >>> # Access voltage delayed by 5ms
    >>> delayed_v = neuron.prefetch('V').delay.at(5.0 * u.ms)
    >>> delayed_value = delayed_v()  # Get the delayed value
    """

    def __init__(self, module: Dynamics, item: str):
        self.module = module
        self.item = item

    def at(self, *delay_time):
        """
        Specifies the delay time for accessing the variable.

        Parameters
        ----------
        time : ArrayLike
            The amount of time to delay the variable access, typically in time units
            (e.g., milliseconds).

        Returns
        -------
        PrefetchDelayAt
            An object that provides access to the variable at the specified delay time.
        """
        return PrefetchDelayAt(self.module, self.item, delay_time)


class PrefetchDelayAt(Node):
    """
    Provides access to a specific delayed state or variable value at the specific time.

    This class represents the final step in the prefetch delay chain, providing
    actual access to state values at a specific delay time. It converts the
    specified time delay into steps and registers the delay with the appropriate
    StateWithDelay handler.

    Parameters
    ----------
    module : Dynamics
        The dynamics module that contains the referenced state or variable.
    item : str
        The name of the state or variable to access with delay.
    time : ArrayLike
        The amount of time to delay access by, typically in time units (e.g., milliseconds).

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> neuron = brainstate.nn.LIF(10)
    >>> # Create a reference to voltage delayed by 5ms
    >>> delayed_v = PrefetchDelayAt(neuron, 'V', 5.0 * u.ms)
    >>> # Get the delayed value
    >>> v_value = delayed_v()
    """

    def __init__(
        self,
        module: Dynamics,
        item: str,
        delay_time: Tuple,
        init: Callable = None
    ):
        """
        Initialize a PrefetchDelayAt object.

        Parameters
        ----------
        module : Dynamics
            The dynamics module that contains the referenced state or variable.
        item : str
            The name of the state or variable to access with delay.
        delay_time : Tuple
            The amount of time to delay access by, typically in time units (e.g., milliseconds).
        """
        super().__init__()
        assert isinstance(module, Dynamics), 'The module should be an instance of Dynamics.'
        self.module = module
        self.item = item
        if not isinstance(delay_time, (tuple, list)):
            delay_time = (delay_time,)
        self.delay_time = delay_time
        if len(delay_time) > 0:
            key = _get_prefetch_delay_key(item)
            if not module._has_after_update(key):
                module._add_after_update(
                    key,
                    not_receive_update_output(
                        StateWithDelay(module, item, init=init)
                    )
                )
            self.state_delay: StateWithDelay = module._get_after_update(key)
            self.delay_info = self.state_delay.register_delay(*delay_time)

    def __call__(self, *args, **kwargs):
        """
        Retrieve the value of the state at the specified delay time.

        Returns
        -------
        Any
            The value of the state or variable at the specified delay time.
        """
        if len(self.delay_time) == 0:
            return _get_prefetch_item(self).value
        else:
            return self.state_delay.retrieve_at_step(*self.delay_info)


class OutputDelayAt(Node):
    """
    Provides access to a specific delayed state or variable value at the specific time.

    This class represents the final step in the prefetch delay chain, providing
    actual access to state values at a specific delay time. It converts the
    specified time delay into steps and registers the delay with the appropriate
    StateWithDelay handler.

    Parameters
    ----------
    module : Dynamics
        The dynamics module that contains the referenced state or variable.
    time : ArrayLike
        The amount of time to delay access by, typically in time units (e.g., milliseconds).

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> neuron = brainstate.nn.LIF(10)
    >>> # Create a reference to voltage delayed by 5ms
    >>> delayed_spike = OutputDelayAt(neuron, 5.0 * u.ms)
    >>> # Get the delayed value
    >>> v_value = delayed_spike()
    """

    def __init__(
        self,
        module: Dynamics,
        delay_time: Tuple,
    ):
        super().__init__()
        assert isinstance(module, Dynamics), 'The module should be an instance of Dynamics.'
        self.module = module
        key = _get_output_delay_key()
        if not module._has_after_update(key):
            delay = Delay(jax.ShapeDtypeStruct(module.out_size, dtype=environ.dftype()), take_aware_unit=True)
            module._add_after_update(key, receive_update_output(delay))
        self.out_delay: Delay = module._get_after_update(key)
        self.delay_info = self.out_delay.register_delay(*delay_time)

    def __call__(self, *args, **kwargs):
        return self.out_delay.retrieve_at_step(*self.delay_info)


def _get_prefetch_delay_key(item) -> str:
    return f'{item}-prefetch-delay'


def _get_output_delay_key() -> str:
    return f'output-delay'


def _get_prefetch_item(target: Union[Prefetch, PrefetchDelayAt]) -> Any:
    item = getattr(target.module, target.item, None)
    if item is None:
        raise AttributeError(f'The target {target.module} should have an `{target.item}` attribute.')
    return item


def _get_prefetch_item_delay(target: Union[Prefetch, PrefetchDelay, PrefetchDelayAt]) -> Delay:
    assert isinstance(target.module, Dynamics), (
        f'The target module should be an instance '
        f'of Dynamics. But got {target.module}.'
    )
    delay = target.module._get_after_update(_get_prefetch_delay_key(target.item))
    if not isinstance(delay, StateWithDelay):
        raise TypeError(f'The prefetch target should be a {StateWithDelay.__name__} when accessing '
                        f'its delay. But got {delay}.')
    return delay


def maybe_init_prefetch(target, *args, **kwargs):
    """
    Initialize a prefetch target if needed, based on its type.

    This function ensures that prefetch references are properly initialized
    and ready to use. It handles different types of prefetch objects by
    performing the appropriate initialization action:
    - For :py:class:`Prefetch` objects: retrieves the referenced item
    - For :py:class:`PrefetchDelay` objects: retrieves the delay handler
    - For :py:class:`PrefetchDelayAt` objects: registers the specified delay

    Parameters
    ----------
    target : Union[Prefetch, PrefetchDelay, PrefetchDelayAt]
        The prefetch target to initialize.
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    None
        This function performs initialization side effects only.

    Notes
    -----
    This function is typically called internally when prefetched references
    are used to ensure they are properly set up before access.
    """
    if isinstance(target, Prefetch):
        _get_prefetch_item(target)

    elif isinstance(target, PrefetchDelay):
        _get_prefetch_item_delay(target)

    elif isinstance(target, PrefetchDelayAt):
        pass
        # delay = _get_prefetch_item_delay(target)
        # delay.register_delay(*target.delay_time)


class DynamicsGroup(Module):
    """
    A group of :py:class:`~.Module` in which the updating order does not matter.

    Args:
      children_as_tuple: The children objects.
      children_as_dict: The children objects.
    """

    __module__ = 'brainstate.nn'

    if not TYPE_CHECKING:
        def __init__(self, *children_as_tuple, **children_as_dict):
            super().__init__()
            self.layers_tuple = tuple(children_as_tuple)
            self.layers_dict = dict(children_as_dict)

    def update(self, *args, **kwargs):
        """
        Update function of a network.

        In this update function, the update functions in children systems are iteratively called.
        """
        projs, dyns, others = self.nodes(allowed_hierarchy=(1, 1)).split(Projection, Dynamics)

        # update nodes of projections
        for node in projs.values():
            node()

        # update nodes of dynamics
        for node in dyns.values():
            node()

        # update nodes with other types, including delays, ...
        for node in others.values():
            node()


def receive_update_output(cls: object):
    """
    The decorator to mark the object (as the after updates) to receive the output of the update function.

    That is, the `aft_update` will receive the return of the update function::

      ret = model.update(*args, **kwargs)
      for fun in model.aft_updates:
        fun(ret)

    """
    # assert isinstance(cls, Module), 'The input class should be instance of Module.'
    if hasattr(cls, '_not_receive_update_output'):
        delattr(cls, '_not_receive_update_output')
    return cls


def not_receive_update_output(cls: T) -> T:
    """
    The decorator to mark the object (as the after updates) to not receive the output of the update function.

    That is, the `aft_update` will not receive the return of the update function::

      ret = model.update(*args, **kwargs)
      for fun in model.aft_updates:
        fun()

    """
    # assert isinstance(cls, Module), 'The input class should be instance of Module.'
    cls._not_receive_update_output = True
    return cls


def receive_update_input(cls: object):
    """
    The decorator to mark the object (as the before updates) to receive the input of the update function.

    That is, the `bef_update` will receive the input of the update function::


      for fun in model.bef_updates:
        fun(*args, **kwargs)
      model.update(*args, **kwargs)

    """
    # assert isinstance(cls, Module), 'The input class should be instance of Module.'
    cls._receive_update_input = True
    return cls


def not_receive_update_input(cls: object):
    """
    The decorator to mark the object (as the before updates) to not receive the input of the update function.

    That is, the `bef_update` will not receive the input of the update function::

        for fun in model.bef_updates:
          fun()
        model.update()

    """
    # assert isinstance(cls, Module), 'The input class should be instance of Module.'
    if hasattr(cls, '_receive_update_input'):
        delattr(cls, '_receive_update_input')
    return cls


def _input_label_start(label: str):
    # unify the input label repr.
    return f'{label} // '


def _input_label_repr(name: str, label: Optional[str] = None):
    # unify the input label repr.
    return name if label is None else (_input_label_start(label) + str(name))
