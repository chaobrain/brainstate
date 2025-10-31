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


import warnings
from collections.abc import Sequence, Mapping
from typing import Callable, TypeVar, Any

import jax

from brainstate._state import catch_new_states
from brainstate._utils import set_module_as
from brainstate.graph import nodes
from brainstate.transform import vmap, vmap_new_states
from brainstate.typing import Filter
from ._module import Module

# the maximum order
MAX_ORDER = 10

T = TypeVar('T', bound=Module)

__all__ = [
    'call_order',
    'call_all_fns',
    'vmap_call_all_fns',
    'init_all_states',
    'vmap_init_all_states',
    'reset_all_states',
    'vmap_reset_all_states',
    'assign_state_values',
]


@set_module_as('brainstate.nn')
def call_order(
    level: int = 0,
    check_order_boundary: bool = True
) -> Callable[[Callable], Callable]:
    """
    Decorator for specifying the execution order of functions in collective operations.

    This decorator attaches a `call_order` attribute to a function, which is used by
    collective operations like `call_all_functions`, `init_all_states`, and `reset_all_states`
    to determine the execution order. Functions with lower order levels are executed first.

    Parameters
    ----------
    level : int, optional
        The execution order level. Lower values indicate earlier execution.
        Must be in the range [0, MAX_ORDER) when `check_order_boundary` is True.
        Default is 0.
    check_order_boundary : bool, optional
        Whether to validate that the order level is within the valid range [0, MAX_ORDER).
        Default is True.

    Returns
    -------
    Callable[[Callable], Callable]
        A decorator function that adds the `call_order` attribute to the decorated function.

    Raises
    ------
    ValueError
        If `check_order_boundary` is True and `level` is not in [0, MAX_ORDER).

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> class MyModule(brainstate.nn.Module):
        ...     @brainstate.nn.call_order(0)
        ...     def reset_state(self):
        ...         print("Reset first")
        ...
        ...     @brainstate.nn.call_order(1)
        ...     def another_reset(self):
        ...         print("Reset second")
    """
    if check_order_boundary and (level < 0 or level >= MAX_ORDER):
        raise ValueError(f'"level" must be an integer in [0, {MAX_ORDER}), but got {level}.')

    def wrap(fun: Callable) -> Callable:
        fun.call_order = level
        return fun

    return wrap


@set_module_as('brainstate.nn')
def call_all_fns(
    target: T,
    fn_name: str,
    args: Sequence[Any] | Any = (),
    kwargs: Mapping[str, Any] | None = None,
    node_to_exclude: Filter = None,
    fn_if_not_exist: str = 'raise',
) -> T:
    """
    Call a specified function on all module nodes within a target, respecting call order.

    This function traverses all module nodes in the target and invokes the specified method
    on each node. Functions decorated with `@call_order()` are executed in ascending order
    of their level values, while functions without the decorator are executed first.

    Parameters
    ----------
    target : Module
        The target module on which to call functions.
    fn_name : str
        The name of the method to call on each module node.
    node_to_exclude : Filter, optional
        A filter to exclude certain nodes from the function call.
        Can be a type, predicate function, or any filter supported by the graph API.
    fn_if_not_exist : str, optional
        Behavior when the specified method doesn't exist on a node:

        - 'raise': Raise an AttributeError (default)
        - 'pass' or 'none': Skip the node silently
        - 'warn': Issue a warning and skip the node
    args
        Positional arguments to pass to the called method. A single non-tuple
        argument will be automatically wrapped in a tuple. Default is ().
    kwargs
        Keyword arguments to pass to the called method. Default is None.

    Raises
    ------
    TypeError
        If `fun_name` is not a string or `kwargs` is not a mapping.
    ValueError
        If `fn_if_not_exist` is not one of the allowed values.
    AttributeError
        If the specified method doesn't exist on a node and `fn_if_not_exist` is 'raise'.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> net = brainstate.nn.Sequential(brainstate.nn.Linear(10, 20), brainstate.nn.ReLU())
        >>> brainstate.nn.call_all_fns(net, 'init_state')
    """
    if not isinstance(fn_name, str):
        raise TypeError(f'fn_name must be a string, but got {type(fn_name).__name__}.')

    args = (args,) if not isinstance(args, tuple) else args
    kwargs = kwargs or {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(f'kwargs must be a mapping, but got {type(kwargs).__name__}.')

    all_nodes = nodes(target).filter(Module)
    if node_to_exclude is not None:
        all_nodes -= all_nodes.filter(node_to_exclude)

    # Separate nodes with and without call_order
    nodes_with_order = []
    for path, node in all_nodes.items():
        try:
            fun = getattr(node, fn_name)
        except AttributeError as e:
            if fn_if_not_exist == 'raise':
                raise AttributeError(
                    f"Module {type(node).__name__} with the path {path} does not have method '{fn_name}'"
                ) from e
            elif fn_if_not_exist in ('pass', 'none'):
                continue
            elif fn_if_not_exist == 'warn':
                warnings.warn(
                    f"Module {type(node).__name__} with the path {path} does not have method '{fn_name}'. "
                    f"Skipping.",
                    UserWarning
                )
                continue
            else:
                raise ValueError(
                    f"fn_if_not_exist must be one of ['raise', 'pass', 'none'], but got '{fn_if_not_exist}'."
                )

        if not callable(fun):
            raise TypeError(f"'{fn_name}' must be callable, but got {type(fun).__name__}.")

        if hasattr(fun, 'call_order'):
            nodes_with_order.append(node)
        else:
            fun(*args, **kwargs)

    # Execute nodes with call_order in sorted order
    for node in sorted(nodes_with_order, key=lambda x: getattr(x, fn_name).call_order):
        getattr(node, fn_name)(*args, **kwargs)
    return target


def vmap_call_all_fns(
    target: T,
    fn_name: str,
    args: Sequence[Any] | Any = (),
    kwargs: Mapping[str, Any] | None = None,
    axis_size: int = None,
    node_to_exclude: Filter = None,
    state_tag: str | None = None,
    fn_if_not_exist: str = 'raise',
) -> T:
    """
    Apply vectorized mapping to call a function on all module nodes with batched state handling.

    This function creates multiple batched instances by applying vmap to the specified method
    call across all module nodes. Each batch element maintains its own random key and state
    values. This is particularly useful for creating ensembles or batched models.

    Parameters
    ----------
    target : Module
        The target module on which to call functions.
    fn_name : str
        The name of the method to call on each module node.
    args : Sequence[Any] or Any, optional
        Positional arguments to pass to the called method. A single non-tuple
        argument will be automatically wrapped in a tuple. Default is ().
    kwargs : Mapping[str, Any], optional
        Keyword arguments to pass to the called method. Default is None.
    axis_size : int
        The size of the batch dimension for vmap. Must be a positive integer.
    node_to_exclude : Filter, optional
        A filter to exclude certain nodes from the function call.
    state_tag : str, optional
        An optional tag to categorize newly created states during the vmap operation.
    fn_if_not_exist : str, optional
        Behavior when the specified method doesn't exist on a node:

        - 'raise': Raise an AttributeError (default)
        - 'pass' or 'none': Skip the node silently
        - 'warn': Issue a warning and skip the node

    Raises
    ------
    ValueError
        If `axis_size` is None or not a positive integer.
    TypeError
        If `kwargs` is not a mapping.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> net = brainstate.nn.Linear(10, 20)
        >>> # Create 5 batched instances with different initializations
        >>> brainstate.nn.vmap_call_all_fns(net, 'init_state', axis_size=5)
    """

    if axis_size is None or axis_size <= 0:
        raise ValueError(f"axis_size must be a positive integer, got {axis_size}")

    if not isinstance(args, tuple):
        args = (args,)
    kwargs = kwargs or {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(f'kwargs must be a mapping, but got {type(kwargs).__name__}.')

    @vmap(axis_size=axis_size)
    def vmapped_fn():
        with catch_new_states(state_tag) as inner_catcher:
            call_all_fns(
                target,
                fn_name=fn_name,
                args=args,
                kwargs=kwargs,
                node_to_exclude=node_to_exclude,
                fn_if_not_exist=fn_if_not_exist
            )
        return inner_catcher.get_state_values()

    with catch_new_states(state_tag) as outer_catcher:
        values = vmapped_fn()
        states = outer_catcher.get_states()
    for state, value in zip(states, values):
        state.value = value
    return target


@set_module_as('brainstate.nn')
def init_all_states(
    target: T,
    *init_args,
    node_to_exclude: Filter = None,
    **init_kwargs,
) -> T:
    """
    Initialize states for all module nodes within the target.

    This is a convenience wrapper around `call_all_functions` that specifically calls
    the `init_state` method on all module nodes. The execution order respects any
    `@call_order()` decorators on the `init_state` methods.

    Parameters
    ----------
    target : Module
        The target module whose states are to be initialized.
    *init_args
        Variable positional arguments to pass to each `init_state` method.
    node_to_exclude : Filter, optional
        A filter to exclude certain nodes from initialization.
        Can be a type, predicate function, or any filter supported by the graph API.
    **init_kwargs
        Variable keyword arguments to pass to each `init_state` method.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> net = brainstate.nn.Sequential(
        ...     brainstate.nn.Linear(10, 20),
        ...     brainstate.nn.Dropout(0.5)
        ... )
        >>> # Initialize all states
        >>> brainstate.nn.init_all_states(net)
        >>>
        >>> # Initialize with custom arguments
        >>> brainstate.nn.init_all_states(net, batch_size=32)

    See Also
    --------
    call_all_functions : The underlying function that executes the calls.
    vmap_init_all_states : Vectorized version for batched initialization.
    """
    call_all_fns(target, 'init_state', init_args, init_kwargs, node_to_exclude)
    return target


@set_module_as('brainstate.nn')
def vmap_init_all_states(
    target: T,
    *init_args,
    axis_size: int = None,
    node_to_exclude: Filter = None,
    state_to_exclude: Filter = None,
    state_tag: str | None = None,
    **init_kwargs
) -> T:
    """
    Initialize states with vectorized mapping for creating batched module instances.

    This function applies vmap to the initialization process, creating multiple batched
    instances of module states. Each batch element will have independent state values
    and random keys. This is useful for ensemble models or parameter sweeps.

    Parameters
    ----------
    target : Module
        The target module whose states are to be initialized.
    *init_args
        Variable positional arguments to pass to each `init_state` method.
    axis_size : int
        The size of the batch dimension. Must be a positive integer.
    node_to_exclude : Filter, optional
        A filter to exclude certain nodes from initialization.
    state_to_exclude : Filter, optional
        A filter to exclude certain states from being vmapped.
        Excluded states will remain shared across all batched instances.
    state_tag : str, optional
        An optional tag to categorize newly created states.
    **init_kwargs
        Variable keyword arguments to pass to each `init_state` method.

    Raises
    ------
    ValueError
        If `axis_size` is None or not a positive integer.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> net = brainstate.nn.Linear(10, 20)
        >>> # Create 8 batched instances with different random initializations
        >>> brainstate.nn.vmap_init_all_states(net, axis_size=8)
        >>>
        >>> # The weight parameter now has shape (8, 20, 10) instead of (20, 10)
        >>> print(net.weight.shape)

    See Also
    --------
    init_all_states : Non-vectorized version.
    vmap_new_states : The underlying vmap transformation for states.
    """

    # vmap_call_all_functions(
    #     target,
    #     fun_name='init_state',
    #     args=init_args,
    #     kwargs=init_kwargs,
    #     axis_size=axis_size,
    #     node_to_exclude=node_to_exclude,
    #     state_tag=state_tag,
    # )

    def init_fn():
        init_all_states(
            target,
            *init_args,
            **init_kwargs,
            node_to_exclude=node_to_exclude,
        )
        return

    vmap_new_states(init_fn, state_tag=state_tag, axis_size=axis_size, state_to_exclude=state_to_exclude)()
    return target


@set_module_as('brainstate.nn')
def reset_all_states(
    target: T,
    *reset_args,
    node_to_exclude: Filter = None,
    **reset_kwargs,
) -> T:
    """
    Reset states for all module nodes within the target.

    This is a convenience wrapper around `call_all_functions` that specifically calls
    the `reset_state` method on all module nodes. The execution order respects any
    `@call_order()` decorators on the `reset_state` methods. This is typically used
    to reset recurrent neural network states between sequences.

    Parameters
    ----------
    target : Module
        The target module whose states are to be reset.
    reset_args
        Positional arguments to pass to each `reset_state` method.
        A single non-tuple argument will be automatically wrapped in a tuple.
        Default is ().
    reset_kwargs
        Keyword arguments to pass to each `reset_state` method.
        Default is None.
    node_to_exclude : Filter, optional
        A filter to exclude certain nodes from reset.
        Can be a type, predicate function, or any filter supported by the graph API.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> rnn = brainstate.nn.RNNCell(10, 20)
        >>> brainstate.nn.init_all_states(rnn, batch_size=32)
        >>>
        >>> # Process a sequence
        >>> for x in sequence:
        ...     output = rnn(x)
        >>>
        >>> # Reset states before processing next sequence
        >>> brainstate.nn.reset_all_states(rnn)

    See Also
    --------
    call_all_functions : The underlying function that executes the calls.
    vmap_reset_all_states : Vectorized version for batched reset.
    """
    call_all_fns(
        target,
        fn_name='reset_state',
        args=reset_args,
        kwargs=reset_kwargs,
        node_to_exclude=node_to_exclude
    )
    return target


def vmap_reset_all_states(
    target: T,
    *reset_args,
    axis_size: int = None,
    node_to_exclude: Filter = None,
    state_tag: str | None = None,
    **reset_kwargs,
) -> T:
    """
    Reset states with vectorized mapping across batched module instances.

    This function applies vmap to the reset process, resetting states across all
    batched instances of the module. Each batch element will have its state reset
    independently with its own random key. This is useful when working with batched
    recurrent models or ensembles.

    Parameters
    ----------
    target : Module
        The target module whose states are to be reset.
    reset_args
        Positional arguments to pass to each `reset_state` method.
        A single non-tuple argument will be automatically wrapped in a tuple.
        Default is ().
    reset_kwargs
        Keyword arguments to pass to each `reset_state` method.
        Default is None.
    axis_size : int
        The size of the batch dimension. Must be a positive integer.
    node_to_exclude : Filter, optional
        A filter to exclude certain nodes from reset.
    state_tag : str, optional
        An optional tag to categorize newly created states during the reset.

    Raises
    ------
    ValueError
        If `axis_size` is None or not a positive integer.
    TypeError
        If `reset_kwargs` is not a mapping.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> rnn = brainstate.nn.RNNCell(10, 20)
        >>> # Initialize with 16 batched instances
        >>> brainstate.nn.vmap_init_all_states(rnn, batch_size=32, axis_size=16)
        >>>
        >>> # Process sequences...
        >>>
        >>> # Reset all 16 batched instances
        >>> brainstate.nn.vmap_reset_all_states(rnn, axis_size=16)

    See Also
    --------
    reset_all_states : Non-vectorized version.
    vmap_call_all_functions : The underlying vmap function call mechanism.
    """
    vmap_call_all_fns(
        target,
        fn_name='reset_state',
        args=reset_args,
        kwargs=reset_kwargs,
        axis_size=axis_size,
        node_to_exclude=node_to_exclude,
        state_tag=state_tag,
    )
    return target


@set_module_as('brainstate.nn')
def assign_state_values(
    target: Module,
    *state_by_abs_path: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """
    Assign state values to a module from one or more state dictionaries.

    This function updates the state values of a module based on provided state dictionaries.
    State dictionaries should use absolute paths as keys (e.g., 'layer1.weight', 'layer2.bias').
    The function handles missing and unexpected keys, returning them for inspection.

    Parameters
    ----------
    target : Module
        The target module whose states will be updated.
    *state_by_abs_path : Mapping[str, Any]
        One or more state dictionaries with absolute path keys mapping to state values.
        If multiple dictionaries are provided, they will be merged (later dictionaries
        override earlier ones for duplicate keys).

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple of (unexpected_keys, missing_keys):

        - unexpected_keys: Keys present in the state dictionaries but not in the module
        - missing_keys: Keys present in the module but not in the state dictionaries

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> net = brainstate.nn.Linear(10, 20)
        >>> brainstate.nn.init_all_states(net)
        >>>
        >>> # Save state values
        >>> state_dict = {path: state.value for path, state in net.states().items()}
        >>>
        >>> # Later, restore state values
        >>> unexpected, missing = brainstate.nn.assign_state_values(net, state_dict)
        >>> print(f"Unexpected keys: {unexpected}")
        >>> print(f"Missing keys: {missing}")

    Notes
    -----
    - All values are automatically converted to JAX arrays using `jax.numpy.asarray`.
    - Only states with matching keys are updated; unexpected and missing keys are
      returned but do not cause errors.
    - If multiple dictionaries contain the same key, the last one takes precedence.
    """
    # Merge all state dictionaries
    all_states = {}
    for state_dict in state_by_abs_path:
        all_states.update(state_dict)

    # Get current module states
    variables = target.states()
    keys1 = set(all_states.keys())
    keys2 = set(variables.keys())

    # Update matching states
    for key in keys2.intersection(keys1):
        variables[key].value = jax.numpy.asarray(all_states[key])

    # Return mismatched keys
    unexpected_keys = sorted(keys1 - keys2)
    missing_keys = sorted(keys2 - keys1)
    return unexpected_keys, missing_keys
