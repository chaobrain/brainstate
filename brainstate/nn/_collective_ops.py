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

from collections import namedtuple
from typing import Callable, TypeVar, Tuple, Any, Dict

import jax

from brainstate._state import catch_new_states
from brainstate._utils import set_module_as
from brainstate.augment import vmap, vmap_new_states
from brainstate.graph import nodes
from brainstate.random import set_key, split_key
from brainstate.typing import Filter
from ._module import Module

# the maximum order
MAX_ORDER = 10

# State Load Results
StateLoadResult = namedtuple('StateLoadResult', ['missing_keys', 'unexpected_keys'])

T = TypeVar('T', bound=Module)

__all__ = [
    'MAX_ORDER',
    'call_order',
    'call_all_functions',
    'vmap_call_all_functions',
    'init_all_states',
    'vmap_init_all_states',
    'reset_all_states',
    'load_all_states',
    'save_all_states',
    'assign_state_values',
]


@set_module_as('brainstate.nn')
def call_order(level: int = 0, check_order_boundary: bool = True):
    """The decorator for indicating the resetting level.

    The function takes an optional integer argument level with a default value of 0.

    The lower the level, the earlier the function is called.

    >>> import brainstate as brainstate
    >>> brainstate.nn.call_order(0)
    >>> brainstate.nn.call_order(-1)
    >>> brainstate.nn.call_order(-2)

    Parameters
    ----------
    level: int
      The call order level.
    check_order_boundary: bool
      Whether check the boundary of function call order. If True,
      the order that not in [0,  10) will raise a ValueError.

    Returns
    -------
    The function to warp.
    """
    if check_order_boundary and (level < 0 or level >= MAX_ORDER):
        raise ValueError(f'"call_order" must be an integer in [0, {MAX_ORDER}). but we got {level}.')

    def wrap(fun: Callable):
        fun.call_order = level
        return fun

    return wrap


@set_module_as('brainstate.nn')
def call_all_functions(
    target: T,
    fun_name: str,
    args: Tuple[Any, ...] | Any = (),
    kwargs: Dict[str, Any] | None = None,
    node_to_exclude: Filter = None,
    fun_if_not_exist: str = 'raise',
) -> T:
    """
    Call a specified function on all nodes of a target module, respecting call order if defined.

    This function iterates through all nodes of the target module, calling a specified function
    on each node. It respects the call order of functions if defined, and provides options for
    handling cases where the specified function does not exist on a node.

    Parameters
    -----------
    target : T
        The target module on which to call functions.
    fun_name : str
        The name of the function to call on each node.
    args : Tuple[Any, ...] | Any, optional
        Positional arguments to pass to the called function. Default is an empty tuple.
    kwargs : Dict[str, Any] | None, optional
        Keyword arguments to pass to the called function. Default is None.
    node_to_exclude : Filter, optional
        A filter function to exclude certain nodes from the function call.
    fun_if_not_exist : str, optional
        Specifies behavior when the function doesn't exist on a node. Options are:

        - 'raise': Raise an exception (default)
        - 'pass' or 'none': Skip the node and continue
        
    Returns
    --------
    T
        The target module after calling the specified function on all applicable nodes.

    Raises
    -------
    AssertionError
        If fun_name is not a string or kwargs is not a dictionary.
    ValueError
        If fun_if_not_exist is not one of the allowed values.
    AttributeError
        If the specified function doesn't exist on a node and fun_if_not_exist is 'raise'.
    """
    assert isinstance(fun_name, str), f'fun_name must be a string, but got {fun_name}.'

    args = (args,) if not isinstance(args, tuple) else args
    kwargs = kwargs or {}
    assert isinstance(kwargs, dict), f'kwargs must be a dict, but got {kwargs}.'

    all_nodes = nodes(target).filter(Module)
    if node_to_exclude is not None:
        all_nodes -= all_nodes.filter(node_to_exclude)

    nodes_with_order = []
    for node in all_nodes.values():
        try:
            fun = getattr(node, fun_name)
        except AttributeError as e:
            if fun_if_not_exist == 'raise':
                raise
            elif fun_if_not_exist in ('pass', 'none'):
                continue
            else:
                raise ValueError(
                    f'fun_if_not_exist must be one of ["raise", "pass", "none"], but got {fun_if_not_exist}.')

        assert callable(fun), f'{fun_name} must be a callable function, but got {fun}.'
        if hasattr(fun, 'call_order'):
            nodes_with_order.append(node)
        else:
            fun(*args, **kwargs)

    for node in sorted(nodes_with_order, key=lambda x: getattr(x, fun_name).call_order):
        getattr(node, fun_name)(*args, **kwargs)

    return target


def vmap_call_all_functions(
    target: T,
    fun_name: str,
    args: Tuple[Any, ...] | Any = (),
    kwargs: Dict[str, Any] | None = None,
    axis_size: int = None,
    node_to_exclude: Filter = None,
    tag: str | None = None,
    fun_if_not_exist: str = 'raise',
) -> T:
    """
    Apply vectorized mapping (vmap) to call a specified function on all nodes of a target module.

    This function vectorizes the process of calling a specified function across multiple instances
    of the target module, effectively batching the operation.

    Parameters
    -----------
    target : T
        The target module on which to call functions.
    fun_name : str
        The name of the function to call on each node.
    args : Tuple[Any, ...] | Any, optional
        Positional arguments to pass to the called function. Default is an empty tuple.
    kwargs : Dict[str, Any] | None, optional
        Keyword arguments to pass to the called function. Default is None.
    axis_size : int, optional
        The size of the batch axis for vmap. Must be a positive integer.
    node_to_exclude : Filter, optional
        A filter function to exclude certain nodes from the function call.
    tag : str | None, optional
        A tag to be used for catching new states.
    fun_if_not_exist : str, optional
        Specifies behavior when the function doesn't exist on a node. Options are:

        - 'raise': Raise an exception (default)
        - 'pass' or 'none': Skip the node and continue

    Returns
    --------
    T
        The target module after applying the vectorized function call on all applicable nodes.

    Raises
    -------
    AssertionError
        If axis_size is not specified or is not a positive integer.
    """
    assert axis_size is not None and axis_size > 0, f"axis_size must be a positive integer, got {axis_size}"

    if not isinstance(args, tuple):
        args = (args,)
    kwargs = kwargs or {}
    assert isinstance(kwargs, dict), f'kwargs must be a dict, but got {kwargs}.'

    @vmap(out_axes=0, axis_size=axis_size)
    def vmapped_fn(key):
        set_key(key)
        with catch_new_states(tag) as inner_catcher:
            call_all_functions(
                target,
                fun_name=fun_name,
                args=args,
                kwargs=kwargs,
                node_to_exclude=node_to_exclude,
                fun_if_not_exist=fun_if_not_exist
            )
        values = inner_catcher.get_state_values()
        return values

    with catch_new_states(tag) as outer_catcher:
        values = vmapped_fn(split_key(axis_size))
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
    Initialize all states for the given target module and its submodules.

    This function initializes the states of the target module and all its submodules,
    respecting any call order decorators that may be present on the init_state methods.

    Parameters
    ----------
    target : T
        The target module whose states are to be initialized.
    init_args : Tuple[Any, ...] | Any, optional
        Positional arguments to be passed to each init_state method.
        If a single non-tuple argument is provided, it will be wrapped in a tuple.
    init_kwargs : Dict[str, Any] | None, optional
        Keyword arguments to be passed to each init_state method.
        If None, an empty dictionary will be used.
    node_to_exclude : Filter, optional
        A filter function or predicate to exclude certain nodes from initialization.

    Returns
    -------
    T
        The target module with all states initialized.

    Raises
    ------
    AssertionError
        If init_kwargs is provided but is not a dictionary.
    """
    return call_all_functions(target, 'init_state', init_args, init_kwargs, node_to_exclude)


@set_module_as('brainstate.nn')
def vmap_init_all_states(
    target: T,
    *init_args: Tuple[Any, ...] | Any,
    axis_size: int = None,
    node_to_exclude: Filter = None,
    state_to_exclude: Filter = None,
    state_tag: str | None = None,
    **init_kwargs: Dict[str, Any] | None
) -> T:
    """
    Initialize all vmap states for the given target module.

    This function applies vectorized mapping (vmap) to initialize states across multiple
    instances of the target module, effectively batching the initialization process.

    Parameters
    -----------
    target : T
        The target module whose states are to be initialized.
    init_args : Tuple[Any, ...] | Any, optional
        Positional arguments to be passed to the init_all_states function. Default is an empty tuple.
    init_kwargs : Dict[str, Any] | None, optional
        Keyword arguments to be passed to the init_all_states function. Default is None.
    axis_size : int, optional
        The size of the batch axis for vmap. This must be specified and should be greater than 0.
    node_to_exclude : Filter, optional
        A filter to exclude certain nodes from initialization.
    state_tag : str | None, optional
        A tag to be used for catching new states.

    Returns
    --------
    T
        The target module with initialized states.

    Raises
    -------
    AssertionError
        If axis_size is not specified or is not greater than 0.
        If init_kwargs is not a dictionary.
    """

    # return vmap_call_all_functions(
    #     target,
    #     'init_state',
    #     args=init_args,
    #     kwargs=init_kwargs,
    #     axis_size=axis_size,
    #     node_to_exclude=node_to_exclude,
    #     tag=tag,
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
    reset_args: Tuple[Any, ...] | Any = (),
    reset_kwargs: Dict[str, Any] | None = None,
    node_to_exclude: Filter = None,
) -> T:
    """
    Reset all states for the given target module and its submodules.

    This function resets the states of the target module and all its submodules,
    respecting any call order decorators that may be present on the reset_state methods.

    Parameters
    ----------
    target : T
        The target module whose states are to be reset.
    reset_args : Tuple[Any, ...] | Any, optional
        Positional arguments to be passed to each reset_state method.
        If a single non-tuple argument is provided, it will be wrapped in a tuple.
    reset_kwargs : Dict[str, Any] | None, optional
        Keyword arguments to be passed to each reset_state method.
        If None, an empty dictionary will be used.
    node_to_exclude : Filter, optional
        A filter function or predicate to exclude certain nodes from reset.

    Returns
    -------
    T
        The target module with all states reset.

    Raises
    ------
    AssertionError
        If init_kwargs is provided but is not a dictionary.
    """
    return call_all_functions(
        target,
        fun_name='reset_state',
        args=reset_args,
        kwargs=reset_kwargs,
        node_to_exclude=node_to_exclude
    )


def vmap_reset_all_states(
    target: T,
    reset_args: Tuple[Any, ...] | Any = (),
    reset_kwargs: Dict[str, Any] | None = None,
    axis_size: int = None,
    node_to_exclude: Filter = None,
    tag: str | None = None,
) -> T:
    """
    Reset all vmap states for the given target module.

    This function applies vectorized mapping (vmap) to reset states across multiple
    instances of the target module, effectively batching the reset process.

    Parameters
    -----------
    target : T
        The target module whose states are to be reset.
    reset_args : Tuple[Any, ...] | Any, optional
        Positional arguments to be passed to the reset_all_states function. Default is an empty tuple.
    reset_kwargs : Dict[str, Any] | None, optional
        Keyword arguments to be passed to the reset_all_states function. Default is None.
    axis_size : int, optional
        The size of the batch axis for vmap. This must be specified and should be greater than 0.
    node_to_exclude : Filter, optional
        A filter to exclude certain nodes from reset.
    tag : str | None, optional
        A tag to be used for catching new states.

    Returns
    --------
    T
        The target module with reset states.

    Raises
    -------
    AssertionError
        If axis_size is not specified or is not greater than 0.
        If reset_kwargs is not a dictionary.
    """
    return vmap_call_all_functions(
        target,
        fun_name='reset_state',
        args=reset_args,
        kwargs=reset_kwargs,
        axis_size=axis_size,
        node_to_exclude=node_to_exclude,
        tag=tag,
    )


@set_module_as('brainstate.nn')
def load_all_states(target: Module, state_dict: Dict, **kwargs):
    """
    Copy parameters and buffers from :attr:`state_dict` into
    this module and its descendants.

    Args:
      target: Module. The dynamical system to load its states.
      state_dict: dict. A dict containing parameters and persistent buffers.

    Returns
    -------
      ``NamedTuple``  with ``missing_keys`` and ``unexpected_keys`` fields:

      * **missing_keys** is a list of str containing the missing keys
      * **unexpected_keys** is a list of str containing the unexpected keys
    """
    missing_keys = []
    unexpected_keys = []
    for path, node in nodes(target).items():
        r = node.load_state(state_dict[path], **kwargs)
        if r is not None:
            missing, unexpected = r
            missing_keys.extend([f'{path}.{key}' for key in missing])
            unexpected_keys.extend([f'{path}.{key}' for key in unexpected])
    return StateLoadResult(missing_keys, unexpected_keys)


@set_module_as('brainstate.nn')
def save_all_states(target: Module, **kwargs) -> Dict:
    """
    Save all states in the ``target`` as a dictionary for later disk serialization.

    Args:
      target: Module. The node to save its states.

    Returns
      Dict. The state dict for serialization.
    """
    return {key: node.save_state(**kwargs) for key, node in target.nodes().items()}


@set_module_as('brainstate.nn')
def assign_state_values(target: Module, *state_by_abs_path: Dict):
    """
    Assign state values according to the given state dictionary.

    Parameters
    ----------
    target: Module
      The target module.
    state_by_abs_path: dict
      The state dictionary which is accessed by the "absolute" accessing method.

    """
    all_states = dict()
    for state in state_by_abs_path:
        all_states.update(state)
    variables = target.states()
    keys1 = set(all_states.keys())
    keys2 = set(variables.keys())
    for key in keys2.intersection(keys1):
        variables[key].value = jax.numpy.asarray(all_states[key])
    unexpected_keys = list(keys1 - keys2)
    missing_keys = list(keys2 - keys1)
    return unexpected_keys, missing_keys
