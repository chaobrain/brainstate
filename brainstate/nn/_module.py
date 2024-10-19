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
All the basic classes for neural networks in ``brainstate``.

The basic classes include:

- ``Module``: The base class for all the objects in the ecosystem.
- ``Sequential``: The class for a sequential of modules, which update the modules sequentially.

"""

import math
import warnings
from collections import namedtuple
from typing import Dict, Callable, Hashable, Sequence, Optional, Tuple, Union

import jax
import numpy as np

from brainstate import environ
from brainstate._state import State
from brainstate._utils import set_module_as
from brainstate.graph import Node, states, nodes
from brainstate.mixin import DelayedInitializer
from brainstate.util import FlattedMapping
from brainstate.typing import PathParts

# maximum integer
max_int = np.iinfo(np.int32).max

# the maximum order
_max_order = 10

# State Load Results
StateLoadResult = namedtuple('StateLoadResult', ['missing_keys', 'unexpected_keys'])

__all__ = [
  'Module', 'ElementWiseBlock', 'Sequential',
  'call_order', 'all_init_states', 'all_reset_states', 'all_load_states', 'all_save_states', 'assign_state_values',
]


class Module(Node):
  """
  The Module class for the whole ecosystem.

  The ``Module`` is the base class for all the objects in the ecosystem. It
  provides the basic functionalities for the objects, including:

  - ``states()``: Collect all states in this node and the children nodes.
  - ``nodes()``: Collect all children nodes.
  - ``update()``: The function to specify the updating rule.
  - ``init_state()``: State initialization function.
  - ``reset_state()``: State resetting function.

  """

  __module__ = 'brainstate.nn'

  _in_size: Optional[Tuple[int, ...]]
  _out_size: Optional[Tuple[int, ...]]
  _name: Optional[str]

  def __init__(self, name: str = None):
    super().__init__()

    # check the name
    if name is not None:
      assert isinstance(name, str), f'The name must be a string, but we got {type(name)}: {name}'
    self._name = name

    # input and output size
    self._in_size = None
    self._out_size = None

  @property
  def name(self):
    """Name of the model."""
    return self._name

  @name.setter
  def name(self, name: str = None):
    raise AttributeError('The name of the model is read-only.')

  @property
  def in_size(self) -> Tuple[int, ...]:
    return self._in_size

  @in_size.setter
  def in_size(self, in_size: Sequence[int] | int):
    if isinstance(in_size, int):
      in_size = (in_size,)
    assert isinstance(in_size, (tuple, list)), f"Invalid type of in_size: {type(in_size)}"
    self._in_size = tuple(in_size)

  @property
  def out_size(self) -> Tuple[int, ...]:
    return self._out_size

  @out_size.setter
  def out_size(self, out_size: Sequence[int] | int):
    if isinstance(out_size, int):
      out_size = (out_size,)
    assert isinstance(out_size, (tuple, list)), f"Invalid type of out_size: {type(out_size)}"
    self._out_size = tuple(out_size)

  def update(self, *args, **kwargs):
    """
    The function to specify the updating rule.
    """
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} must implement "update" function.')

  def __call__(self, *args, **kwargs):
    return self.update(*args, **kwargs)

  def __rrshift__(self, other):
    """
    Support using right shift operator to call modules.

    Examples
    --------

    >>> import brainstate as bst
    >>> x = bst.random.rand((10, 10))
    >>> l = bst.nn.Dropout(0.5)
    >>> y = x >> l
    """
    return self.__call__(other)

  def states(
      self,
      *filters,
      allowed_hierarchy: Tuple[int, int] = (0, max_int),
      level: int = None,
  ) -> FlattedMapping[PathParts, State] | Tuple[FlattedMapping[PathParts, State], ...]:
    """
    Collect all states in this node and the children nodes.

    Parameters
    ----------
    filters : tuple
      The filters to select the states.
    allowed_hierarchy : tuple of int
      The hierarchy of the states to be collected.
    level : int
      The level of the states to be collected. Has been deprecated.

    Returns
    -------
    states : FlattedMapping, tuple of FlattedMapping
      The collection contained (the path, the state).
    """
    if level is not None:
      allowed_hierarchy = (0, level)
      warnings.warn('The "level" argument is deprecated. Please use "allowed_hierarchy" instead.', DeprecationWarning)

    return states(self, *filters, allowed_hierarchy=allowed_hierarchy)

  def nodes(
      self,
      *filters,
      allowed_hierarchy: Tuple[int, int] = (0, max_int),
      level: int = None,
  ) -> FlattedMapping[PathParts, Node] | Tuple[FlattedMapping[PathParts, Node], ...]:
    """
    Collect all children nodes.

    Parameters
    ----------
    filters : tuple
      The filters to select the states.
    allowed_hierarchy : tuple of int
      The hierarchy of the states to be collected.
    level : int
      The level of the states to be collected. Has been deprecated.

    Returns
    -------
    nodes : FlattedMapping, tuple of FlattedMapping
      The collection contained (the path, the node).
    """
    if level is not None:
      allowed_hierarchy = (0, level)
      warnings.warn('The "level" argument is deprecated. Please use "allowed_hierarchy" instead.', DeprecationWarning)

    return nodes(self, *filters, allowed_hierarchy=allowed_hierarchy)

  def init_state(self, *args, **kwargs):
    """
    State initialization function.
    """
    pass

  def reset_state(self, *args, **kwargs):
    """
    State resetting function.
    """
    pass

  def __leaf_fn__(self, name, value):
    if name in ['_in_size', '_out_size', '_name']:
      return (name, value) if value is None else (name[1:], value)  # skip the first `_`
    return name, value


class ElementWiseBlock(Module):
  __module__ = 'brainstate.nn'


@set_module_as('brainstate.nn')
def call_order(level: int = 0, check_order_boundary: bool = True):
  """The decorator for indicating the resetting level.

  The function takes an optional integer argument level with a default value of 0.

  The lower the level, the earlier the function is called.

  >>> import brainstate as bst
  >>> bst.nn.call_order(0)
  >>> bst.nn.call_order(-1)
  >>> bst.nn.call_order(-2)

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
  if check_order_boundary and (level < 0 or level >= _max_order):
    raise ValueError(f'"call_order" must be an integer in [0, {_max_order}). but we got {level}.')

  def wrap(fun: Callable):
    fun.call_order = level
    return fun

  return wrap


@set_module_as('brainstate.nn')
def all_init_states(target: Module, *args, **kwargs) -> Module:
  """
  Collectively initialize states of all children nodes in the given target.

  Args:
    target: The target Module.

  Returns:
    The target Module.
  """
  nodes_with_order = []

  # reset node whose `init_state` has no `call_order`
  for node in list(target.nodes().values()):
    if hasattr(node.init_state, 'call_order'):
      nodes_with_order.append(node)
    else:
      node.init_state(*args, **kwargs)

  # reset the node's states
  for node in sorted(nodes_with_order, key=lambda x: x.init_state.call_order):
    node.init_state(*args, **kwargs)

  return target


@set_module_as('brainstate.nn')
def all_reset_states(target: Module, *args, **kwargs) -> Module:
  """
  Collectively reset states of all children nodes in the given target.

  Args:
    target: The target Module.

  Returns:
    The target Module.
  """
  nodes_with_order = []

  # reset node whose `init_state` has no `call_order`
  for node in list(target.nodes().values()):
    if hasattr(node.reset_state, 'call_order'):
      nodes_with_order.append(node)
    else:
      node.reset_state(*args, **kwargs)

  # reset the node's states
  for node in sorted(nodes_with_order, key=lambda x: x.reset_state.call_order):
    node.reset_state(*args, **kwargs)

  return target


@set_module_as('brainstate.nn')
def all_load_states(target: Module, state_dict: Dict, **kwargs):
  """
  Copy parameters and buffers from :attr:`state_dict` into
  this module and its descendants.

  Args:
    target: Module. The dynamical system to load its states.
    state_dict: dict. A dict containing parameters and persistent buffers.

  Returns:
  -------
    ``NamedTuple``  with ``missing_keys`` and ``unexpected_keys`` fields:

    * **missing_keys** is a list of str containing the missing keys
    * **unexpected_keys** is a list of str containing the unexpected keys
  """
  missing_keys = []
  unexpected_keys = []
  for name, node in target.nodes().items():
    r = node.load_state(state_dict[name], **kwargs)
    if r is not None:
      missing, unexpected = r
      missing_keys.extend([f'{name}.{key}' for key in missing])
      unexpected_keys.extend([f'{name}.{key}' for key in unexpected])
  return StateLoadResult(missing_keys, unexpected_keys)


@set_module_as('brainstate.nn')
def all_save_states(target: Module, **kwargs) -> Dict:
  """
  Save all states in the ``target`` as a dictionary for later disk serialization.

  Args:
    target: Module. The node to save its states.

  Returns:
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


def _input_label_start(label: str):
  # unify the input label repr.
  return f'{label} // '


def _input_label_repr(name: str, label: Optional[str] = None):
  # unify the input label repr.
  return name if label is None else (_input_label_start(label) + str(name))


def _repr_context(repr_str, indent):
  splits = repr_str.split('\n')
  splits = [(s if i == 0 else (indent + s)) for i, s in enumerate(splits)]
  return '\n'.join(splits)


def _get_delay(delay_time, delay_step):
  if delay_time is None:
    if delay_step is None:
      return 0., 0
    else:
      assert isinstance(delay_step, int), '"delay_step" should be an integer.'
      if delay_step == 0:
        return 0., 0
      delay_time = delay_step * environ.get_dt()
  else:
    assert delay_step is None, '"delay_step" should be None if "delay_time" is given.'
    # assert isinstance(delay_time, (int, float))
    delay_step = math.ceil(delay_time / environ.get_dt())
  return delay_time, delay_step


class Sequential(Module):
  """
  A sequential `input-output` module.

  Modules will be added to it in the order they are passed in the
  constructor. Alternatively, an ``dict`` of modules can be
  passed in. The ``update()`` method of ``Sequential`` accepts any
  input and forwards it to the first module it contains. It then
  "chains" outputs to inputs sequentially for each subsequent module,
  finally returning the output of the last module.

  The value a ``Sequential`` provides over manually calling a sequence
  of modules is that it allows treating the whole container as a
  single module, such that performing a transformation on the
  ``Sequential`` applies to each of the modules it stores (which are
  each a registered submodule of the ``Sequential``).

  What's the difference between a ``Sequential`` and a
  :py:class:`Container`? A ``Container`` is exactly what it
  sounds like--a container to store :py:class:`DynamicalSystem` s!
  On the other hand, the layers in a ``Sequential`` are connected
  in a cascading way.

  Examples
  --------

  >>> import jax
  >>> import brainstate as bst
  >>> import brainstate.nn as nn
  >>>
  >>> # composing ANN models
  >>> l = nn.Sequential(nn.Linear(100, 10),
  >>>                   jax.nn.relu,
  >>>                   nn.Linear(10, 2))
  >>> l(bst.random.random((256, 100)))

  Args:
    modules_as_tuple: The children modules.
    modules_as_dict: The children modules.
    name: The object name.
  """
  __module__ = 'brainstate.nn'

  def __init__(self, first: Module, *layers):
    super().__init__()
    self.layers = []

    # add all modules
    assert isinstance(first, Module), 'The first module should be an instance of Module.'
    in_size = first.out_size
    self.layers.append(first)
    for module in layers:
      module, in_size = _format_module(module, in_size)
      self.layers.append(module)

    # the input and output shape
    if first.in_size is not None:
      self.in_size = first.in_size
    self.out_size = tuple(in_size)

  def update(self, x):
    """Update function of a sequential model.
    """
    for m in self.layers:
      x = m(x)
    return x

  def __getitem__(self, key: Union[int, slice]):
    if isinstance(key, slice):
      return Sequential(*self.layers[key])
    elif isinstance(key, int):
      return self.layers[key]
    elif isinstance(key, (tuple, list)):
      return Sequential(*[self.layers[k] for k in key])
    else:
      raise KeyError(f'Unknown type of key: {type(key)}')


def _format_module(module, in_size):
  if isinstance(module, DelayedInitializer):
    module = module(in_size=in_size)
    assert isinstance(module, Module), 'The module should be an instance of Module.'
    out_size = module.out_size
  elif isinstance(module, ElementWiseBlock):
    out_size = in_size
  elif isinstance(module, Module):
    out_size = module.out_size
  else:
    raise TypeError(f"Unsupported type {type(module)}. ")
  return module, out_size
