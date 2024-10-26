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

from collections import namedtuple
from typing import Dict, Callable, TypeVar

import jax

from brainstate._utils import set_module_as
from ._module import Module

# the maximum order
MAX_ORDER = 10

# State Load Results
StateLoadResult = namedtuple('StateLoadResult', ['missing_keys', 'unexpected_keys'])

T = TypeVar('T', bound=Module)

__all__ = [
  'MAX_ORDER', 'call_order', 'init_all_states', 'reset_all_states',
  'load_all_states', 'save_all_states', 'assign_state_values',
]


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
  if check_order_boundary and (level < 0 or level >= MAX_ORDER):
    raise ValueError(f'"call_order" must be an integer in [0, {MAX_ORDER}). but we got {level}.')

  def wrap(fun: Callable):
    fun.call_order = level
    return fun

  return wrap


@set_module_as('brainstate.nn')
def init_all_states(target: T, *args, exclude=None, **kwargs) -> T:
  """
  Collectively initialize states of all children nodes in the given target.

  Args:
    target: The target Module.

  Returns:
    The target Module.
  """
  nodes_with_order = []

  nodes = target.nodes(Module)
  if exclude is not None:
    nodes = nodes - nodes.filter(exclude)

  # reset node whose `init_state` has no `call_order`
  for node in list(nodes.values()):
    if hasattr(node.init_state, 'call_order'):
      nodes_with_order.append(node)
    else:
      node.init_state(*args, **kwargs)

  # reset the node's states
  for node in sorted(nodes_with_order, key=lambda x: x.init_state.call_order):
    node.init_state(*args, **kwargs)

  return target


@set_module_as('brainstate.nn')
def reset_all_states(target: Module, *args, **kwargs) -> Module:
  """
  Collectively reset states of all children nodes in the given target.

  Args:
    target: The target Module.

  Returns:
    The target Module.
  """
  nodes_with_order = []

  # reset node whose `init_state` has no `call_order`
  for node in list(target.nodes(Module).values()):
    if hasattr(node.reset_state, 'call_order'):
      nodes_with_order.append(node)
    else:
      node.reset_state(*args, **kwargs)

  # reset the node's states
  for node in sorted(nodes_with_order, key=lambda x: x.reset_state.call_order):
    node.reset_state(*args, **kwargs)

  return target


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
  for name, node in target.nodes().items():
    r = node.load_state(state_dict[name], **kwargs)
    if r is not None:
      missing, unexpected = r
      missing_keys.extend([f'{name}.{key}' for key in missing])
      unexpected_keys.extend([f'{name}.{key}' for key in unexpected])
  return StateLoadResult(missing_keys, unexpected_keys)


@set_module_as('brainstate.nn')
def save_all_states(target: Module, **kwargs) -> Dict:
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
