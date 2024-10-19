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

- ``ModuleGroup``: The class for a group of modules, which update ``Projection`` first,
                   then ``Dynamics``, finally others.
- ``Projection``: The class for the synaptic projection.
- ``Dynamics``: The class for the dynamical system.

For handling the delays:

- ``Delay``: The class for all delays.
- ``DelayAccess``: The class for the delay access.

"""

from typing import Any, Dict, Callable, Hashable, Optional, Union, TypeVar

import numpy as np

from brainstate._state import State
from brainstate.mixin import DelayedInitializer
from brainstate.nn._module import Module
from brainstate.typing import Size, ArrayLike
from ._state_delay import StateWithDelay, Delay

__all__ = [
  'ModuleGroup', 'Projection', 'Dynamics',
]

T = TypeVar('T')
_max_order = 10


class Projection(Module):
  """
  Base class to model synaptic projections.
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
  Base class to model dynamics.

  .. note::
     In general, every instance of :py:class:`~.Module` implemented in
     BrainPy only defines the evolving function at each time step :math:`t`.

     If users want to define the logic of running models across multiple steps,
     we recommend users to use :py:func:`~.for_loop`, :py:class:`~.LoopOverTime`,
     :py:class:`~.DSRunner`, or :py:class:`~.DSTrainer`.

     To be compatible with previous APIs, :py:class:`~.Module` inherits
     from the :py:class:`~.DelayRegister`. It's worthy to note that the methods of
     :py:class:`~.DelayRegister` will be removed in the future, including:

     - ``.register_delay()``
     - ``.get_delay_data()``
     - ``.update_local_delays()``
     - ``.reset_local_delays()``


  There are several essential attributes:

  - ``size``: the geometry of the neuron group. For example, `(10, )` denotes a line of
    neurons, `(10, 10)` denotes a neuron group aligned in a 2D space, `(10, 15, 4)` denotes
    a 3-dimensional neuron group.
  - ``num``: the flattened number of neurons in the group. For example, `size=(10, )` => \
    `num=10`, `size=(10, 10)` => `num=100`, `size=(10, 15, 4)` => `num=600`.

  Args:
    size: The neuron group geometry.
    name: The name of the dynamic system.
    keep_size: Whether keep the geometry information.
    mode: The computing mode.
  """

  __module__ = 'brainstate.nn'

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
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
  ):
    # initialize
    super().__init__(name=name)

    # geometry size of neuron population
    if isinstance(size, (list, tuple)):
      if len(size) <= 0:
        raise ValueError(f'"size" must be int, or a tuple/list of int. But we got {type(size)}')
      if not isinstance(size[0], (int, np.integer)):
        raise ValueError(f'"size" must be int, or a tuple/list of int. But we got {type(size)}')
      size = tuple(size)
    elif isinstance(size, (int, np.integer)):
      size = (size,)
    else:
      raise ValueError(f'"size" must be int, or a tuple/list of int. But we got {type(size)}')
    self.size = size
    self.keep_size = keep_size

    # number of neurons
    self.num = np.prod(size)

    # current inputs
    self._current_inputs = None

    # delta inputs
    self._delta_inputs = None

    # before updates
    self._before_updates = None

    # after updates
    self._after_updates = None

    # in-/out- size of neuron population
    self.in_size = self.size if self.keep_size else (self.num,)
    self.out_size = self.size if self.keep_size else (self.num,)

  @property
  def varshape(self):
    """The shape of variables in the neuron group."""
    return self.in_size

  @property
  def current_inputs(self):
    """
    The current inputs of the model. It should be a dictionary of the input data.
    """
    return self._current_inputs

  @property
  def delta_inputs(self):
    """
    The delta inputs of the model. It should be a dictionary of the input data.
    """
    return self._delta_inputs

  def add_current_input(
      self,
      key: str,
      inp: Union[Callable, ArrayLike],
      label: Optional[str] = None
  ):
    """
    Add a current input function.

    Args:
      key: str. The dict key.
      inp: Callable, ArrayLike. The currents or the function to generate currents.
      label: str. The input label.
    """
    key = _input_label_repr(key, label)
    if self._current_inputs is None:
      self._current_inputs = dict()
    if key in self._current_inputs:
      raise ValueError(f'Key "{key}" has been defined and used in the current inputs of {self}.')
    self._current_inputs[key] = inp

  def add_delta_input(
      self,
      key: str,
      inp: Union[Callable, ArrayLike],
      label: Optional[str] = None
  ):
    """
    Add a delta input function.

    Args:
      key: str. The dict key.
      inp: Callable, ArrayLike. The currents or the function to generate currents.
      label: str. The input label.
    """
    key = _input_label_repr(key, label)
    if self._delta_inputs is None:
      self._delta_inputs = dict()
    if key in self._delta_inputs:
      raise ValueError(f'Key "{key}" has been defined and used.')
    self._delta_inputs[key] = inp

  def get_input(self, key: str):
    """Get the input function.

    Args:
      key: str. The key.

    Returns:
      The input function which generates currents.
    """
    if self._current_inputs is not None and key in self._current_inputs:
      return self._current_inputs[key]
    elif self._delta_inputs is not None and key in self._delta_inputs:
      return self._delta_inputs[key]
    else:
      raise ValueError(f'Input key {key} is not in current/delta inputs of the module {self}.')

  def sum_current_inputs(self, init: Any, *args, label: Optional[str] = None, **kwargs):
    """
    Summarize all current inputs by the defined input functions ``.current_inputs``.

    Args:
      init: The initial input data.
      *args: The arguments for input functions.
      **kwargs: The arguments for input functions.
      label: str. The input label.

    Returns:
      The total currents.
    """
    if self._current_inputs is None:
      return init
    if label is None:
      # no label
      for key, out in self._current_inputs.items():
        init = init + out(*args, **kwargs)
    else:
      # has label
      label_repr = _input_label_start(label)
      for key, out in self._current_inputs.items():
        if key.startswith(label_repr):
          init = init + out(*args, **kwargs)
    return init

  def sum_delta_inputs(self, init: Any, *args, label: Optional[str] = None, **kwargs):
    """
    Summarize all delta inputs by the defined input functions ``.delta_inputs``.

    Args:
      init: The initial input data.
      *args: The arguments for input functions.
      **kwargs: The arguments for input functions.
      label: str. The input label.

    Returns:
      The total currents.
    """
    if self._delta_inputs is None:
      return init
    if label is None:
      # no label
      for key, out in self._delta_inputs.items():
        init = init + out(*args, **kwargs)
    else:
      # has label
      label_repr = _input_label_start(label)
      for key, out in self._delta_inputs.items():
        if key.startswith(label_repr):
          init = init + out(*args, **kwargs)
    return init

  @property
  def before_updates(self):
    """
    The before updates of the model. It should be a dictionary of the updating functions.
    """
    return self._before_updates

  @property
  def after_updates(self):
    """
    The after updates of the model. It should be a dictionary of the updating functions.
    """
    return self._after_updates

  def _add_before_update(self, key: Any, fun: Callable):
    """
    Add the before update into this node.
    """
    if self._before_updates is None:
      self._before_updates = dict()
    if key in self.before_updates:
      raise KeyError(f'{key} has been registered in before_updates of {self}')
    self.before_updates[key] = fun

  def _add_after_update(self, key: Any, fun: Callable):
    """Add the after update into this node"""
    if self._after_updates is None:
      self._after_updates = dict()
    if key in self.after_updates:
      raise KeyError(f'{key} has been registered in after_updates of {self}')
    self.after_updates[key] = fun

  def _get_before_update(self, key: Any):
    """Get the before update of this node by the given ``key``."""
    if self._before_updates is None:
      raise KeyError(f'{key} is not registered in before_updates of {self}')
    if key not in self.before_updates:
      raise KeyError(f'{key} is not registered in before_updates of {self}')
    return self.before_updates.get(key)

  def _get_after_update(self, key: Any):
    """Get the after update of this node by the given ``key``."""
    if self._after_updates is None:
      raise KeyError(f'{key} is not registered in after_updates of {self}')
    if key not in self.after_updates:
      raise KeyError(f'{key} is not registered in after_updates of {self}')
    return self.after_updates.get(key)

  def _has_before_update(self, key: Any):
    """Whether this node has the before update of the given ``key``."""
    if self._before_updates is None:
      return False
    return key in self.before_updates

  def _has_after_update(self, key: Any):
    """Whether this node has the after update of the given ``key``."""
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
        if hasattr(model, '_receive_input'):
          model(*args, **kwargs)
        else:
          model()

    # update the model self
    ret = self.update(*args, **kwargs)

    # ``after_updates``
    if self.after_updates is not None:
      for model in self.after_updates.values():
        if hasattr(model, '_not_receive_output'):
          model()
        else:
          model(ret)
    return ret

  def prefetch(self, item: str) -> 'Prefetch':
    return Prefetch(self, item)

  def align_pre(
      self, dyn: Union[DelayedInitializer[T], T]
  ) -> T:
    """
    Align the dynamics before the interaction.
    """
    if isinstance(dyn, Dynamics):
      self._add_after_update(dyn.name, dyn)
      return dyn
    elif isinstance(dyn, DelayedInitializer):
      if not isinstance(dyn.cls, Dynamics):
        raise TypeError(f'The input {dyn} should be an instance of {Dynamics}.')
      if not self._has_after_update(dyn.identifier):
        self._add_after_update(dyn.identifier, dyn())
      return self._get_after_update(dyn.identifier)
    else:
      raise TypeError(f'The input {dyn} should be an instance of {Dynamics} or a delayed initializer.')

  def __leaf_fn__(self, name, value):
    if name in ['_in_size', '_out_size', '_name', '_mode',
                '_before_updates', '_after_updates', '_current_inputs', '_delta_inputs']:
      return (name, value) if value is None else (name[1:], value)  # skip the first `_`
    return name, value


class Prefetch:
  """
  Prefetch a variable of the given module.
  """

  def __init__(self, module: Dynamics, item: str):
    self.module = module
    self.item = item

  @property
  def delay(self):
    return PrefetchDelay(self.module, self.item)

  def __call__(self, *args, **kwargs):
    item = _get_prefetch_item(self)
    return item.value if isinstance(item, State) else item


class PrefetchDelay:
  def __init__(self, module: Dynamics, item: str):
    self.module = module
    self.item = item

  def at(self, time: ArrayLike):
    return PrefetchDelayAt(self.module, self.item, time)


class PrefetchDelayAt:
  def __init__(self, module: Dynamics, item: str, time: ArrayLike):
    self.module = module
    self.item = item
    self.time = time

  def __call__(self, *args, **kwargs):
    item = _get_prefetch_item_delay(self)
    return item.retrieve_at_time(self.time)


def _get_prefetch_item(target) -> State:
  item = getattr(target.module, target.item, None)
  if item is None:
    raise AttributeError(f'The target {target} should have an `{target.item}` attribute.')
  if not isinstance(item, State):
    raise TypeError(f'The prefetch target should be a brainstate.State. But got {item}.')
  return item


def _get_prefetch_item_delay(target) -> Delay:
  item = _get_prefetch_item(target)
  if not isinstance(item, StateWithDelay):
    raise TypeError(f'The prefetch target should be a {StateWithDelay.__name__} when accessing '
                    f'its delay. But got {item}.')
  delay = getattr(item, 'delay', None)
  if not isinstance(delay, Delay):
    raise TypeError(f'The delay of the prefetch target should be an instance of {Delay}. But got {delay}.')
  return delay


def maybe_init_prefetch(target, *args, **kwargs):
  if isinstance(target, Prefetch):
    _get_prefetch_item(target)

  elif isinstance(target, PrefetchDelay):
    _get_prefetch_item_delay(target)

  elif isinstance(target, PrefetchDelayAt):
    delay = _get_prefetch_item_delay(target)
    delay.register_delay(target.time)


class ModuleGroup(Module):
  """
  A group of :py:class:`~.Module` in which the updating order does not matter.

  Args:
    children_as_tuple: The children objects.
    children_as_dict: The children objects.
  """

  __module__ = 'brainstate.nn'

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


def not_receive_update_output(cls: object):
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