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

from typing import Union, Optional, Sequence, Callable

import brainunit as u

from brainstate import environ, init, random
from brainstate._state import ShortTermState
from brainstate.compile import while_loop
from brainstate.nn._dynamics._dynamics_base import Dynamics
from brainstate.typing import ArrayLike, Size, DTypeLike

__all__ = [
  'SpikeTime',
  'PoissonSpike',
  'PoissonEncoder',
]


class SpikeTime(Dynamics):
  """The input neuron group characterized by spikes emitting at given times.

  >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
  >>> SpikeTime(2, times=[10, 20])
  >>> # or
  >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
  >>> SpikeTime(2, times=[10, 20], indices=[0, 0])
  >>> # or
  >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
  >>> SpikeTime(2, times=[10, 20, 30], indices=[0, 1, 0])
  >>> # or
  >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
  >>> # at 30 ms, neuron 1 fires.
  >>> SpikeTime(2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])

  Parameters
  ----------
  size : int, tuple, list
      The neuron group geometry.
  indices : list, tuple, ArrayType
      The neuron indices at each time point to emit spikes.
  times : list, tuple, ArrayType
      The time points which generate the spikes.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(
      self,
      size: Size,
      indices: Union[Sequence, ArrayLike],
      times: Union[Sequence, ArrayLike],
      spk_type: DTypeLike = bool,
      name: Optional[str] = None,
      need_sort: bool = True,
  ):
    super().__init__(size=size, name=name, keep_size=True)

    # parameters
    if len(indices) != len(times):
      raise ValueError(f'The length of "indices" and "times" must be the same. '
                       f'However, we got {len(indices)} != {len(times)}.')
    self.num_times = len(times)
    self.spk_type = spk_type

    # data about times and indices
    self.times = u.math.asarray(times)
    self.indices = u.math.asarray(indices, dtype=environ.ditype())
    if need_sort:
      sort_idx = u.math.argsort(self.times)
      self.indices = self.indices[sort_idx]
      self.times = self.times[sort_idx]

  def init_state(self, *args, **kwargs):
    self.i = ShortTermState(-1)

  def reset_state(self, batch_size=None, **kwargs):
    self.i.value = -1

  def update(self):
    t = environ.get('t')

    def _cond_fun(spikes):
      i = self.i.value
      return u.math.logical_and(i < self.num_times, t >= self.times[i])

    def _body_fun(spikes):
      i = self.i.value
      spikes = spikes.at[..., self.indices[i]].set(True)
      self.i.value += 1
      return spikes

    spike = u.math.zeros(self.num, dtype=self.spk_type)
    spike = while_loop(_cond_fun, _body_fun, spike)
    return spike


class PoissonSpike(Dynamics):
  """
  Poisson Neuron Group.
  """

  def __init__(
      self,
      size: Size,
      freqs: Union[ArrayLike, Callable],
      keep_size: bool = False,
      spk_type: DTypeLike = bool,
      name: Optional[str] = None,
  ):
    super().__init__(size=size, name=name, keep_size=keep_size)

    self.spk_type = spk_type

    # parameters
    self.freqs = init.param(freqs, self.varshape, allow_none=False)

  def update(self):
    spikes = random.rand(self.varshape) <= (self.freqs * environ.get_dt())
    spikes = u.math.asarray(spikes, dtype=self.spk_type)
    return spikes


class PoissonEncoder(Dynamics):
  """
  Poisson Neuron Group.
  """

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      spk_type: DTypeLike = bool,
      name: Optional[str] = None,
  ):
    super().__init__(size=size, name=name, keep_size=keep_size)

    self.spk_type = spk_type

  def update(self, freqs: ArrayLike):
    spikes = random.rand(*self.varshape) <= (freqs * environ.get_dt())
    spikes = u.math.asarray(spikes, dtype=self.spk_type)
    return spikes
