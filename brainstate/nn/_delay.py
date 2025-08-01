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

import math
import numbers
from functools import partial
from typing import Optional, Dict, Callable, Union, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate import environ
from brainstate._state import ShortTermState, State
from brainstate.compile import jit_error_if
from brainstate.graph import Node
from brainstate.typing import ArrayLike, PyTree
from ._collective_ops import call_order
from ._module import Module

__all__ = [
    'Delay', 'DelayAccess', 'StateWithDelay',
]

_DELAY_ROTATE = 'rotation'
_DELAY_CONCAT = 'concat'
_INTERP_LINEAR = 'linear_interp'
_INTERP_ROUND = 'round'


def _get_delay(delay_time, delay_step):
    if delay_time is None:
        if delay_step is None:
            return 0., 0
        else:
            assert isinstance(delay_step, int), '"delay_step" should be an integer.'
            if delay_step == 0:
                return 0., 0
            with jax.ensure_compile_time_eval():
                delay_time = delay_step * environ.get_dt()
    else:
        assert delay_step is None, '"delay_step" should be None if "delay_time" is given.'
        # assert isinstance(delay_time, (int, float))
        with jax.ensure_compile_time_eval():
            delay_step = delay_time / environ.get_dt()
        delay_step = math.ceil(float(delay_step))
    return delay_time, delay_step


class DelayAccess(Node):
    """
    The delay access class.

    Args:
      delay: The delay instance.
      time: The delay time.
      indices: The indices of the delay data.
      delay_entry: The delay entry.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        delay: 'Delay',
        time: Union[None, int, float],
        delay_entry: str,
        *indices,
    ):
        super().__init__()
        self.refs = {'delay': delay}
        assert isinstance(delay, Delay), 'The input delay should be an instance of Delay.'
        self._delay_entry = delay_entry
        delay.register_entry(self._delay_entry, time)
        self.indices = indices

    def update(self):
        return self.refs['delay'].at(self._delay_entry, *self.indices)


class Delay(Module):
    """
    Generate Delays for the given :py:class:`~.State` instance.

    The data in this delay variable is arranged as::

         delay = 0             [ data
         delay = 1               data
         delay = 2               data
         ...                     ....
         ...                     ....
         delay = length-1        data
         delay = length          data ]

    Args:
      time: int, float. The delay time.
      init: Any. The delay data. It can be a Python number, like float, int, boolean values.
        It can also be arrays. Or a callable function or instance of ``Connector``.
        Note that ``initial_delay_data`` should be arranged as the following way::

           delay = 1             [ data
           delay = 2               data
           ...                     ....
           ...                     ....
           delay = length-1        data
           delay = length          data ]
      entries: optional, dict. The delay access entries.
      delay_method: str. The method used for updating delay. Default None.
    """

    __module__ = 'brainstate.nn'

    max_time: float  #
    max_length: int
    history: Optional[ShortTermState]

    def __init__(
        self,
        target_info: PyTree,
        time: Optional[Union[int, float, u.Quantity]] = None,  # delay time
        init: Optional[Union[ArrayLike, Callable]] = None,  # delay data before t0
        entries: Optional[Dict] = None,  # delay access entry
        delay_method: Optional[str] = _DELAY_ROTATE,  # delay method
        interp_method: str = _INTERP_LINEAR,  # interpolation method
        take_aware_unit: bool = False
    ):
        # target information
        self.target_info = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), target_info)

        # delay method
        assert delay_method in [_DELAY_ROTATE, _DELAY_CONCAT], (f'Un-supported delay method {delay_method}. '
                                                                f'Only support {_DELAY_ROTATE} and {_DELAY_CONCAT}')
        self.delay_method = delay_method

        # interp method
        assert interp_method in [_INTERP_LINEAR, _INTERP_ROUND], (f'Un-supported interpolation method {interp_method}. '
                                                                  f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}')
        self.interp_method = interp_method

        # delay length and time
        self.max_time, delay_length = _get_delay(time, None)
        self.max_length = delay_length + 1

        super().__init__()

        # delay data
        if init is not None:
            if not isinstance(init, (numbers.Number, jax.Array, np.ndarray, Callable)):
                raise TypeError(f'init should be Array, Callable, or None. But got {init}')
        self._init = init
        self._history = None

        # other info
        self._registered_entries = dict()

        # other info
        if entries is not None:
            for entry, delay_time in entries.items():
                self.register_entry(entry, delay_time)

        self.take_aware_unit = take_aware_unit
        self._unit = None

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    def _f_to_init(self, a, batch_size, length):
        shape = list(a.shape)
        if batch_size is not None:
            shape.insert(0, batch_size)
        shape.insert(0, length)
        if isinstance(self._init, (jax.Array, np.ndarray, numbers.Number)):
            data = jnp.broadcast_to(jnp.asarray(self._init, a.dtype), shape)
        elif callable(self._init):
            data = self._init(shape, dtype=a.dtype)
        else:
            assert self._init is None, f'init should be Array, Callable, or None. but got {self._init}'
            data = jnp.zeros(shape, dtype=a.dtype)
        return data

    @call_order(3)
    def init_state(self, batch_size: int = None, **kwargs):
        fun = partial(self._f_to_init, length=self.max_length, batch_size=batch_size)
        self.history = ShortTermState(jax.tree.map(fun, self.target_info))

    def reset_state(self, batch_size: int = None, **kwargs):
        fun = partial(self._f_to_init, length=self.max_length, batch_size=batch_size)
        self.history.value = jax.tree.map(fun, self.target_info)

    def register_delay(
        self,
        delay_time: Optional[Union[int, float]] = None,
        delay_step: Optional[int] = None,
    ):
        if isinstance(delay_time, (np.ndarray, jax.Array)):
            assert delay_time.size == 1 and delay_time.ndim == 0
            delay_time = delay_time.item()

        _, delay_step = _get_delay(delay_time, delay_step)

        # delay variable
        if self.max_length <= delay_step + 1:
            self.max_length = delay_step + 1
            self.max_time = delay_time
        return self

    def register_entry(
        self,
        entry: str,
        delay_time: Optional[Union[int, float]] = None,
        delay_step: Optional[int] = None,
    ) -> 'Delay':
        """
        Register an entry to access the delay data.

        Args:
          entry: str. The entry to access the delay data.
          delay_time: The delay time of the entry (can be a float).
          delay_step: The delay step of the entry (must be an int). ``delat_step = delay_time / dt``.

        Returns:
          Return the self.
        """
        if entry in self._registered_entries:
            raise KeyError(f'Entry {entry} has been registered. '
                           f'The existing delay for the key {entry} is {self._registered_entries[entry]}. '
                           f'The new delay for the key {entry} is {delay_time}. '
                           f'You can use another key. ')

        if isinstance(delay_time, (np.ndarray, jax.Array)):
            assert delay_time.size == 1 and delay_time.ndim == 0
            delay_time = delay_time.item()

        _, delay_step = _get_delay(delay_time, delay_step)

        # delay variable
        if self.max_length <= delay_step + 1:
            self.max_length = delay_step + 1
            self.max_time = delay_time
        self._registered_entries[entry] = delay_step
        return self

    def access(
        self,
        entry: str = None,
        time: Sequence = None,
    ) -> DelayAccess:
        return DelayAccess(self, time, delay_entry=entry)

    def at(self, entry: str, *indices) -> ArrayLike:
        """
        Get the data at the given entry.

        Args:
          entry: str. The entry to access the data.
          *indices: The slicing indices. Not include the slice at the batch dimension.

        Returns:
          The data.
        """
        assert isinstance(entry, str), (f'entry should be a string for describing the '
                                        f'entry of the delay data. But we got {entry}.')
        if entry not in self._registered_entries:
            raise KeyError(f'Does not find delay entry "{entry}".')
        delay_step = self._registered_entries[entry]
        if delay_step is None:
            delay_step = 0
        return self.retrieve_at_step(delay_step, *indices)

    def retrieve_at_step(self, delay_step, *indices) -> PyTree:
        """
        Retrieve the delay data at the given delay time step (the integer to indicate the time step).

        Parameters
        ----------
        delay_step: int_like
          Retrieve the data at the given time step.
        indices: tuple
          The indices to slice the data.

        Returns
        -------
        delay_data: The delay data at the given delay step.

        """
        assert self.history is not None, 'The delay history is not initialized.'
        assert delay_step is not None, 'The delay step should be given.'

        if environ.get(environ.JIT_ERROR_CHECK, False):
            def _check_delay(delay_len):
                raise ValueError(f'The request delay length should be less than the '
                                 f'maximum delay {self.max_length - 1}. But we got {delay_len}')

            jit_error_if(delay_step >= self.max_length, _check_delay, delay_step)

        # rotation method
        if self.delay_method == _DELAY_ROTATE:
            i = environ.get(environ.I, desc='The time step index.')
            di = i - delay_step
            delay_idx = jnp.asarray(di % self.max_length, dtype=jnp.int32)
            delay_idx = jax.lax.stop_gradient(delay_idx)

        elif self.delay_method == _DELAY_CONCAT:
            delay_idx = delay_step

        else:
            raise ValueError(f'Unknown delay updating method "{self.delay_method}"')

        # the delay index
        if hasattr(delay_idx, 'dtype') and not jnp.issubdtype(delay_idx.dtype, jnp.integer):
            raise ValueError(f'"delay_len" must be integer, but we got {delay_idx}')
        indices = (delay_idx,) + indices

        # the delay data
        if self._unit is None:
            return jax.tree.map(lambda a: a[indices], self.history.value)
        else:
            return jax.tree.map(
                lambda hist, unit: u.maybe_decimal(hist[indices] * unit),
                self.history.value,
                self._unit
            )

    def retrieve_at_time(self, delay_time, *indices) -> PyTree:
        """
        Retrieve the delay data at the given delay time step (the integer to indicate the time step).

        Parameters
        ----------
        delay_time: float
          Retrieve the data at the given time.
        indices: tuple
          The indices to slice the data.

        Returns
        -------
        delay_data: The delay data at the given delay step.

        """
        assert self.history is not None, 'The delay history is not initialized.'
        assert delay_time is not None, 'The delay time should be given.'

        current_time = environ.get(environ.T, desc='The current time.')
        dt = environ.get_dt()

        if environ.get(environ.JIT_ERROR_CHECK, False):
            def _check_delay(t_now, t_delay):
                raise ValueError(f'The request delay time should be within '
                                 f'[{t_now - self.max_time - dt}, {t_now}], '
                                 f'but we got {t_delay}')

            jit_error_if(
                jnp.logical_or(delay_time > current_time,
                               delay_time < current_time - self.max_time - dt),
                _check_delay,
                current_time,
                delay_time
            )

        diff = current_time - delay_time
        float_time_step = diff / dt

        if self.interp_method == _INTERP_LINEAR:  # "linear" interpolation
            data_at_t0 = self.retrieve_at_step(jnp.asarray(jnp.floor(float_time_step), dtype=jnp.int32), *indices)
            data_at_t1 = self.retrieve_at_step(jnp.asarray(jnp.ceil(float_time_step), dtype=jnp.int32), *indices)
            t_diff = float_time_step - jnp.floor(float_time_step)
            return jax.tree.map(lambda a, b: a * (1 - t_diff) + b * t_diff, data_at_t0, data_at_t1)

        elif self.interp_method == _INTERP_ROUND:  # "round" interpolation
            return self.retrieve_at_step(
                jnp.asarray(jnp.round(float_time_step), dtype=jnp.int32),
                *indices
            )

        else:  # raise error
            raise ValueError(f'Un-supported interpolation method {self.interp_method}, '
                             f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}')

    def update(self, current: PyTree) -> None:
        """
        Update delay variable with the new data.
        """
        assert self.history is not None, 'The delay history is not initialized.'

        if self.take_aware_unit and self._unit is None:
            self._unit = jax.tree.map(lambda x: u.get_unit(x), current, is_leaf=u.math.is_quantity)

        # update the delay data at the rotation index
        if self.delay_method == _DELAY_ROTATE:
            i = environ.get(environ.I)
            idx = jnp.asarray(i % self.max_length, dtype=environ.dutype())
            idx = jax.lax.stop_gradient(idx)
            self.history.value = jax.tree.map(
                lambda hist, cur: hist.at[idx].set(cur),
                self.history.value,
                current
            )
        # update the delay data at the first position
        elif self.delay_method == _DELAY_CONCAT:
            current = jax.tree.map(lambda a: jnp.expand_dims(a, 0), current)
            if self.max_length > 1:
                self.history.value = jax.tree.map(
                    lambda hist, cur: jnp.concatenate([cur, hist[:-1]], axis=0),
                    self.history.value,
                    current
                )
            else:
                self.history.value = current

        else:
            raise ValueError(f'Unknown updating method "{self.delay_method}"')




class StateWithDelay(Delay):
    """
    A ``State`` type that defines the state in a differential equation.
    """

    __module__ = 'brainstate.nn'

    state: State  # state

    def __init__(self, target: Node, item: str):
        super().__init__(None)

        self._target = target
        self._target_term = item

    @property
    def state(self) -> State:
        r = getattr(self._target, self._target_term)
        if not isinstance(r, State):
            raise TypeError(f'The term "{self._target_term}" in the module "{self._target}" is not a State.')
        return r

    @call_order(3)
    def init_state(self, *args, **kwargs):
        """
        State initialization function.
        """
        state = self.state
        self.target_info = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state.value)
        super().init_state(*args, **kwargs)

    def update(self, *args) -> None:
        """
        Update the delay variable with the new data.
        """
        value = self.state.value
        return super().update(value)
