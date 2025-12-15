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

import numbers
from functools import partial
from typing import Optional, Dict, Callable, Union, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate import environ
from brainstate._state import ShortTermState, State, DelayState
from brainstate.graph import Node
from brainstate.transform import jit_error_if
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


def _get_delay(delay_time):
    if delay_time is None:
        return 0. * environ.get_dt(), 0
    delay_step = delay_time / environ.get_dt()
    assert u.get_dim(delay_step) == u.DIMENSIONLESS
    delay_step = jnp.ceil(delay_step).astype(environ.ditype())
    return delay_time, delay_step


class DelayAccess(Node):
    """
    Accessor node for a registered entry in a Delay instance.

    This node holds a reference to a Delay and a named entry that was
    registered on that Delay. It is used by graphs to query delayed
    values by delegating to the underlying Delay instance.

    Args:
        delay: The delay instance.
        *time: The delay time.
        entry: The delay entry.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        delay: 'Delay',
        *time,
        entry: str,
    ):
        super().__init__()
        self.delay = delay
        assert isinstance(delay, Delay), 'The input delay should be an instance of Delay.'
        self._delay_entry = entry
        self.delay_info = delay.register_entry(self._delay_entry, *time)

    def update(self):
        return self.delay.at(self._delay_entry)


class Delay(Module):
    """
    Delay variable for storing short-term history data.

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
        assert delay_method in [_DELAY_ROTATE, _DELAY_CONCAT], (
            f'Un-supported delay method {delay_method}. '
            f'Only support {_DELAY_ROTATE} and {_DELAY_CONCAT}'
        )
        self.delay_method = delay_method

        # interp method
        assert interp_method in [_INTERP_LINEAR, _INTERP_ROUND], (
            f'Un-supported interpolation method {interp_method}. '
            f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}'
        )
        self.interp_method = interp_method

        # delay length and time
        with jax.ensure_compile_time_eval():
            self.max_time, delay_length = _get_delay(time)
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
                if isinstance(delay_time, (tuple, list)):
                    self.register_entry(entry, *delay_time)
                else:
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
        self.history = DelayState(jax.tree.map(fun, self.target_info))

    def reset_state(self, batch_size: int = None, **kwargs):
        fun = partial(self._f_to_init, length=self.max_length, batch_size=batch_size)
        self.history.value = jax.tree.map(fun, self.target_info)

    def register_delay(self, *delay_time):
        """
        Register delay times and update the maximum delay configuration.

        This method processes one or more delay times, validates their format and consistency,
        and updates the delay buffer size if necessary. It handles both scalar and vector
        delay times, ensuring all vector delays have the same size.

        Args:
            *delay_time: Variable number of delay time arguments. The first argument should be
                the primary delay time (float, int, or array-like). Additional arguments are
                treated as indices or secondary delay parameters. All delay times should be
                non-negative numbers or arrays of the same size.

        Returns:
            tuple or None: If delay_time[0] is None, returns None. Otherwise, returns a tuple
                containing (delay_step, *delay_time[1:]) where delay_step is the computed
                delay step in integer time units, and the remaining elements are the
                additional delay parameters passed in.

        Raises:
            AssertionError: If no delay time is provided (empty delay_time).
            ValueError: If delay times have inconsistent sizes when using vector delays,
                or if delay times are not scalar or 1D arrays.

        Note:
            - The method updates self.max_time and self.max_length if the new delay
              requires a larger buffer size.
            - Delay steps are computed using the current environment time step (dt).
            - All delay indices (delay_time[1:]) must be integers.
            - Vector delays must all have the same size as the first delay time.

        Example:
            >>> delay_obj.register_delay(5.0)  # Register 5ms delay
            >>> delay_obj.register_delay(jnp.array([2.0, 3.0]), 0, 1)  # Vector delay with indices
        """
        assert len(delay_time) >= 1, 'You should provide at least one delay time.'
        for dt in delay_time[1:]:
            assert jnp.issubdtype(u.math.get_dtype(dt), jnp.integer), f'The index should be integer. But got {dt}.'
        if delay_time[0] is None:
            return None
        with jax.ensure_compile_time_eval():
            time, delay_step = _get_delay(delay_time[0])
            max_delay_step = jnp.max(delay_step)
            self.max_time = u.math.max(time)

            # delay variable
            if self.max_length <= max_delay_step + 1:
                self.max_length = max_delay_step + 1
            return delay_step, *delay_time[1:]

    def register_entry(self, entry: str, *delay_time) -> 'Delay':
        """
        Register an entry to access the delay data.

        Args:
            entry: str. The entry to access the delay data.
            delay_time: The delay time of the entry, the first element is the delay time,
                the second and later element is the index.
        """
        if entry in self._registered_entries:
            raise KeyError(
                f'Entry {entry} has been registered. '
                f'The existing delay for the key {entry} is {self._registered_entries[entry]}. '
                f'The new delay for the key {entry} is {delay_time}. '
                f'You can use another key. '
            )
        delay_info = self.register_delay(*delay_time)
        self._registered_entries[entry] = delay_info
        return delay_info

    def access(self, entry: str, *delay_time) -> DelayAccess:
        """
        Create a DelayAccess object for a specific delay entry and delay time.

        Args:
            entry (str): The name of the delay entry to access.
            delay_time (Sequence): The delay time or parameters associated with the entry.

        Returns:
            DelayAccess: An object that provides access to the delay data for the specified entry and time.
        """
        return DelayAccess(self, *delay_time, entry=entry)

    def at(self, entry: str) -> ArrayLike:
        """
        Get the data at the given entry.

        Args:
          entry: str. The entry to access the data.

        Returns:
          The data.
        """
        assert isinstance(entry, str), (f'entry should be a string for describing the '
                                        f'entry of the delay data. But we got {entry}.')
        if entry not in self._registered_entries:
            raise KeyError(f'Does not find delay entry "{entry}".')
        delay_step = self._registered_entries[entry]
        if delay_step is None:
            delay_step = (0,)
        return self.retrieve_at_step(*delay_step)

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
                raise ValueError(
                    f'The request delay length should be less than the '
                    f'maximum delay {self.max_length - 1}. But we got {delay_len}'
                )

            jit_error_if(delay_step >= self.max_length, _check_delay, delay_step)

        # rotation method
        with jax.ensure_compile_time_eval():
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
                raise ValueError(
                    f'The request delay time should be within '
                    f'[{t_now - self.max_time - dt}, {t_now}], '
                    f'but we got {t_delay}'
                )

            jit_error_if(
                jnp.logical_or(
                    delay_time > current_time,
                    delay_time < current_time - self.max_time - dt
                ),
                _check_delay,
                current_time,
                delay_time
            )

        with jax.ensure_compile_time_eval():
            diff = current_time - delay_time
            float_time_step = diff / dt

            if self.interp_method == _INTERP_LINEAR:  # "linear" interpolation
                data_at_t0 = self.retrieve_at_step(jnp.asarray(jnp.floor(float_time_step), dtype=jnp.int32), *indices)
                data_at_t1 = self.retrieve_at_step(jnp.asarray(jnp.ceil(float_time_step), dtype=jnp.int32), *indices)
                t_diff = float_time_step - jnp.floor(float_time_step)
                return jax.tree.map(lambda a, b: a * (1 - t_diff) + b * t_diff, data_at_t0, data_at_t1)

            elif self.interp_method == _INTERP_ROUND:  # "round" interpolation
                return self.retrieve_at_step(jnp.asarray(jnp.round(float_time_step), dtype=jnp.int32), *indices)

            else:  # raise error
                raise ValueError(f'Un-supported interpolation method {self.interp_method}, '
                                 f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}')

    def update(self, current: PyTree) -> None:
        """
        Update delay variable with the new data.
        """

        with jax.ensure_compile_time_eval():
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
    Delayed history buffer bound to a module state.

    StateWithDelay is a specialized :py:class:`~.Delay` that attaches to a
    concrete :py:class:`~brainstate._state.State` living on a target module
    (for example a membrane potential ``V`` on a neuron). It automatically
    maintains a rolling history of that state and exposes convenient helpers to
    retrieve the value at a given delay either by step or by time.

    In normal usage you rarely instantiate this class directly. It is created
    implicitly when using the prefetch-delay helpers on a Dynamics module, e.g.:

    - ``module.prefetch('V').delay.at(5.0 * u.ms)``
    - ``module.prefetch_delay('V', 5.0 * u.ms)``

    Both will construct a StateWithDelay bound to ``module.V`` under the hood
    and register the requested delay, so you can retrieve the delayed value
    inside your update rules.

    Parameters
    ----------
    target : :py:class:`~brainstate.graph.Node`
        The module object that owns the state to track.
    item : str
        The attribute name of the target state on ``target`` (must be a
        :py:class:`~brainstate._state.State`).
    init : Callable, optional
        Optional initializer used to fill the history buffer before ``t0``
        when delays request values from the past that hasn't been simulated yet.
        The callable receives ``(shape, dtype)`` and must return an array.
        If not provided, zeros are used. You may also pass a scalar/array
        literal via the underlying Delay API when constructing manually.
    delay_method : {"rotation", "concat"}, default "rotation"
        Internal buffering strategy (inherits behavior from :py:class:`~.Delay`).
        "rotation" keeps a ring buffer; "concat" shifts by concatenation.

    Attributes
    ----------
    state : :py:class:`~brainstate._state.State`
        The concrete state object being tracked.
    history : :py:class:`DelayState`
        Rolling time axis buffer with shape ``[length, *state.shape]``.
    max_time : float
        Maximum time span currently supported by the buffer.
    max_length : int
        Buffer length in steps (``ceil(max_time/dt)+1``).

    Notes
    -----
    - This class inherits all retrieval utilities from :py:class:`~.Delay`:
      use :py:meth:`retrieve_at_step` when you know the integer delay steps,
      or :py:meth:`retrieve_at_time` for continuous-time queries with optional
      linear/round interpolation.
    - It is registered as an "after-update" hook on the owning Dynamics so the
      buffer is updated automatically after each simulation step.

    Examples
    --------
    Access a neuron's membrane potential 5 ms in the past:

    >>> import brainunit as u
    >>> import brainstate as brainstate
    >>> lif = brainpy.state.LIF(100)
    >>> # Create a delayed accessor to V(t-5ms)
    >>> v_delay = lif.prefetch_delay('V', 5.0 * u.ms)
    >>> # Inside another module's update you can read the delayed value
    >>> v_t_minus_5ms = v_delay()

    Register multiple delay taps and index-specific delays:

    >>> # Under the hood, a StateWithDelay is created and you can register
    >>> # additional taps (in steps or time) via its Delay interface
    >>> _ = lif.prefetch('V').delay.at(2.0 * u.ms)   # additional delay
    >>> # Direct access to buffer by steps (advanced)
    >>> # lif._get_after_update('V-prefetch-delay').retrieve_at_step(3)
    """

    __module__ = 'brainstate.nn'

    state: State  # state

    def __init__(
        self,
        target: Node,
        item: str,
        init: Callable = None,
        delay_method: Optional[str] = _DELAY_ROTATE,
    ):
        super().__init__(None, init=init, delay_method=delay_method)

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
