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
import threading
import warnings
from functools import partial
from typing import Optional, Dict, Callable, Union, Sequence, Tuple, Protocol

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate import environ
from brainstate._state import ShortTermState, State, DelayState
from brainstate.graph import Node
from brainstate.transform import jit_error_if, cond
from brainstate.transform._mapping2 import INIT_NO_BATCHING
from brainstate.typing import ArrayLike, PyTree
from ._collective_ops import call_order
from ._module import Module

__all__ = [
    'Delay',
    'DelayAccess',
    'StateWithDelay',
    'InterpolationRegistry',
]

_DELAY_ROTATE = 'rotation'
_DELAY_CONCAT = 'concat'
_INTERP_LINEAR = 'linear_interp'
_INTERP_ROUND = 'round'

# Interpolation method aliases for backward compatibility
INTERP_ALIAS_MAP = {
    'linear_interp': 'linear',
    'round': 'nearest',
}


class InterpolationMethod(Protocol):
    """Protocol for custom interpolation methods.

    Custom interpolation methods must implement this protocol to be compatible
    with the Delay system.

    Parameters
    ----------
    history : PyTree
        Ring buffer data with shape [max_length, *data_shape]
    indices : Tuple
        Additional indices to slice the data (for vectorized delays)
    float_idx : float
        Float index for interpolation (e.g., 10.3 steps ago)
    max_length : int
        Maximum buffer size for modulo wrapping

    Returns
    -------
    PyTree
        Interpolated value matching the structure of history elements
    """

    def __call__(
        self,
        history: PyTree,
        indices: Tuple,
        float_idx: float,
        max_length: int
    ) -> PyTree:
        ...


class InterpolationRegistry:
    """Registry for interpolation methods.

    This class manages built-in and custom interpolation methods for the
    Delay system. Users can register custom interpolation functions and
    query available methods.

    Examples
    --------
    Register a custom interpolation method::

        def my_interp(history, indices, float_idx, max_length):
            # Custom interpolation logic
            ...

        InterpolationRegistry.register('my_method', my_interp)

    Use in Delay::

        delay = Delay(target, time=5.0, interpolation='my_method')
    """

    _methods: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, method: Callable):
        """Register a new interpolation method.

        Parameters
        ----------
        name : str
            Name of the interpolation method
        method : Callable
            Interpolation function following the InterpolationMethod protocol
        """
        cls._methods[name] = method

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get an interpolation method by name.

        Parameters
        ----------
        name : str
            Name of the interpolation method

        Returns
        -------
        Callable
            The interpolation function

        Raises
        ------
        ValueError
            If the interpolation method is not registered
        """
        if name not in cls._methods:
            raise ValueError(
                f"Unknown interpolation method: {name}. "
                f"Available methods: {cls.list_methods()}"
            )
        return cls._methods[name]

    @classmethod
    def list_methods(cls) -> list:
        """List all registered interpolation methods.

        Returns
        -------
        list
            List of available interpolation method names
        """
        return list(cls._methods.keys())


def _nearest_interpolation(history: PyTree, indices: Tuple, float_idx: float, max_length: int) -> PyTree:
    """Round to nearest index (no interpolation).

    This is the simplest interpolation method that rounds the float index
    to the nearest integer and retrieves that value.
    """
    i = jnp.round(float_idx).astype(jnp.int32) % max_length
    idx = (i,) + indices
    return jax.tree.map(lambda h: h[idx], history)


def _linear_interpolation(history: PyTree, indices: Tuple, float_idx: float, max_length: int) -> PyTree:
    """Linear interpolation between two adjacent points.

    Interpolates linearly between floor(float_idx) and ceil(float_idx)
    based on the fractional part of float_idx.
    """
    i0 = jnp.floor(float_idx).astype(jnp.int32) % max_length
    i1 = jnp.ceil(float_idx).astype(jnp.int32) % max_length
    t = float_idx - jnp.floor(float_idx)

    idx0 = (i0,) + indices
    idx1 = (i1,) + indices

    v0 = jax.tree.map(lambda h: h[idx0], history)
    v1 = jax.tree.map(lambda h: h[idx1], history)

    return jax.tree.map(lambda a, b: a * (1 - t) + b * t, v0, v1)


def _cubic_interpolation(history: PyTree, indices: Tuple, float_idx: float, max_length: int) -> PyTree:
    """Cubic spline interpolation using 4 neighboring points.

    Uses Catmull-Rom spline (a type of cubic Hermite spline) for smooth
    interpolation. Provides C1 continuity (continuous first derivatives).
    """
    i0 = jnp.floor(float_idx).astype(jnp.int32)
    # Get 4 neighboring points: i0-1, i0, i0+1, i0+2
    i_m1 = (i0 - 1) % max_length
    i_0 = i0 % max_length
    i_1 = (i0 + 1) % max_length
    i_2 = (i0 + 2) % max_length

    # Fractional part
    t = float_idx - jnp.floor(float_idx)

    # Catmull-Rom basis functions
    w_m1 = -0.5 * t ** 3 + t ** 2 - 0.5 * t
    w_0 = 1.5 * t ** 3 - 2.5 * t ** 2 + 1.0
    w_1 = -1.5 * t ** 3 + 2.0 * t ** 2 + 0.5 * t
    w_2 = 0.5 * t ** 3 - 0.5 * t ** 2

    # Gather values
    idx_m1 = (i_m1,) + indices
    idx_0 = (i_0,) + indices
    idx_1 = (i_1,) + indices
    idx_2 = (i_2,) + indices

    def cubic_interp(h):
        v_m1 = h[idx_m1]
        v_0 = h[idx_0]
        v_1 = h[idx_1]
        v_2 = h[idx_2]
        return w_m1 * v_m1 + w_0 * v_0 + w_1 * v_1 + w_2 * v_2

    return jax.tree.map(cubic_interp, history)


def _hermite_interpolation(history: PyTree, indices: Tuple, float_idx: float, max_length: int) -> PyTree:
    """Hermite spline interpolation with smooth derivatives.

    Uses cubic Hermite interpolation with automatically estimated derivatives
    (finite differences). Provides smoother interpolation than linear.
    """
    i0 = jnp.floor(float_idx).astype(jnp.int32)
    i_m1 = (i0 - 1) % max_length
    i_0 = i0 % max_length
    i_1 = (i0 + 1) % max_length
    i_2 = (i0 + 2) % max_length

    t = float_idx - jnp.floor(float_idx)

    # Hermite basis functions
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2

    idx_m1 = (i_m1,) + indices
    idx_0 = (i_0,) + indices
    idx_1 = (i_1,) + indices
    idx_2 = (i_2,) + indices

    def hermite_interp(h):
        v_m1 = h[idx_m1]
        v_0 = h[idx_0]
        v_1 = h[idx_1]
        v_2 = h[idx_2]

        # Estimate derivatives using finite differences
        m0 = 0.5 * (v_1 - v_m1)  # Derivative at i0
        m1 = 0.5 * (v_2 - v_0)  # Derivative at i0+1

        return h00 * v_0 + h10 * m0 + h01 * v_1 + h11 * m1

    return jax.tree.map(hermite_interp, history)


def _polynomial2_interpolation(history: PyTree, indices: Tuple, float_idx: float, max_length: int) -> PyTree:
    """Quadratic polynomial interpolation using 3 points.

    Fits a 2nd degree polynomial through 3 neighboring points and
    evaluates it at the desired position.
    """
    i0 = jnp.floor(float_idx).astype(jnp.int32)
    i_m1 = (i0 - 1) % max_length
    i_0 = i0 % max_length
    i_1 = (i0 + 1) % max_length

    t = float_idx - jnp.floor(float_idx)

    # Lagrange polynomial basis functions for 3 points
    # Points at positions: -1, 0, 1
    L_m1 = 0.5 * t * (t - 1)
    L_0 = (1 - t) * (1 + t)
    L_1 = 0.5 * t * (t + 1)

    idx_m1 = (i_m1,) + indices
    idx_0 = (i_0,) + indices
    idx_1 = (i_1,) + indices

    def poly2_interp(h):
        v_m1 = h[idx_m1]
        v_0 = h[idx_0]
        v_1 = h[idx_1]
        return L_m1 * v_m1 + L_0 * v_0 + L_1 * v_1

    return jax.tree.map(poly2_interp, history)


def _polynomial3_interpolation(history: PyTree, indices: Tuple, float_idx: float, max_length: int) -> PyTree:
    """Cubic polynomial interpolation using 4 points.

    Fits a 3rd degree polynomial through 4 neighboring points and
    evaluates it at the desired position. Similar to cubic spline but
    uses Lagrange polynomial basis.
    """
    i0 = jnp.floor(float_idx).astype(jnp.int32)
    i_m1 = (i0 - 1) % max_length
    i_0 = i0 % max_length
    i_1 = (i0 + 1) % max_length
    i_2 = (i0 + 2) % max_length

    t = float_idx - jnp.floor(float_idx)

    # Lagrange polynomial basis functions for 4 points
    # Points at positions: -1, 0, 1, 2 (relative to i0)
    L_m1 = -t * (t - 1) * (t - 2) / 6
    L_0 = (t + 1) * (t - 1) * (t - 2) / 2
    L_1 = -t * (t + 1) * (t - 2) / 2
    L_2 = t * (t + 1) * (t - 1) / 6

    idx_m1 = (i_m1,) + indices
    idx_0 = (i_0,) + indices
    idx_1 = (i_1,) + indices
    idx_2 = (i_2,) + indices

    def poly3_interp(h):
        v_m1 = h[idx_m1]
        v_0 = h[idx_0]
        v_1 = h[idx_1]
        v_2 = h[idx_2]
        return L_m1 * v_m1 + L_0 * v_0 + L_1 * v_1 + L_2 * v_2

    return jax.tree.map(poly3_interp, history)


# Register all built-in interpolation methods
InterpolationRegistry.register('nearest', _nearest_interpolation)
InterpolationRegistry.register('round', _nearest_interpolation)  # Alias for backward compatibility
InterpolationRegistry.register('linear', _linear_interpolation)
InterpolationRegistry.register('linear_interp', _linear_interpolation)  # Alias for backward compatibility
InterpolationRegistry.register('cubic', _cubic_interpolation)
InterpolationRegistry.register('hermite', _hermite_interpolation)
InterpolationRegistry.register('polynomial2', _polynomial2_interpolation)
InterpolationRegistry.register('polynomial3', _polynomial3_interpolation)


def _get_delay(delay_time, dt):
    if delay_time is None:
        return 0. * dt, 0
    delay_step = delay_time / dt
    assert u.get_dim(delay_step) == u.DIMENSIONLESS, (
        f'The time_and_idx should have time dimension. '
        f'Got delay time unit {u.get_unit(delay_time)}, and dt unit {u.get_unit(dt)}.'
    )
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

    def __call__(self, *args, **kwargs):
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
      time: int, float, or Quantity. The delay time.
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
      interpolation: str or Callable. The interpolation method for continuous-time retrieval.
        Built-in methods: 'nearest', 'linear', 'cubic', 'hermite', 'polynomial2', 'polynomial3'.
        Can also be a custom callable following the InterpolationMethod protocol.
      take_aware_unit: bool. Whether to track and preserve units from brainunit.
      update_every: optional, float or Quantity. Time interval between buffer updates.
        If None (default), the buffer is updated every time update() is called.
        If specified, the buffer is only updated when the accumulated time since the
        last update exceeds this threshold. Supports brainunit quantities (e.g., 5.0*u.ms).
        Example: update_every=5.0 means update every 5 time units.
      update_strategy: str. Strategy for handling updates between threshold crossings.
        Options:
        - 'hold' (default): Skip writes between thresholds, keep last written value.
        - 'latest': Always cache the newest value, write it when threshold is crossed.
        - 'aggregate': Accumulate all values between thresholds and write aggregated result.
      aggregate_fn: optional, str or Callable. Aggregation function for 'aggregate' strategy.
        Built-in options (strings): 'mean', 'sum', 'max', 'min', 'last'.
        Custom: Any callable that takes an array and axis parameter and returns aggregated value.
        Default: 'mean' when update_strategy='aggregate'.
        Ignored for other strategies.
      delay_method: str. Deprecated parameter kept for backward compatibility.
        The unified ring buffer implementation now uses rotation for all delays.
      interp_method: str. Deprecated parameter kept for backward compatibility.
        Use 'interpolation' parameter instead.

    Examples:
      Basic delay with default behavior (update every call)::

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> delay = brainstate.nn.Delay(jnp.zeros((10,)), time=5.0)
        >>> delay.init_state()
        >>> for i in range(100):
        ...     delay.update(jnp.ones((10,)) * i)

      Delay with update frequency control (hold strategy)::

        >>> import brainunit as u
        >>> delay = brainstate.nn.Delay(
        ...     jnp.zeros((10,)),
        ...     time=10.0 * u.ms,
        ...     update_every=5.0 * u.ms,  # Update every 5ms
        ... )
        >>> delay.init_state()

    """

    __module__ = 'brainstate.nn'

    max_time: float  #
    max_length: int
    history: DelayState
    write_ptr: ShortTermState  # Write pointer for ring buffer

    def __init__(
        self,
        target_info: PyTree,
        time: Optional[Union[int, float, u.Quantity]] = None,  # delay time
        init: Optional[Union[ArrayLike, Callable]] = None,  # delay data before t0
        entries: Optional[Dict] = None,  # delay access entry
        interpolation: Optional[Union[str, Callable]] = None,  # interpolation method (new parameter)
        take_aware_unit: bool = False,
        # NEW PARAMETERS for update frequency control
        update_every: Optional[Union[float, u.Quantity]] = None,  # time interval between buffer updates
        # deprecated parameters for backward compatibility
        delay_method: Optional[str] = _DELAY_ROTATE,  # delay method (deprecated, kept for compatibility)
        interp_method: str = _INTERP_LINEAR,  # interpolation method (deprecated, use interpolation)
    ):
        super().__init__()

        # target information
        self.target_info = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), target_info)

        # delay method (backward compatibility)
        if delay_method is None:
            delay_method = _DELAY_ROTATE
        if delay_method == _DELAY_CONCAT:
            warnings.warn(
                "delay_method='concat' is deprecated. The unified ring buffer "
                "implementation now uses rotation for all delays, providing better performance.",
                DeprecationWarning,
                stacklevel=2
            )
        # Always use rotation (unified ring buffer)
        self.delay_method = _DELAY_ROTATE

        # interpolation parameter handling
        if interpolation is None:
            # Use old interp_method for backward compatibility
            interpolation = INTERP_ALIAS_MAP.get(interp_method, interp_method)

        # Validate and set interpolation method
        if isinstance(interpolation, str):
            if interpolation not in InterpolationRegistry.list_methods():
                raise ValueError(
                    f"Unknown interpolation method: {interpolation}. "
                    f"Available methods: {InterpolationRegistry.list_methods()}"
                )
            self.interp_method = interpolation
        elif callable(interpolation):
            # Custom interpolation function
            self.interp_method = interpolation
        else:
            raise TypeError(
                f"interpolation must be str or callable, got {type(interpolation)}"
            )

        # delay length and time
        with jax.ensure_compile_time_eval():
            self.max_time, delay_length = _get_delay(
                time,
                environ.get_dt() if update_every is None else update_every
            )
            self.max_length = delay_length + 1

        # delay data
        if init is not None and not isinstance(init, (numbers.Number, jax.Array, np.ndarray, Callable)):
            raise TypeError(f'init should be Array, Callable, or None. But got {init}')
        self._init = init

        # delay entries
        self._registered_entries = dict()
        if entries is not None:
            for entry, time_and_idx in entries.items():
                if isinstance(time_and_idx, (tuple, list)):
                    self.register_entry(entry, *time_and_idx)
                else:
                    self.register_entry(entry, time_and_idx)

        # unit handling
        self.take_aware_unit = take_aware_unit
        self._unit = None

        # Validate and convert update_every
        with jax.ensure_compile_time_eval():
            if update_every is not None:
                if update_every < environ.get_dt():
                    raise ValueError(f"update_every must be >= dt ({environ.get_dt()}), got {update_every}")
                self.update_every = update_every
                self.update_every_step = int(update_every / environ.get_dt())
            else:
                self.update_every = None
                self.update_every_step = 1

        # Thread safety locks (lazy initialization)
        self._update_lock = threading.RLock()
        self._retrieve_lock = threading.RLock()

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
        # Initialize write pointer as ShortTermState (always scalar, not batched)
        # All batches share the same write pointer as they update synchronously
        self.write_ptr = ShortTermState(jnp.array(0, dtype=environ.ditype()), tag=INIT_NO_BATCHING)

        # Initialize history buffer
        fun = partial(self._f_to_init, length=self.max_length, batch_size=batch_size)
        self.history = DelayState(jax.tree.map(fun, self.target_info))

    def reset_state(self, batch_size: int = None, **kwargs):
        # Reset write pointer to 0 (always scalar)
        self.write_ptr.value = jnp.array(0, dtype=environ.ditype())

        # Reset history buffer
        fun = partial(self._f_to_init, length=self.max_length, batch_size=batch_size)
        self.history.value = jax.tree.map(fun, self.target_info)

    def register_delay(self, *time_and_idx):
        """
        Register delay times and update the maximum delay configuration.

        This method processes one or more delay times, validates their format and consistency,
        and updates the delay buffer size if necessary. It handles both scalar and vector
        delay times, ensuring all vector delays have the same size.

        Args:
            *time_and_idx: Variable number of delay time arguments. The first argument should be
                the primary delay time (float, int, or array-like). Additional arguments are
                treated as indices or secondary delay parameters. All delay times should be
                non-negative numbers or arrays of the same size.

        Returns:
            tuple or None: If time_and_index[0] is None, returns None. Otherwise, returns a tuple
                containing (delay_step, *time_and_index[1:]) where delay_step is the computed
                delay step in integer time units, and the remaining elements are the
                additional delay parameters passed in.

        Raises:
            AssertionError: If no delay time is provided (empty time_and_index).
            ValueError: If delay times have inconsistent sizes when using vector delays,
                or if delay times are not scalar or 1D arrays.

        Note:
            - The method updates self.max_time and self.max_length if the new delay
              requires a larger buffer size.
            - Delay steps are computed using the current environment time step (dt).
            - All delay indices (time_and_index[1:]) must be integers.
            - Vector delays must all have the same size as the first delay time.

        Example:
            >>> delay_obj.register_delay(5.0)  # Register 5ms delay
            >>> delay_obj.register_delay(jnp.array([2.0, 3.0]), 0, 1)  # Vector delay with indices
        """
        assert len(time_and_idx) >= 1, 'You should provide at least one delay time.'
        for dt in time_and_idx[1:]:
            assert jnp.issubdtype(u.math.get_dtype(dt), jnp.integer), f'The index should be integer. But got {dt}.'
        if time_and_idx[0] is None:
            return None
        with jax.ensure_compile_time_eval():
            time, delay_step = _get_delay(
                time_and_idx[0],
                environ.get_dt() if self.update_every is None else self.update_every
            )
            max_delay_step = jnp.max(delay_step)
            self.max_time = u.math.max(time)

            # delay variable
            if self.max_length <= max_delay_step + 1:
                self.max_length = int(max_delay_step + 1)
            return delay_step, *time_and_idx[1:]

    def register_entry(self, entry: str, *time_and_idx) -> 'Delay':
        """
        Register an entry to access the delay data.

        Args:
            entry: str. The entry to access the delay data.
            time_and_idx: The delay time of the entry, the first element is the delay time,
                the second and later element is the index.
        """
        if entry in self._registered_entries:
            raise KeyError(
                f'Entry {entry} has been registered. '
                f'The existing delay for the key {entry} is {self._registered_entries[entry]}. '
                f'The new delay for the key {entry} is {time_and_idx}. '
                f'You can use another key. '
            )
        delay_info = self.register_delay(*time_and_idx)
        self._registered_entries[entry] = delay_info
        return delay_info

    def access(self, entry: str, *time_and_idx) -> DelayAccess:
        """
        Create a DelayAccess object for a specific delay entry and delay time.

        Args:
            entry (str): The name of the delay entry to access.
            time_and_idx (Sequence): The delay time or parameters associated with the entry.

        Returns:
            DelayAccess: An object that provides access to the delay data for the specified entry and time.
        """
        return DelayAccess(self, *time_and_idx, entry=entry)

    def at(self, entry: str) -> ArrayLike:
        """
        Get the data at the given entry.

        Args:
          entry: str. The entry to access the data.

        Returns:
          The data.
        """
        assert isinstance(entry, str), (
            f'entry should be a string for describing the '
            f'entry of the delay data. But we got {entry}.'
        )
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
        # Acquire lock if in multi-threaded context
        with self._retrieve_lock:
            return self._retrieve_at_step_impl(delay_step, *indices)

    def _retrieve_at_step_impl(self, delay_step, *indices) -> PyTree:
        """Internal implementation of retrieve_at_step (lock-free for JAX tracing)."""
        assert self.history is not None, 'The delay history is not initialized.'
        assert delay_step is not None, 'The delay step should be given.'

        if environ.get(environ.JIT_ERROR_CHECK, False):
            def _check_delay(delay_len):
                raise ValueError(
                    f'The request delay length should be less than the '
                    f'maximum delay {self.max_length - 1}. But we got {delay_len}'
                )

            jit_error_if(delay_step >= self.max_length, _check_delay, delay_step)

        # unified ring buffer method using write_ptr
        with jax.ensure_compile_time_eval():
            # Use write_ptr instead of environ.get(environ.I)
            # Note: write_ptr points to the NEXT write position, so current position is write_ptr - 1
            current_ptr = self.write_ptr.value // self.update_every_step - 1
            di = current_ptr - delay_step
            delay_idx = jnp.asarray(di % self.max_length, dtype=jnp.int32)
            delay_idx = jax.lax.stop_gradient(delay_idx)

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

            # Use interpolation methods that call retrieve_at_step for bounds checking
            if (
                self.interp_method == 'linear' or
                self.interp_method == 'linear_interp' or
                self.interp_method == _INTERP_LINEAR
            ):
                # Linear interpolation - call retrieve_at_step for bounds checking
                data_at_t0 = self.retrieve_at_step(jnp.asarray(jnp.floor(float_time_step), dtype=jnp.int32), *indices)
                data_at_t1 = self.retrieve_at_step(jnp.asarray(jnp.ceil(float_time_step), dtype=jnp.int32), *indices)
                t_diff = float_time_step - jnp.floor(float_time_step)
                return jax.tree.map(lambda a, b: a * (1 - t_diff) + b * t_diff, data_at_t0, data_at_t1)

            elif (
                self.interp_method == 'nearest' or
                self.interp_method == 'round' or
                self.interp_method == _INTERP_ROUND
            ):
                # Round interpolation - call retrieve_at_step for bounds checking
                return self.retrieve_at_step(jnp.asarray(jnp.round(float_time_step), dtype=jnp.int32), *indices)

            else:
                # For other interpolation methods (cubic, hermite, polynomial), use the registry
                # Calculate the buffer position accounting for ring buffer
                current_ptr = self.write_ptr.value // self.update_every_step - 1
                float_buffer_idx = current_ptr - float_time_step

                if isinstance(self.interp_method, str):
                    interp_func = InterpolationRegistry.get(self.interp_method)
                else:
                    # Custom callable interpolation method
                    interp_func = self.interp_method

                # Call interpolation function with history, indices, float buffer index, and max_length
                return interp_func(self.history.value, indices, float_buffer_idx, self.max_length)

    def _write_to_buffer(self, value: PyTree) -> None:
        """Write a value to the ring buffer at current write_ptr position."""
        idx = jnp.asarray(self.write_ptr.value // self.update_every_step, dtype=environ.dutype())
        idx = jax.lax.stop_gradient(idx)
        self.history.value = jax.tree.map(
            lambda hist, val: hist.at[idx].set(val),
            self.history.value,
            value
        )
        self.write_ptr.value = (self.write_ptr.value + 1) % self.max_length

    def _frequency_controlled_update(self, current: PyTree) -> None:
        """Handle frequency-controlled updates with different strategies."""

        # Update time accumulator
        should_update = self.write_ptr.value % self.update_every_step == 0

        def do_nothing():
            pass

        # Hold: Only write when threshold crossed
        def write_and_reset():
            self._write_to_buffer(current)

        cond(should_update, write_and_reset, do_nothing)

    def update(self, current: PyTree) -> None:
        """
        Update delay variable with the new data.
        """
        # Acquire lock if in multi-threaded context
        with self._update_lock:
            self._update_impl(current)

    def _update_impl(self, current: PyTree) -> None:
        """Internal implementation of update (lock-free for JAX tracing)."""
        with jax.ensure_compile_time_eval():
            assert self.history is not None, 'The delay history is not initialized.'

            if self.take_aware_unit and self._unit is None:
                self._unit = jax.tree.map(lambda x: u.get_unit(x), current, is_leaf=u.math.is_quantity)

            # Check if frequency control is enabled
            if self.update_every is None:
                # Default: update every call
                self._write_to_buffer(current)
            else:
                # Frequency-controlled update
                self._frequency_controlled_update(current)


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
    interpolation : str or Callable, default "round"
        Interpolation method for continuous-time delay retrieval.
        Built-in methods: 'nearest', 'linear', 'cubic', 'hermite',
        'polynomial2', 'polynomial3'. Can also be a custom callable following
        the InterpolationMethod protocol.

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
    >>> import brainstate
    >>> import brainpy
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
    >>> # lif.get_after_update('V-prefetch-delay').retrieve_at_step(3)
    """

    __module__ = 'brainstate.nn'

    state: State  # state

    def __init__(
        self,
        target: Node,
        item: str,
        init: Callable = None,
        interpolation: Optional[str] = _INTERP_ROUND,
        **kwargs
    ):
        super().__init__(None, init=init, interpolation=interpolation, **kwargs)

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
