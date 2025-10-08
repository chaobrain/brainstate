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

"""
Environment configuration and context management for BrainState.

This module provides comprehensive functionality for managing computational
environments, including platform selection, precision control, mode setting,
and context-based configuration management. It enables flexible configuration
of JAX-based computations with thread-safe context switching.

The module supports:
- Platform configuration (CPU, GPU, TPU)
- Precision control (8, 16, 32, 64 bit and bfloat16)
- Computation mode management
- Context-based temporary settings
- Default data type management
- Custom behavior registration

Examples
--------
Global environment configuration:

.. code-block:: python

    >>> import brainstate as bs
    >>> import brainstate.environ as env
    >>>
    >>> # Set global precision to 32-bit
    >>> env.set(precision=32, dt=0.01, mode=bs.mixin.Training())
    >>>
    >>> # Get current settings
    >>> print(env.get('precision'))  # 32
    >>> print(env.get('dt'))  # 0.01

Context-based temporary settings:

.. code-block:: python

    >>> import brainstate.environ as env
    >>>
    >>> # Temporarily change precision
    >>> with env.context(precision=64, dt=0.001):
    ...     high_precision_result = compute_something()
    ...     print(env.get('precision'))  # 64
    >>> print(env.get('precision'))  # Back to 32
"""

import contextlib
import dataclasses
import functools
import os
import re
import threading
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, Optional, Union, ContextManager, List

import brainunit as u
import numpy as np
from jax import config, devices, numpy as jnp
from jax.typing import DTypeLike

__all__ = [
    # Core environment management
    'set',
    'get',
    'all',
    'pop',
    'context',
    'reset',

    # Platform and device management
    'set_platform',
    'get_platform',
    'set_host_device_count',
    'get_host_device_count',

    # Precision and data type management
    'set_precision',
    'get_precision',
    'dftype',
    'ditype',
    'dutype',
    'dctype',

    # Mode and computation settings
    'get_dt',

    # Utility functions
    'tolerance',
    'register_default_behavior',
    'unregister_default_behavior',
    'list_registered_behaviors',

    # Constants
    'DEFAULT_PRECISION',
    'SUPPORTED_PLATFORMS',
    'SUPPORTED_PRECISIONS',

    # Names
    'I',
    'T',
    'DT',
    'PRECISION',
    'PLATFORM',
    'HOST_DEVICE_COUNT',
    'JIT_ERROR_CHECK',
    'FIT',
]

# Type definitions
# T = TypeVar('T')
PrecisionType = Union[int, str]
PlatformType = str

# Constants for environment keys
I = 'i'  # Index of the current computation
T = 't'  # Current time of the computation
DT = 'dt'  # Time step for numerical integration
PRECISION = 'precision'  # Numerical precision
PLATFORM = 'platform'  # Computing platform
HOST_DEVICE_COUNT = 'host_device_count'  # Number of host devices
JIT_ERROR_CHECK = 'jit_error_check'  # JIT error checking flag
FIT = 'fit'  # Model fitting flag

# Default values
DEFAULT_PRECISION = 32
SUPPORTED_PLATFORMS = ('cpu', 'gpu', 'tpu')
SUPPORTED_PRECISIONS = (8, 16, 32, 64, 'bf16')

# Sentinel value for missing arguments
_NOT_PROVIDED = object()


@dataclasses.dataclass
class EnvironmentState(threading.local):
    """
    Thread-local storage for environment configuration.

    This class maintains separate configuration states for different threads,
    ensuring thread-safe environment management in concurrent applications.

    Attributes
    ----------
    settings : Dict[Hashable, Any]
        Global default environment settings.
    contexts : defaultdict[Hashable, List[Any]]
        Stack of context-specific settings for nested contexts.
    functions : Dict[Hashable, Callable]
        Registered callback functions for environment changes.
    locks : Dict[str, threading.Lock]
        Thread locks for synchronized access to critical sections.
    """
    settings: Dict[Hashable, Any] = dataclasses.field(default_factory=dict)
    contexts: defaultdict[Hashable, List[Any]] = dataclasses.field(default_factory=lambda: defaultdict(list))
    functions: Dict[Hashable, Callable] = dataclasses.field(default_factory=dict)
    locks: Dict[str, threading.Lock] = dataclasses.field(default_factory=lambda: defaultdict(threading.Lock))

    def __post_init__(self):
        """Initialize with default settings."""
        # Set default precision if not already set
        if PRECISION not in self.settings:
            self.settings[PRECISION] = DEFAULT_PRECISION


# Global environment state
_ENV_STATE = EnvironmentState()


def reset() -> None:
    """
    Reset the environment to default settings.

    This function clears all custom settings and restores the environment
    to its initial state. Useful for testing or when starting fresh.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Set custom values
        >>> env.set(dt=0.1, custom_param='value')
        >>> print(env.get('custom_param'))  # 'value'
        >>>
        >>> # Reset to defaults
        >>> env.reset()
        >>> print(env.get('custom_param', default=None))  # None

    Notes
    -----
    This operation cannot be undone. All custom settings will be lost.
    """
    global _ENV_STATE
    _ENV_STATE = EnvironmentState()
    # Re-apply default precision
    _set_jax_precision(DEFAULT_PRECISION)

    warnings.warn(
        "Environment has been reset to default settings. "
        "All custom configurations have been cleared.",
        UserWarning
    )


@contextlib.contextmanager
def context(**kwargs) -> ContextManager[Dict[str, Any]]:
    """
    Context manager for temporary environment settings.

    This context manager allows you to temporarily modify environment settings
    within a specific scope. Settings are automatically restored when exiting
    the context, even if an exception occurs.

    Parameters
    ----------
    **kwargs
        Environment settings to apply within the context.
        Common parameters include:

        - precision : int or str.
            Numerical precision (8, 16, 32, 64, or 'bf16')
        - dt : float.
            Time step for numerical integration
        - mode : :class:`Mode`.
            Computation mode instance
        - Any custom parameters registered via register_default_behavior

    Yields
    ------
    dict
        Current environment settings within the context.

    Raises
    ------
    ValueError
        If attempting to set platform or host_device_count in context
        (these must be set globally).
    TypeError
        If invalid parameter types are provided.

    Examples
    --------
    Basic usage with precision control:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Set global precision
        >>> env.set(precision=32)
        >>>
        >>> # Temporarily use higher precision
        >>> with env.context(precision=64) as ctx:
        ...     print(f"Precision in context: {env.get('precision')}")  # 64
        ...     print(f"Float type: {env.dftype()}")  # float64
        >>>
        >>> print(f"Precision after context: {env.get('precision')}")  # 32

    Nested contexts:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> with env.context(dt=0.1) as ctx1:
        ...     print(f"dt = {env.get('dt')}")  # 0.1
        ...
        ...     with env.context(dt=0.01) as ctx2:
        ...         print(f"dt = {env.get('dt')}")  # 0.01
        ...
        ...     print(f"dt = {env.get('dt')}")  # 0.1

    Error handling in context:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> env.set(value=10)
        >>> try:
        ...     with env.context(value=20):
        ...         print(env.get('value'))  # 20
        ...         raise ValueError("Something went wrong")
        ... except ValueError:
        ...     pass
        >>>
        >>> print(env.get('value'))  # 10 (restored)

    Notes
    -----
    - Platform and host_device_count cannot be set in context
    - Contexts can be nested arbitrarily deep
    - Settings are restored in reverse order when exiting
    - Thread-safe: each thread maintains its own context stack
    """
    # Validate restricted parameters
    if PLATFORM in kwargs:
        raise ValueError(
            f"Cannot set '{PLATFORM}' in context. "
            f"Use set_platform() or set() for global configuration."
        )
    if HOST_DEVICE_COUNT in kwargs:
        raise ValueError(
            f"Cannot set '{HOST_DEVICE_COUNT}' in context. "
            f"Use set_host_device_count() or set() for global configuration."
        )

    # Handle precision changes
    original_precision = None
    if PRECISION in kwargs:
        original_precision = _get_precision()
        _validate_precision(kwargs[PRECISION])
        _set_jax_precision(kwargs[PRECISION])

    try:
        # Push new values onto context stacks
        for key, value in kwargs.items():
            with _ENV_STATE.locks[key]:
                _ENV_STATE.contexts[key].append(value)

                # Trigger registered callbacks
                if key in _ENV_STATE.functions:
                    try:
                        _ENV_STATE.functions[key](value)
                    except Exception as e:
                        warnings.warn(
                            f"Callback for '{key}' raised an exception: {e}",
                            RuntimeWarning
                        )

        # Yield current environment state
        yield all()

    finally:
        # Restore previous values
        for key in kwargs:
            with _ENV_STATE.locks[key]:
                if _ENV_STATE.contexts[key]:
                    _ENV_STATE.contexts[key].pop()

                # Restore callbacks with previous value
                if key in _ENV_STATE.functions:
                    try:
                        prev_value = get(key, default=None)
                        if prev_value is not None:
                            _ENV_STATE.functions[key](prev_value)
                    except Exception as e:
                        warnings.warn(
                            f"Callback restoration for '{key}' raised: {e}",
                            RuntimeWarning
                        )

        # Restore precision if it was changed
        if original_precision is not None:
            _set_jax_precision(original_precision)


def get(key: str, default: Any = _NOT_PROVIDED, desc: Optional[str] = None) -> Any:
    """
    Get a value from the current environment.

    This function retrieves values from the environment, checking first in
    the context stack, then in global settings. Special handling is provided
    for platform and device count parameters.

    Parameters
    ----------
    key : str
        The environment key to retrieve.
    default : Any, optional
        Default value to return if key is not found.
        If not provided, raises KeyError for missing keys.
    desc : str, optional
        Description of the parameter for error messages.

    Returns
    -------
    Any
        The value associated with the key.

    Raises
    ------
    KeyError
        If key is not found and no default is provided.

    Examples
    --------
    Basic retrieval:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> env.set(learning_rate=0.001)
        >>> lr = env.get('learning_rate')
        >>> print(f"Learning rate: {lr}")  # 0.001

    With default value:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Get with default
        >>> batch_size = env.get('batch_size', default=32)
        >>> print(f"Batch size: {batch_size}")  # 32

    Context-aware retrieval:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> env.set(temperature=1.0)
        >>> print(env.get('temperature'))  # 1.0
        >>>
        >>> with env.context(temperature=0.5):
        ...     print(env.get('temperature'))  # 0.5
        >>>
        >>> print(env.get('temperature'))  # 1.0

    Notes
    -----
    Special keys 'platform' and 'host_device_count' are handled separately
    and retrieve system-level information.
    """
    # Special cases for platform-specific parameters
    if key == PLATFORM:
        return get_platform()
    if key == HOST_DEVICE_COUNT:
        return get_host_device_count()

    # Check context stack first (most recent value)
    with _ENV_STATE.locks[key]:
        if key in _ENV_STATE.contexts and _ENV_STATE.contexts[key]:
            return _ENV_STATE.contexts[key][-1]

    # Check global settings
    if key in _ENV_STATE.settings:
        return _ENV_STATE.settings[key]

    # Handle missing key
    if default is _NOT_PROVIDED:
        error_msg = f"Key '{key}' not found in environment."
        if desc:
            error_msg += f" Description: {desc}"
        error_msg += (
            f"\nSet it using:\n"
            f"  - env.set({key}=value) for global setting\n"
            f"  - env.context({key}=value) for temporary setting"
        )
        raise KeyError(error_msg)

    return default


def all() -> Dict[str, Any]:
    """
    Get all current environment settings.

    This function returns a dictionary containing all active environment
    settings, with context values taking precedence over global settings.

    Returns
    -------
    dict
        Dictionary of all current environment settings.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Set various parameters
        >>> env.set(precision=32, dt=0.01, debug=True)
        >>>
        >>> # Get all settings
        >>> settings = env.all()
        >>> print(settings)
        {'precision': 32, 'dt': 0.01, 'debug': True}

        >>> # Context overrides
        >>> with env.context(precision=64, new_param='test'):
        ...     settings = env.all()
        ...     print(settings['precision'])  # 64
        ...     print(settings['new_param'])  # 'test'

    Notes
    -----
    The returned dictionary is a snapshot and modifying it does not
    affect the environment settings.
    """
    result = {}

    # Add global settings
    result.update(_ENV_STATE.settings)

    # Override with context values (most recent)
    for key, values in _ENV_STATE.contexts.items():
        if values:
            result[key] = values[-1]

    return result


def pop(key: str, default: Any = _NOT_PROVIDED) -> Any:
    """
    Remove and return a value from the global environment.

    This function removes a key from the global environment settings and
    returns its value. If the key is not found, it returns the default
    value if provided, or raises KeyError.

    Note that this function only affects global settings, not context values.
    Keys in active contexts are not affected.

    Parameters
    ----------
    key : str
        The environment key to remove.
    default : Any, optional
        Default value to return if key is not found.
        If not provided, raises KeyError for missing keys.

    Returns
    -------
    Any
        The value that was removed from the environment.

    Raises
    ------
    KeyError
        If key is not found and no default is provided.
    ValueError
        If attempting to pop a key that is currently in a context.

    Examples
    --------
    Basic usage:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Set a value
        >>> env.set(temp_param='temporary')
        >>> print(env.get('temp_param'))  # 'temporary'
        >>>
        >>> # Pop the value
        >>> value = env.pop('temp_param')
        >>> print(value)  # 'temporary'
        >>>
        >>> # Value is now gone
        >>> env.get('temp_param', default=None)  # None

    With default value:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Pop non-existent key with default
        >>> value = env.pop('nonexistent', default='default_value')
        >>> print(value)  # 'default_value'

    Pop multiple values:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Set multiple values
        >>> env.set(param1='value1', param2='value2', param3='value3')
        >>>
        >>> # Pop them one by one
        >>> v1 = env.pop('param1')
        >>> v2 = env.pop('param2')
        >>>
        >>> # param3 still exists
        >>> print(env.get('param3'))  # 'value3'

    Context protection:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> env.set(protected='global_value')
        >>>
        >>> with env.context(protected='context_value'):
        ...     # Cannot pop a key that's in active context
        ...     try:
        ...         env.pop('protected')
        ...     except ValueError as e:
        ...         print("Cannot pop key in active context")

    Notes
    -----
    - This function only removes keys from global settings
    - Keys that are currently overridden in active contexts cannot be popped
    - Special keys like 'platform' and 'host_device_count' can be popped but
      their system-level values remain accessible through get_platform() etc.
    - Registered callbacks are NOT triggered when popping values
    """
    # Check if key is currently in any active context
    if key in _ENV_STATE.contexts and _ENV_STATE.contexts[key]:
        raise ValueError(
            f"Cannot pop key '{key}' while it is active in a context. "
            f"The key is currently overridden in {len(_ENV_STATE.contexts[key])} context(s)."
        )

    # Check if key exists in global settings
    if key in _ENV_STATE.settings:
        # Remove and return the value
        value = _ENV_STATE.settings.pop(key)

        # Note: We don't trigger callbacks here as this is a removal operation
        # If needed, users can register callbacks for removal separately

        return value

    # Key not found, handle default
    if default is _NOT_PROVIDED:
        raise KeyError(f"Key '{key}' not found in global environment settings.")

    return default


def set(
    platform: Optional[PlatformType] = None,
    host_device_count: Optional[int] = None,
    precision: Optional[PrecisionType] = None,
    dt: Optional[float] = None,
    **kwargs
) -> None:
    """
    Set global environment configuration.

    This function sets persistent global environment settings that remain
    active until explicitly changed or the program terminates.

    Parameters
    ----------
    platform : str, optional
        Computing platform ('cpu', 'gpu', or 'tpu').
    host_device_count : int, optional
        Number of host devices for parallel computation.
    precision : int or str, optional
        Numerical precision (8, 16, 32, 64, or 'bf16').
    mode : Mode, optional
        Computation mode instance.
    dt : float, optional
        Time step for numerical integration.
    **kwargs
        Additional custom environment parameters.

    Raises
    ------
    ValueError
        If invalid platform or precision is specified.
    TypeError
        If mode is not a Mode instance.

    Examples
    --------
    Basic configuration:

    .. code-block:: python

        >>> import brainstate as bs
        >>> import brainstate.environ as env
        >>>
        >>> # Set multiple parameters
        >>> env.set(
        ...     precision=32,
        ...     dt=0.01,
        ...     mode=bs.mixin.Training(),
        ...     debug=False
        ... )
        >>>
        >>> print(env.get('precision'))  # 32
        >>> print(env.get('dt'))  # 0.01

    Platform configuration:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Configure for GPU computation
        >>> env.set(platform='gpu', precision=16)
        >>>
        >>> # Configure for multi-core CPU
        >>> env.set(platform='cpu', host_device_count=4)

    Custom parameters:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Set custom parameters
        >>> env.set(
        ...     experiment_name='test_001',
        ...     random_seed=42,
        ...     log_level='DEBUG'
        ... )
        >>>
        >>> # Retrieve custom parameters
        >>> print(env.get('experiment_name'))  # 'test_001'

    Notes
    -----
    - Platform changes only take effect at program start
    - Some JAX configurations require restart to take effect
    - Custom parameters can be any hashable key-value pairs
    """
    # Handle special parameters
    if platform is not None:
        set_platform(platform)

    if host_device_count is not None:
        set_host_device_count(host_device_count)

    if precision is not None:
        _validate_precision(precision)
        _set_jax_precision(precision)
        kwargs[PRECISION] = precision

    if dt is not None:
        if not u.math.isscalar(dt):
            raise TypeError(f"'{DT}' must be a scalar number, got {type(dt)}")
        kwargs[DT] = dt

    # Update global settings
    _ENV_STATE.settings.update(kwargs)

    # Trigger registered callbacks
    for key, value in kwargs.items():
        if key in _ENV_STATE.functions:
            try:
                _ENV_STATE.functions[key](value)
            except Exception as e:
                warnings.warn(
                    f"Callback for '{key}' raised an exception: {e}",
                    RuntimeWarning
                )


def get_dt() -> float:
    """
    Get the current numerical integration time step.

    Returns
    -------
    float
        The time step value.

    Raises
    ------
    KeyError
        If dt is not set.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> env.set(dt=0.01)
        >>> dt = env.get_dt()
        >>> print(f"Time step: {dt} ms")  # Time step: 0.01 ms
        >>>
        >>> # Use in computation
        >>> with env.context(dt=0.001):
        ...     fine_dt = env.get_dt()
        ...     print(f"Fine time step: {fine_dt}")  # 0.001
    """
    return get(DT)


def get_platform() -> PlatformType:
    """
    Get the current computing platform.

    Returns
    -------
    str
        Platform name ('cpu', 'gpu', or 'tpu').

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> platform = env.get_platform()
        >>> print(f"Running on: {platform}")
        >>>
        >>> if platform == 'gpu':
        ...     print("GPU acceleration available")
        ... else:
        ...     print(f"Using {platform.upper()}")
    """
    return devices()[0].platform


def get_host_device_count() -> int:
    """
    Get the number of host devices.

    Returns
    -------
    int
        Number of host devices configured.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Get device count
        >>> n_devices = env.get_host_device_count()
        >>> print(f"Host devices: {n_devices}")
        >>>
        >>> # Configure for parallel computation
        >>> if n_devices > 1:
        ...     print(f"Can use {n_devices} devices for parallel computation")
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    match = re.search(r"--xla_force_host_platform_device_count=(\d+)", xla_flags)
    return int(match.group(1)) if match else 1


def set_platform(platform: PlatformType) -> None:
    """
    Set the computing platform.

    Parameters
    ----------
    platform : str
        Platform to use ('cpu', 'gpu', or 'tpu').

    Raises
    ------
    ValueError
        If platform is not supported.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Set to GPU
        >>> env.set_platform('gpu')
        >>>
        >>> # Verify platform
        >>> print(env.get_platform())  # 'gpu'

    Notes
    -----
    Platform changes only take effect at program start. Changing platform
    after JAX initialization may not have the expected effect.
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Platform must be one of {SUPPORTED_PLATFORMS}, got '{platform}'"
        )

    config.update("jax_platform_name", platform)

    # Trigger callbacks
    if PLATFORM in _ENV_STATE.functions:
        _ENV_STATE.functions[PLATFORM](platform)


def set_host_device_count(n: int) -> None:
    """
    Set the number of host (CPU) devices.

    This function configures XLA to treat CPU cores as separate devices,
    enabling parallel computation with jax.pmap on CPU.

    Parameters
    ----------
    n : int
        Number of host devices to configure.

    Raises
    ------
    ValueError
        If n is not a positive integer.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>> import jax
        >>>
        >>> # Configure 4 CPU devices
        >>> env.set_host_device_count(4)
        >>>
        >>> # Use with pmap
        >>> def parallel_fn(x):
        ...     return x * 2
        >>>
        >>> # This will work with 4 devices
        >>> pmapped_fn = jax.pmap(parallel_fn)

    Warnings
    --------
    This setting only takes effect at program start. The effects of using
    xla_force_host_platform_device_count are not fully understood and may
    cause unexpected behavior.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"Host device count must be a positive integer, got {n}")

    # Update XLA flags
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+",
        "",
        xla_flags
    ).split()

    os.environ["XLA_FLAGS"] = " ".join(
        [f"--xla_force_host_platform_device_count={n}"] + xla_flags
    )

    # Trigger callbacks
    if HOST_DEVICE_COUNT in _ENV_STATE.functions:
        _ENV_STATE.functions[HOST_DEVICE_COUNT](n)


def set_precision(precision: PrecisionType) -> None:
    """
    Set the global numerical precision.

    Parameters
    ----------
    precision : int or str
        Precision to use (8, 16, 32, 64, or 'bf16').

    Raises
    ------
    ValueError
        If precision is not supported.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>> import jax.numpy as jnp
        >>>
        >>> # Set to 64-bit precision
        >>> env.set_precision(64)
        >>>
        >>> # Arrays will use float64 by default
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> print(x.dtype)  # float64
        >>>
        >>> # Set to bfloat16 for efficiency
        >>> env.set_precision('bf16')
    """
    _validate_precision(precision)
    _set_jax_precision(precision)
    _ENV_STATE.settings[PRECISION] = precision

    # Trigger callbacks
    if PRECISION in _ENV_STATE.functions:
        _ENV_STATE.functions[PRECISION](precision)


def get_precision() -> int:
    """
    Get the current numerical precision as an integer.

    Returns
    -------
    int
        Precision in bits (8, 16, 32, or 64).

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> env.set_precision(32)
        >>> bits = env.get_precision()
        >>> print(f"Using {bits}-bit precision")  # Using 32-bit precision
        >>>
        >>> # Special handling for bfloat16
        >>> env.set_precision('bf16')
        >>> print(env.get_precision())  # 16

    Notes
    -----
    'bf16' (bfloat16) is reported as 16-bit precision.
    """
    precision = get(PRECISION, default=DEFAULT_PRECISION)

    if precision == 'bf16':
        return 16
    elif isinstance(precision, str):
        return int(precision)
    elif isinstance(precision, int):
        return precision
    else:
        raise ValueError(f"Invalid precision type: {type(precision)}")


def _validate_precision(precision: PrecisionType) -> None:
    """Validate precision value."""
    if precision not in SUPPORTED_PRECISIONS and str(precision) not in map(str, SUPPORTED_PRECISIONS):
        raise ValueError(
            f"Precision must be one of {SUPPORTED_PRECISIONS}, got {precision}"
        )


def _get_precision() -> PrecisionType:
    """Get raw precision value (including 'bf16')."""
    return get(PRECISION, default=DEFAULT_PRECISION)


def _set_jax_precision(precision: PrecisionType) -> None:
    """Configure JAX precision settings."""
    # Enable/disable 64-bit mode
    if precision in (64, '64'):
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)


@functools.lru_cache(maxsize=16)
def _get_uint(precision: PrecisionType) -> DTypeLike:
    """Get unsigned integer type for given precision."""
    if precision in (64, '64'):
        return np.uint64
    elif precision in (32, '32'):
        return np.uint32
    elif precision in (16, '16', 'bf16'):
        return np.uint16
    elif precision in (8, '8'):
        return np.uint8
    else:
        raise ValueError(f"Unsupported precision: {precision}")


@functools.lru_cache(maxsize=16)
def _get_int(precision: PrecisionType) -> DTypeLike:
    """Get integer type for given precision."""
    if precision in (64, '64'):
        return np.int64
    elif precision in (32, '32'):
        return np.int32
    elif precision in (16, '16', 'bf16'):
        return np.int16
    elif precision in (8, '8'):
        return np.int8
    else:
        raise ValueError(f"Unsupported precision: {precision}")


@functools.lru_cache(maxsize=16)
def _get_float(precision: PrecisionType) -> DTypeLike:
    """Get floating-point type for given precision."""
    if precision in (64, '64'):
        return np.float64
    elif precision in (32, '32'):
        return np.float32
    elif precision in (16, '16'):
        return np.float16
    elif precision == 'bf16':
        return jnp.bfloat16
    elif precision in (8, '8'):
        return jnp.float8_e5m2
    else:
        raise ValueError(f"Unsupported precision: {precision}")


@functools.lru_cache(maxsize=16)
def _get_complex(precision: PrecisionType) -> DTypeLike:
    """Get complex type for given precision."""
    if precision in (64, '64'):
        return np.complex128
    elif precision in (32, '32', 16, '16', 'bf16', 8, '8'):
        return np.complex64
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def dftype() -> DTypeLike:
    """
    Get the default floating-point data type.

    This function returns the appropriate floating-point type based on
    the current precision setting, allowing dynamic type selection.

    Returns
    -------
    DTypeLike
        Default floating-point data type.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>> import jax.numpy as jnp
        >>>
        >>> # With 32-bit precision
        >>> env.set(precision=32)
        >>> x = jnp.zeros(10, dtype=env.dftype())
        >>> print(x.dtype)  # float32
        >>>
        >>> # With 64-bit precision
        >>> with env.context(precision=64):
        ...     y = jnp.ones(5, dtype=env.dftype())
        ...     print(y.dtype)  # float64
        >>>
        >>> # With bfloat16
        >>> env.set(precision='bf16')
        >>> z = jnp.array([1, 2, 3], dtype=env.dftype())
        >>> print(z.dtype)  # bfloat16

    See Also
    --------
    ditype : Default integer type
    dutype : Default unsigned integer type
    dctype : Default complex type
    """
    return _get_float(_get_precision())


def ditype() -> DTypeLike:
    """
    Get the default integer data type.

    This function returns the appropriate integer type based on
    the current precision setting.

    Returns
    -------
    DTypeLike
        Default integer data type.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>> import jax.numpy as jnp
        >>>
        >>> # With 32-bit precision
        >>> env.set(precision=32)
        >>> indices = jnp.arange(10, dtype=env.ditype())
        >>> print(indices.dtype)  # int32
        >>>
        >>> # With 64-bit precision
        >>> with env.context(precision=64):
        ...     big_indices = jnp.arange(1000, dtype=env.ditype())
        ...     print(big_indices.dtype)  # int64

    See Also
    --------
    dftype : Default floating-point type
    dutype : Default unsigned integer type
    """
    return _get_int(_get_precision())


def dutype() -> DTypeLike:
    """
    Get the default unsigned integer data type.

    This function returns the appropriate unsigned integer type based on
    the current precision setting.

    Returns
    -------
    DTypeLike
        Default unsigned integer data type.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>> import jax.numpy as jnp
        >>>
        >>> # With 32-bit precision
        >>> env.set(precision=32)
        >>> counts = jnp.array([10, 20, 30], dtype=env.dutype())
        >>> print(counts.dtype)  # uint32
        >>>
        >>> # With 16-bit precision
        >>> with env.context(precision=16):
        ...     small_counts = jnp.array([1, 2, 3], dtype=env.dutype())
        ...     print(small_counts.dtype)  # uint16

    See Also
    --------
    ditype : Default signed integer type
    """
    return _get_uint(_get_precision())


def dctype() -> DTypeLike:
    """
    Get the default complex data type.

    This function returns the appropriate complex type based on
    the current precision setting.

    Returns
    -------
    DTypeLike
        Default complex data type.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>> import jax.numpy as jnp
        >>>
        >>> # With 32-bit precision
        >>> env.set(precision=32)
        >>> z = jnp.array([1+2j, 3+4j], dtype=env.dctype())
        >>> print(z.dtype)  # complex64
        >>>
        >>> # With 64-bit precision
        >>> with env.context(precision=64):
        ...     w = jnp.array([5+6j], dtype=env.dctype())
        ...     print(w.dtype)  # complex128

    Notes
    -----
    Complex128 is only available with 64-bit precision.
    All other precisions use complex64.
    """
    return _get_complex(_get_precision())


def tolerance() -> jnp.ndarray:
    """
    Get numerical tolerance based on current precision.

    This function returns an appropriate tolerance value for numerical
    comparisons based on the current precision setting.

    Returns
    -------
    jnp.ndarray
        Tolerance value as a scalar array.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>> import jax.numpy as jnp
        >>>
        >>> # Different tolerances for different precisions
        >>> env.set(precision=64)
        >>> tol64 = env.tolerance()
        >>> print(f"64-bit tolerance: {tol64}")  # 1e-12
        >>>
        >>> env.set(precision=32)
        >>> tol32 = env.tolerance()
        >>> print(f"32-bit tolerance: {tol32}")  # 1e-5
        >>>
        >>> # Use in numerical comparisons
        >>> def are_close(a, b):
        ...     return jnp.abs(a - b) < env.tolerance()

    Notes
    -----
    Tolerance values:
    - 64-bit: 1e-12
    - 32-bit: 1e-5
    - 16-bit and below: 1e-2
    """
    precision = get_precision()

    if precision == 64:
        return jnp.array(1e-12, dtype=np.float64)
    elif precision == 32:
        return jnp.array(1e-5, dtype=np.float32)
    else:
        return jnp.array(1e-2, dtype=np.float16)


def register_default_behavior(
    key: str,
    behavior: Callable[[Any], None],
    replace_if_exist: bool = False
) -> None:
    """
    Register a callback for environment parameter changes.

    This function allows you to register custom behaviors that are
    triggered whenever a specific environment parameter is modified.

    Parameters
    ----------
    key : str
        Environment parameter key to monitor.
    behavior : Callable[[Any], None]
        Callback function that receives the new value.
    replace_if_exist : bool, default=False
        Whether to replace existing callback for this key.

    Raises
    ------
    TypeError
        If behavior is not callable.
    ValueError
        If key already has a registered behavior and replace_if_exist is False.

    Examples
    --------
    Basic callback registration:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Define a callback
        >>> def on_dt_change(new_dt):
        ...     print(f"Time step changed to: {new_dt}")
        >>>
        >>> # Register the callback
        >>> env.register_default_behavior('dt', on_dt_change)
        >>>
        >>> # Callback is triggered on changes
        >>> env.set(dt=0.01)  # Prints: Time step changed to: 0.01
        >>>
        >>> with env.context(dt=0.001):  # Prints: Time step changed to: 0.001
        ...     pass  # Prints: Time step changed to: 0.01 (on exit)

    Complex behavior with validation:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> def validate_batch_size(size):
        ...     if not isinstance(size, int) or size <= 0:
        ...         raise ValueError(f"Invalid batch size: {size}")
        ...     if size > 1024:
        ...         print(f"Warning: Large batch size {size} may cause OOM")
        >>>
        >>> env.register_default_behavior('batch_size', validate_batch_size)
        >>>
        >>> # Valid setting
        >>> env.set(batch_size=32)  # OK
        >>>
        >>> # Invalid setting
        >>> # env.set(batch_size=-1)  # Raises ValueError

    Replacing existing behavior:

    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> def old_behavior(value):
        ...     print(f"Old: {value}")
        >>>
        >>> def new_behavior(value):
        ...     print(f"New: {value}")
        >>>
        >>> env.register_default_behavior('key', old_behavior)
        >>> env.register_default_behavior('key', new_behavior, replace_if_exist=True)
        >>>
        >>> env.set(key='test')  # Prints: New: test

    See Also
    --------
    unregister_default_behavior : Remove registered callbacks
    list_registered_behaviors : List all registered callbacks
    """
    if not isinstance(key, str):
        raise TypeError(f"Key must be a string, got {type(key)}")

    if not callable(behavior):
        raise TypeError(f"Behavior must be callable, got {type(behavior)}")

    if key in _ENV_STATE.functions and not replace_if_exist:
        raise ValueError(
            f"Behavior for key '{key}' already registered. "
            f"Use replace_if_exist=True to override."
        )

    _ENV_STATE.functions[key] = behavior


def unregister_default_behavior(key: str) -> bool:
    """
    Remove a registered callback for an environment parameter.

    Parameters
    ----------
    key : str
        Environment parameter key.

    Returns
    -------
    bool
        True if a callback was removed, False if no callback existed.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Register a callback
        >>> def callback(value):
        ...     print(f"Value: {value}")
        >>>
        >>> env.register_default_behavior('param', callback)
        >>>
        >>> # Remove the callback
        >>> removed = env.unregister_default_behavior('param')
        >>> print(f"Callback removed: {removed}")  # True
        >>>
        >>> # No callback triggers now
        >>> env.set(param='test')  # No output
        >>>
        >>> # Removing non-existent callback
        >>> removed = env.unregister_default_behavior('nonexistent')
        >>> print(f"Callback removed: {removed}")  # False
    """
    if key in _ENV_STATE.functions:
        del _ENV_STATE.functions[key]
        return True
    return False


def list_registered_behaviors() -> List[str]:
    """
    List all keys with registered callbacks.

    Returns
    -------
    list of str
        Keys that have registered behavior callbacks.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate.environ as env
        >>>
        >>> # Register some callbacks
        >>> env.register_default_behavior('param1', lambda x: None)
        >>> env.register_default_behavior('param2', lambda x: None)
        >>>
        >>> # List registered behaviors
        >>> behaviors = env.list_registered_behaviors()
        >>> print(f"Registered: {behaviors}")  # ['param1', 'param2']
        >>>
        >>> # Check if specific behavior is registered
        >>> if 'dt' in behaviors:
        ...     print("dt has a registered callback")
    """
    return list(_ENV_STATE.functions.keys())


# Initialize default precision on module load
set(precision=DEFAULT_PRECISION)
