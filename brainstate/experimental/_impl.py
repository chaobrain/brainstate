# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Device-specific implementation registry for JIT compilation and loop transformations.

This module provides a registry system for managing device-specific implementations
of core transformations like JIT compilation and for-loops. It allows registration
of custom implementations for different compute devices (CPU, GPU, TPU, etc.).
"""

from typing import Callable, Dict, List, Any

from brainstate.transform._jit import jit
from brainstate.transform._loop_collect_return import for_loop

__all__ = [
    "get_registered_devices",
    'get_forloop_impl',
    'get_jit_impl',
    'register_forloop_impl',
    'register_jit_impl',
    'unregister_device',
    'is_device_registered',
    'clear_all_registrations',
]

# Type alias for device implementation registry
DeviceRegistry = Dict[str, Dict[str, Callable]]

# Global registry for device-specific implementations
registered_devices: DeviceRegistry = {}


def get_registered_devices() -> List[str]:
    """Get a list of all registered device names.

    Parameters
    ----------

    Returns
    -------
    List[str]
        A list of device names that have been registered in the system.
        Common devices include 'cpu', 'gpu', 'tpu'.

    Examples
    --------
    >>> devices = get_registered_devices()
    >>> print(devices)
    ['cpu', 'gpu', 'tpu']
    """
    return list(registered_devices.keys())


def is_device_registered(device: str) -> bool:
    """Check if a device is registered in the system.

    Parameters
    ----------
    device : str
        The device name to check.
    device: str :
        

    Returns
    -------
    bool
        True if the device is registered, False otherwise.

    Examples
    --------
    >>> is_device_registered('gpu')
    True
    >>> is_device_registered('custom_device')
    False
    """
    return device in registered_devices


def get_forloop_impl(device: str) -> Callable:
    """Get the for-loop implementation for a specific device.

    Parameters
    ----------
    device : str
        The target device name (e.g., 'cpu', 'gpu', 'tpu').
    device: str :
        

    Returns
    -------
    Callable
        The for-loop implementation function registered for the device.

    Raises
    ------
    ValueError
        If the device is not registered or if the device doesn't have
        a for-loop implementation.

    Examples
    --------
    >>> forloop_impl = get_forloop_impl('cpu')
    >>> wrapped_fn = forloop_impl(my_function)
    """
    if device not in registered_devices:
        available = get_registered_devices()
        raise ValueError(
            f"Device '{device}' is not registered.\n"
            f"Available devices: {available}\n"
            f"Use register_forloop_impl() to register a new device."
        )

    if 'forloop' not in registered_devices[device]:
        raise ValueError(
            f"Device '{device}' is registered but has no for-loop implementation.\n"
            f"Use register_forloop_impl('{device}', impl) to add one."
        )

    return registered_devices[device]['forloop']


def get_jit_impl(device: str) -> Callable:
    """Get the JIT compilation implementation for a specific device.

    Parameters
    ----------
    device : str
        The target device name (e.g., 'cpu', 'gpu', 'tpu').
    device: str :
        

    Returns
    -------
    Callable
        The JIT implementation function registered for the device.

    Raises
    ------
    ValueError
        If the device is not registered or if the device doesn't have
        a JIT implementation.

    Examples
    --------
    >>> jit_impl = get_jit_impl('gpu')
    >>> compiled_fn = jit_impl(my_function)
    """
    if device not in registered_devices:
        available = get_registered_devices()
        raise ValueError(
            f"Device '{device}' is not registered.\n"
            f"Available devices: {available}\n"
            f"Use register_jit_impl() to register a new device."
        )

    if 'jit' not in registered_devices[device]:
        raise ValueError(
            f"Device '{device}' is registered but has no JIT implementation.\n"
            f"Use register_jit_impl('{device}', impl) to add one."
        )

    return registered_devices[device]['jit']


def register_forloop_impl(device: str, impl: Callable) -> None:
    """Register a for-loop implementation for a specific device.

    Parameters
    ----------
    device : str
        The device name to register (e.g., 'cpu', 'gpu', 'tpu', 'custom').
    impl : Callable
        The implementation function that wraps functions for for-loop execution.
        Should accept a function and return a wrapped callable.
    device: str :
        
    impl: Callable :
        

    Returns
    -------

    Raises
    ------
    TypeError
        If impl is not callable.

    Examples
    --------
    Notes
    -----
    If the device already has a for-loop implementation, it will be overwritten.
    >>> def my_forloop_impl(fn, **kwargs):
    ...     def wrapper(*args, **kw):
    ...         return for_loop(fn, *args, **kw)
    ...     return wrapper
    >>> register_forloop_impl('custom_device', my_forloop_impl)
    """
    if not callable(impl):
        raise TypeError(
            f"Expected a callable implementation, got {type(impl).__name__}.\n"
            f"The implementation should be a function that wraps other functions."
        )

    if device not in registered_devices:
        registered_devices[device] = {}
    registered_devices[device]['forloop'] = impl


def register_jit_impl(device: str, impl: Callable) -> None:
    """Register a JIT compilation implementation for a specific device.

    Parameters
    ----------
    device : str
        The device name to register (e.g., 'cpu', 'gpu', 'tpu', 'custom').
    impl : Callable
        The implementation function that JIT-compiles functions.
        Should accept a function and optional kwargs, returning a compiled callable.
    device: str :
        
    impl: Callable :
        

    Returns
    -------

    Raises
    ------
    TypeError
        If impl is not callable.

    Examples
    --------
    Notes
    -----
    If the device already has a JIT implementation, it will be overwritten.
    >>> def my_jit_impl(fn, **jit_kwargs):
    ...     return jit(fn, **jit_kwargs)
    >>> register_jit_impl('custom_device', my_jit_impl)
    """
    if not callable(impl):
        raise TypeError(
            f"Expected a callable implementation, got {type(impl).__name__}.\n"
            f"The implementation should be a function that compiles other functions."
        )

    if device not in registered_devices:
        registered_devices[device] = {}
    registered_devices[device]['jit'] = impl


def unregister_device(device: str) -> bool:
    """Unregister a device and remove all its implementations.

    Parameters
    ----------
    device : str
        The device name to unregister.
    device: str :
        

    Returns
    -------
    bool
        True if the device was successfully unregistered, False if it wasn't registered.

    Examples
    --------
    >>> register_jit_impl('temp_device', my_impl)
    >>> unregister_device('temp_device')
    True
    >>> unregister_device('nonexistent_device')
    False
    """
    if device in registered_devices:
        del registered_devices[device]
        return True
    return False


def clear_all_registrations() -> None:
    """Clear all device registrations from the system.
    
    This removes all registered devices and their implementations.
    Use with caution as this will affect all code relying on registered devices.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    Notes
    -----
    This is primarily useful for testing or when you need to completely
    reset the device registry state.
    >>> clear_all_registrations()
    >>> get_registered_devices()
    []
    """
    registered_devices.clear()


def _for_loop_wrapper(fn: Callable, **kwargs: Any) -> Callable:
    """Default for-loop wrapper implementation.
    
    This function wraps a given function to execute using BrainState's for_loop
    transformation. It serves as the default implementation for all standard devices.

    Parameters
    ----------
    fn : Callable
        The function to be wrapped for for-loop execution.
    **kwargs : Any
        Additional keyword arguments to pass to the for_loop transformation.
    fn: Callable :
        
    **kwargs: Any :
        

    Returns
    -------
    Callable
        A wrapped function that executes using for_loop.

    Notes
    -----
    This is an internal implementation used as the default for 'cpu', 'gpu', and 'tpu' devices.
    """

    def run(*args: Any, **run_kwargs: Any) -> Any:
        """

        Parameters
        ----------
        *args: Any :
            
        **run_kwargs: Any :
            

        Returns
        -------

        """
        return for_loop(fn, *args, **run_kwargs)

    return run


def _jit_wrapper(fn: Callable, **jit_kwargs: Any) -> Callable:
    """Default JIT compilation wrapper implementation.
    
    This function wraps a given function with BrainState's JIT compiler.
    It serves as the default implementation for all standard devices.

    Parameters
    ----------
    fn : Callable
        The function to be JIT-compiled.
    **jit_kwargs : Any
        Additional keyword arguments to pass to the JIT compiler.
    fn: Callable :
        
    **jit_kwargs: Any :
        

    Returns
    -------
    Callable
        A JIT-compiled version of the function.

    Notes
    -----
    This is an internal implementation used as the default for 'cpu', 'gpu', and 'tpu' devices.
    """
    return jit(fn, **jit_kwargs)


def _register_default_devices() -> None:
    """Register default device implementations for standard compute devices.
    
    This function registers the default for-loop and JIT implementations
    for CPU, GPU, and TPU devices. It is called automatically on module import.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    The default devices are: 'cpu', 'gpu', 'tpu'.
    All use the same underlying BrainState transformations (for_loop and jit).
    """
    for device in ['cpu', 'gpu', 'tpu']:
        register_forloop_impl(device, _for_loop_wrapper)
        register_jit_impl(device, _jit_wrapper)


# Register default implementations on module import
_register_default_devices()
