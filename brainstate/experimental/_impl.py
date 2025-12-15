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
    """Return the identifiers of every registered device implementation.

    Returns
    -------
    list of str
        Device identifiers in the order they were registered.

    Notes
    -----
    The returned list is a copy; mutating it does not affect the registry.

    Examples
    --------
    >>> get_registered_devices()
    ['cpu', 'gpu', 'tpu']
    """
    return list(registered_devices.keys())


def is_device_registered(device: str) -> bool:
    """Check whether a device has registered implementations.

    Parameters
    ----------
    device : str
        Device identifier such as ``"cpu"``, ``"gpu"``, or a custom name.

    Returns
    -------
    bool
        ``True`` when the device exists in the registry, ``False`` otherwise.

    Examples
    --------
    >>> is_device_registered('gpu')
    True
    >>> is_device_registered('custom_device')
    False
    """
    return device in registered_devices


def get_forloop_impl(device: str) -> Callable:
    """Return the device-specific ``for_loop`` wrapper.

    Parameters
    ----------
    device : str
        Target device identifier (for example ``"cpu"`` or ``"gpu"``).

    Returns
    -------
    Callable
        Callable that accepts a user function plus optional keyword arguments
        and returns a wrapped looping function.

    Raises
    ------
    ValueError
        If the device is not registered or no for-loop implementation was
        registered for it.

    Examples
    --------
    >>> loop_impl = get_forloop_impl('cpu')
    >>> compiled = loop_impl(fn)
    >>> compiled(xs)
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
    """Return the device-specific JIT compilation wrapper.

    Parameters
    ----------
    device : str
        Target device identifier (for example ``"cpu"`` or ``"tpu"``).

    Returns
    -------
    Callable
        Callable that accepts a function plus optional keyword arguments
        and returns the compiled callable.

    Raises
    ------
    ValueError
        If the device is not registered or no JIT implementation is available.

    Examples
    --------
    >>> jit_impl = get_jit_impl('gpu')
    >>> compiled_fn = jit_impl(fn, static_argnums=(0,))
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
    """Add or replace the ``for_loop`` implementation for a device.

    Parameters
    ----------
    device : str
        Device identifier to register (e.g., ``"cpu"``, ``"gpu"``, or custom).
    impl : Callable
        Factory that accepts a user function and optional keyword arguments and
        returns the wrapped loop implementation.

    Raises
    ------
    TypeError
        If ``impl`` is not callable.

    Notes
    -----
    A previous implementation for the same device is silently overwritten.

    Examples
    --------
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
    """Add or replace the JIT implementation for a device.

    Parameters
    ----------
    device : str
        Device identifier to register (e.g., ``"cpu"``, ``"tpu"``, or custom).
    impl : Callable
        Factory that accepts a user function plus optional keyword arguments and
        returns its compiled counterpart.

    Raises
    ------
    TypeError
        If ``impl`` is not callable.

    Notes
    -----
    A previous implementation for the same device is silently overwritten.

    Examples
    --------
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
    """Remove a device and all associated implementations.

    Parameters
    ----------
    device : str
        Device identifier to remove from the registry.

    Returns
    -------
    bool
        ``True`` if the device existed and was removed, ``False`` otherwise.

    Examples
    --------
    >>> unregister_device('temp_device')
    True
    >>> unregister_device('unknown')
    False
    """
    if device in registered_devices:
        del registered_devices[device]
        return True
    return False


def clear_all_registrations() -> None:
    """Remove every device registration from the global registry.

    This helper is primarily intended for tests or scenarios where a clean slate
    is required before re-registering device implementations.

    Notes
    -----
    The operation is destructive and affects subsequent calls to ``ForLoop`` and
    ``JIT`` until new implementations are registered.

    Examples
    --------
    >>> clear_all_registrations()
    >>> get_registered_devices()
    []
    """
    registered_devices.clear()


def _for_loop_wrapper(fn: Callable, **kwargs: Any) -> Callable:
    """Create the default ``for_loop`` wrapper for standard devices.

    Parameters
    ----------
    fn : Callable
        User function to execute inside the ``for_loop`` transformation.
    **kwargs : Any
        Extra keyword arguments forwarded to :func:`brainstate.transform.for_loop`.

    Returns
    -------
    Callable
        Function that applies ``for_loop`` to ``fn`` when invoked.

    Notes
    -----
    Used internally for the built-in ``'cpu'``, ``'gpu'``, and ``'tpu'`` devices.
    """

    def run(*args: Any, **run_kwargs: Any) -> Any:
        """Execute ``fn`` using ``for_loop`` with the captured configuration."""
        return for_loop(fn, *args, **run_kwargs)

    return run


def _jit_wrapper(fn: Callable, **jit_kwargs: Any) -> Callable:
    """Create the default JIT wrapper for standard devices.

    Parameters
    ----------
    fn : Callable
        User function to compile.
    **jit_kwargs : Any
        Extra keyword arguments forwarded to :func:`brainstate.transform._jit.jit`.

    Returns
    -------
    Callable
        The compiled callable returned by :func:`jit`.

    Notes
    -----
    Used internally for the built-in ``'cpu'``, ``'gpu'``, and ``'tpu'`` devices.
    """
    return jit(fn, **jit_kwargs)


def _register_default_devices() -> None:
    """Populate the registry with the built-in device implementations.

    Notes
    -----
    Invoked automatically at module import. The default devices (``'cpu'``,
    ``'gpu'``, ``'tpu'``) all share the same wrappers defined in this module.
    """
    for device in ['cpu', 'gpu', 'tpu']:
        register_forloop_impl(device, _for_loop_wrapper)
        register_jit_impl(device, _jit_wrapper)


# Register default implementations on module import
_register_default_devices()
