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

"""Global hook registry for hooks that apply to all State instances."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from ._state_hook_manager import HookManager, HookConfig
from ._state_hook_core import HookHandle
from ._state_hook_context import HookContext

__all__ = [
    'GlobalHookRegistry',
    'register_state_hook',
    'unregister_state_hook',
    'clear_state_hooks',
    'has_state_hooks',
    'list_state_hooks',
]


class GlobalHookRegistry(HookManager):
    """Singleton registry for global hooks that apply to all State instances.

    Global hooks are executed before instance-specific hooks for each operation.
    This is useful for system-wide monitoring, logging, or validation.

    The global registry is a singleton, accessed via GlobalHookRegistry.instance().

    Thread Safety:
        The global registry is thread-safe, using the same locking mechanism
        as HookManager.

    Example:
        >>> # Register a global hook that logs all state reads
        >>> def log_all_reads(ctx):
        ...     print(f"Global: Reading {ctx.state_name}")
        >>> handle = GlobalHookRegistry.instance().register_hook('read', log_all_reads)
        >>>
        >>> # Now all State instances will trigger this hook
        >>> import brainstate
        >>> s1 = brainstate.State(1)
        >>> s2 = brainstate.State(2)
        >>> _ = s1.value  # Prints: Global: Reading None
        >>> _ = s2.value  # Prints: Global: Reading None
    """

    _instance: Optional['GlobalHookRegistry'] = None
    _initialized: bool = False

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[HookConfig] = None):
        """Initialize the global hook registry.

        Note: This is only called once due to the singleton pattern.

        Args:
            config: Optional HookConfig for customizing error handling
        """
        # Only initialize once
        if not GlobalHookRegistry._initialized:
            super().__init__(config)
            GlobalHookRegistry._initialized = True

    @classmethod
    def instance(cls) -> 'GlobalHookRegistry':
        """Get the singleton instance of the global hook registry.

        Returns:
            The GlobalHookRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global hook registry (useful for testing).

        Warning:
            This will clear all global hooks and create a new instance.
            Use with caution in production code.
        """
        cls._instance = None
        cls._initialized = False


# Module-level convenience functions

def register_state_hook(
    hook_type: Literal['read', 'write_before', 'write_after', 'restore', 'init'],
    callback: Callable[[HookContext], Any],
    priority: int = 0,
    name: Optional[str] = None,
    enabled: bool = True,
) -> HookHandle:
    """Register a global hook that applies to all State instances.

    Global hooks execute before instance-specific hooks.

    Args:
        hook_type: Type of hook ('read', 'write_before', 'write_after', 'restore', 'init')
        callback: Callable that receives a HookContext
        priority: Execution priority (higher = earlier, default 0)
        name: Optional name for the hook
        enabled: Whether the hook is enabled initially (default True)

    Returns:
        HookHandle for managing the hook

    Example:
        >>> import brainstate
        >>> def validate_all_writes(ctx):
        ...     if hasattr(ctx.value, 'shape'):
        ...         print(f"Writing array with shape {ctx.value.shape}")
        >>> handle = brainstate.register_state_hook('write_before', validate_all_writes)
    """
    return GlobalHookRegistry.instance().register_hook(
        hook_type, callback, priority, name, enabled
    )


def unregister_state_hook(handle: HookHandle) -> bool:
    """Unregister a global hook using its handle.

    Args:
        handle: The HookHandle returned by register_global_hook

    Returns:
        True if successfully unregistered, False otherwise
    """
    return GlobalHookRegistry.instance().unregister_hook(handle)


def clear_state_hooks(hook_type: Optional[str] = None) -> None:
    """Clear all global hooks, optionally filtered by type.

    Args:
        hook_type: Optional hook type to clear (if None, clears all)
    """
    GlobalHookRegistry.instance().clear_hooks(hook_type)


def has_state_hooks(hook_type: Optional[str] = None) -> bool:
    """Check if any global hooks are registered.

    Args:
        hook_type: Optional hook type to check (if None, checks all)

    Returns:
        True if global hooks are registered, False otherwise
    """
    return GlobalHookRegistry.instance().has_hooks(hook_type)


def list_state_hooks(hook_type: Optional[str] = None):
    """List all registered global hooks, optionally filtered by type.

    Args:
        hook_type: Optional hook type to filter by

    Returns:
        List of Hook objects
    """
    return GlobalHookRegistry.instance().get_hooks(hook_type)
