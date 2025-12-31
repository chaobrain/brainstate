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

"""Core hook classes and exception types."""

from __future__ import annotations

import weakref
from typing import Any, Callable, Optional, TYPE_CHECKING

from ._state_hook_context import HookContext

if TYPE_CHECKING:
    from ._state_hook_manager import HookManager

__all__ = [
    'Hook',
    'HookHandle',
    'HookError',
    'HookExecutionError',
    'HookRegistrationError',
    'HookCancellationError',
    'HookWarning',
]


# Exception classes
class HookError(Exception):
    """Base exception for hook-related errors."""
    pass


class HookExecutionError(HookError):
    """Exception raised when a hook execution fails."""
    pass


class HookRegistrationError(HookError):
    """Exception raised when hook registration fails."""
    pass


class HookCancellationError(HookError):
    """Exception raised when a hook cancels an operation."""
    pass


class HookWarning(UserWarning):
    """Warning for hook-related issues."""
    pass


# Hook class
class Hook:
    """Base hook class for all hook types.

    A hook encapsulates a callback function along with metadata about its
    execution priority, name, and enabled state.

    Attributes:
        callback: The callable to invoke when the hook executes
        priority: Execution priority (higher = executes earlier)
        name: Optional name for the hook (for debugging/logging)
        enabled: Whether the hook is currently enabled
        hook_id: Unique identifier for the hook
    """

    _id_counter = 0

    def __init__(
        self,
        callback: Callable[[HookContext], Any],
        priority: int = 0,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        """Initialize a hook.

        Args:
            callback: Callable that receives a HookContext and optionally returns a value
            priority: Priority for execution order (higher = earlier, default 0)
            name: Optional name for the hook
            enabled: Whether the hook is enabled initially (default True)
        """
        if not callable(callback):
            raise HookRegistrationError(f"Hook callback must be callable, got {type(callback)}")

        self.callback = callback
        self.priority = priority
        self.name = name or f"hook_{Hook._id_counter}"
        self.enabled = enabled
        self._error_count = 0
        self.hook_id = Hook._id_counter
        Hook._id_counter += 1

    def execute(self, context: HookContext) -> Optional[Any]:
        """Execute the hook callback with the given context.

        Args:
            context: The hook context to pass to the callback

        Returns:
            The return value from the callback, if any

        Raises:
            HookExecutionError: If the callback raises an exception
        """
        if not self.enabled:
            return None

        try:
            result = self.callback(context)
            return result
        except Exception as e:
            # Don't increment error count here - it's handled by HookManager._handle_hook_error
            raise HookExecutionError(f"Hook '{self.name}' execution failed") from e

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"Hook(name='{self.name}', priority={self.priority}, {status})"

    def __lt__(self, other: 'Hook') -> bool:
        """Compare hooks by priority (for sorting).

        Higher priority hooks should come first, so we reverse the comparison.
        If priorities are equal, compare by hook_id for stable sorting.
        """
        if self.priority != other.priority:
            return self.priority > other.priority  # Descending priority
        return self.hook_id < other.hook_id  # Ascending ID for stability


class HookHandle:
    """Handle for managing a registered hook.

    This handle provides methods to enable, disable, and remove hooks
    without directly accessing the HookManager.

    Example:
        >>> state = bst.State(0, enable_hooks=True)
        >>> handle = state.register_hook('read', lambda ctx: print(ctx.value))
        >>> handle.disable()
        >>> state.value  # Hook not executed
        >>> handle.enable()
        >>> state.value  # Hook executed
        >>> handle.remove()  # Permanently unregister
    """

    def __init__(self, manager_ref: weakref.ref[HookManager], hook: Hook, hook_type: str):
        """Initialize a hook handle.

        Args:
            manager_ref: Weak reference to the HookManager that owns this hook
            hook: The Hook instance being managed
            hook_type: Type of hook ('read', 'write_before', 'write_after', 'restore')
        """
        self._manager_ref = manager_ref
        self._hook = hook
        self._hook_type = hook_type
        self._removed = False

    def enable(self) -> None:
        """Enable the hook.

        Raises:
            HookError: If the hook has been removed or the manager is gone
        """
        if self._removed:
            raise HookError("Cannot enable a removed hook")

        manager = self._manager_ref()
        if manager is None:
            raise HookError("HookManager has been garbage collected")

        self._hook.enabled = True
        manager._invalidate_cache()

    def disable(self) -> None:
        """Disable the hook.

        Raises:
            HookError: If the hook has been removed or the manager is gone
        """
        if self._removed:
            raise HookError("Cannot disable a removed hook")

        manager = self._manager_ref()
        if manager is None:
            raise HookError("HookManager has been garbage collected")

        self._hook.enabled = False
        manager._invalidate_cache()

    def remove(self) -> bool:
        """Remove the hook permanently.

        Returns:
            True if the hook was successfully removed, False otherwise
        """
        if self._removed:
            return False

        manager = self._manager_ref()
        if manager is None:
            return False

        success = manager.unregister_hook(self)
        if success:
            self._removed = True
        return success

    def is_enabled(self) -> bool:
        """Check if the hook is currently enabled.

        Returns:
            True if enabled, False otherwise (including if removed)
        """
        if self._removed:
            return False
        return self._hook.enabled

    def is_removed(self) -> bool:
        """Check if the hook has been removed.

        Returns:
            True if removed, False otherwise
        """
        return self._removed

    @property
    def name(self) -> str:
        """Get the hook's name."""
        return self._hook.name

    @property
    def priority(self) -> int:
        """Get the hook's priority."""
        return self._hook.priority

    def __repr__(self) -> str:
        if self._removed:
            return f"HookHandle(name='{self.name}', removed)"
        status = "enabled" if self.is_enabled() else "disabled"
        return f"HookHandle(name='{self.name}', priority={self.priority}, {status})"
