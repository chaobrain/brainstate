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

"""Hook manager for managing and executing hooks."""

from __future__ import annotations

import threading
import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

from ._state_hook_context import (
    HookContext,
    ReadHookContext,
    MutableWriteHookContext,
    WriteHookContext,
    RestoreHookContext,
)
from ._state_hook_core import (
    Hook,
    HookHandle,
    HookExecutionError,
    HookCancellationError,
    HookWarning,
    HookRegistrationError,
)

__all__ = [
    'HookConfig',
    'HookManager',
]

allowed_hook_types = ('read', 'write_before', 'write_after', 'restore', 'init')


@dataclass
class HookConfig:
    """Configuration for hook error handling and behavior.

    Attributes:
        on_error: How to handle hook execution errors:
            - 'raise': Propagate the exception to the caller
            - 'log': Log the error and continue (default)
            - 'ignore': Silently ignore errors
        error_logger: Optional custom error logging function
        max_errors_per_hook: Maximum errors before auto-disabling (default 10)
        disable_on_error: Whether to auto-disable hooks after max errors (default False)
    """
    on_error: Literal['raise', 'log', 'ignore'] = 'log'
    error_logger: Optional[Callable[[str, Exception, Hook, HookContext], None]] = None
    max_errors_per_hook: int = 10
    disable_on_error: bool = False


class HookManager:
    """Manager for hooks on a single State instance.

    This class handles registration, unregistration, and execution of hooks
    for all hook types (read, write_before, write_after, restore).

    Features:
        - Priority-based execution ordering
        - Thread-safe hook registration and execution
        - Hook list caching for performance
        - Sequential chaining for write_before hooks
        - Configurable error handling

    Thread Safety:
        All operations are protected by a reentrant lock (RLock), allowing
        hooks to safely trigger other state operations.

    Example:
        >>> from brainstate import HookManager
        >>> manager = HookManager()
        >>> handle = manager.register_hook('read', lambda ctx: print(ctx.value), priority=10)
        >>> manager.has_hooks('read')
        True
        >>> handle.disable()
        >>> manager.clear_hooks()
    """

    def __init__(self, config: Optional[HookConfig] = None):
        """Initialize the hook manager.

        Args:
            config: Optional HookConfig for customizing error handling
        """
        self._lock = threading.RLock()

        # Hook storage (unsorted lists, sorted on-demand)
        self._read_hooks: List[Hook] = []
        self._write_before_hooks: List[Hook] = []
        self._write_after_hooks: List[Hook] = []
        self._restore_hooks: List[Hook] = []
        self._init_hooks: List[Hook] = []

        # Performance optimization
        self._has_hooks = False
        self._hook_cache: Dict[str, List[Hook]] = {}
        self._cache_valid = False

        # Configuration
        self.config = config or HookConfig()

    def register_hook(
        self,
        hook_type: Literal['read', 'write_before', 'write_after', 'restore', 'init'],
        callback: Callable[[HookContext], Any],
        priority: int = 0,
        name: Optional[str] = None,
        enabled: bool = True,
    ) -> HookHandle:
        """Register a new hook.

        Args:
            hook_type: Type of hook to register
            callback: Callable that receives a HookContext
            priority: Execution priority (higher = earlier, default 0)
            name: Optional name for the hook
            enabled: Whether the hook is enabled initially (default True)

        Returns:
            HookHandle for managing the hook

        Raises:
            HookRegistrationError: If hook_type is invalid

        Example:
            >>> def my_hook(ctx):
            ...     print(f"Hook called: {ctx.operation}")
            >>> handle = manager.register_hook('read', my_hook, priority=10, name='my_reader')
        """
        if hook_type not in allowed_hook_types:
            raise HookRegistrationError(f"Invalid hook type: {hook_type}")

        with self._lock:
            # Create the hook
            hook = Hook(callback, priority, name, enabled)

            # Add to appropriate list
            hook_list = self._get_hook_list(hook_type)
            hook_list.append(hook)

            # Update flags and invalidate cache
            self._has_hooks = True
            self._invalidate_cache()

            # Create and return handle
            manager_ref = weakref.ref(self)
            return HookHandle(manager_ref, hook, hook_type)

    def unregister_hook(self, handle: HookHandle) -> bool:
        """Unregister a hook using its handle.

        Args:
            handle: The HookHandle to unregister

        Returns:
            True if the hook was successfully removed, False otherwise
        """
        with self._lock:
            hook_list = self._get_hook_list(handle._hook_type)
            try:
                hook_list.remove(handle._hook)
                self._invalidate_cache()
                self._update_has_hooks_flag()
                return True
            except ValueError:
                return False

    def get_hooks(self, hook_type: Optional[str] = None) -> List[Hook]:
        """Get all registered hooks, optionally filtered by type.

        Args:
            hook_type: Optional hook type to filter by

        Returns:
            List of hooks (copies to prevent external modification)
        """
        with self._lock:
            if hook_type is None:
                # Return all hooks
                all_hooks = (
                    self._read_hooks +
                    self._write_before_hooks +
                    self._write_after_hooks +
                    self._restore_hooks
                )
                return list(all_hooks)
            else:
                hook_list = self._get_hook_list(hook_type)
                return list(hook_list)

    def clear_hooks(self, hook_type: Optional[str] = None) -> None:
        """Clear all hooks, optionally filtered by type.

        Args:
            hook_type: Optional hook type to clear (if None, clears all)
        """
        with self._lock:
            if hook_type is None:
                self._read_hooks.clear()
                self._write_before_hooks.clear()
                self._write_after_hooks.clear()
                self._restore_hooks.clear()
                self._init_hooks.clear()
            else:
                hook_list = self._get_hook_list(hook_type)
                hook_list.clear()

            self._invalidate_cache()
            self._update_has_hooks_flag()

    def has_hooks(self, hook_type: Optional[str] = None) -> bool:
        """Check if any hooks are registered.

        Args:
            hook_type: Optional hook type to check (if None, checks all)

        Returns:
            True if hooks are registered, False otherwise
        """
        # Fast path without lock for common case
        if not self._has_hooks:
            return False

        with self._lock:
            if hook_type is None:
                return self._has_hooks
            else:
                hook_list = self._get_hook_list(hook_type)
                # Check for any enabled hooks
                return any(h.enabled for h in hook_list)

    def execute_read_hooks(self, value: Any, state_ref: weakref.ref) -> None:
        """Execute all read hooks.

        Args:
            value: The value being read
            state_ref: Weak reference to the State instance
        """
        hooks = self._get_cached_hooks('read')
        if not hooks:
            return

        with self._lock:
            context = ReadHookContext(
                operation='read',
                state_ref=state_ref,
                value=value,
            )

            for hook in hooks:
                if not hook.enabled:
                    continue

                try:
                    hook.execute(context)
                except Exception as e:
                    self._handle_hook_error(hook, e, context)

    def execute_write_before_hooks(
        self,
        new_value: Any,
        old_value: Any,
        state_ref: weakref.ref
    ) -> Any:
        """Execute all write_before hooks with sequential chaining.

        Each hook can transform the value, and the next hook receives
        the transformed output.

        Args:
            new_value: The new value being written
            old_value: The previous value
            state_ref: Weak reference to the State instance

        Returns:
            The potentially transformed value

        Raises:
            HookCancellationError: If a hook cancels the operation
        """
        hooks = self._get_cached_hooks('write_before')
        if not hooks:
            return new_value

        with self._lock:
            # Start with the original new value
            transformed_value = new_value

            for hook in hooks:
                if not hook.enabled:
                    continue

                # Create context with current transformation state
                context = MutableWriteHookContext(
                    operation='write_before',
                    state_ref=state_ref,
                    value=new_value,  # Always pass original value
                    old_value=old_value,
                    transformed_value=transformed_value,  # Chain previous transformation
                )

                try:
                    hook.execute(context)

                    # Check for cancellation
                    if context.cancel:
                        reason = context.cancel_reason or "Hook cancelled the operation"
                        raise HookCancellationError(f"{hook.name}: {reason}")

                    # Update transformed value for next hook if this hook set it
                    if context.transformed_value is not None:
                        transformed_value = context.transformed_value

                except HookCancellationError:
                    # Re-raise cancellation errors
                    raise
                except Exception as e:
                    self._handle_hook_error(hook, e, context)

            return transformed_value

    def execute_write_after_hooks(
        self,
        new_value: Any,
        old_value: Any,
        state_ref: weakref.ref
    ) -> None:
        """Execute all write_after hooks.

        Args:
            new_value: The new value that was written
            old_value: The previous value
            state_ref: Weak reference to the State instance
        """
        hooks = self._get_cached_hooks('write_after')
        if not hooks:
            return

        with self._lock:
            context = WriteHookContext(
                operation='write_after',
                state_ref=state_ref,
                value=new_value,
                old_value=old_value,
            )

            for hook in hooks:
                if not hook.enabled:
                    continue

                try:
                    hook.execute(context)
                except Exception as e:
                    self._handle_hook_error(hook, e, context)

    def execute_restore_hooks(
        self,
        new_value: Any,
        old_value: Any,
        state_ref: weakref.ref
    ) -> None:
        """Execute all restore hooks.

        Args:
            new_value: The restored value
            old_value: The previous value before restoration
            state_ref: Weak reference to the State instance
        """
        hooks = self._get_cached_hooks('restore')
        if not hooks:
            return

        with self._lock:
            context = RestoreHookContext(
                operation='restore',
                state_ref=state_ref,
                value=new_value,
                old_value=old_value,
            )

            for hook in hooks:
                if not hook.enabled:
                    continue

                try:
                    hook.execute(context)
                except Exception as e:
                    self._handle_hook_error(hook, e, context)

    def execute_init_hooks(
        self,
        value: Any,
        state_ref: weakref.ref,
        init_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute all init hooks.

        Args:
            value: The initial value of the state
            state_ref: Weak reference to the State instance
            init_metadata: Optional dictionary of initialization metadata
        """
        hooks = self._get_cached_hooks('init')
        if not hooks:
            return

        with self._lock:
            from ._state_hook_context import InitHookContext
            context = InitHookContext(
                operation='init',
                state_ref=state_ref,
                value=value,
                init_metadata=init_metadata or {},
            )

            for hook in hooks:
                if not hook.enabled:
                    continue

                try:
                    hook.execute(context)
                except Exception as e:
                    self._handle_hook_error(hook, e, context)

    # Private methods

    def _get_hook_list(self, hook_type: str) -> List[Hook]:
        """Get the hook list for a given type (internal use)."""
        if hook_type == 'read':
            return self._read_hooks
        elif hook_type == 'write_before':
            return self._write_before_hooks
        elif hook_type == 'write_after':
            return self._write_after_hooks
        elif hook_type == 'restore':
            return self._restore_hooks
        elif hook_type == 'init':
            return self._init_hooks
        else:
            raise ValueError(f"Invalid hook type: {hook_type}")

    def _get_cached_hooks(self, hook_type: str) -> List[Hook]:
        """Get cached, sorted, enabled hooks for a type.

        This method rebuilds the cache if it's invalid.
        """
        if not self._cache_valid:
            self._rebuild_cache()

        return self._hook_cache.get(hook_type, [])

    def _rebuild_cache(self) -> None:
        """Rebuild the hook cache with sorted, enabled hooks."""
        self._hook_cache = {
            'read': sorted([h for h in self._read_hooks if h.enabled]),
            'write_before': sorted([h for h in self._write_before_hooks if h.enabled]),
            'write_after': sorted([h for h in self._write_after_hooks if h.enabled]),
            'restore': sorted([h for h in self._restore_hooks if h.enabled]),
            'init': sorted([h for h in self._init_hooks if h.enabled]),
        }
        self._cache_valid = True

    def _invalidate_cache(self) -> None:
        """Invalidate the hook cache."""
        self._cache_valid = False

    def _update_has_hooks_flag(self) -> None:
        """Update the _has_hooks flag based on current hook lists."""
        self._has_hooks = bool(
            self._read_hooks or
            self._write_before_hooks or
            self._write_after_hooks or
            self._restore_hooks or
            self._init_hooks
        )

    def _handle_hook_error(self, hook: Hook, error: Exception, context: HookContext) -> None:
        """Handle an error that occurred during hook execution.

        Args:
            hook: The hook that raised the error
            error: The exception that was raised
            context: The context in which the error occurred

        Raises:
            HookExecutionError: If on_error='raise'
        """
        # Track errors
        hook._error_count += 1

        # Disable hook if configured and error threshold exceeded
        if self.config.disable_on_error and hook._error_count >= self.config.max_errors_per_hook:
            hook.enabled = False
            self._invalidate_cache()
            error_msg = (
                f"Hook '{hook.name}' disabled after {hook._error_count} errors. "
                f"Last error: {error}"
            )
        else:
            error_msg = f"Hook '{hook.name}' error #{hook._error_count}: {error}"

        # Handle based on configuration
        if self.config.on_error == 'raise':
            raise HookExecutionError(error_msg) from error
        elif self.config.on_error == 'log':
            if self.config.error_logger:
                self.config.error_logger(error_msg, error, hook, context)
            else:
                warnings.warn(error_msg, HookWarning)
        # 'ignore' → do nothing
