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

"""Hook context objects passed to hook callbacks."""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from brainstate._state import State

__all__ = [
    'HookContext',
    'ReadHookContext',
    'WriteHookContext',
    'MutableWriteHookContext',
    'RestoreHookContext',
    'InitHookContext',
]


@dataclass(slots=True)
class HookContext:
    """Base context object passed to all hooks.

    This context provides information about the hook execution environment,
    including the operation type, the state being operated on, timing information,
    and user-defined metadata.

    Attributes:
        operation: Type of operation ('read', 'write_before', 'write_after', 'restore', 'init')
        state_ref: Weak reference to the State instance (avoids circular references)
        timestamp: Unix timestamp when the operation occurred
        metadata: Dictionary for user-defined metadata
    """
    operation: Literal['read', 'write_before', 'write_after', 'restore', 'init']
    state_ref: weakref.ref[State]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> Optional[State]:
        """Get the State instance if it's still alive.

        Returns:
            The State instance, or None if it has been garbage collected.
        """
        return self.state_ref()

    @property
    def state_name(self) -> Optional[str]:
        """Get the name of the State instance.

        Returns:
            The state's name, or None if the state has been garbage collected
            or doesn't have a name.
        """
        s = self.state
        return s.name if s is not None else None


@dataclass(slots=True)
class ReadHookContext(HookContext):
    """Context for read hooks.

    This context is passed to hooks registered for 'read' operations.
    Read hooks can inspect the value but should not modify it.

    Attributes:
        value: The current value being read from the state
    """
    value: Any = None


@dataclass(slots=True)
class WriteHookContext(HookContext):
    """Base context for write operations.

    This context provides information about both the new value being written
    and the previous value that will be replaced.

    Attributes:
        value: The new value being written to the state
        old_value: The previous value before the write operation
    """
    value: Any = None
    old_value: Any = None


@dataclass(slots=True)
class MutableWriteHookContext(WriteHookContext):
    """Context for write_before hooks with transformation capabilities.

    This context allows hooks to transform the value being written or cancel
    the write operation entirely. Hooks execute in priority order, and each
    hook receives the transformed output from the previous hook (sequential chaining).

    Attributes:
        value: The original new value being written
        old_value: The previous value before the write
        transformed_value: The transformed value (set by hooks to modify the value)
        cancel: Set to True to cancel the write operation
        cancel_reason: Optional explanation for why the operation was cancelled

    Example:
        >>> def clip_values(ctx: MutableWriteHookContext):
        ...     # Transform the value by clipping to [-1, 1]
        ...     import jax.numpy as jnp
        ...     input_value = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
        ...     ctx.transformed_value = jnp.clip(input_value, -1.0, 1.0)

        >>> def validate_positive(ctx: MutableWriteHookContext):
        ...     # Cancel if value is negative
        ...     import jax.numpy as jnp
        ...     check_value = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
        ...     if jnp.any(check_value < 0):
        ...         ctx.cancel = True
        ...         ctx.cancel_reason = "Value must be positive"
    """
    transformed_value: Optional[Any] = None
    cancel: bool = False
    cancel_reason: Optional[str] = None


@dataclass(slots=True)
class RestoreHookContext(WriteHookContext):
    """Context for restore hooks.

    This context is passed to hooks registered for 'restore' operations
    (when state.restore_value() is called). It provides both the restored
    value and the previous value for comparison.

    Restore hooks can inspect and log the restoration but cannot modify
    the value or cancel the operation (restoration is considered atomic).

    Attributes:
        value: The new value being restored
        old_value: The previous value before restoration
    """
    pass


@dataclass(slots=True)
class InitHookContext(HookContext):
    """Context for initialization hooks.

    This context is passed to hooks registered for 'init' operations
    (when a State instance is created). It provides the initial value
    and any initialization metadata.

    Init hooks can inspect the initial state and perform actions like
    logging, validation, or registration, but cannot modify the initial
    value or cancel the initialization (use factory functions for that).

    Attributes:
        value: The initial value of the state
        metadata: Dictionary of initialization metadata (name, tag, etc.)

    Example:
        >>> def log_state_creation(ctx: InitHookContext):
        ...     print(f"Created state '{ctx.state_name}' with value: {ctx.value}")

        >>> state = brainstate.State(jnp.array([1, 2, 3]), name="my_state")
        >>> state.register_hook('init', log_state_creation)
    """
    value: Any = None
    init_metadata: Dict[str, Any] = field(default_factory=dict)
