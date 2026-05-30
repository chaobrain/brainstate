``brainstate`` module
=====================

.. currentmodule:: brainstate


Core State Classes
------------------

State classes are the fundamental building blocks for managing dynamic data in BrainState.
They provide a unified interface for tracking, tracing, and transforming stateful computations.


Basic State Types
~~~~~~~~~~~~~~~~~

Basic state types provide semantic distinctions for different data lifecycles in your program.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   State
   ShortTermState
   LongTermState
   ParamState
   BatchState
   DelayState


Hidden State Types
~~~~~~~~~~~~~~~~~~

Hidden state types are designed for recurrent neural networks and eligibility trace-based learning,
with special support for BrainScale online learning integration.

- **HiddenState**: Single hidden state variable for neurons or synapses (equivalent to ``brainstate.HiddenState``)
- **HiddenGroupState**: Multiple hidden states stored in the last dimension of a single array
- **HiddenTreeState**: Multiple hidden states with different units, stored as a PyTree structure

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HiddenState
   HiddenGroupState
   HiddenTreeState


Special State Types
~~~~~~~~~~~~~~~~~~~

Special-purpose state types for advanced use cases and PyTree integration.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FakeState
   TreefyState



State Management
----------------

Tools for managing collections of states and tracking state access patterns during program execution.


State Collections
~~~~~~~~~~~~~~~~~

Organize and manipulate multiple states as cohesive units.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StateDictManager


State Tracing
~~~~~~~~~~~~~

Track state read/write operations for automatic differentiation and program transformation.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StateTraceStack


State Utilities
---------------

Helper functions and context managers for working with states effectively.


Context Managers
~~~~~~~~~~~~~~~~

Control state behavior within specific code blocks.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   check_state_value_tree
   check_state_jax_tracer
   catch_new_states



Helper Functions
~~~~~~~~~~~~~~~~

Utility functions for common state operations.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   maybe_state



Error Handling
--------------

Custom exceptions for state-related errors and debugging.


Exception Classes
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BrainStateError
   BatchAxisError
   TraceContextError


State Hooks
-----------

The hook system observes and intercepts state operations — reads, writes,
initialization, and restoration. Hooks can be registered globally and are managed
per-process, enabling logging, validation, and instrumentation of stateful
computations without modifying the states themselves.


Hook Contexts
~~~~~~~~~~~~~

Context objects passed to hooks describing the operation being performed.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HookContext
   ReadHookContext
   WriteHookContext
   MutableWriteHookContext
   RestoreHookContext
   InitHookContext


Hook Core
~~~~~~~~~

Core hook abstractions: the ``Hook`` base class and the ``HookHandle`` returned on
registration.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Hook
   HookHandle


Hook Manager and Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~

Manage hook lifecycles and global registration.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HookManager
   HookConfig
   GlobalHookRegistry


Global Hook Functions
~~~~~~~~~~~~~~~~~~~~~~

Convenience functions for registering and querying globally-installed hooks.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   register_state_hook
   unregister_state_hook
   clear_state_hooks
   has_state_hooks
   list_state_hooks


Hook Exceptions
~~~~~~~~~~~~~~~

Exceptions and warnings raised by the hook system.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HookError
   HookExecutionError
   HookRegistrationError
   HookCancellationError
   HookWarning


