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
   ArrayParam
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

- **FakeState**: Lightweight state-like object without tracing functionality
- **TreefyState**: PyTree-compatible state reference for functional transformations


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

**StateDictManager** provides dict-like interface for state collections with built-in operations:

- Batch value assignment and collection
- Type-based filtering and splitting
- Integration with training loops and checkpointing


State Tracing
~~~~~~~~~~~~~

Track state read/write operations for automatic differentiation and program transformation.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StateTraceStack

**StateTraceStack** enables automatic state management in JAX transformations:

- Records which states are read vs. written
- Manages state values across transformation boundaries
- Supports state recovery and rollback operations


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

- **check_state_value_tree**: Validates that state value structure remains consistent on assignment
- **check_state_jax_tracer**: Ensures states are properly traced during JAX compilation
- **catch_new_states**: Captures and tags newly created states within a context


Helper Functions
~~~~~~~~~~~~~~~~

Utility functions for common state operations.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   maybe_state

**maybe_state** extracts values from State objects, or returns non-State values unchanged.
Useful for writing functions that accept both states and raw values.


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

- **BrainStateError**: Base exception class for all BrainState-specific errors
- **BatchAxisError**: Raised when batch axis handling fails in transformations

