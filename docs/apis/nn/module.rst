Base Module Classes
===================

.. currentmodule:: brainstate.nn

Core building blocks for neural network construction. ``Module`` is the base class
for all components in BrainState, providing utilities for parameter management,
state traversal, and hierarchical composition. ``Sequential`` enables easy chaining
of modules for feedforward architectures.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Module
   ElementWiseBlock
   Sequential

Common Wrappers
---------------

Utility wrappers for context management and vectorization. ``EnvironContext`` manages
environment-specific configurations, while ``Vmap`` and ``Map`` enable efficient
batching and vectorization of module operations across multiple inputs.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   EnvironContext
   Vmap
   Map
