JIT Compilation
===============

.. currentmodule:: brainstate.transform

Just-In-Time compilation transformation that converts Python functions into optimized
machine code. JIT compilation dramatically accelerates numerical computations by
eliminating Python interpreter overhead and enabling hardware-specific optimizations.

.. autosummary::
   :toctree: ../generated/

   jit
   named_scope
   named_call

Checkpointing
-------------

Memory-efficient gradient computation techniques that trade computation for memory.
These transformations are crucial for training large models by recomputing intermediate
values during backpropagation rather than storing them all in memory.

.. autosummary::
   :toctree: ../generated/

   remat
   checkpoint
