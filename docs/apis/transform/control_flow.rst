Control Flow
============

.. currentmodule:: brainstate.transform

Structured control-flow transformations that are JIT-compilable and state-aware,
covering conditional branching, bounded and dynamic loops, and result-collecting
iteration.

Conditions
----------

Control flow transformations that enable conditional execution of different computation
branches based on runtime conditions. These functions provide efficient, JIT-compilable
alternatives to Python's native if/elif/else statements, ensuring optimal performance
in compiled code.

.. autosummary::
   :toctree: ../generated/

   cond
   switch
   ifelse

For Loop
--------

Transformations for structured iteration with result collection. These functions provide
efficient ways to perform repeated computations while accumulating results into arrays,
with optional checkpointing for memory-efficient training of deep networks.

.. autosummary::
   :toctree: ../generated/

   scan
   checkpointed_scan
   for_loop
   checkpointed_for_loop
   ProgressBar

While Loop
----------

Dynamic iteration transformations that continue execution based on runtime conditions.
These functions enable loops with variable iteration counts, essential for adaptive
algorithms and convergence-based computations.

.. autosummary::
   :toctree: ../generated/

   while_loop
   bounded_while_loop
