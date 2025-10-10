``brainstate.transform`` module
===============================

.. currentmodule:: brainstate.transform
.. automodule:: brainstate.transform

The ``brainstate.transform`` module provides powerful transformations for
neural computation and scientific computing. It extends JAX's transformation capabilities
with stateful computation support, enabling efficient compilation, automatic differentiation,
parallelization, and control flow for brain simulation and machine learning workloads.


Condition
---------

Control flow transformations that enable conditional execution of different computation
branches based on runtime conditions. These functions provide efficient, JIT-compilable
alternatives to Python's native if/elif/else statements, ensuring optimal performance
in compiled code.

.. autosummary::
   :toctree: generated/

   cond
   switch
   ifelse



For Loop
--------

Transformations for structured iteration with result collection. These functions provide
efficient ways to perform repeated computations while accumulating results into arrays,
with optional checkpointing for memory-efficient training of deep networks.

.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   while_loop
   bounded_while_loop



JIT Compilation
---------------

Just-In-Time compilation transformation that converts Python functions into optimized
machine code. JIT compilation dramatically accelerates numerical computations by
eliminating Python interpreter overhead and enabling hardware-specific optimizations.

.. autosummary::
   :toctree: generated/

   jit



Checkpointing
-------------

Memory-efficient gradient computation techniques that trade computation for memory.
These transformations are crucial for training large models by recomputing intermediate
values during backpropagation rather than storing them all in memory.

.. autosummary::
   :toctree: generated/

   remat
   checkpoint



Compilation Tools
-----------------

Advanced utilities for compilation and debugging. These tools provide low-level access
to JAX's compilation pipeline, enabling inspection of intermediate representations and
custom error handling in JIT-compiled code.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StatefulFunction
   StatefulMapping


.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_jaxpr
   jit_error_if




Gradient Computations
---------------------

Automatic differentiation transformations for computing gradients, Jacobians, and
Hessians. These functions extend JAX's autodiff capabilities with support for stateful
computations, making them ideal for training neural networks and optimizing complex
dynamical systems.

.. autosummary::
   :toctree: generated/

   vector_grad
   grad
   jacrev
   jacfwd
   jacobian
   hessian




Batching and Parallelism
------------------------

Transformations for vectorized and parallel computation across multiple data points
or devices. These functions enable efficient batch processing and multi-device
scaling, essential for large-scale simulations and distributed training.

.. autosummary::
   :toctree: generated/

   vmap
   pmap
   map
   vmap_new_states



Shape Evaluation
----------------

Shape inference transformation that determines output shapes without executing the
computation. This function is invaluable for debugging and pre-allocating arrays,
allowing you to understand data flow through complex transformations.

.. autosummary::
   :toctree: generated/

   eval_shape





