Intermediate Representation (IR) Tooling
========================================

.. currentmodule:: brainstate.transform

Tools for optimizing, processing, generating code from, and visualizing JAX
intermediate representations (Jaxpr). These utilities reduce computation overhead and
improve runtime performance while preserving a function's semantics and interface.

IR Optimization
---------------

Optimize Jaxpr intermediate representations by applying compiler optimizations such as
constant folding, dead code elimination, common subexpression elimination, copy
propagation, and algebraic simplification.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   constant_fold
   dead_code_elimination
   common_subexpression_elimination
   copy_propagation
   algebraic_simplification
   optimize_jaxpr

IR Processing and Transformation
--------------------------------

Tools for processing and transforming JAX intermediate representations, including
equation-to-Jaxpr conversion and JIT inlining operations.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   eqns_to_closed_jaxpr
   eqns_to_jaxpr
   inline_jit

Code Generation
--------------

Convert JAX functions and Jaxpr representations into readable Python code for
inspection, debugging, and understanding the underlying computation structure.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   fn_to_python_code
   jaxpr_to_python_code
   register_prim_handler

Visualization
------------

Visualize computation graphs and Jaxpr structures using various graph drawing
libraries and formats, enabling visual inspection of complex transformations.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   draw
   view_pydot
   draw_dot_graph
