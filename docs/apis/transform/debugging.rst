Debugging and Error Checking
============================

.. currentmodule:: brainstate.transform

JIT-compatible debugging utilities for identifying NaN and Inf values during
gradient computations, plus functionalized runtime error checking. These tools
help diagnose numerical issues in compiled code without sacrificing performance.

NaN/Inf Debugging
-----------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   debug_nan
   debug_nan_if
   breakpoint_if

Error Checking
--------------

Performs conditional checks during JIT compilation and raises an error if the
specified condition is met, helping catch exceptional cases at compile or run time.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   jit_error_if
   checkify
   check
   check_error
   all_checks
   user_checks
   nan_checks
   div_checks
   index_checks
   float_checks
   automatic_checks
