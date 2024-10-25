``brainstate.compile`` module for program compilation
=====================================================

.. currentmodule:: brainstate.compile
.. automodule:: brainstate.compile


Condition
---------

.. autosummary::
   :toctree: generated/

   cond
   switch
   ifelse



For Loop
--------


These transformations collect the results of a loop into a single array.

.. autosummary::
   :toctree: generated/

   scan
   checkpointed_scan
   for_loop
   checkpointed_for_loop
   ProgressBar




While Loop
----------


.. autosummary::
   :toctree: generated/

   while_loop
   bounded_while_loop



JIT Compilation
---------------

.. autosummary::
   :toctree: generated/

   jit



Checkpointing
-------------

.. autosummary::
   :toctree: generated/

   remat
   checkpoint



Compilation Tools
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StatefulFunction


.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_jaxpr
   jit_error_if

