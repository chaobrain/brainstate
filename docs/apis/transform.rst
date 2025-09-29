``brainstate.transform`` module
===============================

.. currentmodule:: brainstate.transform
.. automodule:: brainstate.transform



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




Gradient Computations
---------------------

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

.. autosummary::
   :toctree: generated/

   vmap
   pmap
   map
   vmap_new_states



Random State Processing
-----------------------

.. autosummary::
   :toctree: generated/

   restore_rngs



Shape Evaluation
----------------


.. autosummary::
   :toctree: generated/

   eval_shape





