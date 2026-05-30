Core Transformation Infrastructure
==================================

.. currentmodule:: brainstate.transform

Low-level building blocks underpinning the stateful transformations. ``StatefulFunction``
captures a function together with the states it touches; ``make_jaxpr`` generates the
JAX expression (JAXPR) for a function for visualization and debugging; ``eval_shape``
infers output shapes without executing the computation; and ``StateFinder`` locates and
manages state variables in complex stateful workflows.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   StatefulFunction
   StateFinder

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   make_jaxpr
   eval_shape
