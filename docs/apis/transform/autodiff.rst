Gradient Computations
=====================

.. currentmodule:: brainstate.transform

Automatic differentiation transformations for computing gradients, Jacobians, and
Hessians. These functions extend JAX's autodiff capabilities with support for stateful
computations, making them ideal for training neural networks and optimizing complex
dynamical systems.

Gradient Transformations
------------------------

.. autosummary::
   :toctree: ../generated/

   grad
   vector_grad
   fwd_grad

Vector-Jacobian and Jacobian-Vector Products
--------------------------------------------

.. autosummary::
   :toctree: ../generated/

   vjp
   jvp

Jacobian and Hessian
--------------------

.. autosummary::
   :toctree: ../generated/

   jacrev
   jacfwd
   jacobian
   hessian

Base Classes
------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   GradientTransform
