Parameter Containers
====================

.. currentmodule:: brainstate.nn

Flexible parameter containers that integrate with BrainState's module system.
``Param`` supports bijective transformations for constrained optimization and
optional regularization. ``Const`` provides non-trainable constant parameters.
Both support automatic caching of transformed values for improved performance.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Param
   Const

Parameter Transforms
--------------------

Bijective transformations for constrained parameter optimization. These transforms
map between unconstrained and constrained spaces, enabling gradient-based optimization
of parameters with constraints (positivity, boundedness, simplex, etc.). All transforms
implement ``forward()``, ``inverse()``, and optional ``log_abs_det_jacobian()`` for
probabilistic applications. Use with ``Param`` for automatic constraint handling.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Transform
   IdentityT
   ClipT
   AffineT
   SigmoidT
   TanhT
   SoftsignT
   ScaledSigmoidT
   SoftplusT
   NegSoftplusT
   LogT
   ExpT
   ReluT
   PositiveT
   NegativeT
   PowerT
   OrderedT
   SimplexT
   UnitVectorT
   ChainT
   MaskedT
