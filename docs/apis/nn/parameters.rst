Parameter containers
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

Parameter transforms
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


Standard regularizations
------------------------

Classical regularization methods for parameter penalization and constraint enforcement.
These regularizations add penalty terms to the loss function to encourage desired
properties like sparsity (L1), smoothness (L2), or structural constraints (orthogonality,
spectral norms). Use with ``Param`` to automatically include regularization losses in
training objectives.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Regularization
   L1Reg
   L2Reg
   ElasticNetReg
   HuberReg
   GroupLassoReg
   TotalVariationReg
   MaxNormReg
   EntropyReg
   OrthogonalReg
   SpectralNormReg
   ChainedReg

Prior distribution-based regularizations
-----------------------------------------

Probabilistic regularizations based on prior distributions for Bayesian-inspired
parameter estimation. These regularizations encode domain knowledge or assumptions
about parameter distributions (Gaussian, heavy-tailed, bounded, etc.). Particularly
useful for variational inference, maximum a posteriori (MAP) estimation, and
uncertainty quantification. Each regularization implements ``loss()``, ``sample_init()``,
and ``reset_value()`` for prior-based parameter initialization.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   GaussianReg
   StudentTReg
   CauchyReg
   UniformReg
   BetaReg
   LogNormalReg
   ExponentialReg
   GammaReg
   InverseGammaReg
   LogUniformReg
   HorseshoeReg
   SpikeAndSlabReg
   DirichletReg
