Standard Regularizations
========================

.. currentmodule:: brainstate.nn

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

Prior Distribution-Based Regularizations
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
