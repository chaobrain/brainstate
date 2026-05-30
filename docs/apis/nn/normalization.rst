Normalization Layers
====================

.. currentmodule:: brainstate.nn

Normalization techniques for stabilizing training and improving convergence.
Includes batch normalization variants (0D-3D), layer normalization, RMS normalization,
group normalization, and weight standardization. Each normalization strategy addresses
different aspects of internal covariate shift and gradient flow.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   BatchNorm0d
   BatchNorm1d
   BatchNorm2d
   BatchNorm3d
   LayerNorm
   RMSNorm
   GroupNorm
   weight_standardization
