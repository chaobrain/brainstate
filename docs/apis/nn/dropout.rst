Dropout Layers
==============

.. currentmodule:: brainstate.nn

Regularization through stochastic neuron dropping during training. Includes standard
dropout, spatial dropout variants (1D-3D), alpha dropout for self-normalizing networks,
and fixed dropout with deterministic masking. Prevents overfitting by encouraging
robust feature learning.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Dropout
   Dropout1d
   Dropout2d
   Dropout3d
   AlphaDropout
   FeatureAlphaDropout
   DropoutFixed
