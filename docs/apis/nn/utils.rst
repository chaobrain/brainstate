Utility Functions
=================

.. currentmodule:: brainstate.nn

General-purpose utilities for neural network operations. ``count_parameters`` tallies
trainable and total parameters in a model, while ``clip_grad_norm`` implements gradient
clipping for training stability. ``HiData`` provides utilities for organizing and
accessing hierarchical, tree-structured data for compositional models.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   count_parameters
   clip_grad_norm

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   HiData
