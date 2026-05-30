Linear Layers
=============

.. currentmodule:: brainstate.nn

Fully-connected linear transformation layers with various specializations.
Includes standard dense layers, weight-standardized variants for improved training
stability, sparse connections for efficiency, and low-rank adaptation (LoRA) for
parameter-efficient fine-tuning.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Linear
   ScaledWSLinear
   SignedWLinear
   SparseLinear
   LoRA
   AllToAll
   OneToOne
