Convolutional Layers
====================

.. currentmodule:: brainstate.nn

Convolutional layers for 1D, 2D, and 3D spatial feature extraction. Includes standard
convolutions, weight-standardized variants for improved normalization, and transposed
convolutions for upsampling operations. Essential for processing sequential data, images,
and volumetric inputs.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Conv1d
   Conv2d
   Conv3d
   ScaledWSConv1d
   ScaledWSConv2d
   ScaledWSConv3d
   ConvTranspose1d
   ConvTranspose2d
   ConvTranspose3d
