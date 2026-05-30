Pooling and Reshaping
=====================

.. currentmodule:: brainstate.nn

Downsampling, upsampling, and shape manipulation operations for spatial data.
Includes average pooling, max pooling, Lp-norm pooling, unpooling for reconstruction,
and adaptive pooling for fixed output sizes. ``Flatten`` and ``Unflatten`` enable
seamless transitions between spatial and flat representations.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Flatten
   Unflatten
   AvgPool1d
   AvgPool2d
   AvgPool3d
   MaxPool1d
   MaxPool2d
   MaxPool3d
   MaxUnpool1d
   MaxUnpool2d
   MaxUnpool3d
   LPPool1d
   LPPool2d
   LPPool3d
   AdaptiveAvgPool1d
   AdaptiveAvgPool2d
   AdaptiveAvgPool3d
   AdaptiveMaxPool1d
   AdaptiveMaxPool2d
   AdaptiveMaxPool3d
