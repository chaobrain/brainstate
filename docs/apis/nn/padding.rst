Padding Layers
==============

.. currentmodule:: brainstate.nn

Spatial padding operations with various boundary conditions. Supports reflection,
replication, zero, constant value, and circular padding for 1D, 2D, and 3D inputs.
Essential for controlling output sizes in convolutional networks and handling edge effects.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   ReflectionPad1d
   ReflectionPad2d
   ReflectionPad3d
   ReplicationPad1d
   ReplicationPad2d
   ReplicationPad3d
   ZeroPad1d
   ZeroPad2d
   ZeroPad3d
   ConstantPad1d
   ConstantPad2d
   ConstantPad3d
   CircularPad1d
   CircularPad2d
   CircularPad3d
