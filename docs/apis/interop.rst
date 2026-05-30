``brainstate.interop`` module
=============================

.. currentmodule:: brainstate.interop
.. automodule:: brainstate.interop

Interoperability utilities for converting models between ``brainstate.nn`` and other
JAX-based frameworks (Flax NNX, Flax Linen, and Equinox). The module also exposes the
layer-mapping registry used to extend conversion support and a hierarchy of errors
raised during conversion.

Conversion Functions
--------------------

Convert models in either direction between ``brainstate.nn`` and a target framework.
The ``from_*`` functions import an external model into ``brainstate.nn``; the ``to_*``
functions export a ``brainstate.nn`` model to the target framework.

.. autosummary::
   :toctree: generated/

   from_nnx
   to_nnx
   from_linen
   to_linen
   from_equinox
   to_equinox

Layer Mapping Registry
---------------------

Register and inspect the layer mappings that drive conversion. ``register_layer_mapping``
adds support for a new layer type, ``supported_layers`` lists the currently supported
layers, and ``LayerMapping`` describes an individual mapping entry.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LayerMapping

.. autosummary::
   :toctree: generated/
   :nosignatures:

   register_layer_mapping
   supported_layers

Errors
------

Exceptions raised when a conversion cannot be completed, e.g. a missing optional
dependency, an unmapped or unsupported layer, an unsupported model structure, or a
missing input shape.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   InteropError
   MissingDependencyError
   UnmappedLayerError
   UnsupportedLayerError
   UnsupportedStructureError
   MissingShapeError
   ConversionError
