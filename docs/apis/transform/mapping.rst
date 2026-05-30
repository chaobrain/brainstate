Mapping and Vectorization
=========================

.. currentmodule:: brainstate.transform

Transformations for vectorized and parallel computation across multiple data points
or devices. These functions enable efficient batch processing and multi-device
scaling, essential for large-scale simulations and distributed training.

Basic Vectorization
-------------------

Vectorize computations across batch dimensions. ``vmap2`` is the recommended API
with enhanced state handling and control over batching axes.

.. autosummary::
   :toctree: ../generated/

   vmap
   vmap_new_states
   vmap2
   vmap2_new_states
   map

Parallel Mapping
---------------

Execute computations in parallel across devices, or shard them with explicit mesh
control.

.. autosummary::
   :toctree: ../generated/

   pmap2
   pmap2_new_states
   shard_map

Base Classes and Utilities
--------------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   StatefulMapping

.. autosummary::
   :toctree: ../generated/

   unvmap
