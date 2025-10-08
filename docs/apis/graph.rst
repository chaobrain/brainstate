``brainstate.graph`` moduel
===========================

Most of these APIs are adapted from Flax (https://github.com/google/flax/blob/main/flax/nnx/graph.py).
It enables the structure-preserving ``State`` retrieval and manipulatio in the ``brainstate``.


.. currentmodule:: brainstate.graph 
.. automodule:: brainstate.graph 

Graph Node
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Node


Graph Operation
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   pop_states
   nodes
   states
   treefy_states
   update_states
   flatten
   unflatten
   treefy_split
   treefy_merge
   iter_leaf
   iter_node
   clone
   graphdef
   call

