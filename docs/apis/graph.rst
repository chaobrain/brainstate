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


Context Management
------------------

Context managers for handling complex state updates during graph transformations.
These utilities enable splitting and merging graph states in a thread-safe manner.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   split_context
   merge_context


Graph Conversion
----------------

Utilities for converting between graph and tree representations, enabling
flexible manipulation of nested module structures.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   graph_to_tree
   tree_to_graph
   NodeStates


Graph Definition Classes
------------------------

Core classes for representing graph structure, node definitions, and references.
These classes provide the foundation for graph operations and state management.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   GraphDef
   NodeDef
   NodeRef
   RefMap
   register_graph_node_type

