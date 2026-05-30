``brainstate.util`` module
==========================

.. currentmodule:: brainstate.util 
.. automodule:: brainstate.util 

Dict Operation
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   flat_mapping
   nest_mapping
   NestedDict
   FlattedDict

Functions for flattening, merging, and freezing nested dictionaries.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   flatten_dict
   unflatten_dict
   merge_dicts
   freeze
   unfreeze
   copy
   pop
   is_dataclass


Filter Operation
----------------

``to_predicate`` converts a filter specification into a predicate; the remaining
classes are composable filter primitives for selecting states and tree leaves.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   to_predicate

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   All
   Any
   Not
   Nothing
   Everything
   OfType
   WithTag
   PathContains


Pretty Representation
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   PrettyType
   PrettyAttr
   PrettyRepr
   PrettyMapping
   MappingReprMixin
   PrettyDict
   PrettyList
   PrettyObject

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pretty_repr
   yield_unique_pretty_repr_items


Struct Operation
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   dataclass
   field
   PyTreeNode
   FrozenDict


Other Operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   clear_buffer_memory
   not_instance_eval
   is_instance_eval
   DictManager
   DotDict
   StateJaxTracer
   BoundedCache

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_unique_name
   split_total


