``brainstate.typing`` module
============================

.. currentmodule:: brainstate.typing
.. automodule:: brainstate.typing

The :mod:`brainstate.typing` module provides comprehensive type annotations specifically
designed for scientific computing, neural network modeling, and array operations within
the BrainState ecosystem. It offers JAX-compatible, NumPy-compatible, and BrainUnit-compatible
type hints that enhance code clarity and enable better static analysis.

Key Features
------------

- **JAX Compatibility**: Full support for JAX arrays, PRNG keys, and functional programming patterns
- **NumPy Integration**: Compatible with NumPy arrays and data types
- **BrainUnit Support**: Type annotations for physical quantities with units
- **PyTree Annotations**: Advanced type system for tree-structured data
- **Array Shape Annotations**: Flexible array type system with shape specifications
- **Filter System**: Sophisticated filtering types for PyTree operations

Quick Start
-----------

Basic type annotations:

.. code-block:: python

    import brainstate
    from brainstate.typing import ArrayLike, Shape, DTypeLike

    def process_array(data: ArrayLike, shape: Shape, dtype: DTypeLike) -> brainstate.Array:
        return brainstate.asarray(data, dtype=dtype).reshape(shape)

Advanced array annotations with shape information:

.. code-block:: python

    from brainstate.typing import Array

    def matrix_multiply(a: Array["m, n"], b: Array["n, k"]) -> Array["m, k"]:
        return a @ b

PyTree type annotations:

.. code-block:: python

    from brainstate.typing import PyTree

    def tree_operation(tree: PyTree[float, "T"]) -> PyTree[float, "T"]:
        return brainstate.tree_map(lambda x: x * 2, tree)


Array Type Annotations
----------------------

Advanced array type system with support for shape and dtype specifications.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Array
   ArrayLike

Shape and Size Types
~~~~~~~~~~~~~~~~~~~~

Types for specifying array dimensions and sizes.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Shape
   Size
   Axes

Data Type Annotations
~~~~~~~~~~~~~~~~~~~~~

Types for specifying array data types and dtype-like objects.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DType
   DTypeLike
   SupportsDType


PyTree Type System
------------------

Sophisticated type annotations for tree-structured data with support for structural constraints.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   PyTree

Path and Filter System
~~~~~~~~~~~~~~~~~~~~~~

Types for navigating and filtering PyTree structures.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Key
   PathParts
   Predicate
   Filter
   FilterLiteral


Random Number Generation Types
------------------------------

Type annotations for random number generation and seeding.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SeedOrKey


Type Variables and Utilities
----------------------------

Generic type variables and utility types for advanced type annotations.

Type Variables
~~~~~~~~~~~~~~

Generic type variables for creating flexible type annotations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   K
   _T
   _Annotation

Utility Types
~~~~~~~~~~~~~

Helper types for advanced use cases and sentinel values.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Missing
