``brainstate.mixin`` module
===========================

.. currentmodule:: brainstate.mixin
.. automodule:: brainstate.mixin

The :mod:`brainstate.mixin` module provides mixin classes and utility types that enhance
the functionality of BrainState components through multiple inheritance. It includes
parameter description systems and advanced type definitions for expressing complex
type requirements.

Key Features
------------

- **Parameter Descriptors**: Reusable parameter templates for object instantiation
- **Type System**: Advanced union and intersection type utilities (JointTypes, OneOfTypes)
- **Deferred Instantiation**: Templates for creating multiple objects with shared configurations
- **Utility Functions**: Helper functions for hashability checks and marking unimplemented methods

Quick Start
-----------

Using parameter descriptors for reusable configurations:

.. code-block:: python

    import brainstate

    class NeuronModel(brainstate.mixin.ParamDesc):
        def __init__(self, size, tau=10.0, threshold=1.0):
            self.size = size
            self.tau = tau
            self.threshold = threshold

    # Create a reusable template
    neuron_template = NeuronModel.desc(size=100, tau=20.0)

    # Create multiple instances with different thresholds
    excitatory = neuron_template(threshold=1.0)
    inhibitory = neuron_template(threshold=0.5)

Using advanced type system:

.. code-block:: python

    from typing import Protocol

    # Define protocols/interfaces
    class Trainable(Protocol):
        def train(self): ...

    class Evaluable(Protocol):
        def evaluate(self): ...

    # Require both interfaces (intersection type)
    TrainableEvaluableModel = brainstate.mixin.JointTypes[Trainable, Evaluable]

    # Allow either type (union type)
    NumericType = brainstate.mixin.OneOfTypes[int, float]


Base Mixin Classes
------------------

Core mixin classes providing foundational functionality.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Mixin

Parameter Description System
----------------------------

Classes and utilities for creating reusable parameter templates and deferred object instantiation.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ParamDesc
   ParamDescriber
   HashableDict
   NoSubclassMeta

Advanced Type System
--------------------

Utilities for creating complex type annotations and requirements.

Type Combinators
~~~~~~~~~~~~~~~~

Classes and functions for creating union and intersection types.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   JointTypes
   OneOfTypes
   _JointGenericAlias
   _OneOfGenericAlias

Utility Functions and Decorators
---------------------------------

Helper functions and decorators for enhanced functionality.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   hashable
   not_implemented
