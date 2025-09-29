``brainstate.mixin`` module
===========================

.. currentmodule:: brainstate.mixin
.. automodule:: brainstate.mixin

The :mod:`brainstate.mixin` module provides mixin classes and utility types that enhance
the functionality of BrainState components through multiple inheritance. It includes
parameter description systems, alignment interfaces, computation modes, and advanced
type definitions for expressing complex behavioral requirements.

Key Features
------------

- **Parameter Descriptors**: Reusable parameter templates for object instantiation
- **Behavioral Interfaces**: Mixins for post-synaptic alignment and conductance binding
- **Computation Modes**: Training, batching, and joint mode configurations
- **Type System**: Advanced union and intersection type utilities
- **Deferred Instantiation**: Templates for creating multiple objects with shared configurations

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

Using computation modes:

.. code-block:: python

    # Create combined training and batching mode
    training = brainstate.mixin.Training()
    batching = brainstate.mixin.Batching(batch_size=32)
    mode = brainstate.mixin.JointMode(training, batching)

    # Check mode properties
    if mode.has(brainstate.mixin.Training):
        print("Model is in training mode")
    if mode.has(brainstate.mixin.Batching):
        print(f"Batch size: {mode.batch_size}")


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

Behavioral Interface Mixins
---------------------------

Mixins that provide specific behavioral interfaces for neural computation components.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   AlignPost
   BindCondData

Computation Modes
-----------------

Classes for representing different computational contexts and behaviors.

Base Mode Classes
~~~~~~~~~~~~~~~~~

Fundamental mode classes for computation context management.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Mode
   JointMode

Specific Mode Types
~~~~~~~~~~~~~~~~~~~

Concrete mode implementations for common computation scenarios.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Training
   Batching

Advanced Type System
--------------------

Utilities for creating complex type annotations and requirements.

Type Combinators
~~~~~~~~~~~~~~~~

Functions for creating union and intersection types.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   JointTypes
   OneOfTypes

Utility Functions and Decorators
---------------------------------

Helper functions and decorators for enhanced functionality.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   hashable
   not_implemented
