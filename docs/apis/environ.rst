``brainstate.environ`` module
=============================

.. currentmodule:: brainstate.environ
.. automodule:: brainstate.environ

The ``brainstate.environ`` module provides a comprehensive environment management system for
configuring computational settings, platform preferences, numerical precision, and runtime
behaviors. It offers thread-safe configuration management with support for both global
settings and context-specific overrides.

Environment Management
----------------------

Functions for managing environment configuration and settings. These utilities allow you to
set, retrieve, and manage environment variables that control the behavior of brainstate
computations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   reset
   context
   get
   set
   pop
   all


Environment State
-----------------

Thread-local environment container used for custom or isolated configurations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   EnvironmentState


Platform and Device Settings
-----------------------------

Configure the computing platform (CPU, GPU, TPU) and control device allocation. These
functions help optimize performance by allowing you to select appropriate hardware backends
and manage device resources.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   get_platform
   set_platform
   get_host_device_count
   set_host_device_count


Precision and Data Types
-------------------------

Control numerical precision and retrieve appropriate data types for computations. These
functions ensure consistent precision across your calculations and provide easy access to
the correct NumPy/JAX data types based on the current precision setting.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   get_precision
   set_precision
   dftype
   ditype
   dutype
   dctype
   tolerance


Mode and Timing
----------------

Access computation mode settings and numerical integration parameters. These functions
provide information about the current execution mode and time step settings for simulations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   get_dt


Environment Keys and Flags
--------------------------

Common environment key constants used with ``get``/``set``/``context``.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   I
   T
   DT
   PRECISION
   PLATFORM
   HOST_DEVICE_COUNT
   JIT_ERROR_CHECK
   FIT


Defaults and Supported Values
------------------------------

Default values and supported platform/precision settings.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DEFAULT_PRECISION
   SUPPORTED_PLATFORMS
   SUPPORTED_PRECISIONS


Behavior Registration
---------------------

Register custom callbacks that respond to environment parameter changes. This system allows
you to define automatic behaviors when specific environment settings are modified, enabling
reactive configuration management.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   register_default_behavior
   unregister_default_behavior
   list_registered_behaviors

