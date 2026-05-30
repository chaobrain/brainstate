Dynamics and Simulation
=======================

.. currentmodule:: brainstate.nn

Primitives for building dynamical systems and time-evolving neural models:
differential-equation base classes, prefetch/delay utilities, ring-buffer delays,
collective operations over module hierarchies, and numerical integration.

Dynamics Base Class
-------------------

``Dynamics`` provides the foundation for differential equation-based models and
time-stepped simulation. The helper decorators control how update inputs and
outputs are received during a simulation step.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Dynamics

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   receive_update_output
   not_receive_update_output
   receive_update_input
   not_receive_update_input

Dynamics Utilities
------------------

Utilities for managing temporal dynamics, prefetching, and delayed outputs in
dynamical systems. Enable efficient handling of time-stepped simulations and
asynchronous signal processing in recurrent and spiking neural networks.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Prefetch
   PrefetchDelay
   PrefetchDelayAt
   OutputDelayAt

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   init_maybe_prefetch

Delay Utilities
---------------

Temporal delay buffers and state management for neural dynamics with synaptic delays.
``Delay`` provides ring buffer storage, ``DelayAccess`` enables retrieval of past values,
and ``StateWithDelay`` integrates delay mechanisms with state variables for realistic
neural modeling. ``InterpolationRegistry`` manages interpolation methods for delays.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Delay
   DelayAccess
   StateWithDelay
   InterpolationRegistry

Collective Operations
---------------------

Batch operations for managing states and function calls across module hierarchies.
Includes utilities for initialization, resetting, and vectorized execution (vmap)
of all states and functions in a network. Essential for efficient batch processing
and state management in complex neural architectures.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   call_order
   call_all_fns
   vmap_call_all_fns
   init_all_states
   vmap_init_all_states
   reset_all_states
   vmap_reset_all_states
   assign_state_values

Numerical Integration
---------------------

Numerical integration methods for solving ordinary differential equations (ODEs)
in dynamical systems. ``exp_euler_step`` implements the exponential Euler method
for stable integration of linear and nonlinear dynamics in neuronal models.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   exp_euler_step
