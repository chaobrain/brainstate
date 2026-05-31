``brainstate`` documentation
============================

`brainstate <https://github.com/chaobrain/brainstate>`_ implements a ``State``-based transformation system for programming compilation.



----

Features
^^^^^^^^^

.. grid::


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: State-based Transformation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainState`` provides a concise interface to write `State-based <./apis/brainstate.html>`__
            programs with composable `transformation <./apis/transform/index.html>`__ capabilities.


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Neural Network Support
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainState`` implements a neural network module system for building and training `ANNs/SNNs <./apis/nn/index.html>`__.


----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainstate[cpu]

    .. tab-item:: GPU

       .. code-block:: bash

          pip install -U brainstate[cuda12]
          pip install -U brainstate[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainstate[tpu]


----

See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

``brainstate`` is one part of our `brain simulation ecosystem <https://brainx.chaobrain.com/>`_.


----



Quick Start
^^^^^^^^^^^

Wrap mutable arrays in a ``State``, build models from ``brainstate.nn.Module``, and compose
JAX transformations that track state automatically:

.. code-block:: python

   import brainstate
   import jax.numpy as jnp


   class Linear(brainstate.nn.Module):
       def __init__(self, din, dout):
           super().__init__()
           self.w = brainstate.ParamState(brainstate.random.randn(din, dout) * 0.1)
           self.b = brainstate.ParamState(jnp.zeros(dout))

       def __call__(self, x):
           return x @ self.w.value + self.b.value


   model = Linear(3, 2)
   params = model.states(brainstate.ParamState)

   def loss_fn(x, y):
       return jnp.mean((model(x) - y) ** 2)

   @brainstate.transform.jit
   def train_step(x, y):
       grads = brainstate.transform.grad(loss_fn, grad_states=params)(x, y)
       for key, grad in grads.items():
           params[key].value -= 0.1 * grad
       return loss_fn(x, y)

   x = brainstate.random.randn(16, 3)
   y = brainstate.random.randn(16, 2)
   for step in range(100):
       loss = train_step(x, y)

Parameters are the only states differentiated; ``jit``, ``grad``, and ``vmap`` thread state
through automatically, with no manual bookkeeping. For a guided walkthrough, start with
:doc:`getting_started/quickstart`.




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/thinking_in_brainstate

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorials/core/index
   tutorials/transformations/index
   tutorials/brain_dynamics/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: How-to Guides

   how_to/checkpoint_and_restore
   how_to/inspect_and_edit_state_graph
   how_to/filter_and_organize_states
   how_to/collective_operations
   how_to/custom_states_and_mixins
   how_to/state_hooks
   how_to/constrain_and_regularize_parameters
   how_to/interoperate_with_flax_equinox
   how_to/migrate_from_pytorch

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Concepts

   concepts/why_state_based
   concepts/the_state_model
   concepts/the_parameter_model
   concepts/the_graph_model
   concepts/transformation_semantics
   concepts/time_and_environment
   concepts/the_typing_system

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/deep_learning/index
   examples/brain_dynamics/index



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   apis/brainstate.rst
   apis/graph.rst
   apis/nn/index.rst
   apis/transform/index.rst
   apis/interop.rst
   apis/random.rst
   apis/util.rst
   apis/typing.rst
   apis/mixin.rst
   apis/environ.rst

