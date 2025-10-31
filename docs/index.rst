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

            ``BrainState`` provides an intuitive interface to write `State-based <./apis/brainstate.html>`__
            programs with powerful `transformation <./apis/transform.html>`__ capabilities.


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Neural Network Support
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainState`` implements a neural network module system for building and training `ANNs/SNNs <./apis/nn.html>`__.


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

``brainstate`` is one part of our `brain simulation ecosystem <https://brainmodeling.readthedocs.io/>`_.


----



Quick Start
^^^^^^^^^^^

``BrainState`` enables **Pythonic State-based programming** with automatic state management
in JAX transformations. Here's a complete example demonstrating the key concepts:


**1. States vs Constants: Defining Dynamic Variables**

.. code-block:: python

   import brainstate
   import jax.numpy as jnp

   # Dynamic variables are defined with State
   # These will be automatically tracked and managed through transformations
   state_var = brainstate.State(jnp.array([1.0, 2.0, 3.0]))
   param = brainstate.ParamState(jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

   # Regular Python variables are constants
   # These remain fixed and are not tracked
   learning_rate = 0.01
   num_steps = 100


**2. JIT Compilation with Automatic State Management**

.. code-block:: python

   # Define a stateful computation
   def step_forward(x):
       # Access state value
       y = state_var.value @ param.value
       # Update state in-place
       state_var.value = state_var.value + 0.1
       return y

   # JIT compilation automatically handles state
   step_jit = brainstate.transform.jit(step_forward)

   # States are tracked and updated automatically!
   result1 = step_jit(jnp.array(1.0))
   result2 = step_jit(jnp.array(1.0))  # state_var has been updated


**3. Gradient Computation with State Awareness**

.. code-block:: python

   # Define a loss function with states
   def loss_fn(x):
       pred = state_var.value @ param.value
       return jnp.sum((pred - x) ** 2)

   # Compute gradients - only for ParamState, not all States!
   grad_fn = brainstate.transform.grad(loss_fn, grad_states=param)

   # Get gradients
   target = jnp.array([1.0, 2.0])
   grads = grad_fn(target)

   # Update parameters
   param.value = param.value - learning_rate * grads


**4. Vectorization (vmap) with State Broadcasting**

.. code-block:: python

   # Reset states for demo
   state_var.value = jnp.array([1.0, 2.0, 3.0])

   # Process batch of inputs with vmap
   def process_single(x):
       return jnp.sum(state_var.value * x)

   # vmap automatically broadcasts states across batch dimension
   process_batch = brainstate.transform.vmap(
       process_single,
       in_axes=0,  # batch over first axis of input
       out_axes=0  # batch over first axis of output
   )

   # Process entire batch - state is shared/broadcasted
   batch_inputs = jnp.array([[1., 2., 3.],
                             [4., 5., 6.],
                             [7., 8., 9.]])
   batch_results = process_batch(batch_inputs)


**5. Complete Training Loop Example**

.. code-block:: python

   import brainstate as bst
   import jax.numpy as jnp

   # Define a simple model with states
   class SimpleModel(brainstate.nn.Module):
       def __init__(self, in_size, out_size):
           super().__init__()
           self.weight = brainstate.ParamState(brainstate.random.randn(in_size, out_size) * 0.01)
           self.bias = brainstate.ParamState(jnp.zeros(out_size))

       def __call__(self, x):
           return x @ self.weight.value + self.bias.value

   # Create model
   model = SimpleModel(3, 2)

   # Define loss with automatic state tracking
   def loss_fn(x, y):
       pred = model(x)
       return jnp.mean((pred - y) ** 2)

   # JIT + Grad: compose transformations seamlessly
   params = model.states(brainstate.ParamState)

   @brainstate.transform.jit
   def train_step(x, y):
       # Compute gradients only for ParamStates
       grads = brainstate.transform.grad(loss_fn, grad_states=params)(x, y)

       # Update parameters
       for key in params.keys():
           params[key].value -= 0.01 * grads[key]

       return loss_fn(x, y)

   # Training loop - states updated automatically!
   for i in range(100):
       x_batch = brainstate.random.randn(32, 3)
       y_batch = brainstate.random.randn(32, 2)
       loss = train_step(x_batch, y_batch)
       if i % 20 == 0:
           print(f"Step {i}, Loss: {loss:.4f}")


**Key Advantages:**

- ✅ **Intuitive**: States behave like normal Python variables with ``.value``
- ✅ **Automatic**: No manual state threading through transformations
- ✅ **Composable**: JIT, grad, vmap work together seamlessly
- ✅ **Type-safe**: Different state types (State, ParamState, RandomState) for different purposes
- ✅ **Efficient**: Zero-overhead state management compiled by JAX




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorial Series

   tutorials/basics/index.rst
   tutorials/neural_networks/index.rst
   tutorials/transforms/index.rst
   tutorials/utilities/index.rst
   tutorials/examples/index.rst
   tutorials/migration/index.rst




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/core_examples.rst
   examples/ann_training-en.ipynb
   examples/ann_training-zh.ipynb
   examples/snn_simulation-en.ipynb
   examples/snn_simulation-zh.ipynb
   examples/snn_training-en.ipynb
   examples/snn_training-zh.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   apis/brainstate.rst
   apis/graph.rst
   apis/transform.rst
   apis/nn.rst
   apis/random.rst
   apis/util.rst
   apis/typing.rst
   apis/mixin.rst
   apis/environ.rst

