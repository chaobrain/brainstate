``brainstate`` documentation
============================

`brainstate <https://github.com/chaobrain/brainstate>`_ implements a ``State``-based Transformation System for Program Compilation and Augmentation.

``BrainState`` is specifically designed to work with models that have states, including rate-based recurrent neural networks, spiking neural networks, and other dynamical systems.

``BrainState`` is the foundation of our establishing `Brain Dynamics Programming (BDP) ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.

----

Features
^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Pythonic
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            ``BrainState`` provides a Pythonic interface to brain dynamics programming.


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Event-driven Computation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            ``BrainState`` enables `event-driven computation <./apis/event.rst>`__ for spiking neural networks,
            and thus obtains unprecedented performance on CPU and GPU devices.



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Program Compilation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            ``BrainState`` supports `program compilation <./apis/compile.rst>`__ (such as just-in-time compilation) with its `state-based <./apis/brainstate.rst>`__ IR construction.



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Program Augmentation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            ``BrainState`` supports program `functionality augmentation <./apis/augment.rst>`__ (such batching) with its `graph-based <./apis/graph.rst>`__ Python objects.




----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainstate[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainstate[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainstate[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `BDP ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts-en.ipynb
   quickstart/concepts-zh.ipynb
   quickstart/ann_training.ipynb
   quickstart/snn_training.ipynb
   quickstart/snn_training.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorials/pygraph-zh.ipynb
   tutorials/random_numbers.ipynb
   tutorials/event_driven_computation.ipynb
   tutorials/gspmd.ipynb
   tutorials/optimizers.ipynb
   tutorials/program_compilation.ipynb
   tutorials/program_augmentation.ipynb
   tutorials/checkpointing.ipynb
   tutorials/align_pre_align_post.ipynb
   tutorials/artificial_neural_networks.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/core_examples.rst



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   api.rst

