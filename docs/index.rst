``brainstate`` documentation
============================

`brainstate <https://github.com/chaobrain/brainstate>`_ implements a ``State``-based Transformation System for Program Compilation and Augmentation.

``BrainState`` is specifically designed to work with models that have states, including rate-based recurrent neural networks, spiking neural networks, and other dynamical systems.

----

Features
^^^^^^^^^

.. grid::



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Program Compilation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainState`` supports `program compilation <./apis/compile.html>`__ (such as just-in-time compilation) with its `state-based <./apis/brainstate.html>`__ IR construction.



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Program Augmentation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainState`` supports program `functionality augmentation <./apis/augment.html>`__ (such batching) with its `graph-based <./apis/graph.html>`__ Python objects.




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


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts-en.ipynb
   quickstart/concepts-zh.ipynb




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials


   tutorials/state-en.ipynb
   tutorials/state-zh.ipynb
   tutorials/pygraph-en.ipynb
   tutorials/pygraph-zh.ipynb
   tutorials/program_compilation-en.ipynb
   tutorials/program_compilation-zh.ipynb
   tutorials/program_augmentation-en.ipynb
   tutorials/program_augmentation-zh.ipynb
   tutorials/gspmd-en.ipynb
   tutorials/gspmd-zh.ipynb
   tutorials/random_numbers-en.ipynb
   tutorials/random_numbers-zh.ipynb




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

