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

