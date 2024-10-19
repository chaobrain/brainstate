``brainstate`` documentation
============================

`brainstate <https://github.com/brainpy/brainstate>`_ implements a ``State``-based transformation system for brain dynamics programming (BDP).

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
   :maxdepth: 2

   api.rst

