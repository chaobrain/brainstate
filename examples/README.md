# Examples for ``brainstate`` library

We provide several kinds of examples to demonstrate the usage of the ``brainstate`` library. The examples are organized
in the following categories:

- The files with name started with ``0__`` are the examples for deep neural networks and the functional /
  transform API.
- The files with name started with ``1__`` are the examples for brain simulation models, especially spiking neural
  networks.
- The files with name started with ``2__`` are the examples for brain-inspired computing models, especially training
  spiking neural networks.
- The files with name started with ``3__`` are the examples for rate-based recurrent neural networks.

## State-aware transforms (autodiff & parallelism)

Self-contained, runnable walkthroughs of the state-aware
``brainstate.transform`` primitives:

- ``012_shard_map_spmd.py`` — single-program multiple-data sharding across a
  device mesh with ``shard_map`` (data parallelism, replicated/sharded states,
  collectives, ``jit`` composition, and 2-D data + model parallelism).
- ``013_vjp_reverse_mode.py`` — reverse-mode autodiff with ``vjp`` (state
  gradients, full Jacobians, Hessian-vector products, a training loop).
- ``014_jvp_forward_mode.py`` — forward-mode autodiff with ``jvp`` (directional
  derivatives, column-by-column Jacobians, forward-over-reverse HVPs).



