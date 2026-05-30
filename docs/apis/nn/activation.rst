Activations
===========

.. currentmodule:: brainstate.nn

Non-linear activations, available both as stateful layer modules and as pure
functions for flexible composition.

Element-wise Layers
-------------------

Non-linear activation layers that operate element-wise on input tensors. Includes
rectified linear units (ReLU and variants), sigmoid functions, hyperbolic tangent,
softmax for probability distributions, and specialized activations for specific
architectures (SELU, GELU, SiLU, Mish). These introduce non-linearity enabling
networks to learn complex patterns.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Threshold
   ReLU
   RReLU
   Hardtanh
   ReLU6
   Sigmoid
   Hardsigmoid
   Tanh
   SiLU
   Mish
   Hardswish
   ELU
   CELU
   SELU
   GLU
   GELU
   Hardshrink
   LeakyReLU
   LogSigmoid
   Softplus
   Softshrink
   PReLU
   Softsign
   Tanhshrink
   Softmin
   Softmax
   Softmax2d
   LogSoftmax
   Identity
   SpikeBitwise

Functional Activations
----------------------

Functional (non-module) activation functions for flexible composition. These are
pure functions that can be used directly in ``update()`` methods or combined with
JAX transformations. Provides the same activations as the layer-based equivalents
but without state or module overhead.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   tanh
   relu
   squareplus
   softplus
   soft_sign
   sigmoid
   silu
   swish
   log_sigmoid
   elu
   leaky_relu
   hard_tanh
   celu
   selu
   gelu
   glu
   logsumexp
   log_softmax
   softmax
   standardize
   one_hot
   relu6
   hard_sigmoid
   hard_silu
   hard_swish
   hard_shrink
   rrelu
   mish
   soft_shrink
   prelu
   tanh_shrink
   softmin
   sparse_plus
   sparse_sigmoid
