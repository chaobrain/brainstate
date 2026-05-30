Parameter Initialization
========================

.. currentmodule:: brainstate.nn.init

Weight initialization strategies for neural network parameters. Includes zero and
constant initialization, random distributions (normal, uniform, truncated normal),
and variance-scaling methods (Kaiming/He, Xavier/Glorot, LeCun) designed for specific
activation functions. Orthogonal initialization supports recurrent networks. Proper
initialization is crucial for training stability and convergence.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   param
   calculate_init_gain
   ZeroInit
   Constant
   Identity
   Normal
   TruncatedNormal
   Uniform
   VarianceScaling
   KaimingUniform
   KaimingNormal
   XavierUniform
   XavierNormal
   LecunUniform
   LecunNormal
   Orthogonal
   DeltaOrthogonal
