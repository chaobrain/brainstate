``brainstate.nn`` module
========================

.. currentmodule:: brainstate.nn
.. automodule:: brainstate.nn

Base Module Classes
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Module
   ElementWiseBlock
   Sequential

Common Wrappers
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   EnvironContext
   Vmap

Linear Layers
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Linear
   ScaledWSLinear
   SignedWLinear
   SparseLinear
   LoRA
   AllToAll
   OneToOne

Convolutional Layers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Conv1d
   Conv2d
   Conv3d
   ScaledWSConv1d
   ScaledWSConv2d
   ScaledWSConv3d
   ConvTranspose1d
   ConvTranspose2d
   ConvTranspose3d

Pooling and Reshaping
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Flatten
   Unflatten
   AvgPool1d
   AvgPool2d
   AvgPool3d
   MaxPool1d
   MaxPool2d
   MaxPool3d
   MaxUnpool1d
   MaxUnpool2d
   MaxUnpool3d
   LPPool1d
   LPPool2d
   LPPool3d
   AdaptiveAvgPool1d
   AdaptiveAvgPool2d
   AdaptiveAvgPool3d
   AdaptiveMaxPool1d
   AdaptiveMaxPool2d
   AdaptiveMaxPool3d

Padding Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ReflectionPad1d
   ReflectionPad2d
   ReflectionPad3d
   ReplicationPad1d
   ReplicationPad2d
   ReplicationPad3d
   ZeroPad1d
   ZeroPad2d
   ZeroPad3d
   ConstantPad1d
   ConstantPad2d
   ConstantPad3d
   CircularPad1d
   CircularPad2d
   CircularPad3d

Normalization Layers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BatchNorm0d
   BatchNorm1d
   BatchNorm2d
   BatchNorm3d
   LayerNorm
   RMSNorm
   GroupNorm
   weight_standardization

Dropout Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Dropout
   Dropout1d
   Dropout2d
   Dropout3d
   AlphaDropout
   FeatureAlphaDropout
   DropoutFixed

Embedding
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Embedding

Element-wise Layers
-------------------

.. autosummary::
   :toctree: generated/
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

Activation Functions
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

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

Event-based Connectivity
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FixedNumConn
   EventFixedNumConn
   EventFixedProb
   EventLinear

Recurrent Cells
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   RNNCell
   ValinaRNNCell
   GRUCell
   MGUCell
   LSTMCell
   URLSTMCell

Dynamics Base Classes
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DynamicsGroup
   Dynamics

Dynamics Utilities
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Prefetch
   PrefetchDelay
   PrefetchDelayAt
   OutputDelayAt

Delay Utilities
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Delay
   DelayState
   DelayAccess
   StateWithDelay

Collective Operations
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

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

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   exp_euler_step

Metrics
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   MetricState
   Metric
   AverageMetric
   WelfordMetric
   AccuracyMetric
   MultiMetric
   PrecisionMetric
   RecallMetric
   F1ScoreMetric
   ConfusionMatrix

Utility Functions
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   count_parameters
   clip_grad_norm

Parameter Initialization
------------------------

.. currentmodule:: brainstate.nn.init

.. autosummary::
   :toctree: generated/
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

.. currentmodule:: brainstate.nn
