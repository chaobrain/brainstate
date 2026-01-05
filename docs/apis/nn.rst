``brainstate.nn`` module
========================

.. currentmodule:: brainstate.nn
.. automodule:: brainstate.nn

Base Module Classes
-------------------

Core building blocks for neural network construction. ``Module`` is the base class
for all components in BrainState, providing utilities for parameter management,
state traversal, and hierarchical composition. ``Sequential`` enables easy chaining
of modules for feedforward architectures.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Module
   ElementWiseBlock
   Sequential

Parameter Containers
--------------------

Flexible parameter containers that integrate with BrainState's module system.
``Param`` supports bijective transformations for constrained optimization and
optional regularization. ``Const`` provides non-trainable constant parameters.
Both support automatic caching of transformed values for improved performance.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Param
   Const

Parameter Transforms
--------------------

Bijective transformations for constrained parameter optimization. These transforms
map between unconstrained and constrained spaces, enabling gradient-based optimization
of parameters with constraints (positivity, boundedness, simplex, etc.). All transforms
implement ``forward()``, ``inverse()``, and optional ``log_abs_det_jacobian()`` for
probabilistic applications. Use with ``Param`` for automatic constraint handling.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Transform
   IdentityT
   ClipT
   AffineT
   SigmoidT
   TanhT
   SoftsignT
   ScaledSigmoidT
   SoftplusT
   NegSoftplusT
   LogT
   ExpT
   ReluT
   PositiveT
   NegativeT
   PowerT
   OrderedT
   SimplexT
   UnitVectorT
   ChainT
   MaskedT

Standard Regularizations
------------------------

Classical regularization methods for parameter penalization and constraint enforcement.
These regularizations add penalty terms to the loss function to encourage desired
properties like sparsity (L1), smoothness (L2), or structural constraints (orthogonality,
spectral norms). Use with ``Param`` to automatically include regularization losses in
training objectives.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Regularization
   L1Reg
   L2Reg
   ElasticNetReg
   HuberReg
   GroupLassoReg
   TotalVariationReg
   MaxNormReg
   EntropyReg
   OrthogonalReg
   SpectralNormReg
   ChainedReg

Prior Distribution-Based Regularizations
-----------------------------------------

Probabilistic regularizations based on prior distributions for Bayesian-inspired
parameter estimation. These regularizations encode domain knowledge or assumptions
about parameter distributions (Gaussian, heavy-tailed, bounded, etc.). Particularly
useful for variational inference, maximum a posteriori (MAP) estimation, and
uncertainty quantification. Each regularization implements ``loss()``, ``sample_init()``,
and ``reset_value()`` for prior-based parameter initialization.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   GaussianReg
   StudentTReg
   CauchyReg
   UniformReg
   BetaReg
   LogNormalReg
   ExponentialReg
   GammaReg
   InverseGammaReg
   LogUniformReg
   HorseshoeReg
   SpikeAndSlabReg
   DirichletReg

Common Wrappers
---------------

Utility wrappers for context management and vectorization. ``EnvironContext`` manages
environment-specific configurations, while ``Vmap`` and ``ModuleMapper`` enable efficient
batching and vectorization of module operations across multiple inputs.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   EnvironContext
   Vmap
   ModuleMapper

Linear Layers
-------------

Fully-connected linear transformation layers with various specializations.
Includes standard dense layers, weight-standardized variants for improved training
stability, sparse connections for efficiency, and low-rank adaptation (LoRA) for
parameter-efficient fine-tuning.

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

Convolutional layers for 1D, 2D, and 3D spatial feature extraction. Includes standard
convolutions, weight-standardized variants for improved normalization, and transposed
convolutions for upsampling operations. Essential for processing sequential data, images,
and volumetric inputs.

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

Downsampling, upsampling, and shape manipulation operations for spatial data.
Includes average pooling, max pooling, Lp-norm pooling, unpooling for reconstruction,
and adaptive pooling for fixed output sizes. ``Flatten`` and ``Unflatten`` enable
seamless transitions between spatial and flat representations.

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

Spatial padding operations with various boundary conditions. Supports reflection,
replication, zero, constant value, and circular padding for 1D, 2D, and 3D inputs.
Essential for controlling output sizes in convolutional networks and handling edge effects.

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

Normalization techniques for stabilizing training and improving convergence.
Includes batch normalization variants (0D-3D), layer normalization, RMS normalization,
group normalization, and weight standardization. Each normalization strategy addresses
different aspects of internal covariate shift and gradient flow.

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

Regularization through stochastic neuron dropping during training. Includes standard
dropout, spatial dropout variants (1D-3D), alpha dropout for self-normalizing networks,
and fixed dropout with deterministic masking. Prevents overfitting by encouraging
robust feature learning.

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

Learnable embedding layers for mapping discrete tokens to continuous vector representations.
Essential for processing categorical inputs, text, and discrete symbols in neural networks.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Embedding

Element-wise Layers
-------------------

Non-linear activation layers that operate element-wise on input tensors. Includes
rectified linear units (ReLU and variants), sigmoid functions, hyperbolic tangent,
softmax for probability distributions, and specialized activations for specific
architectures (SELU, GELU, SiLU, Mish). These introduce non-linearity enabling
networks to learn complex patterns.

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

Functional (non-module) activation functions for flexible composition. These are
pure functions that can be used directly in ``update()`` methods or combined with
JAX transformations. Provides the same activations as the layer-based equivalents
but without state or module overhead.

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

Sparse, event-driven connectivity patterns for neuromorphic computing and spiking
neural networks. Supports fixed connection counts, probabilistic connectivity, and
event-based linear transformations for efficient processing of sparse temporal signals.

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

Recurrent neural network cells for sequential data processing and temporal modeling.
Includes vanilla RNN, gated recurrent units (GRU), minimal gated units (MGU), long
short-term memory (LSTM), and unbalanced LSTM variants. Each cell maintains internal
state across time steps for memory-dependent computations.

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

Base classes for implementing dynamical systems and time-evolving neural models.
``Dynamics`` provides the foundation for differential equation-based models, while
``DynamicsGroup`` enables hierarchical composition of multiple dynamical components.
Essential for neuromorphic computing and brain-inspired architectures.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DynamicsGroup
   Dynamics

Dynamics Utilities
------------------

Utilities for managing temporal dynamics, prefetching, and delayed outputs in
dynamical systems. Enable efficient handling of time-stepped simulations and
asynchronous signal processing in recurrent and spiking neural networks.

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

Temporal delay buffers and state management for neural dynamics with synaptic delays.
``Delay`` provides ring buffer storage, ``DelayAccess`` enables retrieval of past values,
and ``StateWithDelay`` integrates delay mechanisms with state variables for realistic
neural modeling.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Delay
   DelayAccess
   StateWithDelay

Collective Operations
---------------------

Batch operations for managing states and function calls across module hierarchies.
Includes utilities for initialization, resetting, and vectorized execution (vmap)
of all states and functions in a network. Essential for efficient batch processing
and state management in complex neural architectures.

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

Numerical integration methods for solving ordinary differential equations (ODEs)
in dynamical systems. ``exp_euler_step`` implements the exponential Euler method
for stable integration of linear and nonlinear dynamics in neuronal models.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   exp_euler_step

Metrics
-------

Performance metrics for model evaluation and monitoring during training. Includes
accuracy, precision, recall, F1 score, confusion matrices, and running statistics
(average, Welford variance). ``MetricState`` provides state containers, while
``MultiMetric`` enables tracking multiple metrics simultaneously.

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

Hierarchical Data
-----------------

Data structures for managing hierarchical and nested information in neural networks.
``HiData`` provides utilities for organizing and accessing tree-structured data,
useful for compositional models and hierarchical state management.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HiData

Utility Functions
-----------------

General-purpose utilities for neural network operations. ``count_parameters`` tallies
trainable and total parameters in a model, while ``clip_grad_norm`` implements gradient
clipping for training stability.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   count_parameters
   clip_grad_norm

Parameter Initialization
------------------------

Weight initialization strategies for neural network parameters. Includes zero and
constant initialization, random distributions (normal, uniform, truncated normal),
and variance-scaling methods (Kaiming/He, Xavier/Glorot, LeCun) designed for specific
activation functions. Orthogonal initialization supports recurrent networks. Proper
initialization is crucial for training stability and convergence.

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
