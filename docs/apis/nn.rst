``brainstate.nn`` for neural network building
=============================================

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


Synaptic Interaction Layers
---------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Linear
   ScaledWSLinear
   SignedWLinear
   SparseLinear
   AllToAll
   OneToOne

   Conv1d
   Conv2d
   Conv3d
   ScaledWSConv1d
   ScaledWSConv2d
   ScaledWSConv3d

   BatchNorm0d
   BatchNorm1d
   BatchNorm2d
   BatchNorm3d

   Flatten
   Unflatten

   AvgPool1d
   AvgPool2d
   AvgPool3d
   MaxPool1d
   MaxPool2d
   MaxPool3d
   AdaptiveAvgPool1d
   AdaptiveAvgPool2d
   AdaptiveAvgPool3d
   AdaptiveMaxPool1d
   AdaptiveMaxPool2d
   AdaptiveMaxPool3d

   Embedding


Element-wise Layers
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DropoutFixed
   Dropout
   Dropout1d
   Dropout2d
   Dropout3d
   AlphaDropout
   FeatureAlphaDropout
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


Base Dynamics Classes
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DynamicsGroup
   Projection
   Dynamics
   Prefetch
   AlignPostProj
   DeltaProj
   CurrentProj
   Delay
   DelayAccess
   StateWithDelay
   SynOut
   COBA
   CUBA
   MgBlock


Neuronal/Synaptic Dynamics
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Neuron
   IF
   LIF
   LIFRef
   ALIF
   Synapse
   Expon
   STP
   STD
   AMPA
   GABAa
   SpikeTime
   PoissonSpike
   PoissonEncoder
   RNNCell
   ValinaRNNCell
   GRUCell
   MGUCell
   LSTMCell
   URLSTMCell
   LeakyRateReadout
   LeakySpikeReadout


Numerical Integration Methods
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   exp_euler_step


Collective Operations
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   call_order
   init_all_states
   reset_all_states
   load_all_states
   save_all_states
   assign_state_values
   MAX_ORDER


