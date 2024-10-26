``brainstate.optim`` for optimization algorithms
================================================

.. currentmodule:: brainstate.optim 
.. automodule:: brainstate.optim 



SGD Optimizer
--------------

Stochastic Gradient Descent (SGD) optimizers.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Optimizer
   SGD
   Momentum
   MomentumNesterov
   Adagrad
   Adadelta
   RMSProp
   Adam
   LARS
   Adan
   AdamW



Learning Rate Scheduler
-----------------------

Learning rate schedulers.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LearningRateScheduler
   ConstantLR
   StepLR
   MultiStepLR
   CosineAnnealingLR
   CosineAnnealingWarmRestarts
   ExponentialLR
   ExponentialDecayLR
   InverseTimeDecayLR
   PolynomialDecayLR
   PiecewiseConstantLR


``optax`` Optimizer
-------------------

Optimizers from the `optax <https://github.com/google-deepmind/optax>`_ library.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   OptaxOptimizer


Helper Functions
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   to_same_dict_tree
   OptimState
