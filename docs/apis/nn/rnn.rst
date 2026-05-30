Recurrent Cells
===============

.. currentmodule:: brainstate.nn

Recurrent neural network cells for sequential data processing and temporal modeling.
Includes vanilla RNN, gated recurrent units (GRU), minimal gated units (MGU), long
short-term memory (LSTM), and unbalanced LSTM variants. Each cell maintains internal
state across time steps for memory-dependent computations.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   RNNCell
   ValinaRNNCell
   GRUCell
   MGUCell
   LSTMCell
   URLSTMCell
