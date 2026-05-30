Metrics
=======

.. currentmodule:: brainstate.nn

Performance metrics for model evaluation and monitoring during training. Includes
accuracy, precision, recall, F1 score, confusion matrices, and running statistics
(average, Welford variance). ``MetricState`` provides state containers, while
``MultiMetric`` enables tracking multiple metrics simultaneously.

.. autosummary::
   :toctree: ../generated/
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
