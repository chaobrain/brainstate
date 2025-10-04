# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from brainstate._state import LongTermState

__all__ = [
    'MetricState',
    'Metric',
    'AverageMetric',
    'WelfordMetric',
    'AccuracyMetric',
    'MultiMetric',
    'PrecisionMetric',
    'RecallMetric',
    'F1ScoreMetric',
    'ConfusionMatrix',
]


class MetricState(LongTermState):
    """
    Wrapper class for Metric Variables.

    This class extends ``State`` to provide a container for metric state variables
    that need to be tracked and updated during training or evaluation.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> state = brainstate.nn.MetricState(jnp.array(0.0))
        >>> state.value
        Array(0., dtype=float32)
        >>> state.value = jnp.array(1.5)
        >>> state.value
        Array(1.5, dtype=float32)
    """
    __module__ = "brainstate.nn"


class Metric(object):
    """
    Base class for metrics.

    Any class that subclasses ``Metric`` should implement ``compute``, ``reset``,
    and ``update`` methods to track and compute evaluation metrics.

    Methods
    -------
    reset()
        Reset the metric state to initial values.
    update(**kwargs)
        Update the metric state with new data.
    compute()
        Compute and return the current metric value.

    Notes
    -----
    This is an abstract base class and should not be instantiated directly.
    Subclasses must implement all three methods.
    """
    __module__ = "brainstate.nn"

    def reset(self) -> None:
        """
        In-place reset the metric state to initial values.

        This method should restore all internal state variables to their
        initial values as if the metric was just constructed.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError('Must override `reset()` method.')

    def update(self, **kwargs) -> None:
        """
        In-place update the metric with new data.

        Parameters
        ----------
        **kwargs
            Keyword arguments containing the data to update the metric.
            The specific arguments depend on the metric implementation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError('Must override `update()` method.')

    def compute(self):
        """
        Compute and return the current value of the metric.

        Returns
        -------
        metric_value
            The computed metric value. The type depends on the specific metric.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError('Must override `compute()` method.')


class AverageMetric(Metric):
    """
    Average metric for computing running mean of values.

    This metric maintains a running sum and count to compute the average
    of all values passed to it via the ``update`` method.

    Parameters
    ----------
    argname : str, optional
        The keyword argument name that ``update`` will use to derive the new value.
        Defaults to ``'values'``.

    Attributes
    ----------
    argname : str
        The keyword argument name for updates.
    total : MetricState
        Cumulative sum of all values.
    count : MetricState
        Total number of elements processed.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> batch_loss = jnp.array([1, 2, 3, 4])
        >>> batch_loss2 = jnp.array([3, 2, 1, 0])
        >>> metrics = brainstate.nn.AverageMetric()
        >>> metrics.compute()
        Array(nan, dtype=float32)
        >>> metrics.update(values=batch_loss)
        >>> metrics.compute()
        Array(2.5, dtype=float32)
        >>> metrics.update(values=batch_loss2)
        >>> metrics.compute()
        Array(2., dtype=float32)
        >>> metrics.reset()
        >>> metrics.compute()
        Array(nan, dtype=float32)

    Notes
    -----
    The metric returns NaN when no values have been added (count = 0).
    This metric can handle scalar values, arrays, or tensors.
    """
    __module__ = "brainstate.nn"

    def __init__(self, argname: str = 'values'):
        self.argname = argname
        self.total = MetricState(jnp.array(0, dtype=jnp.float32))
        self.count = MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """
        Reset the metric state to zero.

        This sets both the total sum and count to zero.
        """
        self.total.value = jnp.array(0, dtype=jnp.float32)
        self.count.value = jnp.array(0, dtype=jnp.int32)

    def update(self, **kwargs) -> None:
        """
        Update the metric with new values.

        Parameters
        ----------
        **kwargs
            Must contain ``self.argname`` as a key, mapping to the values
            to be averaged. Values can be scalars, arrays, or tensors.

        Raises
        ------
        TypeError
            If the expected keyword argument is not provided.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainstate
            >>> metric = brainstate.nn.AverageMetric('loss')
            >>> metric.update(loss=jnp.array([1.0, 2.0, 3.0]))
            >>> metric.compute()
            Array(2., dtype=float32)
        """
        if self.argname not in kwargs:
            raise TypeError(f"Expected keyword argument '{self.argname}'")
        values: tp.Union[int, float, jax.Array] = kwargs[self.argname]
        self.total.value += (
            values if isinstance(values, (int, float)) else values.sum()
        )
        self.count.value += 1 if isinstance(values, (int, float)) else values.size

    def compute(self) -> jax.Array:
        """
        Compute and return the average.

        Returns
        -------
        jax.Array
            The average of all values provided to ``update``.
            Returns NaN if no values have been added.
        """
        return self.total.value / self.count.value


@partial(
    jax.tree_util.register_dataclass,
    data_fields=['mean', 'standard_error_of_mean', 'standard_deviation'],
    meta_fields=[]
)
@dataclass
class Statistics:
    """
    Dataclass for statistical measurements.

    Attributes
    ----------
    mean : float32
        The mean value.
    standard_error_of_mean : float32
        The standard error of the mean (SEM).
    standard_deviation : float32
        The standard deviation.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> stats = brainstate.nn.Statistics(
        ...     mean=jnp.float32(2.5),
        ...     standard_error_of_mean=jnp.float32(0.5),
        ...     standard_deviation=jnp.float32(1.0)
        ... )
        >>> stats.mean
        Array(2.5, dtype=float32)
    """
    __module__ = "brainstate.nn"
    mean: jnp.float32
    standard_error_of_mean: jnp.float32
    standard_deviation: jnp.float32


class WelfordMetric(Metric):
    """
    Welford's algorithm for computing mean and variance of streaming data.

    This metric uses Welford's online algorithm to compute running statistics
    (mean, variance, standard deviation) in a numerically stable way.

    Parameters
    ----------
    argname : str, optional
        The keyword argument name that ``update`` will use to derive the new value.
        Defaults to ``'values'``.

    Attributes
    ----------
    argname : str
        The keyword argument name for updates.
    count : MetricState
        Total number of elements processed.
    mean : MetricState
        Running mean estimate.
    m2 : MetricState
        Running sum of squared deviations from the mean.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> batch_loss = jnp.array([1, 2, 3, 4])
        >>> batch_loss2 = jnp.array([3, 2, 1, 0])
        >>> metrics = brainstate.nn.WelfordMetric()
        >>> metrics.compute()
        Statistics(mean=Array(0., dtype=float32), standard_error_of_mean=Array(nan, dtype=float32), standard_deviation=Array(nan, dtype=float32))
        >>> metrics.update(values=batch_loss)
        >>> metrics.compute()
        Statistics(mean=Array(2.5, dtype=float32), standard_error_of_mean=Array(0.559017, dtype=float32), standard_deviation=Array(1.118034, dtype=float32))
        >>> metrics.update(values=batch_loss2)
        >>> metrics.compute()
        Statistics(mean=Array(2., dtype=float32), standard_error_of_mean=Array(0.43301272, dtype=float32), standard_deviation=Array(1.2247449, dtype=float32))
        >>> metrics.reset()
        >>> metrics.compute()
        Statistics(mean=Array(0., dtype=float32), standard_error_of_mean=Array(nan, dtype=float32), standard_deviation=Array(nan, dtype=float32))

    Notes
    -----
    Welford's algorithm is numerically stable and computes variance in a single pass.
    The algorithm updates the mean and variance incrementally as new data arrives.

    References
    ----------
    .. [1] Welford, B. P. (1962). "Note on a method for calculating corrected sums
           of squares and products". Technometrics. 4 (3): 419-420.
    """
    __module__ = "brainstate.nn"

    def __init__(self, argname: str = 'values'):
        self.argname = argname
        self.count = MetricState(jnp.array(0, dtype=jnp.int32))
        self.mean = MetricState(jnp.array(0, dtype=jnp.float32))
        self.m2 = MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self) -> None:
        """
        Reset the metric state to zero.

        This resets count, mean, and the sum of squared deviations (m2).
        """
        self.count.value = jnp.array(0, dtype=jnp.uint32)
        self.mean.value = jnp.array(0, dtype=jnp.float32)
        self.m2.value = jnp.array(0, dtype=jnp.float32)

    def update(self, **kwargs) -> None:
        """
        Update the metric using Welford's algorithm.

        Parameters
        ----------
        **kwargs
            Must contain ``self.argname`` as a key, mapping to the values
            to be processed. Values can be scalars, arrays, or tensors.

        Raises
        ------
        TypeError
            If the expected keyword argument is not provided.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainstate
            >>> metric = brainstate.nn.WelfordMetric('data')
            >>> metric.update(data=jnp.array([1.0, 2.0, 3.0]))
            >>> stats = metric.compute()
            >>> stats.mean
            Array(2., dtype=float32)
        """
        if self.argname not in kwargs:
            raise TypeError(f"Expected keyword argument '{self.argname}'")
        values: tp.Union[int, float, jax.Array] = kwargs[self.argname]
        count = 1 if isinstance(values, (int, float)) else values.size
        original_count = self.count.value
        self.count.value += count
        delta = (
                    values if isinstance(values, (int, float)) else values.mean()
                ) - self.mean.value
        self.mean.value += delta * count / self.count.value
        m2 = 0.0 if isinstance(values, (int, float)) else values.var() * count
        self.m2.value += (
            m2 + delta * delta * count * original_count / self.count.value
        )

    def compute(self) -> Statistics:
        """
        Compute and return statistical measurements.

        Returns
        -------
        Statistics
            A dataclass containing mean, standard error of mean, and standard deviation.
            Returns NaN for error metrics when count is 0.
        """
        variance = self.m2.value / self.count.value
        standard_deviation = variance ** 0.5
        sem = standard_deviation / (self.count.value ** 0.5)
        return Statistics(
            mean=self.mean.value,
            standard_error_of_mean=sem,
            standard_deviation=standard_deviation,
        )


class AccuracyMetric(AverageMetric):
    """
    Accuracy metric for classification tasks.

    This metric computes the accuracy by comparing predicted labels (derived from
    logits using argmax) with ground truth labels. It inherits from ``AverageMetric``
    and shares the same ``reset`` and ``compute`` implementations.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>> logits = jax.random.normal(jax.random.key(0), (5, 2))
        >>> labels = jnp.array([1, 1, 0, 1, 0])
        >>> logits2 = jax.random.normal(jax.random.key(1), (5, 2))
        >>> labels2 = jnp.array([0, 1, 1, 1, 1])
        >>> metrics = brainstate.nn.AccuracyMetric()
        >>> metrics.compute()
        Array(nan, dtype=float32)
        >>> metrics.update(logits=logits, labels=labels)
        >>> metrics.compute()
        Array(0.6, dtype=float32)
        >>> metrics.update(logits=logits2, labels=labels2)
        >>> metrics.compute()
        Array(0.7, dtype=float32)
        >>> metrics.reset()
        >>> metrics.compute()
        Array(nan, dtype=float32)

    Notes
    -----
    The accuracy is computed as the fraction of correct predictions:
    accuracy = (number of correct predictions) / (total predictions)

    Logits are converted to predictions using argmax along the last dimension.
    """
    __module__ = "brainstate.nn"

    def update(self, *, logits: jax.Array, labels: jax.Array, **_) -> None:
        """
        Update the accuracy metric with predictions and labels.

        Parameters
        ----------
        logits : jax.Array
            Predicted activations/logits with shape (..., num_classes).
            The last dimension represents class scores.
        labels : jax.Array
            Ground truth integer labels with shape (...,).
            Must be one dimension less than logits.
        **_
            Additional keyword arguments are ignored.

        Raises
        ------
        ValueError
            If logits and labels have incompatible shapes, or if labels have
            incorrect dtype.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainstate
            >>> logits = jnp.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
            >>> labels = jnp.array([1, 0, 1])
            >>> metric = brainstate.nn.AccuracyMetric()
            >>> metric.update(logits=logits, labels=labels)
            >>> metric.compute()
            Array(1., dtype=float32)
        """
        if logits.ndim != labels.ndim + 1:
            raise ValueError(
                f'Expected logits.ndim==labels.ndim+1, got {logits.ndim} and {labels.ndim}'
            )
        elif labels.dtype in (jnp.int64, np.int32, np.int64):
            labels = jnp.astype(labels, jnp.int32)
        elif labels.dtype != jnp.int32:
            raise ValueError(f'Expected labels.dtype==jnp.int32, got {labels.dtype}')

        super().update(values=(logits.argmax(axis=-1) == labels))


class PrecisionMetric(Metric):
    """
    Precision metric for binary and multi-class classification.

    Precision is the ratio of true positives to all positive predictions:
    precision = TP / (TP + FP)

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. If None, assumes binary classification. Default is None.
    average : str, optional
        Type of averaging for multi-class: 'micro', 'macro', or 'weighted'.
        Default is 'macro'. Ignored for binary classification.

    Attributes
    ----------
    num_classes : int or None
        Number of classes.
    average : str
        Averaging method for multi-class.
    true_positives : MetricState
        Count of true positive predictions.
    false_positives : MetricState
        Count of false positive predictions.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> predictions = jnp.array([1, 0, 1, 1, 0])
        >>> labels = jnp.array([1, 0, 0, 1, 0])
        >>> metric = brainstate.nn.PrecisionMetric()
        >>> metric.update(predictions=predictions, labels=labels)
        >>> metric.compute()
        Array(0.6666667, dtype=float32)

    Notes
    -----
    For multi-class classification, the metric supports different averaging strategies:
    - 'micro': Calculate metrics globally by counting total TP and FP
    - 'macro': Calculate metrics for each class and find their unweighted mean
    - 'weighted': Calculate metrics for each class and find their weighted mean
    """
    __module__ = "brainstate.nn"

    def __init__(self, num_classes: tp.Optional[int] = None, average: str = 'macro'):
        self.num_classes = num_classes
        self.average = average
        if num_classes is None:
            self.true_positives = MetricState(jnp.array(0, dtype=jnp.int32))
            self.false_positives = MetricState(jnp.array(0, dtype=jnp.int32))
        else:
            self.true_positives = MetricState(jnp.zeros(num_classes, dtype=jnp.int32))
            self.false_positives = MetricState(jnp.zeros(num_classes, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state to zero."""
        if self.num_classes is None:
            self.true_positives.value = jnp.array(0, dtype=jnp.int32)
            self.false_positives.value = jnp.array(0, dtype=jnp.int32)
        else:
            self.true_positives.value = jnp.zeros(self.num_classes, dtype=jnp.int32)
            self.false_positives.value = jnp.zeros(self.num_classes, dtype=jnp.int32)

    def update(self, *, predictions: jax.Array, labels: jax.Array, **_) -> None:
        """
        Update the precision metric.

        Parameters
        ----------
        predictions : jax.Array
            Predicted class labels (integers).
        labels : jax.Array
            Ground truth class labels (integers).
        **_
            Additional keyword arguments are ignored.
        """
        if self.num_classes is None:
            # Binary classification
            self.true_positives.value += jnp.sum((predictions == 1) & (labels == 1))
            self.false_positives.value += jnp.sum((predictions == 1) & (labels == 0))
        else:
            # Multi-class classification
            for c in range(self.num_classes):
                self.true_positives.value = self.true_positives.value.at[c].add(
                    jnp.sum((predictions == c) & (labels == c))
                )
                self.false_positives.value = self.false_positives.value.at[c].add(
                    jnp.sum((predictions == c) & (labels != c))
                )

    def compute(self) -> jax.Array:
        """
        Compute and return the precision.

        Returns
        -------
        jax.Array
            The precision value(s). For binary classification, returns a scalar.
            For multi-class, returns per-class or averaged precision based on
            the ``average`` parameter.
        """
        denominator = self.true_positives.value + self.false_positives.value
        precision = jnp.where(
            denominator > 0,
            self.true_positives.value / denominator,
            jnp.zeros_like(denominator, dtype=jnp.float32)
        )

        if self.num_classes is not None and self.average == 'macro':
            return jnp.mean(precision)
        elif self.num_classes is not None and self.average == 'micro':
            total_tp = jnp.sum(self.true_positives.value)
            total_fp = jnp.sum(self.false_positives.value)
            return jnp.where(
                total_tp + total_fp > 0,
                total_tp / (total_tp + total_fp),
                jnp.float32(0.0)
            )
        return precision


class RecallMetric(Metric):
    """
    Recall (sensitivity) metric for binary and multi-class classification.

    Recall is the ratio of true positives to all actual positives:
    recall = TP / (TP + FN)

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. If None, assumes binary classification. Default is None.
    average : str, optional
        Type of averaging for multi-class: 'micro', 'macro', or 'weighted'.
        Default is 'macro'. Ignored for binary classification.

    Attributes
    ----------
    num_classes : int or None
        Number of classes.
    average : str
        Averaging method for multi-class.
    true_positives : MetricState
        Count of true positive predictions.
    false_negatives : MetricState
        Count of false negative predictions.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> predictions = jnp.array([1, 0, 1, 1, 0])
        >>> labels = jnp.array([1, 0, 0, 1, 0])
        >>> metric = brainstate.nn.RecallMetric()
        >>> metric.update(predictions=predictions, labels=labels)
        >>> metric.compute()
        Array(1., dtype=float32)

    Notes
    -----
    Recall measures the fraction of actual positive cases that were correctly identified.
    Also known as sensitivity or true positive rate (TPR).
    """
    __module__ = "brainstate.nn"

    def __init__(self, num_classes: tp.Optional[int] = None, average: str = 'macro'):
        self.num_classes = num_classes
        self.average = average
        if num_classes is None:
            self.true_positives = MetricState(jnp.array(0, dtype=jnp.int32))
            self.false_negatives = MetricState(jnp.array(0, dtype=jnp.int32))
        else:
            self.true_positives = MetricState(jnp.zeros(num_classes, dtype=jnp.int32))
            self.false_negatives = MetricState(jnp.zeros(num_classes, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state to zero."""
        if self.num_classes is None:
            self.true_positives.value = jnp.array(0, dtype=jnp.int32)
            self.false_negatives.value = jnp.array(0, dtype=jnp.int32)
        else:
            self.true_positives.value = jnp.zeros(self.num_classes, dtype=jnp.int32)
            self.false_negatives.value = jnp.zeros(self.num_classes, dtype=jnp.int32)

    def update(self, *, predictions: jax.Array, labels: jax.Array, **_) -> None:
        """
        Update the recall metric.

        Parameters
        ----------
        predictions : jax.Array
            Predicted class labels (integers).
        labels : jax.Array
            Ground truth class labels (integers).
        **_
            Additional keyword arguments are ignored.
        """
        if self.num_classes is None:
            # Binary classification
            self.true_positives.value += jnp.sum((predictions == 1) & (labels == 1))
            self.false_negatives.value += jnp.sum((predictions == 0) & (labels == 1))
        else:
            # Multi-class classification
            for c in range(self.num_classes):
                self.true_positives.value = self.true_positives.value.at[c].add(
                    jnp.sum((predictions == c) & (labels == c))
                )
                self.false_negatives.value = self.false_negatives.value.at[c].add(
                    jnp.sum((predictions != c) & (labels == c))
                )

    def compute(self) -> jax.Array:
        """
        Compute and return the recall.

        Returns
        -------
        jax.Array
            The recall value(s). For binary classification, returns a scalar.
            For multi-class, returns per-class or averaged recall based on
            the ``average`` parameter.
        """
        denominator = self.true_positives.value + self.false_negatives.value
        recall = jnp.where(
            denominator > 0,
            self.true_positives.value / denominator,
            jnp.zeros_like(denominator, dtype=jnp.float32)
        )

        if self.num_classes is not None and self.average == 'macro':
            return jnp.mean(recall)
        elif self.num_classes is not None and self.average == 'micro':
            total_tp = jnp.sum(self.true_positives.value)
            total_fn = jnp.sum(self.false_negatives.value)
            return jnp.where(
                total_tp + total_fn > 0,
                total_tp / (total_tp + total_fn),
                jnp.float32(0.0)
            )
        return recall


class F1ScoreMetric(Metric):
    """
    F1 score metric for binary and multi-class classification.

    F1 score is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. If None, assumes binary classification. Default is None.
    average : str, optional
        Type of averaging for multi-class: 'micro', 'macro', or 'weighted'.
        Default is 'macro'. Ignored for binary classification.

    Attributes
    ----------
    precision_metric : PrecisionMetric
        Internal precision metric.
    recall_metric : RecallMetric
        Internal recall metric.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> predictions = jnp.array([1, 0, 1, 1, 0])
        >>> labels = jnp.array([1, 0, 0, 1, 0])
        >>> metric = brainstate.nn.F1ScoreMetric()
        >>> metric.update(predictions=predictions, labels=labels)
        >>> metric.compute()
        Array(0.8, dtype=float32)

    Notes
    -----
    The F1 score balances precision and recall, providing a single metric that
    considers both false positives and false negatives.
    """
    __module__ = "brainstate.nn"

    def __init__(self, num_classes: tp.Optional[int] = None, average: str = 'macro'):
        self.precision_metric = PrecisionMetric(num_classes, average)
        self.recall_metric = RecallMetric(num_classes, average)

    def reset(self) -> None:
        """Reset the metric state to zero."""
        self.precision_metric.reset()
        self.recall_metric.reset()

    def update(self, *, predictions: jax.Array, labels: jax.Array, **_) -> None:
        """
        Update the F1 score metric.

        Parameters
        ----------
        predictions : jax.Array
            Predicted class labels (integers).
        labels : jax.Array
            Ground truth class labels (integers).
        **_
            Additional keyword arguments are ignored.
        """
        self.precision_metric.update(predictions=predictions, labels=labels)
        self.recall_metric.update(predictions=predictions, labels=labels)

    def compute(self) -> jax.Array:
        """
        Compute and return the F1 score.

        Returns
        -------
        jax.Array
            The F1 score value(s). Returns 0 when both precision and recall are 0.
        """
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        denominator = precision + recall
        return jnp.where(
            denominator > 0,
            2 * precision * recall / denominator,
            jnp.float32(0.0)
        )


class ConfusionMatrix(Metric):
    """
    Confusion matrix metric for multi-class classification.

    A confusion matrix shows the counts of predicted vs. actual class labels,
    where rows represent true labels and columns represent predicted labels.

    Parameters
    ----------
    num_classes : int
        Number of classes in the classification task.

    Attributes
    ----------
    num_classes : int
        Number of classes.
    matrix : MetricState
        The confusion matrix of shape (num_classes, num_classes).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> predictions = jnp.array([0, 1, 2, 1, 0])
        >>> labels = jnp.array([0, 1, 1, 1, 2])
        >>> metric = brainstate.nn.ConfusionMatrix(num_classes=3)
        >>> metric.update(predictions=predictions, labels=labels)
        >>> metric.compute()
        Array([[1, 0, 1],
               [0, 2, 0],
               [1, 0, 0]], dtype=int32)

    Notes
    -----
    The confusion matrix is useful for understanding which classes are being
    confused with each other and for computing class-specific metrics.
    """
    __module__ = "brainstate.nn"

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = MetricState(jnp.zeros((num_classes, num_classes), dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the confusion matrix to zeros."""
        self.matrix.value = jnp.zeros((self.num_classes, self.num_classes), dtype=jnp.int32)

    def update(self, *, predictions: jax.Array, labels: jax.Array, **_) -> None:
        """
        Update the confusion matrix.

        Parameters
        ----------
        predictions : jax.Array
            Predicted class labels (integers) with shape (batch_size,).
        labels : jax.Array
            Ground truth class labels (integers) with shape (batch_size,).
        **_
            Additional keyword arguments are ignored.

        Raises
        ------
        ValueError
            If predictions or labels contain values outside [0, num_classes).
        """
        predictions = jnp.asarray(predictions, dtype=jnp.int32).flatten()
        labels = jnp.asarray(labels, dtype=jnp.int32).flatten()

        if jnp.any((predictions < 0) | (predictions >= self.num_classes)):
            raise ValueError(f"Predictions contain values outside [0, {self.num_classes})")
        if jnp.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError(f"Labels contain values outside [0, {self.num_classes})")

        for true_label in range(self.num_classes):
            for pred_label in range(self.num_classes):
                count = jnp.sum((labels == true_label) & (predictions == pred_label))
                self.matrix.value = self.matrix.value.at[true_label, pred_label].add(count)

    def compute(self) -> jax.Array:
        """
        Compute and return the confusion matrix.

        Returns
        -------
        jax.Array
            The confusion matrix of shape (num_classes, num_classes).
            Element [i, j] represents the count of samples with true label i
            that were predicted as label j.
        """
        return self.matrix.value


class MultiMetric(Metric):
    """
    Container for multiple metrics updated simultaneously.

    This class allows you to group multiple metrics together and update them
    all with a single call. It's useful for tracking multiple evaluation metrics
    (e.g., accuracy, loss, F1 score) during training or evaluation.

    Parameters
    ----------
    **metrics
        Keyword arguments where keys are metric names (strings) and values
        are Metric instances.

    Attributes
    ----------
    _metric_names : list of str
        List of metric names in the order they were added.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>> metrics = brainstate.nn.MultiMetric(
        ...     accuracy=brainstate.nn.AccuracyMetric(),
        ...     loss=brainstate.nn.AverageMetric(),
        ... )
        >>> logits = jax.random.normal(jax.random.key(0), (5, 2))
        >>> labels = jnp.array([1, 1, 0, 1, 0])
        >>> batch_loss = jnp.array([1, 2, 3, 4])
        >>> metrics.compute()
        {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}
        >>> metrics.update(logits=logits, labels=labels, values=batch_loss)
        >>> metrics.compute()
        {'accuracy': Array(0.6, dtype=float32), 'loss': Array(2.5, dtype=float32)}
        >>> metrics.reset()
        >>> metrics.compute()
        {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}

    Notes
    -----
    All keyword arguments passed to ``update`` are forwarded to all underlying metrics.
    Each metric will extract the arguments it needs based on its implementation.

    Reserved method names ('reset', 'update', 'compute') cannot be used as metric names.
    """
    __module__ = "brainstate.nn"

    def __init__(self, **metrics):
        # Validate that no reserved names are used
        reserved_names = {'reset', 'update', 'compute'}
        for metric_name in metrics.keys():
            if metric_name in reserved_names:
                raise ValueError(
                    f"Metric name '{metric_name}' is reserved for class methods. "
                    f"Please use a different name. Reserved names: {reserved_names}"
                )

        self._metric_names = []
        for metric_name, metric in metrics.items():
            if not isinstance(metric, Metric):
                raise TypeError(
                    f"All metrics must be instances of Metric, got {type(metric)} "
                    f"for metric '{metric_name}'"
                )
            self._metric_names.append(metric_name)
            vars(self)[metric_name] = metric

    def reset(self) -> None:
        """
        Reset all underlying metrics.

        This calls the ``reset`` method on each metric in the collection.
        """
        for metric_name in self._metric_names:
            getattr(self, metric_name).reset()

    def update(self, **updates) -> None:
        """
        Update all underlying metrics.

        All keyword arguments are passed to the ``update`` method of each metric.
        Individual metrics will extract the arguments they need.

        Parameters
        ----------
        **updates
            Keyword arguments to be passed to all underlying metrics.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainstate
            >>> metrics = brainstate.nn.MultiMetric(
            ...     accuracy=brainstate.nn.AccuracyMetric(),
            ...     loss=brainstate.nn.AverageMetric('loss_value'),
            ... )
            >>> logits = jnp.array([[0.2, 0.8], [0.9, 0.1]])
            >>> labels = jnp.array([1, 0])
            >>> loss = jnp.array([0.5, 0.3])
            >>> metrics.update(logits=logits, labels=labels, loss_value=loss)
        """
        for metric_name in self._metric_names:
            getattr(self, metric_name).update(**updates)

    def compute(self) -> dict[str, tp.Any]:
        """
        Compute and return all metric values.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping metric names to their computed values.
            The value type depends on the specific metric implementation.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>> metrics = brainstate.nn.MultiMetric(
            ...     loss=brainstate.nn.AverageMetric(),
            ...     stats=brainstate.nn.WelfordMetric(),
            ... )
            >>> # After updates...
            >>> results = metrics.compute()
            >>> results['loss']  # Returns a scalar
            >>> results['stats']  # Returns a Statistics object
        """
        return {
            metric_name: getattr(self, metric_name).compute()
            for metric_name in self._metric_names
        }
