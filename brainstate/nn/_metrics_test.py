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

"""Comprehensive tests for metrics module."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import brainstate as bst


class MetricStateTest(absltest.TestCase):
    """Test cases for MetricState class."""

    def test_metric_state_creation(self):
        """Test creating a MetricState with different values."""
        state = bst.nn.MetricState(jnp.array(0.0))
        self.assertEqual(state.value, 0.0)

    def test_metric_state_update(self):
        """Test updating MetricState value."""
        state = bst.nn.MetricState(jnp.array(0.0))
        state.value = jnp.array(5.5)
        self.assertAlmostEqual(float(state.value), 5.5)

    def test_metric_state_module_attribute(self):
        """Test that MetricState has correct __module__ attribute."""
        state = bst.nn.MetricState(jnp.array(0.0))
        self.assertEqual(bst.nn.MetricState.__module__, "brainstate.nn")


class AverageMetricTest(parameterized.TestCase):
    """Test cases for AverageMetric class."""

    def test_average_metric_initial_state(self):
        """Test that initial average is NaN."""
        metric = bst.nn.AverageMetric()
        result = metric.compute()
        self.assertTrue(jnp.isnan(result))

    def test_average_metric_single_update(self):
        """Test average after single update."""
        metric = bst.nn.AverageMetric()
        metric.update(values=jnp.array([1, 2, 3, 4]))
        result = metric.compute()
        self.assertAlmostEqual(float(result), 2.5)

    def test_average_metric_multiple_updates(self):
        """Test average after multiple updates."""
        metric = bst.nn.AverageMetric()
        metric.update(values=jnp.array([1, 2, 3, 4]))
        metric.update(values=jnp.array([3, 2, 1, 0]))
        result = metric.compute()
        self.assertAlmostEqual(float(result), 2.0)

    def test_average_metric_scalar_values(self):
        """Test average with scalar values."""
        metric = bst.nn.AverageMetric()
        metric.update(values=5.0)
        metric.update(values=3.0)
        result = metric.compute()
        self.assertAlmostEqual(float(result), 4.0)

    def test_average_metric_reset(self):
        """Test reset functionality."""
        metric = bst.nn.AverageMetric()
        metric.update(values=jnp.array([1, 2, 3, 4]))
        metric.reset()
        result = metric.compute()
        self.assertTrue(jnp.isnan(result))

    def test_average_metric_custom_argname(self):
        """Test using custom argument name."""
        metric = bst.nn.AverageMetric('loss')
        metric.update(loss=jnp.array([1.0, 2.0, 3.0]))
        result = metric.compute()
        self.assertAlmostEqual(float(result), 2.0)

    def test_average_metric_missing_argname(self):
        """Test error when expected argument is missing."""
        metric = bst.nn.AverageMetric('loss')
        with self.assertRaises(TypeError):
            metric.update(values=jnp.array([1.0, 2.0]))

    def test_average_metric_module_attribute(self):
        """Test that AverageMetric has correct __module__ attribute."""
        self.assertEqual(bst.nn.AverageMetric.__module__, "brainstate.nn")


class WelfordMetricTest(parameterized.TestCase):
    """Test cases for WelfordMetric class."""

    def test_welford_metric_initial_state(self):
        """Test initial statistics."""
        metric = bst.nn.WelfordMetric()
        stats = metric.compute()
        self.assertEqual(float(stats.mean), 0.0)
        self.assertTrue(jnp.isnan(stats.standard_error_of_mean))
        self.assertTrue(jnp.isnan(stats.standard_deviation))

    def test_welford_metric_single_update(self):
        """Test statistics after single update."""
        metric = bst.nn.WelfordMetric()
        metric.update(values=jnp.array([1, 2, 3, 4]))
        stats = metric.compute()
        self.assertAlmostEqual(float(stats.mean), 2.5, places=5)
        # Population std of [1,2,3,4] is sqrt(1.25) ≈ 1.118
        self.assertAlmostEqual(float(stats.standard_deviation), 1.118, places=3)

    def test_welford_metric_multiple_updates(self):
        """Test statistics after multiple updates."""
        metric = bst.nn.WelfordMetric()
        metric.update(values=jnp.array([1, 2, 3, 4]))
        metric.update(values=jnp.array([3, 2, 1, 0]))
        stats = metric.compute()
        # Mean of all 8 values: [1,2,3,4,3,2,1,0] is 2.0
        self.assertAlmostEqual(float(stats.mean), 2.0, places=5)

    def test_welford_metric_reset(self):
        """Test reset functionality."""
        metric = bst.nn.WelfordMetric()
        metric.update(values=jnp.array([1, 2, 3, 4]))
        metric.reset()
        stats = metric.compute()
        self.assertEqual(float(stats.mean), 0.0)
        self.assertTrue(jnp.isnan(stats.standard_deviation))

    def test_welford_metric_custom_argname(self):
        """Test using custom argument name."""
        metric = bst.nn.WelfordMetric('data')
        metric.update(data=jnp.array([1.0, 2.0, 3.0]))
        stats = metric.compute()
        self.assertAlmostEqual(float(stats.mean), 2.0, places=5)

    def test_welford_metric_module_attribute(self):
        """Test that WelfordMetric has correct __module__ attribute."""
        self.assertEqual(bst.nn.WelfordMetric.__module__, "brainstate.nn")


class AccuracyMetricTest(parameterized.TestCase):
    """Test cases for AccuracyMetric class."""

    def test_accuracy_metric_initial_state(self):
        """Test that initial accuracy is NaN."""
        metric = bst.nn.AccuracyMetric()
        result = metric.compute()
        self.assertTrue(jnp.isnan(result))

    def test_accuracy_metric_perfect_accuracy(self):
        """Test with perfect predictions."""
        metric = bst.nn.AccuracyMetric()
        logits = jnp.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = jnp.array([1, 0, 1])
        metric.update(logits=logits, labels=labels)
        result = metric.compute()
        self.assertAlmostEqual(float(result), 1.0)

    def test_accuracy_metric_partial_accuracy(self):
        """Test with partial accuracy."""
        metric = bst.nn.AccuracyMetric()
        logits = jnp.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
        labels = jnp.array([1, 0, 1])  # First one wrong
        metric.update(logits=logits, labels=labels)
        result = metric.compute()
        self.assertAlmostEqual(float(result), 2.0 / 3.0, places=5)

    def test_accuracy_metric_multiple_updates(self):
        """Test accuracy after multiple updates."""
        metric = bst.nn.AccuracyMetric()
        logits1 = jax.random.normal(jax.random.key(0), (5, 2))
        labels1 = jnp.array([1, 1, 0, 1, 0])
        logits2 = jax.random.normal(jax.random.key(1), (5, 2))
        labels2 = jnp.array([0, 1, 1, 1, 1])

        metric.update(logits=logits1, labels=labels1)
        metric.update(logits=logits2, labels=labels2)
        result = metric.compute()
        # Result should be between 0 and 1
        self.assertGreaterEqual(float(result), 0.0)
        self.assertLessEqual(float(result), 1.0)

    def test_accuracy_metric_reset(self):
        """Test reset functionality."""
        metric = bst.nn.AccuracyMetric()
        logits = jnp.array([[0.1, 0.9], [0.8, 0.2]])
        labels = jnp.array([1, 0])
        metric.update(logits=logits, labels=labels)
        metric.reset()
        result = metric.compute()
        self.assertTrue(jnp.isnan(result))

    def test_accuracy_metric_shape_validation(self):
        """Test shape validation."""
        metric = bst.nn.AccuracyMetric()
        logits = jnp.array([[0.1, 0.9]])
        labels = jnp.array([[1, 0]])  # Wrong shape
        with self.assertRaises(ValueError):
            metric.update(logits=logits, labels=labels)

    def test_accuracy_metric_module_attribute(self):
        """Test that AccuracyMetric has correct __module__ attribute."""
        self.assertEqual(bst.nn.AccuracyMetric.__module__, "brainstate.nn")


class PrecisionMetricTest(parameterized.TestCase):
    """Test cases for PrecisionMetric class."""

    def test_precision_metric_binary_perfect(self):
        """Test binary precision with perfect predictions."""
        metric = bst.nn.PrecisionMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 1, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        self.assertAlmostEqual(float(result), 1.0)

    def test_precision_metric_binary_partial(self):
        """Test binary precision with partial accuracy."""
        metric = bst.nn.PrecisionMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 0, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        # TP=2, FP=1, Precision=2/3
        self.assertAlmostEqual(float(result), 2.0 / 3.0, places=5)

    def test_precision_metric_multiclass_macro(self):
        """Test multi-class precision with macro averaging."""
        metric = bst.nn.PrecisionMetric(num_classes=3, average='macro')
        predictions = jnp.array([0, 1, 2, 1, 0])
        labels = jnp.array([0, 1, 1, 1, 2])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        # Should compute precision for each class and average
        self.assertGreaterEqual(float(result), 0.0)
        self.assertLessEqual(float(result), 1.0)

    def test_precision_metric_multiclass_micro(self):
        """Test multi-class precision with micro averaging."""
        metric = bst.nn.PrecisionMetric(num_classes=3, average='micro')
        predictions = jnp.array([0, 1, 2, 1, 0])
        labels = jnp.array([0, 1, 1, 1, 2])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        self.assertGreaterEqual(float(result), 0.0)
        self.assertLessEqual(float(result), 1.0)

    def test_precision_metric_reset(self):
        """Test reset functionality."""
        metric = bst.nn.PrecisionMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 0, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        metric.reset()
        result = metric.compute()
        # After reset, with no data, precision should be 0
        self.assertEqual(float(result), 0.0)

    def test_precision_metric_module_attribute(self):
        """Test that PrecisionMetric has correct __module__ attribute."""
        self.assertEqual(bst.nn.PrecisionMetric.__module__, "brainstate.nn")


class RecallMetricTest(parameterized.TestCase):
    """Test cases for RecallMetric class."""

    def test_recall_metric_binary_perfect(self):
        """Test binary recall with perfect predictions."""
        metric = bst.nn.RecallMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 1, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        self.assertAlmostEqual(float(result), 1.0)

    def test_recall_metric_binary_partial(self):
        """Test binary recall with partial accuracy."""
        metric = bst.nn.RecallMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 0, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        # TP=2, FN=0, Recall=2/2=1.0
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_recall_metric_multiclass_macro(self):
        """Test multi-class recall with macro averaging."""
        metric = bst.nn.RecallMetric(num_classes=3, average='macro')
        predictions = jnp.array([0, 1, 2, 1, 0])
        labels = jnp.array([0, 1, 1, 1, 2])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        self.assertGreaterEqual(float(result), 0.0)
        self.assertLessEqual(float(result), 1.0)

    def test_recall_metric_multiclass_micro(self):
        """Test multi-class recall with micro averaging."""
        metric = bst.nn.RecallMetric(num_classes=3, average='micro')
        predictions = jnp.array([0, 1, 2, 1, 0])
        labels = jnp.array([0, 1, 1, 1, 2])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        self.assertGreaterEqual(float(result), 0.0)
        self.assertLessEqual(float(result), 1.0)

    def test_recall_metric_reset(self):
        """Test reset functionality."""
        metric = bst.nn.RecallMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 0, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        metric.reset()
        result = metric.compute()
        self.assertEqual(float(result), 0.0)

    def test_recall_metric_module_attribute(self):
        """Test that RecallMetric has correct __module__ attribute."""
        self.assertEqual(bst.nn.RecallMetric.__module__, "brainstate.nn")


class F1ScoreMetricTest(parameterized.TestCase):
    """Test cases for F1ScoreMetric class."""

    def test_f1_score_binary_perfect(self):
        """Test binary F1 score with perfect predictions."""
        metric = bst.nn.F1ScoreMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 1, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        self.assertAlmostEqual(float(result), 1.0)

    def test_f1_score_binary_partial(self):
        """Test binary F1 score with partial accuracy."""
        metric = bst.nn.F1ScoreMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 0, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        # Precision=2/3, Recall=1.0, F1=2*(2/3*1)/(2/3+1)=0.8
        self.assertAlmostEqual(float(result), 0.8, places=5)

    def test_f1_score_multiclass(self):
        """Test multi-class F1 score."""
        metric = bst.nn.F1ScoreMetric(num_classes=3, average='macro')
        predictions = jnp.array([0, 1, 2, 1, 0])
        labels = jnp.array([0, 1, 1, 1, 2])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        self.assertGreaterEqual(float(result), 0.0)
        self.assertLessEqual(float(result), 1.0)

    def test_f1_score_reset(self):
        """Test reset functionality."""
        metric = bst.nn.F1ScoreMetric()
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 0, 1, 0])
        metric.update(predictions=predictions, labels=labels)
        metric.reset()
        result = metric.compute()
        self.assertEqual(float(result), 0.0)

    def test_f1_score_module_attribute(self):
        """Test that F1ScoreMetric has correct __module__ attribute."""
        self.assertEqual(bst.nn.F1ScoreMetric.__module__, "brainstate.nn")


class ConfusionMatrixTest(parameterized.TestCase):
    """Test cases for ConfusionMatrix class."""

    def test_confusion_matrix_basic(self):
        """Test basic confusion matrix computation."""
        metric = bst.nn.ConfusionMatrix(num_classes=3)
        predictions = jnp.array([0, 1, 2, 1, 0])
        labels = jnp.array([0, 1, 1, 1, 2])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()

        # Check shape
        self.assertEqual(result.shape, (3, 3))

        # Check specific values
        # True label 0, Predicted 0: 1
        # True label 0, Predicted 2: 1
        # True label 1, Predicted 1: 2
        self.assertEqual(int(result[0, 0]), 1)
        self.assertEqual(int(result[1, 1]), 2)

    def test_confusion_matrix_perfect(self):
        """Test confusion matrix with perfect predictions."""
        metric = bst.nn.ConfusionMatrix(num_classes=2)
        predictions = jnp.array([0, 0, 1, 1])
        labels = jnp.array([0, 0, 1, 1])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()

        # Should be diagonal matrix
        self.assertEqual(int(result[0, 0]), 2)
        self.assertEqual(int(result[0, 1]), 0)
        self.assertEqual(int(result[1, 0]), 0)
        self.assertEqual(int(result[1, 1]), 2)

    def test_confusion_matrix_reset(self):
        """Test reset functionality."""
        metric = bst.nn.ConfusionMatrix(num_classes=2)
        predictions = jnp.array([0, 1])
        labels = jnp.array([0, 1])
        metric.update(predictions=predictions, labels=labels)
        metric.reset()
        result = metric.compute()

        # Should be all zeros
        self.assertTrue(jnp.all(result == 0))

    def test_confusion_matrix_invalid_predictions(self):
        """Test error handling for invalid predictions."""
        metric = bst.nn.ConfusionMatrix(num_classes=2)
        predictions = jnp.array([0, 1, 2])  # 2 is out of range
        labels = jnp.array([0, 1, 1])
        with self.assertRaises(ValueError):
            metric.update(predictions=predictions, labels=labels)

    def test_confusion_matrix_invalid_labels(self):
        """Test error handling for invalid labels."""
        metric = bst.nn.ConfusionMatrix(num_classes=2)
        predictions = jnp.array([0, 1, 1])
        labels = jnp.array([0, 1, 3])  # 3 is out of range
        with self.assertRaises(ValueError):
            metric.update(predictions=predictions, labels=labels)

    def test_confusion_matrix_module_attribute(self):
        """Test that ConfusionMatrix has correct __module__ attribute."""
        self.assertEqual(bst.nn.ConfusionMatrix.__module__, "brainstate.nn")


class MultiMetricTest(parameterized.TestCase):
    """Test cases for MultiMetric class."""

    def test_multimetric_creation(self):
        """Test creating a MultiMetric with multiple metrics."""
        metrics = bst.nn.MultiMetric(
            accuracy=bst.nn.AccuracyMetric(),
            loss=bst.nn.AverageMetric(),
        )
        self.assertIsNotNone(metrics.accuracy)
        self.assertIsNotNone(metrics.loss)

    def test_multimetric_compute(self):
        """Test computing all metrics."""
        metrics = bst.nn.MultiMetric(
            accuracy=bst.nn.AccuracyMetric(),
            loss=bst.nn.AverageMetric(),
        )

        logits = jax.random.normal(jax.random.key(0), (5, 2))
        labels = jnp.array([1, 1, 0, 1, 0])
        batch_loss = jnp.array([1, 2, 3, 4])

        result = metrics.compute()
        self.assertIn('accuracy', result)
        self.assertIn('loss', result)
        self.assertTrue(jnp.isnan(result['accuracy']))
        self.assertTrue(jnp.isnan(result['loss']))

    def test_multimetric_update(self):
        """Test updating all metrics."""
        metrics = bst.nn.MultiMetric(
            accuracy=bst.nn.AccuracyMetric(),
            loss=bst.nn.AverageMetric(),
        )

        logits = jnp.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = jnp.array([1, 0, 1])
        batch_loss = jnp.array([1, 2, 3])

        metrics.update(logits=logits, labels=labels, values=batch_loss)
        result = metrics.compute()

        self.assertGreaterEqual(float(result['accuracy']), 0.0)
        self.assertLessEqual(float(result['accuracy']), 1.0)
        self.assertAlmostEqual(float(result['loss']), 2.0)

    def test_multimetric_reset(self):
        """Test resetting all metrics."""
        metrics = bst.nn.MultiMetric(
            accuracy=bst.nn.AccuracyMetric(),
            loss=bst.nn.AverageMetric(),
        )

        logits = jnp.array([[0.1, 0.9], [0.8, 0.2]])
        labels = jnp.array([1, 0])
        batch_loss = jnp.array([1, 2])

        metrics.update(logits=logits, labels=labels, values=batch_loss)
        metrics.reset()
        result = metrics.compute()

        self.assertTrue(jnp.isnan(result['accuracy']))
        self.assertTrue(jnp.isnan(result['loss']))

    def test_multimetric_reserved_name_validation(self):
        """Test that reserved method names cannot be used."""
        with self.assertRaises(ValueError):
            bst.nn.MultiMetric(
                reset=bst.nn.AverageMetric(),
            )

        with self.assertRaises(ValueError):
            bst.nn.MultiMetric(
                update=bst.nn.AverageMetric(),
            )

        with self.assertRaises(ValueError):
            bst.nn.MultiMetric(
                compute=bst.nn.AverageMetric(),
            )

    def test_multimetric_type_validation(self):
        """Test that all metrics must be Metric instances."""
        with self.assertRaises(TypeError):
            bst.nn.MultiMetric(
                accuracy=bst.nn.AccuracyMetric(),
                invalid="not a metric",
            )

    def test_multimetric_module_attribute(self):
        """Test that MultiMetric has correct __module__ attribute."""
        self.assertEqual(bst.nn.MultiMetric.__module__, "brainstate.nn")


class MetricBaseClassTest(absltest.TestCase):
    """Test cases for base Metric class."""

    def test_metric_not_implemented_errors(self):
        """Test that base Metric class raises NotImplementedError."""
        metric = bst.nn.Metric()

        with self.assertRaises(NotImplementedError):
            metric.reset()

        with self.assertRaises(NotImplementedError):
            metric.update()

        with self.assertRaises(NotImplementedError):
            metric.compute()

    def test_metric_module_attribute(self):
        """Test that Metric has correct __module__ attribute."""
        self.assertEqual(bst.nn.Metric.__module__, "brainstate.nn")


class EdgeCasesTest(parameterized.TestCase):
    """Test edge cases and boundary conditions."""

    def test_average_metric_large_numbers(self):
        """Test AverageMetric with very large numbers."""
        metric = bst.nn.AverageMetric()
        metric.update(values=jnp.array([1e10, 2e10, 3e10]))
        result = metric.compute()
        self.assertAlmostEqual(float(result), 2e10, places=-5)

    def test_average_metric_small_numbers(self):
        """Test AverageMetric with very small numbers."""
        metric = bst.nn.AverageMetric()
        metric.update(values=jnp.array([1e-10, 2e-10, 3e-10]))
        result = metric.compute()
        self.assertAlmostEqual(float(result), 2e-10, places=15)

    def test_confusion_matrix_single_class(self):
        """Test ConfusionMatrix with single class."""
        metric = bst.nn.ConfusionMatrix(num_classes=1)
        predictions = jnp.array([0, 0, 0])
        labels = jnp.array([0, 0, 0])
        metric.update(predictions=predictions, labels=labels)
        result = metric.compute()
        self.assertEqual(int(result[0, 0]), 3)

    def test_precision_recall_no_positives(self):
        """Test precision/recall when there are no positive predictions."""
        precision_metric = bst.nn.PrecisionMetric()
        recall_metric = bst.nn.RecallMetric()

        predictions = jnp.array([0, 0, 0, 0, 0])
        labels = jnp.array([0, 0, 0, 0, 0])

        precision_metric.update(predictions=predictions, labels=labels)
        recall_metric.update(predictions=predictions, labels=labels)

        # Should handle gracefully without division by zero
        precision = precision_metric.compute()
        recall = recall_metric.compute()

        self.assertEqual(float(precision), 0.0)
        self.assertEqual(float(recall), 0.0)


if __name__ == '__main__':
    absltest.main()
