import math
import unittest

from app.evaluation.metrics import calculate_binary_metrics, calculate_cost, threshold_predictions


class MetricAndCostTests(unittest.TestCase):
    def test_threshold_metrics_and_confusion_counts(self) -> None:
        metrics = calculate_binary_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], threshold=0.5)
        self.assertEqual((metrics.tp, metrics.fp, metrics.tn, metrics.fn), (1, 1, 1, 1))
        self.assertEqual(metrics.support, 2)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)
        self.assertAlmostEqual(metrics.false_negative_rate, 0.5)
        self.assertGreater(metrics.average_precision, 0.0)
        self.assertGreater(metrics.pr_auc, 0.0)

    def test_historical_threshold_is_strictly_greater_than(self) -> None:
        self.assertEqual(threshold_predictions([0.5, 0.50001, 0.49], 0.5).tolist(), [0, 1, 0])

    def test_cost_uses_the_specified_linear_formula(self) -> None:
        self.assertEqual(calculate_cost(fn=3, fp=4, fn_cost=10, fp_cost=2), 38.0)

    def test_cost_rejects_negative_inputs(self) -> None:
        with self.assertRaises(ValueError):
            calculate_cost(fn=-1, fp=0, fn_cost=1, fp_cost=1)

    def test_no_positive_class_has_explicitly_undefined_rank_metrics(self) -> None:
        metrics = calculate_binary_metrics([0, 0], [0.3, 0.2], threshold=0.5)
        self.assertTrue(math.isnan(metrics.average_precision))
        self.assertTrue(math.isnan(metrics.pr_auc))


if __name__ == "__main__":
    unittest.main()
