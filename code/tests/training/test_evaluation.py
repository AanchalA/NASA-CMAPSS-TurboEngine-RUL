"""Tests for model evaluation metrics."""

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.training.evaluation import evaluate_predictions, life_ratio_to_rul


class EvaluationTests(unittest.TestCase):
    def test_converts_life_ratio_to_rul_and_bounds_predictions(self) -> None:
        self.assertAlmostEqual(life_ratio_to_rul(80, 0.8), 20.0)
        self.assertEqual(life_ratio_to_rul(80, 1.2), 0.0)
        self.assertTrue(math.isfinite(life_ratio_to_rul(80, 0.0)))

    def test_evaluates_regression_predictions(self) -> None:
        metrics = evaluate_predictions(
            y_true=[100, 100, 100],
            y_pred=[87, 100, 110],
        )

        self.assertAlmostEqual(metrics["rmse"], math.sqrt(269 / 3))
        self.assertAlmostEqual(metrics["mae"], 23 / 3)
        self.assertAlmostEqual(metrics["nasa_score"], 2 * math.expm1(1))


if __name__ == "__main__":
    unittest.main()
