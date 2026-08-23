import unittest

import pandas as pd

from src.training.evaluation import evaluate_prediction_diagnostics, prepare_pseudo_test_validation


class PseudoTestValidationTests(unittest.TestCase):
    def test_selects_one_reproducible_prefailure_endpoint_per_engine(self):
        dataframe = pd.DataFrame(
            [
                {"unit_id": unit, "cycle": cycle, "RUL": 10 - cycle, "sensor": cycle * unit}
                for unit in (1, 2)
                for cycle in range(1, 11)
            ]
        )

        X_first, y_first, metadata_first = prepare_pseudo_test_validation(
            dataframe, ["sensor"], seed=42
        )
        X_second, y_second, metadata_second = prepare_pseudo_test_validation(
            dataframe, ["sensor"], seed=42
        )

        pd.testing.assert_frame_equal(X_first, X_second)
        pd.testing.assert_series_equal(y_first, y_second)
        pd.testing.assert_frame_equal(metadata_first, metadata_second)
        self.assertEqual(metadata_first["unit_id"].tolist(), [1, 2])
        self.assertTrue(metadata_first["cycle"].between(5, 9).all())
        self.assertTrue((metadata_first["RUL"] == 10 - metadata_first["cycle"]).all())

    def test_builds_diagnostics_sorted_by_nasa_contribution(self):
        metadata = pd.DataFrame(
            {"unit_id": [1, 2], "cycle": [50, 60], "RUL": [10, 20]}
        )
        diagnostics, metrics = evaluate_prediction_diagnostics(metadata, [11, 50])
        contributions = diagnostics["nasa_contribution"].tolist()

        self.assertEqual(diagnostics["unit_id"].tolist(), [2, 1])
        self.assertEqual(diagnostics["error"].tolist(), [30, 1])
        self.assertGreater(contributions[0], contributions[1])
        self.assertEqual(metrics["worst_positive_error"], 30.0)


if __name__ == "__main__":
    unittest.main()
