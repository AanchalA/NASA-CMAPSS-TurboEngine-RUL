"""Tests for pandas model-input selection."""

import unittest
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing import select_tabular_model_inputs


class TabularInputsTests(unittest.TestCase):
    def test_selects_ordered_features_and_target_without_leakage(self) -> None:
        dataframe = pd.DataFrame({
            "unit_id": [1, 1], "cycle": [1, 2], "RUL": [125, 124],
            "sensor_2": [0.2, 0.5], "sensor_3": [0.3, 0.6], "sensor_4": [0.4, 0.7],
        })
        retained = ["sensor_4", "sensor_2", "sensor_3"]
        features, target = select_tabular_model_inputs(dataframe, retained)

        self.assertEqual(list(features.columns), retained)
        self.assertEqual(target.tolist(), [125, 124])
        self.assertEqual(features.iloc[0].tolist(), [0.4, 0.2, 0.3])
        self.assertTrue({"unit_id", "cycle", "RUL"}.isdisjoint(features.columns))

    def test_infers_scaled_and_temporal_features_from_processed_schema(self) -> None:
        dataframe = pd.DataFrame(
            {
                "unit_id": [1],
                "cycle": [2],
                "RUL": [10],
                "operating_regime": [1],
                "sensor_2": [0.5],
                "sensor_2_lag_1": [0.2],
                "sensor_2_rolling_mean_5": [0.35],
                "sensor_2_rolling_slope_5": [0.3],
                "sensor_2_ewma": [0.4],
            }
        )

        features, target = select_tabular_model_inputs(dataframe)

        self.assertEqual(
            list(features.columns),
            [
                "sensor_2",
                "sensor_2_lag_1",
                "sensor_2_rolling_mean_5",
                "sensor_2_rolling_slope_5",
                "sensor_2_ewma",
            ],
        )
        self.assertEqual(target.tolist(), [10])


if __name__ == "__main__":
    unittest.main()
