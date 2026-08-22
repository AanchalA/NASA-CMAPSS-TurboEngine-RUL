"""Tests for pandas model-input selection."""

import unittest

import pandas as pd


class TabularInputsTests(unittest.TestCase):
    def test_selects_ordered_features_and_target_without_leakage(self) -> None:
        dataframe = pd.DataFrame({
            "unit_id": [1, 1], "cycle": [1, 2], "RUL": [125, 124],
            "sensor_2": [0.2, 0.5], "sensor_3": [0.3, 0.6], "sensor_4": [0.4, 0.7],
        })
        retained = ["sensor_4", "sensor_2", "sensor_3"]

        features = dataframe.loc[:, retained]
        target = dataframe.loc[:, "RUL"]

        self.assertEqual(list(features.columns), retained)
        self.assertEqual(target.tolist(), [125, 124])
        self.assertEqual(features.iloc[0].tolist(), [0.4, 0.2, 0.3])
        self.assertTrue({"unit_id", "cycle", "RUL"}.isdisjoint(features.columns))


if __name__ == "__main__":
    unittest.main()
