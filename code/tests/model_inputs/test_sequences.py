import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing import build_endpoint_sequences, build_sliding_window_sequences


class SequenceInputsTests(unittest.TestCase):
    def test_sequences_are_ordered_and_never_cross_engine_boundaries(self) -> None:
        dataframe = pd.DataFrame(
            {
                "unit_id": [2, 1, 2, 1, 1, 2],
                "cycle": [2, 3, 1, 1, 2, 3],
                "RUL": [8, 7, 9, 9, 8, 7],
                "sensor_2": [200.0, 30.0, 100.0, 10.0, 20.0, 300.0],
                "sensor_3": [20.0, 3.0, 10.0, 1.0, 2.0, 30.0],
            }
        )

        sequences, targets = build_sliding_window_sequences(
            dataframe,
            ("sensor_3", "sensor_2"),
            sequence_length=2,
        )

        np.testing.assert_array_equal(
            sequences,
            np.array(
                [
                    [[1.0, 10.0], [2.0, 20.0]],
                    [[2.0, 20.0], [3.0, 30.0]],
                    [[10.0, 100.0], [20.0, 200.0]],
                    [[20.0, 200.0], [30.0, 300.0]],
                ]
            ),
        )
        np.testing.assert_array_equal(targets, np.array([8, 7, 8, 7]))
        self.assertEqual(sequences.shape, (4, 2, 2))

    def test_default_length_omits_short_engines_without_padding(self) -> None:
        dataframe = pd.DataFrame(
            {
                "unit_id": [1] * 31 + [2] * 29,
                "cycle": list(range(1, 32)) + list(range(1, 30)),
                "RUL": list(range(30, -1, -1)) + list(range(28, -1, -1)),
                "sensor_2": list(range(1, 32)) + list(range(101, 130)),
            }
        )

        sequences, targets = build_sliding_window_sequences(dataframe, ("sensor_2",))

        self.assertEqual(sequences.shape, (2, 30, 1))
        np.testing.assert_array_equal(targets, np.array([1, 0]))
        np.testing.assert_array_equal(sequences[0, :, 0], np.arange(1, 31))
        np.testing.assert_array_equal(sequences[1, :, 0], np.arange(2, 32))

    def test_endpoint_sequences_end_at_the_supplied_cutoffs(self) -> None:
        dataframe = pd.DataFrame(
            {
                "unit_id": [1] * 4 + [2] * 4,
                "cycle": [1, 2, 3, 4] * 2,
                "RUL": [3, 2, 1, 0] * 2,
                "sensor_2": [10, 20, 30, 40, 100, 200, 300, 400],
            }
        )
        endpoints = pd.DataFrame({"unit_id": [1, 2], "cycle": [3, 4]})

        sequences = build_endpoint_sequences(
            dataframe, endpoints, ["sensor_2"], sequence_length=3
        )

        np.testing.assert_array_equal(sequences[0][:, 0], [10, 20, 30])
        np.testing.assert_array_equal(sequences[1][:, 0], [200, 300, 400])

    def test_short_endpoint_sequences_contain_only_real_observations(self) -> None:
        dataframe = pd.DataFrame(
            {
                "unit_id": [1, 1],
                "cycle": [1, 2],
                "RUL": [2, 1],
                "sensor_2": [10.0, 20.0],
            }
        )
        endpoints = pd.DataFrame({"unit_id": [1], "cycle": [2]})

        sequences = build_endpoint_sequences(
            dataframe, endpoints, ["sensor_2"], sequence_length=4
        )

        np.testing.assert_array_equal(sequences[0][:, 0], [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
