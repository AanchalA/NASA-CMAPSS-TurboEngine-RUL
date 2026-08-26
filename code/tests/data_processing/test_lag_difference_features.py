"""Tests for engine-local lag and difference features."""

import os
import shutil
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.feature_engineering import add_lag_difference_features


class LagDifferenceFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-lag-difference-feature-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_lags_and_differences_are_ordered_and_engine_local(self) -> None:
        frame = self.spark.createDataFrame(
            [
                (2, 2, 10.0),
                (1, 3, 18.0),
                (1, 1, 10.0),
                (2, 1, 7.0),
                (1, 2, 13.0),
            ],
            "unit_id int, cycle int, sensor_2 double",
        )

        rows = (
            add_lag_difference_features(frame, ("sensor_2",), lags=(1, 2))
            .orderBy("unit_id", "cycle")
            .collect()
        )

        self.assertEqual(
            [
                (row["sensor_2_lag_1"], row["sensor_2_lag_2"], row["sensor_2_diff_1"], row["sensor_2_diff_2"])
                for row in rows
            ],
            [
                (None, None, None, None),
                (10.0, None, 3.0, None),
                (13.0, 10.0, 5.0, 8.0),
                (None, None, None, None),
                (7.0, None, 3.0, None),
            ],
        )

    def test_defaults_add_lags_one_and_five_without_changing_input(self) -> None:
        frame = self.spark.createDataFrame(
            [(1, cycle, float(cycle * 2)) for cycle in range(1, 7)],
            "unit_id int, cycle int, sensor_2 double",
        )
        original_rows = frame.collect()

        transformed = add_lag_difference_features(frame, ("sensor_2",))
        final_row = transformed.orderBy("cycle").collect()[-1]

        self.assertEqual(final_row["sensor_2_lag_1"], 10.0)
        self.assertEqual(final_row["sensor_2_lag_5"], 2.0)
        self.assertEqual(final_row["sensor_2_diff_1"], 2.0)
        self.assertEqual(final_row["sensor_2_diff_5"], 10.0)
        self.assertEqual(frame.collect(), original_rows)

if __name__ == "__main__":
    unittest.main()
