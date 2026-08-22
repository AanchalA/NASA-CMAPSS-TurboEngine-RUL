"""Tests for causal engine-local rolling degradation slopes."""

import os
import shutil
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing import rolling_slope_and_statistics_features


class RollingSlopeFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-rolling-slope-feature-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_slopes_are_causal_ordered_and_engine_local(self) -> None:
        frame = self.spark.createDataFrame(
            [
                (2, 3, 21.0),
                (1, 2, 9.0),
                (2, 1, 27.0),
                (1, 3, 11.0),
                (1, 1, 7.0),
                (2, 2, 24.0),
                (1, 4, 13.0),
            ],
            "unit_id int, cycle int, sensor_2 double",
        )

        rows = (
            rolling_slope_and_statistics_features(frame, ("sensor_2",), windows=(3,))
            .orderBy("unit_id", "cycle")
            .collect()
        )

        self.assertIsNone(rows[0]["sensor_2_rolling_slope_3"])
        self.assertEqual(
            [row["sensor_2_rolling_slope_3"] for row in rows[1:4]],
            [2.0, 2.0, 2.0],
        )
        self.assertIsNone(rows[4]["sensor_2_rolling_slope_3"])
        self.assertEqual(
            [row["sensor_2_rolling_slope_3"] for row in rows[5:]],
            [-3.0, -3.0],
        )

    def test_defaults_append_one_slope_per_window(self) -> None:
        frame = self.spark.createDataFrame(
            [(1, cycle, float(cycle)) for cycle in range(1, 21)],
            "unit_id int, cycle int, sensor_2 double",
        )

        transformed = rolling_slope_and_statistics_features(frame, ("sensor_2",))

        for window in (5, 10, 20):
            self.assertIn(f"sensor_2_rolling_slope_{window}", transformed.columns)


if __name__ == "__main__":
    unittest.main()
