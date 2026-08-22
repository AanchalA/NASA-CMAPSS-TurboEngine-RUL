"""Tests for causal engine-local rolling statistics."""

import math
import os
import shutil
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing import rolling_slope_and_statistics_features


class RollingStatisticsFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-rolling-statistics-feature-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_statistics_are_causal_ordered_and_engine_local(self) -> None:
        frame = self.spark.createDataFrame(
            [
                (2, 2, 104.0),
                (1, 3, 8.0),
                (1, 1, 2.0),
                (2, 1, 100.0),
                (1, 2, 4.0),
            ],
            "unit_id int, cycle int, sensor_2 double",
        )

        rows = (
            rolling_slope_and_statistics_features(frame, ("sensor_2",), windows=(2,))
            .orderBy("unit_id", "cycle")
            .collect()
        )

        self.assertEqual(
            [row["sensor_2_rolling_mean_2"] for row in rows],
            [2.0, 3.0, 6.0, 100.0, 102.0],
        )
        self.assertIsNone(rows[0]["sensor_2_rolling_std_2"])
        self.assertAlmostEqual(rows[1]["sensor_2_rolling_std_2"], math.sqrt(2.0))
        self.assertAlmostEqual(rows[2]["sensor_2_rolling_std_2"], math.sqrt(8.0))
        self.assertIsNone(rows[3]["sensor_2_rolling_std_2"])
        self.assertAlmostEqual(rows[4]["sensor_2_rolling_std_2"], math.sqrt(8.0))

    def test_defaults_append_mean_and_std_for_each_window(self) -> None:
        frame = self.spark.createDataFrame(
            [(1, cycle, float(cycle)) for cycle in range(1, 21)],
            "unit_id int, cycle int, sensor_2 double",
        )

        transformed = rolling_slope_and_statistics_features(frame, ("sensor_2",))
        second_cycle = transformed.orderBy("cycle").collect()[1]

        self.assertEqual(second_cycle["sensor_2_rolling_mean_5"], 1.5)
        self.assertAlmostEqual(second_cycle["sensor_2_rolling_std_5"], math.sqrt(0.5))

        for window in (5, 10, 20):
            self.assertIn(f"sensor_2_rolling_mean_{window}", transformed.columns)
            self.assertIn(f"sensor_2_rolling_std_{window}", transformed.columns)


if __name__ == "__main__":
    unittest.main()
