"""Tests for causal engine-local EWMA features."""

import os
import shutil
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.feature_engineering import add_ewma_features


class EwmaFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-ewma-feature-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_ewma_is_causal_ordered_and_engine_local(self) -> None:
        frame = self.spark.createDataFrame(
            [
                (2, 2, 80.0),
                (1, 3, 14.0),
                (1, 1, 10.0),
                (2, 1, 100.0),
                (1, 2, 20.0),
            ],
            "unit_id int, cycle int, sensor_2 double",
        )

        rows = (
            add_ewma_features(frame, ("sensor_2",), alpha=0.5)
            .orderBy("unit_id", "cycle")
            .collect()
        )

        self.assertEqual(
            [row["sensor_2_ewma"] for row in rows],
            [10.0, 15.0, 14.5, 100.0, 90.0],
        )

    def test_default_alpha_adds_one_feature_per_sensor(self) -> None:
        frame = self.spark.createDataFrame(
            [(1, 1, 10.0, 20.0), (1, 2, 20.0, 10.0)],
            "unit_id int, cycle int, sensor_2 double, sensor_3 double",
        )

        rows = (
            add_ewma_features(frame, ("sensor_2", "sensor_3"))
            .orderBy("cycle")
            .collect()
        )

        self.assertEqual(rows[1]["sensor_2_ewma"], 12.0)
        self.assertEqual(rows[1]["sensor_3_ewma"], 18.0)


if __name__ == "__main__":
    unittest.main()
