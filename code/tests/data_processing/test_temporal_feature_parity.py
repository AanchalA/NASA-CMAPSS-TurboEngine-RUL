"""Offline Spark and online pandas temporal-feature parity."""

import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing import add_pandas_temporal_features  # noqa: E402
from src.data_processing.feature_engineering import (  # noqa: E402
    add_ewma_features,
    add_lag_difference_features,
    rolling_slope_and_statistics_features,
)


class TemporalFeatureParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-temporal-feature-parity-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_spark_and_pandas_features_match_cycle_by_cycle(self) -> None:
        columns = ("unit_id", "cycle", "sensor_2", "sensor_3")
        rows = [
            (1, 8, 19.0, 48.0),
            (1, 1, 10.0, 50.0),
            (1, 12, 25.0, 43.0),
            (1, 4, 15.0, 47.0),
            (1, 2, 13.0, 49.0),
            (1, 9, 21.0, 46.0),
            (1, 5, 16.0, 52.0),
        ]
        sensor_columns = ("sensor_2", "sensor_3")

        offline = self.spark.createDataFrame(rows, schema=columns)
        offline = add_lag_difference_features(offline, sensor_columns)
        offline = rolling_slope_and_statistics_features(offline, sensor_columns)
        offline = add_ewma_features(offline, sensor_columns)
        offline = offline.orderBy("cycle").toPandas()

        online = add_pandas_temporal_features(
            pd.DataFrame(rows, columns=columns),
            sensor_columns,
        )
        feature_columns = [
            column
            for column in offline.columns
            if column not in columns
        ]

        self.assertEqual(offline["cycle"].tolist(), online["cycle"].tolist())
        np.testing.assert_allclose(
            offline.loc[:, feature_columns].to_numpy(),
            online.loc[:, feature_columns].to_numpy(),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
