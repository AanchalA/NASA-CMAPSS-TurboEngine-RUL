import os
import shutil
import sys
import unittest
from pathlib import Path

import pytest
from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.constants import CMAPSS_COLUMNS
from src.data_processing.data_loader import load_cmapss_raw


@pytest.mark.requires_data
class LoadCmapssRawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-raw-loader-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def assert_standardized_file(self, filename: str) -> None:
        dataframe = load_cmapss_raw(
            self.spark,
            PROJECT_ROOT / "Data" / "CMAPSSData" / filename,
        )

        self.assertEqual(dataframe.columns, list(CMAPSS_COLUMNS))
        self.assertEqual(len(dataframe.columns), 26)
        self.assertNotIn("", dataframe.columns)

        first = dataframe.orderBy("unit_id", "cycle").first()
        if first is None:
            self.fail(f"{filename} unexpectedly produced no rows")
        self.assertEqual(first["unit_id"], 1)
        self.assertEqual(first["cycle"], 1)

    def test_loads_training_file(self) -> None:
        self.assert_standardized_file("train_FD001.txt")

    def test_loads_test_file(self) -> None:
        self.assert_standardized_file("test_FD001.txt")


if __name__ == "__main__":
    unittest.main()
