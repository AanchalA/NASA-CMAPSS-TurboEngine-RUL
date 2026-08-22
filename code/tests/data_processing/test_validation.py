"""Tests for C-MAPSS Spark data-quality validation."""

import os
import shutil
import sys
import unittest
from pathlib import Path

import pytest
from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.schema import CMAPSS_SCHEMA
from src.data_processing.data_loader import load_cmapss_raw
from src.data_processing.validation import validate_cmapss_data


class ValidateCmapssDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-validation-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    @staticmethod
    def row(unit_id: int = 1, cycle: int = 1, value: float = 1.0) -> tuple:
        return (unit_id, cycle, *(value for _ in range(24)))

    def dataframe(self, rows: list[tuple]):
        return self.spark.createDataFrame(rows, schema=CMAPSS_SCHEMA)

    @pytest.mark.requires_data
    def test_actual_parsed_file_is_valid(self) -> None:
        dataframe = load_cmapss_raw(
            self.spark,
            PROJECT_ROOT / "Data" / "CMAPSSData" / "train_FD001.txt",
        )

        self.assertIsNone(validate_cmapss_data(dataframe))

    def test_rejects_null_and_nan_values(self) -> None:
        null_row = list(self.row())
        null_row[2] = None
        nan_row = list(self.row(cycle=2))
        nan_row[3] = float("nan")

        with self.assertRaisesRegex(ValueError, "missing-value validation failed"):
            validate_cmapss_data(self.dataframe([tuple(null_row), tuple(nan_row)]))

    def test_rejects_exact_duplicate_rows_as_invalid_cycles(self) -> None:
        row = self.row()

        with self.assertRaisesRegex(ValueError, "cycle-sequence validation failed"):
            validate_cmapss_data(self.dataframe([row, row]))

    def test_rejects_duplicate_unit_cycle_keys_as_invalid_cycles(self) -> None:
        with self.assertRaisesRegex(ValueError, "cycle-sequence validation failed"):
            validate_cmapss_data(
                self.dataframe([self.row(value=1.0), self.row(value=2.0)])
            )

    def test_rejects_infinite_values(self) -> None:
        infinite_row = list(self.row())
        infinite_row[2] = float("inf")

        with self.assertRaisesRegex(ValueError, "finite-value validation failed"):
            validate_cmapss_data(self.dataframe([tuple(infinite_row)]))

    def test_rejects_missing_cycle(self) -> None:
        dataframe = self.dataframe(
            [self.row(cycle=1), self.row(cycle=2), self.row(cycle=4)]
        )

        with self.assertRaisesRegex(ValueError, "cycle-sequence validation failed"):
            validate_cmapss_data(dataframe)

    def test_rejects_trajectory_not_starting_at_one(self) -> None:
        dataframe = self.dataframe([self.row(cycle=2), self.row(cycle=3)])

        with self.assertRaisesRegex(ValueError, "cycle-sequence validation failed"):
            validate_cmapss_data(dataframe)


if __name__ == "__main__":
    unittest.main()
