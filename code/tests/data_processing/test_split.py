"""Tests for deterministic engine-level train/validation splitting."""

import os
import shutil
import sys
import unittest
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.data_loader import load_cmapss_raw, add_rul_target
from src.data_processing.split import split_train_validation_by_unit



class SplitTrainValidationByUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-engine-split-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def synthetic_data(self, engines: int, cycles: int = 3) -> DataFrame:
        rows = [
            (unit_id, cycle, cycles - cycle, float(unit_id * 100 + cycle))
            for unit_id in range(1, engines + 1)
            for cycle in range(1, cycles + 1)
        ]
        return self.spark.createDataFrame(
            rows, schema="unit_id int, cycle int, RUL int, measurement double"
        )

    @staticmethod
    def unit_ids(df: DataFrame) -> set[int]:
        return {row["unit_id"] for row in df.select("unit_id").distinct().collect()}

    def test_engine_coverage_no_overlap_no_row_loss_and_preserved_values(self) -> None:
        original = self.synthetic_data(10, cycles=4)
        original_rows = set(original.collect())

        train_df, validation_df = split_train_validation_by_unit(
            original, validation_fraction=0.3, seed=42
        )

        original_ids = self.unit_ids(original)
        train_ids = self.unit_ids(train_df)
        validation_ids = self.unit_ids(validation_df)
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertEqual(train_ids | validation_ids, original_ids)
        self.assertEqual(train_df.count() + validation_df.count(), original.count())
        self.assertEqual(set(train_df.collect()) | set(validation_df.collect()), original_rows)
        self.assertEqual(train_df.columns, original.columns)
        self.assertEqual(validation_df.columns, original.columns)
        self.assertEqual(original.columns, ["unit_id", "cycle", "RUL", "measurement"])

        original_cycle_counts = {
            row["unit_id"]: row["count"]
            for row in original.groupBy("unit_id").count().collect()
        }
        output_cycle_counts = {
            row["unit_id"]: row["count"]
            for row in train_df.unionByName(validation_df)
            .groupBy("unit_id")
            .count()
            .collect()
        }
        self.assertEqual(output_cycle_counts, original_cycle_counts)

    def test_split_is_deterministic(self) -> None:
        original = self.synthetic_data(30)

        first_train, first_validation = split_train_validation_by_unit(
            original, validation_fraction=0.2, seed=17
        )
        second_train, second_validation = split_train_validation_by_unit(
            original.repartition(4), validation_fraction=0.2, seed=17
        )

        self.assertEqual(self.unit_ids(first_train), self.unit_ids(second_train))
        self.assertEqual(
            self.unit_ids(first_validation), self.unit_ids(second_validation)
        )

    def test_different_seed_changes_assignment(self) -> None:
        original = self.synthetic_data(50)

        _, first_validation = split_train_validation_by_unit(
            original, validation_fraction=0.2, seed=1
        )
        _, second_validation = split_train_validation_by_unit(
            original, validation_fraction=0.2, seed=2
        )

        self.assertNotEqual(self.unit_ids(first_validation), self.unit_ids(second_validation))

    def test_invalid_validation_fractions_are_rejected(self) -> None:
        original = self.synthetic_data(4)

        for fraction in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(validation_fraction=fraction):
                with self.assertRaisesRegex(
                    ValueError, "validation_fraction must be strictly between 0 and 1"
                ):
                    split_train_validation_by_unit(original, fraction)

    @pytest.mark.requires_data
    def test_fd001_training_split(self) -> None:
        original = add_rul_target(
            load_cmapss_raw(
                self.spark,
                PROJECT_ROOT / "Data" / "CMAPSSData" / "train_FD001.txt",
            )
        )

        train_df, validation_df = split_train_validation_by_unit(
            original, validation_fraction=0.2, seed=42
        )

        train_ids = self.unit_ids(train_df)
        validation_ids = self.unit_ids(validation_df)
        self.assertEqual(len(train_ids), 80)
        self.assertEqual(len(validation_ids), 20)
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertEqual(len(train_ids | validation_ids), 100)
        self.assertEqual(train_df.count() + validation_df.count(), 20_631)
        self.assertIn("RUL", train_df.columns)
        self.assertIn("RUL", validation_df.columns)
        self.assertEqual(
            train_df.select("unit_id", "cycle", "RUL")
            .unionByName(validation_df.select("unit_id", "cycle", "RUL"))
            .exceptAll(original.select("unit_id", "cycle", "RUL"))
            .count(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
