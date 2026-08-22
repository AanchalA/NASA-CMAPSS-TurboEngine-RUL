"""Tests for deterministic C-MAPSS training RUL target construction."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import IntegerType


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.data_loader import (
    add_rul_target,
    add_test_rul_target,
    load_cmapss_test_rul,
    load_cmapss_raw,
)
from src.data_processing.constants import CMAPSS_COLUMNS, FINAL_OBSERVED_RUL_COLUMN
from src.data_processing.schema import (
    CMAPSS_SCHEMA,
    TEST_RUL_SCHEMA,
)


class AddRulTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-rul-target-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    @staticmethod
    def row(unit_id: int, cycle: int, value: float = 1.0) -> tuple:
        return (unit_id, cycle, *(value for _ in range(24)))

    def dataframe(self, rows: list[tuple]):
        return self.spark.createDataFrame(rows, schema=CMAPSS_SCHEMA)

    def final_rul_dataframe(self, records: list[tuple[int, int]]):
        return self.spark.createDataFrame(records, schema=TEST_RUL_SCHEMA)

    def test_simple_trajectory(self) -> None:
        original = self.dataframe([self.row(1, cycle) for cycle in range(1, 5)])

        result = add_rul_target(original)

        actual = [row["RUL"] for row in result.orderBy("cycle").collect()]
        self.assertEqual(actual, [3, 2, 1, 0])
        self.assertIsInstance(result.schema["RUL"].dataType, IntegerType)

    def test_multiple_engines_use_independent_lifetimes(self) -> None:
        original = self.dataframe(
            [self.row(1, cycle) for cycle in range(1, 4)]
            + [self.row(2, cycle) for cycle in range(1, 6)]
        )

        result = add_rul_target(original)

        actual = {
            (row["unit_id"], row["cycle"]): row["RUL"]
            for row in result.select("unit_id", "cycle", "RUL").collect()
        }
        self.assertEqual(
            actual,
            {
                (1, 1): 2,
                (1, 2): 1,
                (1, 3): 0,
                (2, 1): 4,
                (2, 2): 3,
                (2, 3): 2,
                (2, 4): 1,
                (2, 5): 0,
            },
        )

    def test_original_columns_rows_and_values_are_preserved(self) -> None:
        original = self.dataframe(
            [self.row(1, 1, 1.25), self.row(1, 2, 2.5)]
        )
        original_rows = original.orderBy("unit_id", "cycle").collect()

        result = add_rul_target(original)

        self.assertEqual(result.columns, [*CMAPSS_COLUMNS, "RUL"])
        self.assertEqual(result.count(), original.count())
        self.assertEqual(
            result.select(*CMAPSS_COLUMNS).orderBy("unit_id", "cycle").collect(),
            original_rows,
        )
        self.assertEqual(original.columns, list(CMAPSS_COLUMNS))

    def test_fd001_training_data_invariants(self) -> None:
        original = load_cmapss_raw(
            self.spark,
            PROJECT_ROOT / "Data" / "CMAPSSData" / "train_FD001.txt",
        )

        result = add_rul_target(original)

        summary = result.agg(
            F.count("*").alias("rows"),
            F.min("RUL").alias("minimum_rul"),
            F.max("RUL").alias("maximum_rul"),
        ).first()
        if summary is None:
            self.fail("train_FD001.txt unexpectedly produced no rows")

        final_cycles = result.groupBy("unit_id").agg(
            F.max(F.struct("cycle", "RUL")).alias("final")
        )
        invalid_final_cycles = final_cycles.where(F.col("final.RUL") != 0).count()

        rul_steps = (
            result.withColumn(
                "next_rul",
                F.lead("RUL").over(
                    Window.partitionBy("unit_id").orderBy("cycle")
                ),
            )
            .where(F.col("next_rul").isNotNull())
            .where((F.col("RUL") - F.col("next_rul")) != 1)
            .count()
        )

        self.assertEqual(summary["rows"], original.count())
        self.assertEqual(result.select("unit_id").distinct().count(), 100)
        self.assertEqual(summary["minimum_rul"], 0)
        self.assertEqual(summary["maximum_rul"], 361)
        self.assertEqual(invalid_final_cycles, 0)
        self.assertEqual(rul_steps, 0)
        self.assertEqual(result.where(F.col("RUL") < 0).count(), 0)

    def test_truncated_test_engine_uses_official_terminal_rul(self) -> None:
        original = self.dataframe([self.row(1, cycle) for cycle in range(1, 5)])

        result = add_test_rul_target(
            original,
            self.final_rul_dataframe([(1, 2)]),
        )

        actual = [row["RUL"] for row in result.orderBy("cycle").collect()]
        self.assertEqual(actual, [5, 4, 3, 2])
        self.assertEqual(actual[-1], 2)
        self.assertNotEqual(actual[-1], 0)
        self.assertIsInstance(result.schema["RUL"].dataType, IntegerType)

    def test_multiple_test_engines_are_reconstructed_independently(self) -> None:
        original = self.dataframe(
            [self.row(1, cycle) for cycle in range(1, 4)]
            + [self.row(2, cycle) for cycle in range(1, 6)]
        )

        result = add_test_rul_target(
            original,
            self.final_rul_dataframe([(1, 5), (2, 10)]),
        )

        actual = {
            (row["unit_id"], row["cycle"]): row["RUL"]
            for row in result.select("unit_id", "cycle", "RUL").collect()
        }
        self.assertEqual(
            actual,
            {
                (1, 1): 7,
                (1, 2): 6,
                (1, 3): 5,
                (2, 1): 14,
                (2, 2): 13,
                (2, 3): 12,
                (2, 4): 11,
                (2, 5): 10,
            },
        )

        terminal_rows = (
            result.groupBy("unit_id")
            .agg(F.max(F.struct("cycle", "RUL")).alias("terminal"))
            .orderBy("unit_id")
            .collect()
        )
        self.assertEqual(
            [row["terminal"]["RUL"] for row in terminal_rows],
            [5, 10],
        )

        step_errors = (
            result.withColumn(
                "next_rul",
                F.lead("RUL").over(
                    Window.partitionBy("unit_id").orderBy("cycle")
                ),
            )
            .where(F.col("next_rul").isNotNull())
            .where(F.col("RUL") - F.col("next_rul") != 1)
            .count()
        )
        self.assertEqual(step_errors, 0)

    def test_test_target_preserves_original_rows_columns_and_values(self) -> None:
        original = self.dataframe(
            [self.row(1, 1, 1.25), self.row(1, 2, 2.5)]
        )
        original_rows = original.orderBy("unit_id", "cycle").collect()

        result = add_test_rul_target(
            original,
            self.final_rul_dataframe([(1, 25)]),
        )

        self.assertEqual(result.columns, [*CMAPSS_COLUMNS, "RUL"])
        self.assertEqual(result.count(), len(original_rows))
        self.assertEqual(
            result.select(*CMAPSS_COLUMNS).orderBy("unit_id", "cycle").collect(),
            original_rows,
        )
        self.assertEqual(original.columns, list(CMAPSS_COLUMNS))
        terminal_rul = result.where(F.col("cycle") == 2).first()
        if terminal_rul is None:
            self.fail("synthetic terminal test row is missing")
        self.assertEqual(terminal_rul["RUL"], 25)
        self.assertNotEqual(terminal_rul["RUL"], 0)

    def test_test_target_mapping_rejects_missing_and_excess_records(
        self,
    ) -> None:
        three_engines = self.dataframe(
            [self.row(unit_id, 1) for unit_id in range(1, 4)]
        )
        with self.assertRaisesRegex(
            ValueError,
            "engine mapping mismatch",
        ):
            add_test_rul_target(
                three_engines,
                self.final_rul_dataframe([(1, 5), (2, 10)]),
            )

        two_engines = self.dataframe(
            [self.row(unit_id, 1) for unit_id in range(1, 3)]
        )
        with self.assertRaisesRegex(
            ValueError,
            "engine mapping mismatch",
        ):
            add_test_rul_target(
                two_engines,
                self.final_rul_dataframe([(1, 5), (2, 10), (3, 15)]),
            )

    def test_test_target_rejects_incompatible_unit_ids(self) -> None:
        original = self.dataframe([self.row(1, 1), self.row(3, 1)])

        with self.assertRaisesRegex(ValueError, "engine mapping mismatch"):
            add_test_rul_target(
                original,
                self.final_rul_dataframe([(1, 5), (2, 10)]),
            )

    def test_official_rul_loader_maps_non_empty_lines_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "RUL.txt"
            path.write_text(" 25  \n\n10\n", encoding="utf-8")

            result = load_cmapss_test_rul(self.spark, path)

        self.assertEqual(
            [tuple(row) for row in result.collect()],
            [
                (1, 25),
                (2, 10),
            ],
        )
        self.assertEqual(
            result.columns,
            ["unit_id", FINAL_OBSERVED_RUL_COLUMN],
        )
        self.assertTrue(
            all(not field.nullable for field in result.schema.fields)
        )

    def test_official_rul_loader_rejects_negative_and_malformed_values(self) -> None:
        invalid_contents = {
            "negative": ("5\n-1\n", "must be non-negative"),
            "non_numeric": ("5\nnot-a-number\n", "expected an integer"),
            "multiple_values": ("5 10\n", "expected exactly one value"),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            for name, (content, message) in invalid_contents.items():
                with self.subTest(name=name):
                    path = Path(temporary_directory) / f"{name}.txt"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_cmapss_test_rul(self.spark, path)

    def test_all_official_test_subsets_use_nasa_terminal_targets(self) -> None:
        for subset in ("FD001", "FD002", "FD003", "FD004"):
            with self.subTest(subset=subset):
                original = load_cmapss_raw(
                    self.spark,
                    PROJECT_ROOT / "Data" / "CMAPSSData" / f"test_{subset}.txt",
                )
                official = load_cmapss_test_rul(
                    self.spark,
                    PROJECT_ROOT / "Data" / "CMAPSSData" / f"RUL_{subset}.txt",
                )
                result = add_test_rul_target(original, official)

                engine_count = original.select("unit_id").distinct().count()
                self.assertEqual(official.count(), engine_count)
                self.assertEqual(result.count(), original.count())
                self.assertEqual(result.where(F.col("RUL") < 0).count(), 0)

                terminal = result.groupBy("unit_id").agg(
                    F.max(F.struct("cycle", "RUL")).alias("terminal")
                )
                terminal_mismatches = (
                    terminal.join(official, on="unit_id", how="full")
                    .where(
                        F.col("terminal").isNull()
                        | F.col(FINAL_OBSERVED_RUL_COLUMN).isNull()
                        | (
                            F.col("terminal.RUL")
                            != F.col(FINAL_OBSERVED_RUL_COLUMN)
                        )
                    )
                    .count()
                )
                self.assertEqual(terminal_mismatches, 0)

                incorrect_forced_zeros = (
                    terminal.join(official, on="unit_id", how="inner")
                    .where(F.col(FINAL_OBSERVED_RUL_COLUMN) != 0)
                    .where(F.col("terminal.RUL") == 0)
                    .count()
                )
                self.assertEqual(incorrect_forced_zeros, 0)


if __name__ == "__main__":
    unittest.main()
