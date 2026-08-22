"""Tests for deterministic training-fitted operating regimes."""

import os
import shutil
import sys
import unittest
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.feature_engineering import (
    assign_operating_regimes,
    fit_operating_regime_mapping,
)
from src.data_processing.data_loader import load_cmapss_raw, add_rul_target
from src.data_processing.split import split_train_validation_by_unit
from src.data_processing.validation import validate_cmapss_data


class OperatingRegimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-operating-regime-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def synthetic_frame(self, settings: list[tuple[float, float, float]]) -> DataFrame:
        rows = [
            (index, index * 2, 50 - index, *values, float(index * 100))
            for index, values in enumerate(settings, start=1)
        ]
        return self.spark.createDataFrame(
            rows,
            schema=(
                "unit_id int, cycle int, RUL int, setting_1 double, "
                "setting_2 double, setting_3 double, sensor_1 double"
            ),
        )

    def test_multiple_regimes_are_deterministic_and_settings_only(self) -> None:
        frame = self.synthetic_frame(
            [(20.2, 0.701, 100.1), (0.2, 0.004, 0.1), (20.4, 0.699, 99.9)]
        )
        expected = {(0.0, 0.0, 0.0): 1, (20.0, 0.7, 100.0): 2}

        self.assertEqual(fit_operating_regime_mapping(frame), expected)
        self.assertEqual(
            fit_operating_regime_mapping(frame.orderBy("unit_id", ascending=False)),
            expected,
        )
        self.assertEqual(fit_operating_regime_mapping(frame.repartition(3)), expected)

        assigned = assign_operating_regimes(frame, expected)
        regimes = [row["regime"] for row in assigned.orderBy("unit_id").collect()]
        self.assertEqual(regimes, [2, 1, 2])
        self.assertEqual(assigned.columns[2], "regime")
        self.assertEqual(assigned.schema["regime"].dataType.typeName(), "integer")

    def test_validation_reuses_labels_and_preserves_rows_and_values(self) -> None:
        train = self.synthetic_frame(
            [(0.1, 0.001, 0.1), (10.1, 0.501, 50.1), (20.1, 0.999, 100.1)]
        )
        validation = self.synthetic_frame([(20.2, 1.001, 100.2), (0.2, 0.002, 0.2)])
        original_rows = validation.collect()

        mapping = fit_operating_regime_mapping(train)
        assigned = assign_operating_regimes(validation, mapping)

        self.assertEqual(
            [row["regime"] for row in assigned.orderBy("unit_id").collect()], [3, 1]
        )
        self.assertEqual(assigned.count(), validation.count())
        self.assertEqual(
            assigned.drop("regime").orderBy("unit_id").collect(), original_rows
        )
        self.assertEqual(validation.collect(), original_rows)

    def test_unseen_validation_key_fails_clearly(self) -> None:
        mapping = fit_operating_regime_mapping(
            self.synthetic_frame([(0.1, 0.001, 0.1)])
        )

        with self.assertRaisesRegex(
            ValueError, "operating-regime keys unseen in training"
        ):
            assign_operating_regimes(
                self.synthetic_frame([(99.0, 9.99, 999.0)]), mapping
            )

    def test_real_subsets_and_official_tests_use_training_mapping(self) -> None:
        expected_counts = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

        for subset, expected_count in expected_counts.items():
            with self.subTest(subset=subset):
                raw_train = load_cmapss_raw(
                    self.spark,
                    PROJECT_ROOT / "Data" / "CMAPSSData" / f"train_{subset}.txt",
                )
                validate_cmapss_data(raw_train)
                enriched = add_rul_target(raw_train)
                train_df, validation_df = split_train_validation_by_unit(enriched)

                mapping = fit_operating_regime_mapping(train_df)
                self.assertEqual(len(mapping), expected_count)
                assigned_train = assign_operating_regimes(train_df, mapping)
                assigned_validation = assign_operating_regimes(validation_df, mapping)
                self.assertEqual(assigned_train.count(), train_df.count())
                self.assertEqual(assigned_validation.count(), validation_df.count())
                self.assertEqual(
                    assigned_train.select("regime").distinct().count(), expected_count
                )
                self.assertEqual(
                    assigned_validation.filter("regime is null").count(), 0
                )

                official_test = load_cmapss_raw(
                    self.spark,
                    PROJECT_ROOT / "Data" / "CMAPSSData" / f"test_{subset}.txt",
                )
                assigned_test = assign_operating_regimes(official_test, mapping)
                self.assertEqual(assigned_test.count(), official_test.count())
                self.assertNotIn("RUL", assigned_test.columns)


if __name__ == "__main__":
    unittest.main()
