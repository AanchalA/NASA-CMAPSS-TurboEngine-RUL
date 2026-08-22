"""Tests for training-fitted exact constant feature removal."""

import os
import shutil
import sys
import unittest
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.feature_engineering import (
    drop_feature_columns,
    find_constant_feature_columns,
)
from src.data_processing.constants import CMAPSS_FEATURE_COLUMNS
from src.data_processing.data_loader import load_cmapss_raw, add_rul_target
from src.data_processing.split import split_train_validation_by_unit
from src.data_processing.validation import validate_cmapss_data


class ConstantFeatureSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-constant-feature-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def cmapss_frame(
        self,
        feature_rows: list[dict[str, float]],
        unit_id: int = 1,
    ) -> DataFrame:
        rows = []
        for cycle, overrides in enumerate(feature_rows, start=1):
            features = {
                column: float(cycle + index)
                for index, column in enumerate(CMAPSS_FEATURE_COLUMNS)
            }
            features.update(overrides)
            rows.append(
                (
                    unit_id,
                    cycle,
                    len(feature_rows) - cycle,
                    *(features[column] for column in CMAPSS_FEATURE_COLUMNS),
                )
            )

        schema = "unit_id int, cycle int, RUL int, " + ", ".join(
            f"{column} double" for column in CMAPSS_FEATURE_COLUMNS
        )
        return self.spark.createDataFrame(rows, schema=schema)

    def test_exact_detection_is_ordered_repeatable_and_protects_metadata(self) -> None:
        train_df = self.cmapss_frame(
            [
                {"setting_3": 5.0, "sensor_1": 9.0},
                {"setting_3": 5.0, "sensor_1": 9.0},
                {"setting_3": 5.0, "sensor_1": 9.0},
                {"setting_3": 5.0, "sensor_1": 9.0},
            ]
        )

        expected = ["setting_3", "sensor_1"]
        self.assertEqual(find_constant_feature_columns(train_df), expected)
        self.assertEqual(find_constant_feature_columns(train_df), expected)
        self.assertNotIn("unit_id", expected)
        self.assertNotIn("cycle", expected)
        self.assertNotIn("RUL", expected)

        for protected in ("unit_id", "cycle", "RUL"):
            with self.subTest(protected=protected):
                with self.assertRaisesRegex(ValueError, "protected columns"):
                    drop_feature_columns(train_df, [protected])

    def test_empty_learned_list_is_a_no_op(self) -> None:
        original = self.cmapss_frame([{}, {}, {}])

        learned = find_constant_feature_columns(original)
        transformed = drop_feature_columns(original, learned)

        self.assertEqual(learned, [])
        self.assertEqual(transformed.columns, original.columns)
        self.assertEqual(transformed.collect(), original.collect())

    def test_training_decision_is_applied_unchanged_to_validation(self) -> None:
        train_df = self.cmapss_frame(
            [
                {"sensor_1": 5.0, "sensor_2": 1.0},
                {"sensor_1": 5.0, "sensor_2": 2.0},
                {"sensor_1": 5.0, "sensor_2": 3.0},
                {"sensor_1": 5.0, "sensor_2": 4.0},
            ]
        )
        validation_df = self.cmapss_frame(
            [
                {"sensor_1": 5.0, "sensor_2": 7.0},
                {"sensor_1": 6.0, "sensor_2": 7.0},
            ],
            unit_id=2,
        )
        original_validation = validation_df.collect()

        learned = find_constant_feature_columns(train_df)
        transformed_train = drop_feature_columns(train_df, learned)
        transformed_validation = drop_feature_columns(validation_df, learned)

        self.assertEqual(learned, ["sensor_1"])
        self.assertNotIn("sensor_1", transformed_validation.columns)
        self.assertIn("sensor_2", transformed_validation.columns)
        self.assertEqual(transformed_train.columns, transformed_validation.columns)
        self.assertEqual(transformed_validation.count(), validation_df.count())
        self.assertEqual(
            transformed_validation.select("unit_id", "cycle", "RUL", "sensor_2").collect(),
            validation_df.select("unit_id", "cycle", "RUL", "sensor_2").collect(),
        )
        self.assertEqual(validation_df.collect(), original_validation)

    def test_fd001_training_constants_and_aligned_partitions(self) -> None:
        original = load_cmapss_raw(
            self.spark,
            PROJECT_ROOT / "Data" / "CMAPSSData" / "train_FD001.txt",
        )
        validate_cmapss_data(original)
        enriched = add_rul_target(original)
        train_df, validation_df = split_train_validation_by_unit(enriched)
        train_rows = train_df.count()
        validation_rows = validation_df.count()

        learned = find_constant_feature_columns(train_df)
        transformed_train = drop_feature_columns(train_df, learned)
        transformed_validation = drop_feature_columns(validation_df, learned)

        self.assertEqual(
            learned,
            [
                "setting_3",
                "sensor_1",
                "sensor_5",
                "sensor_10",
                "sensor_16",
                "sensor_18",
                "sensor_19",
            ],
        )
        self.assertEqual(transformed_train.count(), train_rows)
        self.assertEqual(transformed_validation.count(), validation_rows)
        self.assertEqual(
            transformed_train.select("unit_id").distinct().count(), 80
        )
        self.assertEqual(
            transformed_validation.select("unit_id").distinct().count(), 20
        )
        self.assertEqual(transformed_train.schema, transformed_validation.schema)
        for protected in ("unit_id", "cycle", "RUL"):
            self.assertIn(protected, transformed_train.columns)
        for removed in learned:
            self.assertNotIn(removed, transformed_train.columns)
            self.assertNotIn(removed, transformed_validation.columns)


if __name__ == "__main__":
    unittest.main()
