"""Tests for the production C-MAPSS preprocessing composition layer."""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.preprocessing import (  # noqa: E402
    PreprocessingRunResult,
    finalize_preprocessing_output,
    run_subset_preprocessing,
)
from src.data_processing.scaler_state import RegimeSensorScaler  # noqa: E402
from src.data_processing.scaling import (  # noqa: E402
    apply_global_sensor_scaler,
    apply_sensor_scaler,
    fit_regime_sensor_scaler,
)
from src.tracking.mlflow_tracking import configure_mlflow, load_preprocessing_state  # noqa: E402
from src.tracking.spark_mlflow_tracking import load_preprocessing_model  # noqa: E402


def _raw_row(unit_id: int, cycle: int) -> str:
    settings = [0.0, 0.0, 100.0]
    sensors = [float(number) for number in range(1, 22)]
    sensors[1] = float(unit_id * 10 + cycle)
    values = [unit_id, cycle, *settings, *sensors]
    return " ".join(str(value) for value in values)


def _write_synthetic_subset(root: Path, subset: str = "FD001") -> None:
    root.mkdir(parents=True, exist_ok=True)
    train_rows = [
        _raw_row(unit_id, cycle)
        for unit_id in range(1, 5)
        for cycle in range(1, 4)
    ]
    test_rows = [
        _raw_row(unit_id, cycle)
        for unit_id in range(1, 3)
        for cycle in range(1, 3)
    ]
    (root / f"train_{subset}.txt").write_text(
        "\n".join(train_rows) + "\n", encoding="utf-8"
    )
    (root / f"test_{subset}.txt").write_text(
        "\n".join(test_rows) + "\n", encoding="utf-8"
    )
    (root / f"RUL_{subset}.txt").write_text("5\n7\n", encoding="utf-8")


class PreprocessingInputTests(unittest.TestCase):
    def test_missing_input_fails_before_spark_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "train_FD001.txt"):
                run_subset_preprocessing(
                    spark=None,  # type: ignore[arg-type]
                    subset="FD001",
                    raw_data_dir=directory,
                    output_dir=Path(directory) / "processed",
                )


class PreprocessingOutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-preprocessing-output-contract-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def frame(self, unit_id, sensor_value):
        frame = self.spark.createDataFrame(
            [(unit_id, 1, 5, 1, 0.0, sensor_value)],
            "unit_id int, cycle int, RUL int, regime int, "
            "setting_1 double, sensor_2 double",
        )
        return VectorAssembler(
            inputCols=["sensor_2"], outputCol="assembled_sensor_features"
        ).transform(frame).withColumn(
            "scaled_sensor_features", F.col("assembled_sensor_features")
        )

    def test_output_is_scalar_and_identical_across_splits(self) -> None:
        expected_columns = [
            "unit_id",
            "cycle",
            "RUL",
            "operating_regime",
            "sensor_2",
        ]
        outputs = [
            finalize_preprocessing_output(self.frame(1, 10.0), ("sensor_2",)),
            finalize_preprocessing_output(self.frame(2, 20.0), ("sensor_2",)),
        ]

        self.assertEqual(outputs[0].columns, expected_columns)
        self.assertEqual(outputs[0].schema, outputs[1].schema)
        for output in outputs:
            self.assertNotIn("assembled_sensor_features", output.columns)
            self.assertNotIn("scaled_sensor_features", output.columns)
            self.assertNotIn("setting_1", output.columns)
            self.assertTrue(
                all(
                    isinstance(field.dataType, NumericType)
                    and not isinstance(field.dataType, VectorUDT)
                    for field in output.schema.fields
                )
            )

    def test_global_and_regime_scaling_have_the_same_output_contract(self) -> None:
        frame = self.spark.createDataFrame(
            [
                (1, 1, 2, 1, 10.0),
                (1, 2, 1, 1, 20.0),
                (2, 1, 2, 2, 30.0),
                (2, 2, 1, 2, 40.0),
            ],
            "unit_id int, cycle int, RUL int, regime int, sensor_2 double",
        )
        global_scaler = Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=["sensor_2"],
                    outputCol="assembled_sensor_features",
                ),
                StandardScaler(
                    inputCol="assembled_sensor_features",
                    outputCol="scaled_sensor_features",
                    withMean=True,
                    withStd=True,
                ),
            ]
        ).fit(frame)
        global_output = finalize_preprocessing_output(
            apply_global_sensor_scaler(frame, global_scaler, ("sensor_2",)),
            ("sensor_2",),
        )
        regime_output = finalize_preprocessing_output(
            apply_sensor_scaler(
                frame, fit_regime_sensor_scaler(frame, ("sensor_2",))
            ),
            ("sensor_2",),
        )

        self.assertEqual(global_output.schema, regime_output.schema)
        self.assertEqual(
            global_output.columns,
            ["unit_id", "cycle", "RUL", "operating_regime", "sensor_2"],
        )

class PreprocessingRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("cmapss-production-preprocessing-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_runner_persists_outputs_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            processed = root / "processed"
            _write_synthetic_subset(raw)

            with patch.dict(
                os.environ,
                {
                    "MLFLOW_TRACKING_URI": (root / "mlruns").as_uri(),
                    "CMAPSS_MLFLOW_EXPERIMENT": "test-cmapss-preprocessing",
                },
            ):
                first = run_subset_preprocessing(
                    self.spark, "FD001", raw, processed
                )
                second = run_subset_preprocessing(
                    self.spark, "FD001", raw, processed
                )
                configure_mlflow()
                loaded = load_preprocessing_state(first.run_id)
                loaded_model = load_preprocessing_model(first.run_id)

            self.assertNotEqual(first.run_id, second.run_id)
            self.assertEqual(first.train_row_count, second.train_row_count)
            self.assertEqual(first.train_row_count + first.validation_row_count, 12)
            self.assertEqual(first.test_row_count, 4)
            self.assertEqual(first.scaling_strategy, "global")
            self.assertEqual(first.regime_count, 1)
            self.assertEqual(first.feature_count, 1)
            self.assertEqual(
                self.spark.read.parquet(first.train_output_path).count(),
                first.train_row_count,
            )
            self.assertEqual(
                self.spark.read.parquet(first.validation_output_path).count(),
                first.validation_row_count,
            )
            self.assertEqual(
                self.spark.read.parquet(first.test_output_path).count(),
                first.test_row_count,
            )

            train_units = {
                row["unit_id"]
                for row in self.spark.read.parquet(first.train_output_path)
                .select("unit_id")
                .distinct()
                .collect()
            }
            validation_units = {
                row["unit_id"]
                for row in self.spark.read.parquet(first.validation_output_path)
                .select("unit_id")
                .distinct()
                .collect()
            }
            self.assertTrue(train_units.isdisjoint(validation_units))
            self.assertIsNone(loaded.regime_sensor_scaler)
            self.assertIsInstance(loaded_model, PipelineModel)


@pytest.mark.requires_data
class RealPreprocessingRunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("cmapss-real-production-preprocessing-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_real_fd001_and_fd002_outputs(self) -> None:
        raw = PROJECT_ROOT / "Data" / "CMAPSSData"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subset, strategy, regime_count, feature_count in (
                ("FD001", "global", 1, 15),
                ("FD002", "regime-aware", 6, 21),
            ):
                with self.subTest(subset=subset):
                    with patch.dict(
                        os.environ,
                        {
                            "MLFLOW_TRACKING_URI": (root / "mlruns").as_uri(),
                            "CMAPSS_MLFLOW_EXPERIMENT": "test-real-cmapss-preprocessing",
                        },
                    ):
                        result = run_subset_preprocessing(
                            self.spark,
                            subset,
                            raw,
                            root / "processed",
                        )
                        configure_mlflow()
                        loaded = load_preprocessing_state(result.run_id)
                    self.assertEqual(result.scaling_strategy, strategy)
                    self.assertEqual(result.regime_count, regime_count)
                    self.assertEqual(result.feature_count, feature_count)
                    self.assertTrue(result.run_id)
                    for path in (
                        result.train_output_path,
                        result.validation_output_path,
                        result.test_output_path,
                    ):
                        self.assertTrue(Path(path, "_SUCCESS").is_file())
                    train_output = self.spark.read.parquet(result.train_output_path)
                    validation_output = self.spark.read.parquet(
                        result.validation_output_path
                    )
                    retained_output_sensors = tuple(
                        column
                        for column in train_output.columns
                        if column.startswith("sensor_")
                    )
                    self.assertEqual(
                        loaded.retained_sensor_columns, retained_output_sensors
                    )
                    self.assertEqual(result.feature_count, len(retained_output_sensors))
                    self.assertEqual(train_output.schema, validation_output.schema)
                    self.assertEqual(
                        train_output.columns,
                        [
                            "unit_id",
                            "cycle",
                            "RUL",
                            "operating_regime",
                            *(
                                column
                                for column in train_output.columns
                                if column.startswith("sensor_")
                            ),
                        ],
                    )
                    self.assertTrue(
                        all(
                            isinstance(field.dataType, NumericType)
                            and not isinstance(field.dataType, VectorUDT)
                            for field in train_output.schema.fields
                        )
                    )
                    self.assertEqual(
                        result.train_row_count + result.validation_row_count,
                        self.spark.read.text(str(raw / f"train_{subset}.txt")).count(),
                    )
                    self.assertEqual(
                        result.test_row_count,
                        self.spark.read.text(str(raw / f"test_{subset}.txt")).count(),
                    )
                    if subset == "FD001":
                        self.assertEqual(
                            {
                                column
                                for column in train_output.columns
                                if column.startswith("sensor_")
                            },
                            {
                                "sensor_2",
                                "sensor_3",
                                "sensor_4",
                                "sensor_6",
                                "sensor_7",
                                "sensor_8",
                                "sensor_9",
                                "sensor_11",
                                "sensor_12",
                                "sensor_13",
                                "sensor_14",
                                "sensor_15",
                                "sensor_17",
                                "sensor_20",
                                "sensor_21",
                            },
                        )
                        self.assertIsNone(loaded.regime_sensor_scaler)
                        self.assertIsInstance(
                            load_preprocessing_model(result.run_id), PipelineModel
                        )
                    else:
                        self.assertIsInstance(
                            loaded.regime_sensor_scaler, RegimeSensorScaler
                        )


if __name__ == "__main__":
    unittest.main()
