"""Tests for custom preprocessing-state serialization."""

import os
import shutil
import sys
import unittest
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.artifacts import (  
    PreprocessingArtifacts,
    preprocessing_artifacts_data,
    preprocessing_artifacts_from_data,
)
from src.data_processing.feature_engineering import (  
    drop_feature_columns,
    find_constant_feature_columns,
)
from src.data_processing.feature_engineering import (  
    assign_operating_regimes,
    fit_operating_regime_mapping,
)
from src.data_processing.scaler_state import RegimeSensorScaler, SensorStatistics
from src.data_processing.scaling import (  
    apply_sensor_scaler,
    fit_regime_sensor_scaler,
)
from src.data_processing.data_loader import load_cmapss_raw, add_rul_target    
from src.data_processing.split import split_train_validation_by_unit  
from src.data_processing.validation import validate_cmapss_data  


def regime_artifact() -> PreprocessingArtifacts:
    statistics = {
        1: {
            "sensor_2": SensorStatistics(mean=10.0, std=2.0),
            "sensor_3": SensorStatistics(mean=50.0, std=0.0),
        },
        2: {
            "sensor_2": SensorStatistics(mean=20.0, std=4.0),
            "sensor_3": SensorStatistics(mean=60.0, std=3.0),
        },
        3: {
            "sensor_2": SensorStatistics(mean=30.0, std=6.0),
            "sensor_3": SensorStatistics(mean=70.0, std=0.0),
        },
    }
    return PreprocessingArtifacts(
        regime_mapping={
            (0.0, 0.0, 100.0): 1,
            (10.0, 0.25, 100.0): 2,
            (20.0, 0.50, 100.0): 3,
        },
        dropped_feature_columns=(),
        retained_sensor_columns=("sensor_2", "sensor_3"),
        regime_sensor_scaler=RegimeSensorScaler(
            statistics=statistics,
        ),
    )


class ArtifactSerializationTests(unittest.TestCase):
    def round_trip(self, artifacts: PreprocessingArtifacts) -> PreprocessingArtifacts:
        return preprocessing_artifacts_from_data(
            preprocessing_artifacts_data(artifacts)
        )

    def test_regime_scaler_round_trip_preserves_zero_variance_statistics(self) -> None:
        artifacts = regime_artifact()
        loaded = self.round_trip(artifacts)

        self.assertEqual(loaded, artifacts)
        self.assertIsInstance(loaded.regime_sensor_scaler, RegimeSensorScaler)
        assert isinstance(loaded.regime_sensor_scaler, RegimeSensorScaler)
        self.assertEqual(loaded.regime_sensor_scaler.statistics[1]["sensor_3"].std, 0.0)
        self.assertEqual(loaded.regime_sensor_scaler.statistics[3]["sensor_3"].std, 0.0)

    def test_global_custom_state_has_no_regime_scaler(self) -> None:
        artifacts = PreprocessingArtifacts(
            regime_mapping={(0.0, 0.0, 100.0): 1},
            dropped_feature_columns=("setting_3", "sensor_1"),
            retained_sensor_columns=("sensor_2",),
        )

        loaded = self.round_trip(artifacts)

        self.assertEqual(loaded, artifacts)
        self.assertIsNone(loaded.regime_sensor_scaler)

    def test_serialized_state_uses_explicit_records(self) -> None:
        document = preprocessing_artifacts_data(regime_artifact())

        self.assertIsInstance(document["regime_mapping"], list)
        self.assertIsInstance(document["regime_sensor_scaler"], list)
        self.assertEqual(document["retained_sensor_columns"], ["sensor_2", "sensor_3"])
        self.assertNotIn("sensor_scaler", document)

    def test_missing_regime_setting_raises_key_error(self) -> None:
        document = preprocessing_artifacts_data(regime_artifact())
        document["regime_mapping"][0].pop("setting_2")

        with self.assertRaisesRegex(KeyError, "setting_2"):
            preprocessing_artifacts_from_data(document)

    def test_retained_columns_must_be_canonical_sensors(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-canonical sensors.*setting_1"):
            PreprocessingArtifacts(
                regime_mapping={(0.0, 0.0, 100.0): 1},
                dropped_feature_columns=(),
                retained_sensor_columns=("setting_1", "sensor_2"),
            )

class RealArtifactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-artifact-integration-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    @staticmethod
    def transform(df: DataFrame, artifacts: PreprocessingArtifacts) -> DataFrame:
        assigned = assign_operating_regimes(df, artifacts.regime_mapping)
        selected = drop_feature_columns(assigned, artifacts.dropped_feature_columns)
        return apply_sensor_scaler(selected, artifacts.regime_sensor_scaler)

    def test_real_fd002_artifact_reproduces_output(self) -> None:
        raw = load_cmapss_raw(
            self.spark,
            PROJECT_ROOT / "Data" / "CMAPSSData" / "train_FD002.txt",
        )
        validate_cmapss_data(raw)
        enriched = add_rul_target(raw)
        train, validation = split_train_validation_by_unit(enriched)
        mapping = fit_operating_regime_mapping(train)
        assigned_train = assign_operating_regimes(train, mapping)
        constants = find_constant_feature_columns(assigned_train)
        selected_train = drop_feature_columns(assigned_train, constants).cache()
        sensors = tuple(
            f"sensor_{number}"
            for number in range(1, 22)
            if f"sensor_{number}" in selected_train.columns
        )
        scaler = fit_regime_sensor_scaler(selected_train, sensors)
        artifacts = PreprocessingArtifacts(
            regime_mapping=mapping,
            dropped_feature_columns=tuple(constants),
            retained_sensor_columns=sensors,
            regime_sensor_scaler=scaler,
        )
        loaded = preprocessing_artifacts_from_data(
            preprocessing_artifacts_data(artifacts)
        )

        sample = validation.orderBy("unit_id", "cycle").limit(25)
        in_memory_rows = self.transform(sample, artifacts).collect()
        loaded_rows = self.transform(sample, loaded).collect()

        self.assertEqual(loaded, artifacts)
        self.assertEqual(loaded_rows, in_memory_rows)
        self.assertEqual(len(loaded.regime_mapping), 6)
        self.assertEqual(len(loaded.regime_sensor_scaler.sensor_columns), 21)
        self.assertIsInstance(loaded.regime_sensor_scaler, RegimeSensorScaler)
        self.assertTrue(
            any(
                values.std == 0.0
                for statistics in loaded.regime_sensor_scaler.statistics.values()
                for values in statistics.values()
            )
        )
        selected_train.unpersist()


if __name__ == "__main__":
    unittest.main()
