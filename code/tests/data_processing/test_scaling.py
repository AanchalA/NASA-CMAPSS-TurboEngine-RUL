"""Tests for leakage-safe training-fitted sensor standardization."""

import math
import os
import shutil
import sys
import unittest
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.feature_engineering import (  
    drop_feature_columns,
    find_constant_feature_columns,
)
from src.data_processing.feature_engineering import (  
    assign_operating_regimes,
    fit_operating_regime_mapping,
)
from src.data_processing.scaler_state import RegimeSensorScaler
from src.data_processing.scaling import (  
    apply_global_sensor_scaler,
    apply_sensor_scaler,
    fit_regime_sensor_scaler,
)
from src.data_processing.data_loader import load_cmapss_raw, add_rul_target  
from src.data_processing.split import split_train_validation_by_unit  
from src.data_processing.validation import validate_cmapss_data  


class SensorScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cmapss-sensor-scaling-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def frame(
        self,
        values: list[tuple[int, float, float]],
        *,
        unit_id: int = 1,
    ) -> DataFrame:
        rows = [
            (
                unit_id,
                cycle,
                10 - cycle,
                regime,
                0.25,
                0.005,
                100.0,
                sensor_2,
                sensor_3,
            )
            for cycle, (regime, sensor_2, sensor_3) in enumerate(values, start=1)
        ]
        return self.spark.createDataFrame(
            rows,
            schema=(
                "unit_id int, cycle int, RUL int, regime int, setting_1 double, "
                "setting_2 double, setting_3 double, sensor_2 double, sensor_3 double"
            ),
        )

    def test_global_fit_application_and_training_moments(self) -> None:
        train = self.frame([(1, 10.0, 2.0), (1, 20.0, 4.0), (1, 30.0, 6.0)])
        scaler = Pipeline(stages=[
            VectorAssembler(inputCols=["sensor_2", "sensor_3"], outputCol="assembled_sensor_features"),
            StandardScaler(inputCol="assembled_sensor_features", outputCol="scaled_sensor_features",
                           withMean=True, withStd=True),
        ]).fit(train)

        scaled = apply_global_sensor_scaler(train, scaler, ["sensor_2", "sensor_3"])
        moments = scaled.agg(
            F.mean("sensor_2").alias("mean"),
            F.stddev_samp("sensor_2").alias("std"),
        ).first()
        self.assertIsNotNone(moments)
        assert moments is not None
        self.assertAlmostEqual(moments["mean"], 0.0, places=12)
        self.assertAlmostEqual(moments["std"], 1.0, places=12)

    def test_global_validation_uses_training_statistics(self) -> None:
        train = self.frame([(1, 10.0, 1.0), (1, 20.0, 2.0), (1, 30.0, 3.0)])
        validation = self.frame([(1, 100.0, 10.0), (1, 200.0, 20.0)], unit_id=2)
        original = validation.collect()

        scaler = Pipeline(stages=[
            VectorAssembler(inputCols=["sensor_2"], outputCol="assembled_sensor_features"),
            StandardScaler(inputCol="assembled_sensor_features", outputCol="scaled_sensor_features",
                           withMean=True, withStd=True),
        ]).fit(train)
        scaled = apply_global_sensor_scaler(validation, scaler, ["sensor_2"])
        actual = [row["sensor_2"] for row in scaled.orderBy("cycle").collect()]
        training_std = 10.0

        self.assertAlmostEqual(actual[0], (100.0 - 20.0) / training_std)
        self.assertAlmostEqual(actual[1], (200.0 - 20.0) / training_std)
        self.assertNotAlmostEqual(sum(actual) / len(actual), 0.0)
        self.assertEqual(validation.collect(), original)

    def test_global_transform_preserves_non_features_shape_and_is_deterministic(self) -> None:
        frame = self.frame([(1, 10.0, 2.0), (1, 20.0, 4.0), (1, 30.0, 6.0)])
        scaler = Pipeline(stages=[
            VectorAssembler(inputCols=["sensor_2"], outputCol="assembled_sensor_features"),
            StandardScaler(inputCol="assembled_sensor_features", outputCol="scaled_sensor_features",
                           withMean=True, withStd=True),
        ]).fit(frame)
        first = apply_global_sensor_scaler(frame, scaler, ["sensor_2"])
        second = apply_global_sensor_scaler(frame, scaler, ["sensor_2"])

        self.assertEqual(first.columns, frame.columns)
        self.assertEqual(first.count(), frame.count())
        self.assertEqual(first.orderBy("cycle").collect(), second.orderBy("cycle").collect())
        unchanged = [
            "unit_id",
            "cycle",
            "RUL",
            "regime",
            "setting_1",
            "setting_2",
            "setting_3",
            "sensor_3",
        ]
        self.assertEqual(
            first.select(*unchanged).orderBy("cycle").collect(),
            frame.select(*unchanged).orderBy("cycle").collect(),
        )

    def test_regime_fit_application_leakage_and_training_moments(self) -> None:
        train = self.frame(
            [
                (1, 10.0, 1.0),
                (1, 20.0, 2.0),
                (1, 30.0, 3.0),
                (2, 100.0, 4.0),
                (2, 110.0, 5.0),
                (2, 120.0, 6.0),
            ]
        )
        validation = self.frame([(1, 40.0, 8.0), (2, 130.0, 9.0)], unit_id=2)
        scaler = fit_regime_sensor_scaler(train, ["sensor_2"])

        self.assertIsInstance(scaler, RegimeSensorScaler)
        self.assertEqual(scaler.statistics[1]["sensor_2"].mean, 20.0)
        self.assertEqual(scaler.statistics[2]["sensor_2"].mean, 110.0)

        training_moments = (
            apply_sensor_scaler(train, scaler)
            .groupBy("regime")
            .agg(
                F.mean("sensor_2").alias("mean"),
                F.stddev_pop("sensor_2").alias("std"),
            )
            .collect()
        )
        for row in training_moments:
            self.assertAlmostEqual(row["mean"], 0.0, places=12)
            self.assertAlmostEqual(row["std"], 1.0, places=12)

        actual = [
            row["sensor_2"]
            for row in apply_sensor_scaler(validation, scaler).orderBy("cycle").collect()
        ]
        group_std = math.sqrt(200.0 / 3.0)
        self.assertAlmostEqual(actual[0], (40.0 - 20.0) / group_std)
        self.assertAlmostEqual(actual[1], (130.0 - 110.0) / group_std)
        self.assertNotAlmostEqual(sum(actual) / len(actual), 0.0)

    def test_regime_zero_variance_maps_to_zero(self) -> None:
        train = self.frame(
            [
                (1, 50.0, 1.0),
                (1, 50.0, 2.0),
                (1, 50.0, 3.0),
                (2, 10.0, 4.0),
                (2, 20.0, 5.0),
                (2, 30.0, 6.0),
            ]
        )
        validation = self.frame([(1, 999.0, 8.0), (2, 40.0, 9.0)], unit_id=2)
        scaler = fit_regime_sensor_scaler(train, ["sensor_2"])

        self.assertEqual(scaler.statistics[1]["sensor_2"].std, 0.0)
        train_values = apply_sensor_scaler(train, scaler)
        self.assertEqual(
            {row["sensor_2"] for row in train_values.filter("regime = 1").collect()},
            {0.0},
        )
        validation_values = apply_sensor_scaler(validation, scaler).orderBy("cycle").collect()
        self.assertEqual(validation_values[0]["sensor_2"], 0.0)
        self.assertAlmostEqual(
            validation_values[1]["sensor_2"],
            (40.0 - 20.0) / math.sqrt(200.0 / 3.0),
        )

    def test_real_subsets_follow_global_and_regime_strategies(self) -> None:
        modes = {"FD001": "global", "FD002": "regime", "FD003": "global", "FD004": "regime"}

        for subset, mode in modes.items():
            with self.subTest(subset=subset):
                raw = load_cmapss_raw(
                    self.spark,
                    PROJECT_ROOT / "Data" / "CMAPSSData" / f"train_{subset}.txt",
                )
                validate_cmapss_data(raw)
                enriched = add_rul_target(raw)
                train, validation = split_train_validation_by_unit(enriched)
                mapping = fit_operating_regime_mapping(train)
                train = assign_operating_regimes(train, mapping)
                validation = assign_operating_regimes(validation, mapping)
                constants = find_constant_feature_columns(train)
                train = drop_feature_columns(train, constants).cache()
                validation = drop_feature_columns(validation, constants).cache()
                sensors = [
                    f"sensor_{number}"
                    for number in range(1, 22)
                    if f"sensor_{number}" in train.columns
                ]
                train_rows = train.count()
                validation_rows = validation.count()
                train_engines = train.select("unit_id").distinct().count()
                validation_engines = validation.select("unit_id").distinct().count()

                if mode == "global":
                    scaler = Pipeline(stages=[
                        VectorAssembler(inputCols=sensors, outputCol="assembled_sensor_features"),
                        StandardScaler(inputCol="assembled_sensor_features",
                                       outputCol="scaled_sensor_features",
                                       withMean=True, withStd=True),
                    ]).fit(train)
                else:
                    scaler = fit_regime_sensor_scaler(train, sensors)
                if mode == "global":
                    scaled_train = apply_global_sensor_scaler(train, scaler, sensors).cache()
                    scaled_validation = apply_global_sensor_scaler(validation, scaler, sensors).cache()
                else:
                    scaled_train = apply_sensor_scaler(train, scaler).cache()
                    scaled_validation = apply_sensor_scaler(validation, scaler).cache()

                self.assertEqual(scaled_train.count(), train_rows)
                self.assertEqual(scaled_validation.count(), validation_rows)
                self.assertEqual(
                    scaled_train.select("unit_id").distinct().count(), train_engines
                )
                self.assertEqual(
                    scaled_validation.select("unit_id").distinct().count(),
                    validation_engines,
                )
                self.assertEqual(scaled_train.schema, scaled_validation.schema)
                self.assertEqual(
                    scaled_train.select("regime").distinct().count(), len(mapping)
                )

                grouping = ["regime"] if mode == "regime" else []
                standard_deviation = F.stddev_samp if mode == "global" else F.stddev_pop
                rows = scaled_train.groupBy(*grouping).agg(
                    *(
                        expression
                        for index, sensor in enumerate(sensors)
                        for expression in (
                            F.mean(sensor).alias(f"mean_{index}"),
                            standard_deviation(sensor).alias(f"std_{index}"),
                        )
                    )
                ).collect()
                zero_pairs = {
                    (regime, sensor)
                    for regime, statistics in scaler.statistics.items()
                    for sensor, values in statistics.items()
                    if values.std == 0.0
                } if isinstance(scaler, RegimeSensorScaler) else set()
                for row in rows:
                    regime = int(row["regime"]) if mode == "regime" else None
                    for index, sensor in enumerate(sensors):
                        self.assertAlmostEqual(row[f"mean_{index}"], 0.0, places=7)
                        expected_std = 0.0 if (regime, sensor) in zero_pairs else 1.0
                        self.assertAlmostEqual(row[f"std_{index}"], expected_std, places=7)

                scaled_train.unpersist()
                scaled_validation.unpersist()
                train.unpersist()
                validation.unpersist()


if __name__ == "__main__":
    unittest.main()
