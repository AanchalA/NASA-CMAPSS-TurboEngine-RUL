import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlflow
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.data_processing.artifacts import PreprocessingArtifacts
from src.tracking.mlflow_tracking import (configure_mlflow,
                                          load_preprocessing_state,
                                          log_preprocessing_state,
                                          load_training_feature_columns,
                                          load_training_model)

from src.tracking.spark_mlflow_tracking import (
    load_preprocessing_model,
    log_preprocessing_model,
)


class MlflowTrackingTests(unittest.TestCase):
    @patch("src.tracking.mlflow_tracking.mlflow_pyfunc.load_model")
    def test_training_model_loading_uses_the_run_reference(self, load_model) -> None:
        expected = object()
        load_model.return_value = expected

        loaded = load_training_model("training-run")

        self.assertIs(loaded, expected)
        load_model.assert_called_once_with("runs:/training-run/model")

    @patch("src.tracking.mlflow_tracking.mlflow_artifacts.load_dict")
    def test_training_feature_loading_uses_the_run_reference(self, load_dict) -> None:
        expected = ["sensor_2", "sensor_3"]
        load_dict.return_value = expected

        loaded = load_training_feature_columns("training-run")

        self.assertIs(loaded, expected)
        load_dict.assert_called_once_with("runs:/training-run/feature_names.json")

    def test_custom_state_round_trips_through_local_mlflow(self) -> None:
        artifacts = PreprocessingArtifacts(
            regime_mapping={(0.0, 0.0, 100.0): 1},
            dropped_feature_columns=("setting_3", "sensor_1"),
            retained_sensor_columns=("sensor_2",),
        )

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "MLFLOW_TRACKING_URI": Path(directory, "mlruns").as_uri(),
                "CMAPSS_MLFLOW_EXPERIMENT": "tracking-test",
            },
        ):
            configure_mlflow()
            with mlflow.start_run() as run:
                log_preprocessing_state(artifacts)

            loaded = load_preprocessing_state(run.info.run_id)

        self.assertEqual(loaded, artifacts)

    @patch("src.tracking.spark_mlflow_tracking.mlflow_spark.log_model")
    def test_spark_model_logging_uses_the_stable_artifact_name(self, log_model) -> None:
        model = object()

        log_preprocessing_model(model)

        self.assertEqual(log_model.call_args.args, (model, "preprocessing_pipeline"))
        self.assertEqual(
            log_model.call_args.kwargs["pip_requirements"],
            mlflow.spark.get_default_pip_requirements(),
        )
        self.assertNotIn("dfs_tmpdir", log_model.call_args.kwargs)

    @patch("src.tracking.spark_mlflow_tracking.mlflow_spark.load_model")
    def test_spark_model_loading_uses_the_run_reference(self, load_model) -> None:
        expected = object()
        load_model.return_value = expected

        loaded = load_preprocessing_model("run-123")

        self.assertIs(loaded, expected)
        load_model.assert_called_once_with("runs:/run-123/preprocessing_pipeline")


class SparkModelTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
            raise unittest.SkipTest("a Java runtime is required to start local Spark")
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("mlflow-spark-preprocessing-test")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_spark_preprocessing_model_is_loadable(self) -> None:
        train = self.spark.createDataFrame([(1.0,), (2.0,), (3.0,)], ["sensor_2"])
        model = Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=["sensor_2"], outputCol="assembled_sensor_features"
                ),
                StandardScaler(
                    inputCol="assembled_sensor_features",
                    outputCol="scaled_sensor_features",
                    withMean=True,
                    withStd=True,
                ),
            ]
        ).fit(train)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "MLFLOW_TRACKING_URI": Path(directory, "mlruns").as_uri(),
                "CMAPSS_MLFLOW_EXPERIMENT": "spark-tracking-test",
            },
        ):
            configure_mlflow()
            with mlflow.start_run() as run:
                log_preprocessing_model(model)

            loaded = load_preprocessing_model(run.info.run_id)

        self.assertIsInstance(loaded, PipelineModel)


if __name__ == "__main__":
    unittest.main()
