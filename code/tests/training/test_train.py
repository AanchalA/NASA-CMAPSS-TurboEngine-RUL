"""Tests for the Random Forest training flow."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor


from src.data_processing.artifacts import PreprocessingArtifacts
from src.tracking.mlflow_tracking import configure_mlflow, log_preprocessing_state
from src.training.run_training import run_random_forest_baseline
from src.training.train import train_random_forest


class RandomForestTrainingTests(unittest.TestCase):
    def test_trains_evaluates_and_logs_subset_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "MLFLOW_TRACKING_URI": (Path(directory) / "mlruns").as_uri(),
                "CMAPSS_MLFLOW_EXPERIMENT": "preprocessing-test",
                "CMAPSS_MLFLOW_TRAINING_EXPERIMENT": "training-test",
            },
        ):
            configure_mlflow()
            with mlflow.start_run() as preprocessing_run:
                log_preprocessing_state(
                    PreprocessingArtifacts(
                        regime_mapping={(0.0, 0.0, 100.0): 1},
                        dropped_feature_columns=("sensor_1",),
                        retained_sensor_columns=("sensor_2", "sensor_3"),
                    )
                )

            processed_data_dir = Path(directory) / "processed"
            (processed_data_dir / "FD002").mkdir(parents=True)
            columns = ["unit_id", "cycle", "RUL", "sensor_2", "sensor_3"]
            pd.DataFrame(
                [
                    (1, 1, 30, 0.0, 3.0),
                    (1, 2, 20, 1.0, 2.0),
                    (2, 1, 10, 2.0, 1.0),
                    (2, 2, 0, 3.0, 0.0),
                ],
                columns=columns,
            ).to_parquet(processed_data_dir / "FD002" / "train")
            pd.DataFrame(
                [(3, 1, 25, 0.5, 2.5), (3, 2, 5, 2.5, 0.5)], columns=columns
            ).to_parquet(processed_data_dir / "FD002" / "validation")

            run_id = train_random_forest(
                subset_id="fd002",
                preprocessing_run_id=preprocessing_run.info.run_id,
                processed_data_dir=processed_data_dir,
                parameters={"n_estimators": 3, "random_state": 42},
            )
            run = MlflowClient().get_run(run_id)

            self.assertEqual(run.data.params["model_type"], "RandomForestRegressor")
            self.assertEqual(run.data.params["subset"], "FD002")
            self.assertEqual(
                run.data.params["preprocessing_run_id"], preprocessing_run.info.run_id
            )
            self.assertEqual(run.data.params["feature_count"], "2")
            self.assertEqual(
                json.loads(run.data.params["feature_names"]),
                ["sensor_2", "sensor_3"],
            )
            self.assertEqual(run.data.params["n_estimators"], "3")
            self.assertEqual(
                set(run.data.metrics),
                {"rmse", "mae", "nasa_score", "bias", "late_prediction_rate"},
            )
            self.assertIsInstance(
                mlflow.sklearn.load_model(f"runs:/{run_id}/model"),
                RandomForestRegressor,
            )

    @patch("src.training.run_training.evaluate_test_data")
    @patch("src.training.run_training.train_random_forest")
    def test_runs_training_then_official_test_evaluation(
        self,
        train_random_forest_mock,
        evaluate_test_mock,
    ) -> None:
        train_random_forest_mock.return_value = "training-run"
        evaluate_test_mock.return_value = {
            "rmse": 1.0,
            "mae": 0.5,
            "nasa_score": 2.0,
        }
        parameters = {"n_estimators": 100, "random_state": 42}

        result = run_random_forest_baseline(
            subset_id="fd003",
            preprocessing_run_id="preprocessing-run",
            processed_data_dir="processed",
            raw_data_dir="raw",
            parameters=parameters,
        )

        self.assertEqual(
            result,
            {
                "subset_id": "FD003",
                "preprocessing_run_id": "preprocessing-run",
                "training_run_id": "training-run",
                "test_metrics": {
                    "rmse": 1.0,
                    "mae": 0.5,
                    "nasa_score": 2.0,
                },
            },
        )
        self.assertEqual(
            train_random_forest_mock.call_args,
            call(
                subset_id="fd003",
                preprocessing_run_id="preprocessing-run",
                processed_data_dir="processed",
                parameters=parameters,
            ),
        )
        self.assertEqual(
            evaluate_test_mock.call_args,
            call(
                subset_id="fd003",
                training_run_id="training-run",
                processed_data_dir="processed",
                raw_data_dir="raw",
            ),
        )


if __name__ == "__main__":
    unittest.main()
