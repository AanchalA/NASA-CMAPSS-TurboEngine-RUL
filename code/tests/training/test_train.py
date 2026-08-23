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

from src.training.tree_models import train_random_forest
from src.training.tree_models.run_tabular_training import run_training


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
            processed_data_dir = Path(directory) / "processed"
            preprocessing_run_id = "preprocessing-run"
            preprocessing_path = processed_data_dir / "FD002" / preprocessing_run_id
            preprocessing_path.mkdir(parents=True)
            columns = [
                "unit_id",
                "cycle",
                "RUL",
                "operating_regime",
                "sensor_2",
                "sensor_3",
                "sensor_2_lag_1",
                "sensor_3_ewma",
            ]
            pd.DataFrame(
                [
                    (1, 1, 30, 1, 0.0, 3.0, 0.0, 3.0),
                    (1, 2, 20, 1, 1.0, 2.0, 0.0, 2.8),
                    (2, 1, 10, 1, 2.0, 1.0, 1.0, 1.0),
                    (2, 2, 0, 1, 3.0, 0.0, 1.0, 0.8),
                ],
                columns=columns,
            ).to_parquet(preprocessing_path / "train")
            pd.DataFrame(
                [
                    (3, 1, 25, 1, 0.5, 2.5, 0.5, 2.5),
                    (3, 2, 5, 1, 2.5, 0.5, 2.0, 2.1),
                    (3, 3, 0, 1, 3.0, 0.0, 2.5, 1.5),
                ],
                columns=columns,
            ).to_parquet(preprocessing_path / "validation")

            run_id = train_random_forest(
                subset_id="fd002",
                preprocessing_run_id=preprocessing_run_id,
                processed_data_dir=processed_data_dir,
                parameters={"n_estimators": 3, "random_state": 42},
            )
            run = MlflowClient().get_run(run_id)

            self.assertEqual(run.data.params["model_type"], "RandomForestRegressor")
            self.assertEqual(run.data.params["subset"], "FD002")
            self.assertEqual(
                run.data.params["preprocessing_run_id"], preprocessing_run_id
            )
            self.assertEqual(run.data.params["feature_count"], "4")
            self.assertEqual(run.data.params["validation_protocol"], "pseudo_test_endpoint")
            self.assertEqual(run.data.params["validation_endpoint_count"], "1")
            self.assertEqual(run.data.tags["feature_set"], "temporal")
            self.assertNotIn("feature_names", run.data.params)
            self.assertEqual(
                mlflow.artifacts.load_dict(f"runs:/{run_id}/feature_names.json"),
                ["sensor_2", "sensor_3", "sensor_2_lag_1", "sensor_3_ewma"],
            )
            self.assertEqual(run.data.params["n_estimators"], "3")
            self.assertEqual(
                set(run.data.metrics),
                {
                    "rmse", "mae", "nasa_score", "bias", "late_prediction_rate",
                    "validation_row_rmse", "validation_row_mae",
                    "validation_row_nasa_score", "validation_row_bias",
                    "validation_row_late_prediction_rate",
                    "worst_positive_error", "worst_negative_error",
                    "largest_nasa_contribution", "top_3_nasa_contribution",
                },
            )
            diagnostics = mlflow.artifacts.load_text(
                f"runs:/{run_id}/validation_endpoint_predictions.csv"
            )
            self.assertIn("unit_id,cutoff_cycle,true_RUL,prediction,error,nasa_contribution", diagnostics)
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

        result = run_training(
            model_type="random_forest",
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
