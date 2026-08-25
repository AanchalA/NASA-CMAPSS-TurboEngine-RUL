import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd


from src.training.evaluation.test_evaluation import evaluate_test_data


class TestEvaluationTests(unittest.TestCase):
    @patch("src.training.evaluation.test_evaluation.load_training_feature_columns")
    @patch("src.training.evaluation.test_evaluation.load_training_model")
    @patch("src.training.evaluation.test_evaluation.MlflowClient")
    @patch("src.training.evaluation.test_evaluation.configure_mlflow")
    def test_uses_final_cycles_and_life_ratio(
        self,
        configure_mlflow,
        mlflow_client,
        load_training_model,
        load_training_feature_columns,
    ) -> None:
        del configure_mlflow
        client = mlflow_client.return_value
        client.get_run.return_value = SimpleNamespace(
            data=SimpleNamespace(
                params={
                    "subset": "FD004",
                    "preprocessing_run_id": "preprocessing-run",
                    "feature_names": '["sensor_2", "sensor_3"]',
                }
            )
        )
        model = MagicMock()
        model.predict.return_value = [0.7, 0.9]
        load_training_model.return_value = model
        load_training_feature_columns.return_value = ["sensor_2", "sensor_3"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed_data_dir = root / "processed"
            raw_data_dir = root / "raw"
            raw_data_dir.mkdir()
            (raw_data_dir / "RUL_FD004.txt").write_text(
                "25\n5\n", encoding="utf-8"
            )
            preprocessing_path = processed_data_dir / "FD004" / "preprocessing-run"
            preprocessing_path.mkdir(parents=True)
            pd.DataFrame(
                [
                    (1, 1, 0.5, 0.1, 0.2),
                    (1, 2, 0.6, 0.3, 0.4),
                    (2, 1, 0.7, 0.5, 0.6),
                    (2, 3, 0.8, 0.7, 0.8),
                ],
                columns=["unit_id", "cycle", "life_ratio", "sensor_2", "sensor_3"],
            ).to_parquet(preprocessing_path / "test")

            metrics = evaluate_test_data(
                subset_id="fd004",
                training_run_id="training-run",
                processed_data_dir=processed_data_dir,
                raw_data_dir=raw_data_dir,
            )

        X_test = model.predict.call_args.args[0]
        self.assertEqual(X_test.values.tolist(), [[0.3, 0.4], [0.7, 0.8]])
        expected_errors = [2 * (1 / 0.7 - 1) - 2 * (1 / 0.6 - 1),
                           3 * (1 / 0.9 - 1) - 3 * (1 / 0.8 - 1)]
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(sum(error ** 2 for error in expected_errors) / 2))
        self.assertAlmostEqual(metrics["mae"], sum(abs(error) for error in expected_errors) / 2)
        client.log_metric.assert_has_calls(
            [
                call("training-run", "test_rmse", metrics["rmse"]),
                call("training-run", "test_mae", metrics["mae"]),
                call("training-run", "test_nasa_score", metrics["nasa_score"]),
            ]
        )
        diagnostics = client.log_text.call_args.args[1]
        self.assertIn("unit_id,cutoff_cycle,true_RUL,prediction,error,nasa_contribution", diagnostics)
        self.assertEqual(client.log_text.call_args.args[2], "test_predictions.csv")
        load_training_model.assert_called_once_with("training-run")
        load_training_feature_columns.assert_called_once_with("training-run")


if __name__ == "__main__":
    unittest.main()
