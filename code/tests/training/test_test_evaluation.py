"""Tests for official C-MAPSS test-set evaluation."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd


from src.training.test_evaluation import evaluate_test_data


class TestEvaluationTests(unittest.TestCase):
    @patch("src.training.test_evaluation.mlflow_sklearn.load_model")
    @patch("src.training.test_evaluation.MlflowClient")
    @patch("src.training.test_evaluation.configure_mlflow")
    def test_uses_final_cycles_and_official_rul(
        self,
        configure_mlflow,
        mlflow_client,
        load_model,
    ) -> None:
        del configure_mlflow
        client = mlflow_client.return_value
        client.get_run.return_value = SimpleNamespace(
            data=SimpleNamespace(
                params={
                    "subset": "FD004",
                    "feature_names": '["sensor_2", "sensor_3"]',
                }
            )
        )
        model = MagicMock()
        model.predict.return_value = [20.0, 10.0]
        load_model.return_value = model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed_data_dir = root / "processed"
            raw_data_dir = root / "raw"
            raw_data_dir.mkdir()
            (raw_data_dir / "RUL_FD004.txt").write_text(
                "25\n5\n", encoding="utf-8"
            )
            (processed_data_dir / "FD004").mkdir(parents=True)
            pd.DataFrame(
                [
                    (1, 1, 999, 0.1, 0.2),
                    (1, 2, 999, 0.3, 0.4),
                    (2, 1, 999, 0.5, 0.6),
                    (2, 3, 999, 0.7, 0.8),
                ],
                columns=["unit_id", "cycle", "RUL", "sensor_2", "sensor_3"],
            ).to_parquet(processed_data_dir / "FD004" / "test")

            metrics = evaluate_test_data(
                subset_id="fd004",
                training_run_id="training-run",
                processed_data_dir=processed_data_dir,
                raw_data_dir=raw_data_dir,
            )

        X_test = model.predict.call_args.args[0]
        self.assertEqual(X_test.values.tolist(), [[0.3, 0.4], [0.7, 0.8]])
        self.assertAlmostEqual(metrics["rmse"], 5.0)
        self.assertAlmostEqual(metrics["mae"], 5.0)
        self.assertAlmostEqual(
            metrics["nasa_score"], math.expm1(5 / 13) + math.expm1(5 / 10)
        )
        client.log_metric.assert_has_calls(
            [
                call("training-run", "test_rmse", metrics["rmse"]),
                call("training-run", "test_mae", metrics["mae"]),
                call("training-run", "test_nasa_score", metrics["nasa_score"]),
            ]
        )
        load_model.assert_called_once_with("runs:/training-run/model")


if __name__ == "__main__":
    unittest.main()
