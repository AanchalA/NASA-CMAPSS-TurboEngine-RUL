"""Tests for Random Forest inference."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.data_processing.artifacts import PreprocessingArtifacts
from src.data_processing.scaler_state import GlobalSensorScaler, SensorStatistics
from src.inference.predict import predict_rul, process_observations


class PredictionTests(unittest.TestCase):
    def test_processes_engine_trajectory_with_temporal_training_schema(self) -> None:
        artifacts = PreprocessingArtifacts(
            regime_mapping={(0.0, 0.0, 100.0): 1},
            dropped_feature_columns=(),
            retained_sensor_columns=("sensor_2", "sensor_3"),
            global_sensor_scaler=GlobalSensorScaler(
                statistics={
                    "sensor_2": SensorStatistics(mean=10.0, std=2.0),
                    "sensor_3": SensorStatistics(mean=100.0, std=10.0),
                }
            ),
        )
        observations = [
            {"unit_id": 7, "cycle": 2, "setting_1": 0.0, "setting_2": 0.0,
             "setting_3": 100.0, "sensor_2": 14.0, "sensor_3": 120.0},
            {"unit_id": 7, "cycle": 1, "setting_1": 0.0, "setting_2": 0.0,
             "setting_3": 100.0, "sensor_2": 10.0, "sensor_3": 100.0},
        ]

        with patch("src.training.predict.load_preprocessing_state", return_value=artifacts):
            model_input = process_observations(
                observations,
                "preprocessing-run",
                [
                    "sensor_2",
                    "sensor_3",
                    "sensor_2_lag_1",
                    "sensor_2_diff_1",
                    "sensor_3_rolling_mean_5",
                    "sensor_3_rolling_std_5",
                    "sensor_2_rolling_slope_5",
                    "sensor_3_ewma",
                ],
            )

        self.assertEqual(
            list(model_input.columns),
            [
                "sensor_2",
                "sensor_3",
                "sensor_2_lag_1",
                "sensor_2_diff_1",
                "sensor_3_rolling_mean_5",
                "sensor_3_rolling_std_5",
                "sensor_2_rolling_slope_5",
                "sensor_3_ewma",
            ],
        )
        values = model_input.iloc[0]
        self.assertEqual(values["sensor_2"], 2.0)
        self.assertEqual(values["sensor_3"], 2.0)
        self.assertEqual(values["sensor_2_lag_1"], 0.0)
        self.assertEqual(values["sensor_2_diff_1"], 2.0)
        self.assertEqual(values["sensor_3_rolling_mean_5"], 1.0)
        self.assertAlmostEqual(values["sensor_3_rolling_std_5"], math.sqrt(2.0))
        self.assertEqual(values["sensor_2_rolling_slope_5"], 2.0)
        self.assertAlmostEqual(values["sensor_3_ewma"], 0.4)

    @patch("src.training.predict.load_training_model")
    @patch("src.training.predict.load_preprocessing_state")
    @patch("src.training.predict.MlflowClient")
    @patch("src.training.predict.configure_mlflow")
    def test_predicts_with_matching_training_run(
        self, configure_mlflow, mlflow_client, load_preprocessing_state, load_training_model
    ) -> None:
        del configure_mlflow
        mlflow_client.return_value.get_run.return_value = SimpleNamespace(
            data=SimpleNamespace(params={
                "subset": "FD001",
                "preprocessing_run_id": "preprocessing-run",
                "feature_names": '["sensor_2"]',
            })
        )
        load_preprocessing_state.return_value = PreprocessingArtifacts(
            regime_mapping={(0.0, 0.0, 100.0): 1},
            dropped_feature_columns=(),
            retained_sensor_columns=("sensor_2",),
            global_sensor_scaler=GlobalSensorScaler(
                statistics={"sensor_2": SensorStatistics(mean=10.0, std=2.0)}
            ),
        )
        model = MagicMock()
        model.predict.return_value = [42.5]
        load_training_model.return_value = model

        prediction = predict_rul(
            "fd001", "training-run",
            [{"unit_id": 7, "cycle": 1, "setting_1": 0.0, "setting_2": 0.0,
              "setting_3": 100.0, "sensor_2": 14.0}],
        )

        self.assertEqual(prediction, 42.5)
        load_training_model.assert_called_once_with("training-run")


if __name__ == "__main__":
    unittest.main()
