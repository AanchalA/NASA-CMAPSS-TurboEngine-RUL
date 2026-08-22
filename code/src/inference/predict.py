import json
from functools import cache
from typing import cast

import pandas as pd
from mlflow import MlflowClient
import mlflow.sklearn as mlflow_sklearn

from src.data_processing import add_pandas_temporal_features
from src.data_processing.constants import SUPPORTED_SUBSETS
from src.data_processing.scaler_state import RegimeSensorScaler
from src.tracking import configure_mlflow, load_preprocessing_state


@cache
def load_training_model(training_run_id):
    return mlflow_sklearn.load_model(f"runs:/{training_run_id}/model")


def process_observations(observations, preprocessing_run_id, feature_columns):

    if len({observation["unit_id"] for observation in observations}) != 1:
        raise ValueError("inference requires observations from exactly one engine")

    artifacts = load_preprocessing_state(preprocessing_run_id)

    trajectory = pd.DataFrame(observations).sort_values("cycle").reset_index(drop=True)
    regime_keys = zip(trajectory["setting_1"].round(0),
                      trajectory["setting_2"].round(2),
                      trajectory["setting_3"].round(0),
                      strict=True)

    try:
        regimes = [artifacts.regime_mapping[tuple(float(value) for value in key)] for key in regime_keys]
    except KeyError as error:
        raise ValueError(f"operating regime key unseen in training: {error.args[0]}") from error    

    for sensor in artifacts.retained_sensor_columns:
        
        if artifacts.global_sensor_scaler is not None:
            sensor_statistics = [artifacts.global_sensor_scaler.statistics[sensor]] * len(trajectory)
        else:
            regime_sensor_scaler = cast(RegimeSensorScaler, artifacts.regime_sensor_scaler)
            sensor_statistics = [regime_sensor_scaler.statistics[regime][sensor] for regime in regimes]

        means = pd.Series([statistics.mean for statistics in sensor_statistics])
        standard_deviations = pd.Series([statistics.std for statistics in sensor_statistics])

        trajectory[sensor] = (trajectory[sensor] - means).where(
            standard_deviations != 0.0, 0.0) / standard_deviations.where(standard_deviations != 0.0, 1.0)

    include_temporal_features = tuple(feature_columns) != artifacts.retained_sensor_columns
    if include_temporal_features:
        trajectory = add_pandas_temporal_features(trajectory, artifacts.retained_sensor_columns)

    return trajectory.iloc[[-1]].loc[:, feature_columns]


def predict_rul(subset_id, training_run_id, observations):
    
    normalized_subset = subset_id.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset_id}")

    configure_mlflow()
    
    training_run = MlflowClient().get_run(training_run_id)
    training_subset = training_run.data.params["subset"].upper()
    
    if training_subset != normalized_subset:
        raise ValueError(f"training run ({training_run_id}) subset {training_subset} does not match {normalized_subset}")

    preprocessing_run_id = training_run.data.params["preprocessing_run_id"]
    feature_columns = json.loads(training_run.data.params["feature_names"])
    
    model_input = process_observations(observations=observations,
                                       preprocessing_run_id=preprocessing_run_id,
                                       feature_columns=feature_columns)

    model = load_training_model(training_run_id)
    
    return float(model.predict(model_input)[0])
