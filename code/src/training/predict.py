import pandas as pd
from functools import cache
from mlflow import MlflowClient
import mlflow.sklearn as mlflow_sklearn

from src.data_processing.constants import SUPPORTED_SUBSETS
from src.tracking import configure_mlflow, load_preprocessing_state


@cache
def load_training_model(training_run_id):
    return mlflow_sklearn.load_model(f"runs:/{training_run_id}/model")


def process_observations(observations, preprocessing_run_id):

    if len({observation["unit_id"] for observation in observations}) != 1:
        raise ValueError("inference requires observations from exactly one engine")

    artifacts = load_preprocessing_state(preprocessing_run_id)
    latest = max(observations, key=lambda observation: observation["cycle"])

    regime_key = (float(round(latest["setting_1"], 0)),
                  float(round(latest["setting_2"], 2)),
                  float(round(latest["setting_3"], 0)))

    try:
        regime = artifacts.regime_mapping[regime_key]
    except KeyError as error:
        raise ValueError(f"operating regime key unseen in training: {regime_key}") from error

    scaler = artifacts.global_sensor_scaler

    if scaler is None and artifacts.regime_sensor_scaler is not None:
        scaler = artifacts.regime_sensor_scaler
        statistics = scaler.statistics[regime]

    elif scaler is not None:
        statistics = scaler.statistics

    else:
        raise ValueError("preprocessing run does not contain online global-scaler statistics. rerun needed for preprocessing and training")

    transformed = {sensor: 0.0 if statistics[sensor].std == 0.0 
                   else (latest[sensor] - statistics[sensor].mean) / statistics[sensor].std 
                   for sensor in artifacts.retained_sensor_columns}
    
    return pd.DataFrame([transformed], columns=artifacts.retained_sensor_columns)


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
    
    model_input = process_observations(observations=observations, preprocessing_run_id=preprocessing_run_id)

    model = load_training_model(training_run_id)
    
    return float(model.predict(model_input)[0])
