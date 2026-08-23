from functools import cache
from os import environ, getenv

import mlflow
import mlflow.pyfunc as mlflow_pyfunc
import mlflow.artifacts as mlflow_artifacts

from src.data_processing.artifacts import preprocessing_artifacts_data, preprocessing_artifacts_from_data


PREPROCESSING_PIPELINE_ARTIFACT = "preprocessing_pipeline"
PREPROCESSING_STATE_ARTIFACT = "preprocessing_state.json"


def configure_mlflow():
    environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(getenv("CMAPSS_MLFLOW_EXPERIMENT", "cmapss-preprocessing"))


def log_preprocessing_state(artifacts):
    mlflow.log_dict(preprocessing_artifacts_data(artifacts), PREPROCESSING_STATE_ARTIFACT)


@cache
def load_preprocessing_state(run_id):
    data = mlflow_artifacts.load_dict(f"runs:/{run_id}/{PREPROCESSING_STATE_ARTIFACT}")
    return preprocessing_artifacts_from_data(data)


@cache
def load_training_model(training_run_id):
    return mlflow_pyfunc.load_model(f"runs:/{training_run_id}/model")


def load_training_feature_columns(training_run_id):
    return mlflow_artifacts.load_dict(f"runs:/{training_run_id}/feature_names.json")
