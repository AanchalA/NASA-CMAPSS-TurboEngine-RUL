from os import environ, getenv

import mlflow
import mlflow.spark as mlflow_spark
from mlflow import artifacts as mlflow_artifacts

from src.data_processing.artifacts import (
    preprocessing_artifacts_data,
    preprocessing_artifacts_from_data,
)


PREPROCESSING_PIPELINE_ARTIFACT = "preprocessing_pipeline"
PREPROCESSING_STATE_ARTIFACT = "preprocessing_state.json"


def configure_mlflow():
    environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(getenv("CMAPSS_MLFLOW_EXPERIMENT", "cmapss-preprocessing"))

def log_preprocessing_model(model):
    mlflow_spark.log_model(model, PREPROCESSING_PIPELINE_ARTIFACT, 
                           pip_requirements=mlflow_spark.get_default_pip_requirements())

def log_preprocessing_state(artifacts):
    mlflow.log_dict(preprocessing_artifacts_data(artifacts), PREPROCESSING_STATE_ARTIFACT)

def load_preprocessing_model(run_id):
    return mlflow_spark.load_model(f"runs:/{run_id}/{PREPROCESSING_PIPELINE_ARTIFACT}")

def load_preprocessing_state(run_id):
    data = mlflow_artifacts.load_dict(f"runs:/{run_id}/{PREPROCESSING_STATE_ARTIFACT}")
    return preprocessing_artifacts_from_data(data)
