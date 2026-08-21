from functools import cache

import mlflow.spark as mlflow_spark

from src.tracking.mlflow_tracking import PREPROCESSING_PIPELINE_ARTIFACT


def log_preprocessing_model(model):
    mlflow_spark.log_model(model, PREPROCESSING_PIPELINE_ARTIFACT, 
                           pip_requirements=mlflow_spark.get_default_pip_requirements())


@cache
def load_preprocessing_model(run_id):
    return mlflow_spark.load_model(f"runs:/{run_id}/{PREPROCESSING_PIPELINE_ARTIFACT}")
