from os import getenv

from mlflow import MlflowClient

from src.data_processing.schema import SUPPORTED_SUBSETS
from src.tracking.mlflow_tracking import configure_mlflow


def fetch_preprocessing_run_id(subset_id):
    normalized_subset = subset_id.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset_id}")

    configure_mlflow()
    experiment_name = getenv("CMAPSS_MLFLOW_EXPERIMENT", "cmapss-preprocessing")
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string=("attributes.status = 'FINISHED' AND params.subset = '{normalized_subset}'"),
                              order_by=["start_time DESC"],
                              max_results=1)
    
    if not runs:
        raise ValueError(f"no successful preprocessing run found for {normalized_subset}")

    return runs[0].info.run_id
