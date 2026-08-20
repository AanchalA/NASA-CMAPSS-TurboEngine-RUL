from os import getenv

from mlflow import MlflowClient

from src.data_processing.schema import SUPPORTED_SUBSETS
from src.tracking.mlflow_tracking import configure_mlflow


def set_experiment_name(experiment_type):
    if experiment_type == "preprocessing":
        return getenv("CMAPSS_MLFLOW_EXPERIMENT", "cmapss-preprocessing")
    
    if experiment_type == "training":
        return getenv("CMAPSS_MLFLOW_EXPERIMENT", "cmapss-training")
    
    raise ValueError(f"unsupported experiment type: {experiment_type}")
    


def fetch_run_id(subset_id, model_type=None, experiment_type="preprocessing"):
    
    normalized_subset = subset_id.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset_id}")

    configure_mlflow()
    experiment_name = set_experiment_name(experiment_type)
    
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)    
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    
    filters = ["attributes.status = 'FINISHED'", 
               f"params.subset = '{normalized_subset}'"]

    if model_type is not None:
        filters.append(f"params.model_type = '{model_type}'")


    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string=" AND ".join(filters),
                              order_by=["start_time DESC"],
                              max_results=1)
    
    if not runs:
        raise ValueError(f"no successful {experiment_type} run found for {normalized_subset}")

    return runs[0].info.run_id
