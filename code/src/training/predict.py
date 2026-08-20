import mlflow.sklearn as mlflow_sklearn
from mlflow import MlflowClient
from pyspark.sql import functions as F

from src.data_processing import validate_cmapss_data, SUPPORTED_SUBSETS
from src.data_processing.preprocessing import transform_with_artifacts
from src.tracking import configure_mlflow, load_preprocessing_model, load_preprocessing_state


def process_raw_trajectory(subset_id, raw_trajectory, preprocessing_run_id):
    
    validate_cmapss_data(raw_trajectory)
    if raw_trajectory.select("unit_id").distinct().count() != 1:
        raise ValueError("inference requires exactly one engine trajectory")
    
    artifacts = load_preprocessing_state(preprocessing_run_id)
    
    sensor_scaler = (load_preprocessing_model(preprocessing_run_id) 
                     if subset_id.upper() in ["FD001", "FD003"] else artifacts.regime_sensor_scaler
    )

    transformed = transform_with_artifacts(raw_trajectory, artifacts, sensor_scaler)
    feature_columns = list(artifacts.retained_sensor_columns)
    latest_observation = transformed.orderBy(F.desc("cycle")).limit(1)
    model_input = latest_observation.select(*feature_columns).toPandas()

    return model_input


def predict_rul(subset_id, training_run_id, raw_trajectory):
    
    normalized_subset = subset_id.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset_id}")

    configure_mlflow()
    
    training_run = MlflowClient().get_run(training_run_id)
    training_subset = training_run.data.params["subset"].upper()
    
    if training_subset != normalized_subset:
        raise ValueError(f"training run ({training_run_id}) subset {training_subset} does not match {normalized_subset}")

    preprocessing_run_id = training_run.data.params["preprocessing_run_id"]
    
    model_input = process_raw_trajectory(subset_id=normalized_subset,
                                         raw_trajectory=raw_trajectory,
                                         preprocessing_run_id=preprocessing_run_id)

    model = mlflow_sklearn.load_model(f"runs:/{training_run_id}/model")
    
    return float(model.predict(model_input)[0])
