import json
import pandas as pd
from os import getenv
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.ensemble import RandomForestRegressor

from src.data_processing.schema import SUPPORTED_SUBSETS
from src.training.model_evaluation_metrics import evaluate_predictions
from src.tracking import configure_mlflow, load_preprocessing_state


def train_random_forest(subset_id, preprocessing_run_id, processed_data_dir, parameters=None):
    normalized_subset = subset_id.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset_id}")

    configure_mlflow()
    
    artifacts = load_preprocessing_state(preprocessing_run_id)
    feature_columns = list(artifacts.retained_sensor_columns)
    subset_path = Path(processed_data_dir) / normalized_subset

    train_dataframe = pd.read_parquet(subset_path / "train")
    validation_dataframe = pd.read_parquet(subset_path / "validation")
     
    X_train, y_train = train_dataframe.loc[:, feature_columns], train_dataframe.loc[:, "RUL"]
    X_validation, y_validation = validation_dataframe.loc[:, feature_columns], validation_dataframe.loc[:, "RUL"]

    model = RandomForestRegressor(**(parameters or {}))
    
    mlflow.set_experiment(getenv("CMAPSS_MLFLOW_TRAINING_EXPERIMENT", "cmapss-training"))
    
    with mlflow.start_run(run_name=f"{normalized_subset}-random-forest") as run:
        
        model.fit(X_train, y_train)
        metrics = evaluate_predictions(y_validation, model.predict(X_validation))

        mlflow.log_params(model.get_params())
        mlflow.log_params({"model_type": type(model).__name__,
                           "subset": normalized_subset,
                           "preprocessing_run_id": preprocessing_run_id,
                           "feature_count": len(feature_columns),
                           "feature_names": json.dumps(feature_columns)})
        
        mlflow.log_metrics(metrics)
        mlflow_sklearn.log_model(model, name="model",
                                 pip_requirements=mlflow_sklearn.get_default_pip_requirements())

    return run.info.run_id
