import json
from os import getenv
from pathlib import Path

import mlflow
import mlflow.lightgbm as mlflow_lightgbm
import mlflow.sklearn as mlflow_sklearn
import mlflow.xgboost as mlflow_xgboost
import numpy as np
import pandas as pd

from src.data_processing import select_tabular_model_inputs
from src.tracking import configure_mlflow
from src.training.evaluation import (evaluate_predictions, 
                                     evaluate_prediction_diagnostics,
                                     prepare_pseudo_test_validation)



def train_tabular_model( model, model_name, subset_id, preprocessing_run_id, processed_data_dir, rul_cap=None):
    
    normalized_subset = subset_id.upper()    
    
    configure_mlflow()
    
    subset_path = Path(processed_data_dir) / normalized_subset / preprocessing_run_id
    train_dataframe = pd.read_parquet(subset_path / "train")
    validation_dataframe = pd.read_parquet(subset_path / "validation")

    X_train, y_train = select_tabular_model_inputs(train_dataframe)
    
    feature_columns = list(X_train.columns)
    feature_set = "temporal" if any(marker in column for column in feature_columns 
                                    for marker in ["_lag_", "_diff_", "_rolling_", "_ewma"]) else "baseline"
    
    if rul_cap is not None:
        y_train = np.minimum(y_train, rul_cap)

    X_validation_rows, y_validation_rows = select_tabular_model_inputs(validation_dataframe, feature_columns)    
    X_validation, y_validation, validation_metadata = prepare_pseudo_test_validation(validation_dataframe, feature_columns)
    
    target_name = "raw_rul" if rul_cap is None else f"capped_rul_{rul_cap:g}"

    mlflow.set_experiment(getenv("CMAPSS_MLFLOW_TRAINING_EXPERIMENT", "cmapss-training"))
    
    with mlflow.start_run(run_name=f"{normalized_subset}-{model_name}-{feature_set}-{target_name}") as run:
        print(f"[{normalized_subset}] fitting {model_name} on {len(X_train):,} rows and {len(feature_columns):,}"
              f"{feature_set} features with target={target_name}...", flush=True)
        
        model.fit(X_train, y_train)

        validation_predictions = model.predict(X_validation)
        diagnostics, tail_metrics = evaluate_prediction_diagnostics(validation_metadata, validation_predictions)
        metrics = evaluate_predictions(y_validation, validation_predictions)
        metrics.update(tail_metrics)
        
        row_metrics = evaluate_predictions(y_validation_rows, model.predict(X_validation_rows))

        mlflow.log_params(model.get_params())
        mlflow.log_params({"model_type": type(model).__name__,
                           "subset": normalized_subset,
                           "preprocessing_run_id": preprocessing_run_id,
                           "feature_count": len(feature_columns),
                           "target": target_name,
                           "rul_cap": "none" if rul_cap is None else rul_cap,
                           "validation_protocol": "pseudo_test_endpoint",
                           "validation_cutoff_seed": 42,
                           "validation_minimum_observed_fraction": 0.5,
                           "validation_endpoint_count": len(y_validation)})
        
        mlflow.set_tag("feature_set", feature_set)
        mlflow.log_text(json.dumps(feature_columns), "feature_names.json")
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"validation_row_{name}": value for name, value in row_metrics.items()})
        mlflow.log_text(diagnostics.to_csv(index=False), "validation_endpoint_predictions.csv")
        
        if model_name == "xgboost":
            mlflow_xgboost.log_model(model, name="model")
        elif model_name == "lightgbm":
            mlflow_lightgbm.log_model(model, name="model")
        else:
            mlflow_sklearn.log_model(model, name="model", pip_requirements=mlflow_sklearn.get_default_pip_requirements())

    return run.info.run_id
