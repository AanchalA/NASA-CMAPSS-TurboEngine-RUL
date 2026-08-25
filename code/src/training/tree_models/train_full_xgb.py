import json
from os import getenv
from pathlib import Path

import mlflow
import mlflow.xgboost as mlflow_xgboost
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.data_processing import select_tabular_model_inputs
from src.data_processing.constants import SUPPORTED_SUBSETS
from src.tracking import configure_mlflow
from src.training.evaluation import (evaluate_prediction_diagnostics,
                                     evaluate_predictions,
                                     prepare_pseudo_test_validation)
from src.training.tree_models.train_xgboost import DEFAULT_PARAMETERS


TEMPORAL_FEATURE_MARKERS = ("_lag_", "_diff_", "_rolling_", "_ewma")


def train_full_xgboost(preprocessing_run_ids, processed_data_dir, parameters=None, rul_cap=125):
    missing_subsets = set(SUPPORTED_SUBSETS) - preprocessing_run_ids.keys()
    if missing_subsets:
        raise ValueError(f"Missing preprocessing runs: {sorted(missing_subsets)}")

    train_dataframes = {
        subset: pd.read_parquet(Path(processed_data_dir) / subset / preprocessing_run_ids[subset] / "train")
        for subset in SUPPORTED_SUBSETS
    }
    validation_dataframes = {
        subset: pd.read_parquet(Path(processed_data_dir) / subset / preprocessing_run_ids[subset] / "validation")
        for subset in SUPPORTED_SUBSETS
    }

    feature_columns = sorted(set().union(*(
        select_tabular_model_inputs(dataframe)[0].columns for dataframe in train_dataframes.values()
    )))
    if any(marker in column for column in feature_columns for marker in TEMPORAL_FEATURE_MARKERS):
        raise ValueError("Full XGBoost training requires baseline preprocessing runs")

    subset_columns = [f"subset_{subset}" for subset in SUPPORTED_SUBSETS]
    model_feature_columns = [*feature_columns, *subset_columns]
    training_inputs = []
    training_targets = []
    validation_inputs = []
    validation_targets = []
    validation_metadata = []

    for subset in SUPPORTED_SUBSETS:
        X_train, y_train = select_tabular_model_inputs(train_dataframes[subset])
        X_train = X_train.reindex(columns=feature_columns)
        for subset_column in subset_columns:
            X_train[subset_column] = float(subset_column == f"subset_{subset}")
        training_inputs.append(X_train)
        training_targets.append(y_train)

        validation_dataframe = validation_dataframes[subset].copy()
        for feature in feature_columns:
            if feature not in validation_dataframe:
                validation_dataframe[feature] = np.nan
        for subset_column in subset_columns:
            validation_dataframe[subset_column] = float(subset_column == f"subset_{subset}")

        X_validation, y_validation, metadata = prepare_pseudo_test_validation(
            validation_dataframe, model_feature_columns
        )
        metadata.insert(0, "subset", subset)
        validation_inputs.append(X_validation)
        validation_targets.append(y_validation)
        validation_metadata.append(metadata)

    X_train = pd.concat(training_inputs, ignore_index=True)
    y_train = pd.concat(training_targets, ignore_index=True)
    X_validation = pd.concat(validation_inputs, ignore_index=True)
    y_validation = pd.concat(validation_targets, ignore_index=True)
    metadata = pd.concat(validation_metadata, ignore_index=True)
    if rul_cap is not None:
        y_train = np.minimum(y_train, rul_cap)

    model = XGBRegressor(**(parameters or DEFAULT_PARAMETERS))
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    metrics = evaluate_predictions(y_validation, predictions)
    diagnostics, tail_metrics = evaluate_prediction_diagnostics(metadata, predictions)
    metrics.update(tail_metrics)

    configure_mlflow()
    mlflow.set_experiment(getenv("CMAPSS_MLFLOW_TRAINING_EXPERIMENT", "cmapss-training"))
    target_name = "raw_rul" if rul_cap is None else f"capped_rul_{rul_cap:g}"

    with mlflow.start_run(run_name=f"all-xgboost-baseline-{target_name}") as run:
        mlflow.log_params(model.get_params())
        mlflow.log_params({"model_type": type(model).__name__,
                           "subsets": ",".join(SUPPORTED_SUBSETS),
                           "preprocessing_run_ids": json.dumps(preprocessing_run_ids, sort_keys=True),
                           "feature_count": len(model_feature_columns),
                           "target": target_name,
                           "rul_cap": "none" if rul_cap is None else rul_cap,
                           "validation_protocol": "pooled_pseudo_test_endpoint",
                           "validation_cutoff_seed": 42,
                           "validation_minimum_observed_fraction": 0.5,
                           "validation_endpoint_count": len(y_validation)})
        mlflow.set_tag("feature_set", "baseline")
        mlflow.log_text(json.dumps(model_feature_columns), "feature_names.json")
        mlflow.log_metrics(metrics)
        mlflow.log_text(diagnostics.to_csv(index=False), "validation_endpoint_predictions.csv")
        mlflow_xgboost.log_model(model, name="model")

    return run.info.run_id


if __name__ == "__main__":
    training_run_id = train_full_xgboost(
        preprocessing_run_ids={
            "FD001": "208b1c6c99264e74a42f35df2edfd4dd",
            "FD002": "1030d9d2baf7491e901e85ed69d5655f",
            "FD003": "61dd784103e24f408aea4486cb982a35",
            "FD004": "432ff4df92c7451b97e52832049b5a70",
        },
        processed_data_dir="Data/processed",
        rul_cap=125,
    )
    print({"training_run_id": training_run_id})
