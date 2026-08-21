import json
import pandas as pd
from pathlib import Path
from mlflow import MlflowClient
import mlflow.sklearn as mlflow_sklearn

from src.tracking import configure_mlflow
from src.data_processing.constants import SUPPORTED_SUBSETS
from src.training.model_evaluation_metrics import evaluate_predictions


def prepare_test_data(subset_id, feature_columns, processed_data_dir, raw_data_dir):

    test_dataframe = pd.read_parquet(Path(processed_data_dir) / subset_id / "test")

    final_cycle_rows = (test_dataframe
                        .sort_values(["unit_id", "cycle"])
                        .groupby("unit_id", sort=True).tail(1)
                        .sort_values("unit_id").reset_index(drop=True))

    official_rul = pd.read_csv(Path(raw_data_dir) / f"RUL_{subset_id}.txt", sep=r"\s+", header=None, names=["RUL"])
    official_rul.insert(0, "unit_id", range(1, len(official_rul) + 1))

    final_cycle_rows = (final_cycle_rows
                        .drop(columns=["RUL"], errors="ignore")
                        .merge(official_rul, on="unit_id", how="inner", validate="one_to_one"))

    X_test = final_cycle_rows.loc[:, feature_columns]
    y_test = final_cycle_rows.loc[:, "RUL"]

    return X_test, y_test


def evaluate_test_data(subset_id, training_run_id, processed_data_dir, raw_data_dir):
    
    normalized_subset = subset_id.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset_id}")

    configure_mlflow()
    
    client = MlflowClient()
    training_run = client.get_run(training_run_id)
    training_subset = training_run.data.params["subset"]
    if training_subset != normalized_subset:
        raise ValueError(f"training run ({training_run_id}) subset {training_subset} does not match {normalized_subset}")
    
    feature_columns = json.loads(training_run.data.params["feature_names"])    
    
    X_test, y_test = prepare_test_data(subset_id=normalized_subset, feature_columns=feature_columns,
                                       processed_data_dir=processed_data_dir, raw_data_dir=raw_data_dir)
    
    model = mlflow_sklearn.load_model(f"runs:/{training_run_id}/model")
    metrics = evaluate_predictions(y_test, model.predict(X_test))

    for metric_name, value in metrics.items():
        client.log_metric(training_run_id, f"test_{metric_name}", value)

    return metrics
