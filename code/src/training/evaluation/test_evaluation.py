import pandas as pd
from pathlib import Path
from mlflow import MlflowClient

from src.tracking import (
    configure_mlflow,
    load_training_feature_columns,
    load_training_model,
)
from src.data_processing.constants import SUPPORTED_SUBSETS
from src.training.evaluation.engine_endpoint_evaluation import (evaluate_prediction_diagnostics,
                                                                prepare_official_test_metadata)
from src.training.evaluation.model_evaluation_metrics import evaluate_predictions


def prepare_test_data(subset_id, preprocessing_run_id, feature_columns, processed_data_dir, raw_data_dir):

    test_dataframe = pd.read_parquet(Path(processed_data_dir) / subset_id / preprocessing_run_id / "test")
    test_metadata = prepare_official_test_metadata(test_dataframe, subset_id, raw_data_dir)

    final_cycle_rows = test_dataframe.merge(test_metadata, on=["unit_id", "cycle"], how="inner",
                                            suffixes=("", "_official"))
    final_cycle_rows["RUL"] = final_cycle_rows.pop("RUL_official")

    X_test = final_cycle_rows.loc[:, feature_columns]
    y_test = final_cycle_rows.loc[:, "RUL"]

    metadata = final_cycle_rows.loc[:, ["unit_id", "cycle", "RUL"]]

    return X_test, y_test, metadata


def evaluate_test_data(subset_id, training_run_id, processed_data_dir, raw_data_dir):

    normalized_subset = subset_id.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset_id}")

    configure_mlflow()

    client = MlflowClient()
    training_run = client.get_run(training_run_id)
    training_subset = training_run.data.params["subset"]
    preprocessing_run_id = training_run.data.params["preprocessing_run_id"]

    if training_subset != normalized_subset:
        raise ValueError(f"training run ({training_run_id}) subset {training_subset} does not match {normalized_subset}")

    model = load_training_model(training_run_id)
    feature_columns = load_training_feature_columns(training_run_id)

    X_test, y_test, metadata = prepare_test_data(subset_id=normalized_subset,
                                                 preprocessing_run_id=preprocessing_run_id,
                                                 feature_columns=feature_columns,
                                                 processed_data_dir=processed_data_dir,
                                                 raw_data_dir=raw_data_dir)

    predictions = model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    for metric_name, value in metrics.items():
        client.log_metric(training_run_id, f"test_{metric_name}", value)

    diagnostics, tail_metrics = evaluate_prediction_diagnostics(metadata, predictions)
    metrics.update(tail_metrics)

    for metric_name, value in tail_metrics.items():
        client.log_metric(training_run_id, f"test_{metric_name}", value)
    client.log_text(training_run_id, diagnostics.to_csv(index=False), "test_predictions.csv")

    return metrics
