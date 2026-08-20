from pathlib import Path

import mlflow.sklearn as mlflow_sklearn
from mlflow import MlflowClient
from pyspark.sql import Window, functions as F

from src.data_processing.data_loader import load_cmapss_test_rul
from src.data_processing.schema import FINAL_OBSERVED_RUL_COLUMN, SUPPORTED_SUBSETS
from src.data_processing.validation import validate_test_target_inputs
from src.tracking import configure_mlflow, load_preprocessing_state
from src.training.model_evaluation_metrics import evaluate_predictions
from src.training.model_input import build_tabular_inputs


def evaluate_test_data(spark, subset_id, training_run_id, processed_data_dir, raw_data_dir):
    
    normalized_subset = subset_id.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset_id}")

    configure_mlflow()
    
    client = MlflowClient()
    training_run = client.get_run(training_run_id)
    training_subset = training_run.data.params["subset"]
    if training_subset != normalized_subset:
        raise ValueError(f"training run subset {training_subset} does not match {normalized_subset}")

    preprocessing_run_id = training_run.data.params["preprocessing_run_id"]
    
    artifacts = load_preprocessing_state(preprocessing_run_id)
    feature_columns = list(artifacts.retained_sensor_columns)
    test_dataframe = spark.read.parquet(str(Path(processed_data_dir) / normalized_subset / "test"))
    
    official_rul = load_cmapss_test_rul(spark, Path(raw_data_dir) / f"RUL_{normalized_subset}.txt")
    
    validate_test_target_inputs(test_dataframe.drop("RUL"), official_rul)

    final_cycle_window = Window.partitionBy("unit_id").orderBy(F.desc("cycle"))
    final_cycle_rows = (test_dataframe.withColumn("cycle_rank", F.row_number().over(final_cycle_window))
                        .where(F.col("cycle_rank") == 1)
                        .drop("cycle_rank", "RUL")
                        .join(official_rul, on="unit_id", how="inner")
                        .withColumnRenamed(FINAL_OBSERVED_RUL_COLUMN, "RUL")
                        .orderBy("unit_id"))

    X_test, y_test, _ = build_tabular_inputs(final_cycle_rows, feature_columns)
    model = mlflow_sklearn.load_model(f"runs:/{training_run_id}/model")
    metrics = evaluate_predictions(y_test, model.predict(X_test))

    for metric_name, value in metrics.items():
        client.log_metric(training_run_id, f"test_{metric_name}", value)

    return metrics
