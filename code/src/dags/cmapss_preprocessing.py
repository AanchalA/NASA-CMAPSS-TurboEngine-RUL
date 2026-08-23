from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task

from src.dags import get_preprocessing_runtime
from src.build_spark import spark_session_context


DAG_ID = "cmapss_preprocessing"
SPARK_PREPROCESSING_POOL = "spark_preprocessing"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


@dag(dag_id=DAG_ID,
    description="Preprocess C-MAPSS data with native WSL Spark and MLflow",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={"subset": Param("FD001", type="string",
                            enum=[*SUBSETS, "ALL"],
                            description="C-MAPSS subset to process, or ALL")},
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["cmapss", "preprocessing", "spark", "mlflow"])
def cmapss_preprocessing_dag():
    
    @task
    def resolve_runtime():
        return get_preprocessing_runtime()

    @task
    def select_subsets(runtime):
        del runtime
        params = get_current_context().get("params")
        if params is None:
            raise RuntimeError("Airflow task context is missing DAG parameters")

        requested = str(params["subset"]).upper()
        if requested == "ALL":
            return list(SUBSETS)
        if requested not in SUBSETS:
            raise ValueError(f"Unsupported C-MAPSS subset: {requested}")
        return [requested]

    @task(pool=SPARK_PREPROCESSING_POOL)
    def preprocess_subset(subset, runtime):        
        from src.data_processing.preprocessing import run_subset_preprocessing

        os.environ["MLFLOW_TRACKING_URI"] = runtime["mlflow_tracking_uri"]
        os.environ["CMAPSS_MLFLOW_EXPERIMENT"] = runtime["mlflow_experiment"]

        with spark_session_context(app_name=f"cmapss-preprocessing-{subset}") as spark:
            result = run_subset_preprocessing(spark=spark,
                                              subset=subset,
                                              raw_data_dir=Path(runtime["raw_data_dir"]),
                                              output_dir=Path(runtime["processed_data_dir"]),
                                              include_temporal_features=False)

        return result.run_id

    @task
    def collect_run_ids(subsets, run_ids):        
        return dict(zip(subsets, run_ids, strict=True))

    runtime = resolve_runtime()
    selected_subsets = select_subsets(runtime)
    run_ids = preprocess_subset.partial(runtime=runtime).expand(subset=selected_subsets)
    collect_run_ids(selected_subsets, run_ids)


cmapss_preprocessing = cmapss_preprocessing_dag()
