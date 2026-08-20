from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import os
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task


DAG_ID = "cmapss_preprocessing"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
ALL_SUBSETS = "ALL"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def spark_session_context(spark_master, subset):
    from pyspark.sql import SparkSession

    spark = None
    try:
        spark = (SparkSession.builder.master(spark_master)
                 .appName(f"cmapss-preprocessing-{subset}")
                 .config("spark.ui.enabled", "false")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("WARN")
        yield spark
    finally:
        if spark is not None:
            spark.stop()


@dag(dag_id=DAG_ID,
    description="Preprocess C-MAPSS data with native WSL Spark and MLflow",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={"subset": Param("FD001", type="string",
                            enum=[*SUBSETS, ALL_SUBSETS],
                            description="C-MAPSS subset to process, or ALL")},
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["cmapss", "preprocessing", "spark", "mlflow"])
def cmapss_preprocessing_dag():
    
    @task
    def resolve_runtime():     
           
        raw_data_dir = Path(os.environ.get("CMAPSS_RAW_DATA_DIR", PROJECT_ROOT / "Data" / "CMAPSSData")).expanduser().resolve()
        if not raw_data_dir.is_dir():
            raise FileNotFoundError(f"C-MAPSS raw data directory not found: {raw_data_dir}")

        processed_data_dir = Path(os.environ.get("CMAPSS_PROCESSED_DATA_DIR", PROJECT_ROOT / "Data" / "processed")).expanduser().resolve()
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri())
        experiment = os.environ.get("CMAPSS_MLFLOW_EXPERIMENT", "cmapss-preprocessing")

        return {"raw_data_dir": str(raw_data_dir),
                "processed_data_dir": str(processed_data_dir),
                "mlflow_tracking_uri": tracking_uri,
                "mlflow_experiment": experiment}

    @task
    def select_subsets(runtime):        
        del runtime  
        requested = str(get_current_context()["params"]["subset"]).upper()
        if requested == ALL_SUBSETS:
            return list(SUBSETS)
        if requested not in SUBSETS:
            raise ValueError(f"Unsupported C-MAPSS subset: {requested}")
        return [requested]

    @task
    def preprocess_subset(subset, runtime):        
        from src.data_processing.preprocessing import run_subset_preprocessing

        os.environ["MLFLOW_TRACKING_URI"] = runtime["mlflow_tracking_uri"]
        os.environ["CMAPSS_MLFLOW_EXPERIMENT"] = runtime["mlflow_experiment"]

        with spark_session_context("local[*]", subset) as spark:
            result = run_subset_preprocessing(spark=spark,
                                              subset=subset,
                                              raw_data_dir=Path(runtime["raw_data_dir"]),
                                              output_dir=Path(runtime["processed_data_dir"]))

        return result.run_id

    @task
    def collect_run_ids(subsets, run_ids):        
        return dict(zip(subsets, run_ids, strict=True))

    runtime = resolve_runtime()
    selected_subsets = select_subsets(runtime)
    run_ids = preprocess_subset.partial(runtime=runtime).expand(subset=selected_subsets)
    collect_run_ids(selected_subsets, run_ids)


cmapss_preprocessing = cmapss_preprocessing_dag()
