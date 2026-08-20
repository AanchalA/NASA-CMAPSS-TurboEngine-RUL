import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_preprocessing_runtime():
    
    raw_data_dir = Path(os.environ.get("CMAPSS_RAW_DATA_DIR", PROJECT_ROOT / "Data" / "CMAPSSData")).expanduser().resolve()
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"C-MAPSS raw data directory not found: {raw_data_dir}")

    processed_data_dir = Path(os.environ.get("CMAPSS_PROCESSED_DATA_DIR", PROJECT_ROOT / "Data" / "processed")).expanduser().resolve()

    return {"raw_data_dir": str(raw_data_dir),
            "processed_data_dir": str(processed_data_dir),
            "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri()),
            "mlflow_experiment": os.environ.get("CMAPSS_MLFLOW_EXPERIMENT", "cmapss-preprocessing")}
