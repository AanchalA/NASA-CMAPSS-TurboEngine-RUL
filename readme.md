# Predictive Maintenance System for Turbofan Engines using Remaining Useful Life Estimation

Predicting the Remaining Useful Life (RUL) of turbofan engines from multivariate sensor data.

Model Input: Last N cycles of sensor measurements for an engine
Model Output: Estimated RUL - the predicted number of operating cycles before failure.

## DATASET

- NASA PCoE Data Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

NASA provides the original CMAPSSData.zip. The dataset comes from the **NASA Ames Prognostics Center of Excellence (PCoE)** and contains **four simulated turbofan degradation datasets** under different operating conditions and fault modes.

NASA's C-MAPSS Turbofan Engine Degradation dataset contains **run-to-failure time-series data from simulated aircraft engines**. Each engine produces operational settings and sensor readings over repeated cycles. 
- Training engines are observed until failure; 
- test engines stop before failure, 
- and the task is to predict how many operational cycles remain.
The raw data has multiple engine trajectories, operational conditions, sensor noise, and degradation over time. NASA provides four subsets with different combinations of operating conditions and fault modes.

## Project Setup

Prerequisites: Linux or WSL, Python 3.12, Java 17+, and `uv`. 

Install the project dependencies:
```bash
cd nasa_c_mapss
uv sync
```

In every new terminal first run:
```bash
cd nasa_c_mapss
source .venv/bin/activate
export PYTHONPATH="$PWD/code"
```

## Run preprocessing with Airflow

Extract `CMAPSSData.zip` into `Data/CMAPSSData`.

Configure the DAG folder once from the repository root:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
airflow db migrate
sed -i "s|^dags_folder = .*|dags_folder = $PWD/code/src/dags|" "$AIRFLOW_HOME/airflow.cfg"
```

Start Airflow:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
airflow standalone
```

In a second terminal, trigger preprocessing:

```bash
export AIRFLOW_HOME="$PWD/.airflow"

airflow dags list-import-errors
airflow dags unpause cmapss_preprocessing
airflow dags trigger --conf '{"subset":"ALL"}' cmapss_preprocessing
airflow dags list-runs cmapss_preprocessing
```

Use `FD001`, `FD002`, `FD003`, or `FD004` instead of `ALL` to process one subset. Processed Parquet data is written to `Data/processed/<subset>` and preprocessing runs are recorded in `mlruns` under `cmapss-preprocessing`.

## Run model training

Run preprocessing first. Then, from the repository root:

Run the Random Forest baseline, replacing `FD001` with the subset you want to train:

```bash
python code/src/training/run_training.py --subset-id FD001
```

The command trains on the processed data, evaluates validation and official test RUL, and records the model and metrics in MLflow.

## Run model serving

Complete preprocessing and model training for the subset first. From the repository root, start the API:

```bash
uv run uvicorn api.app:app --app-dir code --host 0.0.0.0 --port 8000
```

Verify that the API and Spark are ready:

```bash
curl http://localhost:8000/health
```

Send one complete engine trajectory to the latest matching trained model:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "subset_id": "FD004",
    "model_type": "RandomForestRegressor",
    "observations": [
      {
        "unit_id": 1,
        "cycle": 1,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
        "sensor_1": 0.0,
        "sensor_2": 0.0,
        "sensor_3": 0.0,
        "sensor_4": 0.0,
        "sensor_5": 0.0,
        "sensor_6": 0.0,
        "sensor_7": 0.0,
        "sensor_8": 0.0,
        "sensor_9": 0.0,
        "sensor_10": 0.0,
        "sensor_11": 0.0,
        "sensor_12": 0.0,
        "sensor_13": 0.0,
        "sensor_14": 0.0,
        "sensor_15": 0.0,
        "sensor_16": 0.0,
        "sensor_17": 0.0,
        "sensor_18": 0.0,
        "sensor_19": 0.0,
        "sensor_20": 0.0,
        "sensor_21": 0.0
      }
    ]
  }'
```

Include every observed cycle for one engine in `observations`. 

The API resolves the latest completed training run for the requested subset and model type.
