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

## Run preprocessing with Airflow

Prerequisites: Linux or WSL, Python 3.12, Java 17+, and `uv`. Extract `CMAPSSData.zip` into `Data/CMAPSSData`.

Install the project dependencies:

```bash
cd /path/to/nasa_c_mapss
uv sync
```

Configure the DAG folder once from the repository root:

```bash
source .venv/bin/activate
export AIRFLOW_HOME="$PWD/.airflow"
airflow db migrate
sed -i "s|^dags_folder = .*|dags_folder = $PWD/code/src/dags|" "$AIRFLOW_HOME/airflow.cfg"
```

Start Airflow:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
export PYTHONPATH="$PWD/code"
airflow standalone
```

In a second terminal, trigger preprocessing:

```bash
cd /path/to/nasa_c_mapss
source .venv/bin/activate
export PYTHONPATH="$PWD/code"
export AIRFLOW_HOME="$PWD/.airflow"

airflow dags list-import-errors
airflow dags unpause cmapss_preprocessing
airflow dags trigger --conf '{"subset":"ALL"}' cmapss_preprocessing
airflow dags list-runs cmapss_preprocessing
```

Use `FD001`, `FD002`, `FD003`, or `FD004` instead of `ALL` to process one subset. Processed Parquet data is written to `Data/processed/<subset>` and preprocessing runs are recorded in `mlruns`.

## Local MLflow configuration

The prototype uses MLflow directly with local file-backed storage; no tracking server is required.

```text
MLFLOW_TRACKING_URI=file:./mlruns
CMAPSS_MLFLOW_EXPERIMENT=cmapss-preprocessing
```

Both values can be overridden through environment variables. Preprocessing artifacts use these stable paths within each MLflow run:

```text
preprocessing_pipeline/
preprocessing_state.json
```
