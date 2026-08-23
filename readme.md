# Predictive Maintenance System for Turbofan Engines using Remaining Useful Life Estimation

A machine-learning-based predictive maintenance system for estimating the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor time-series data.

The system covers the complete ML lifecycle:

**Raw NASA C-MAPSS data → Data preprocessing → Feature engineering → Time-series processing → Model training → Evaluation → MLflow tracking → FastAPI inference → Docker deployment → Prometheus/Grafana monitoring → CI/CD testing**

---

## 1. Project Information

### Project Title

**Predictive Maintenance System for Turbofan Engines using Remaining Useful Life Estimation**

### Team Members

|       Name        | Roll Number     
| ----------------- | --------------- 
|    Aanchal A      | 
|    Tanya Suri     | Da25M628 

> **Note:** Replace the placeholders above with the actual team-member names, roll numbers, and contributions.

### Main Branch for Evaluation

**`time-series-processing-and-modeling-experiments`**

This branch contains the time-series preprocessing/modeling experiments together with the preprocessing pipeline, model training, MLflow tracking, API serving, Docker deployment, monitoring, and CI/CD configuration.

---

# 2. Project Overview

Modern aircraft engines generate large volumes of sensor measurements during operation. As an engine degrades, patterns in these measurements can provide information about its health and expected remaining operating lifetime.

The objective of this project is to build a **Remaining Useful Life (RUL) estimation system** that predicts how many operational cycles an engine has remaining before failure.

### Problem Statement

Given the historical sensor measurements of a turbofan engine:

> **Predict the number of operational cycles remaining before the engine reaches failure.**

The system receives the latest available sensor observations for an engine and produces an estimated RUL.

### Input

The model uses:

* Engine/unit ID
* Operating cycle
* 3 operating settings
* 21 sensor measurements

The project uses the NASA C-MAPSS turbofan engine degradation datasets, which contain multivariate time-series trajectories from simulated engines operating until failure in the training data.

### Output

The primary model output is:

```text
Predicted Remaining Useful Life (RUL)
```

RUL is expressed as the estimated number of operating cycles remaining before failure.

---

# 3. Dataset Overview

## NASA C-MAPSS

The project uses the **NASA C-MAPSS Turbofan Engine Degradation Dataset** from the NASA Prognostics Center of Excellence (PCoE).

Dataset source:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

The dataset contains four subsets:

| Subset | Description                                          |
| ------ | ---------------------------------------------------- |
| FD001  | One operating condition / one fault mode             |
| FD002  | Multiple operating conditions / one fault mode       |
| FD003  | One operating condition / multiple fault modes       |
| FD004  | Multiple operating conditions / multiple fault modes |

Each dataset contains:

* Training engine trajectories
* Test engine trajectories
* Operating settings
* Sensor measurements
* Engine cycle information
* RUL information derived from run-to-failure trajectories

### Training Data

Training trajectories contain engine observations from the beginning of operation until failure.

### Test Data

Test trajectories stop before failure. The objective is to estimate the remaining number of cycles at the end of the observed trajectory.

### Dataset Directory

After downloading the NASA dataset, extract it into:

```text
Data/CMAPSSData/
```

The expected directory should contain the original C-MAPSS files for:

```text
FD001
FD002
FD003
FD004
```

The dataset itself is not required to be committed to Git because it is large and is handled separately from the source code.

---

# 4. System Architecture

The overall system consists of the following stages:

```text
                         ┌──────────────────────┐
                         │ NASA C-MAPSS Dataset │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Airflow Preprocessing│
                         │        DAG           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ Data Processing & Validation │
                    │                              │
                    │ • Data loading               │
                    │ • RUL generation             │
                    │ • Feature engineering        │
                    │ • Scaling                    │
                    │ • Sequence generation        │
                    │ • Train/validation split     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Processed Parquet Dataset    │
                    │ Data/processed/<subset>/     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Model Training               │
                    │                              │
                    │ Random Forest baseline       │
                    │ Additional experiments       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Model Evaluation             │
                    │                              │
                    │ Validation metrics           │
                    │ Official test evaluation     │
                    │ Engine-level evaluation      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │      MLflow          │
                         │                      │
                         │ • Parameters         │
                         │ • Metrics            │
                         │ • Artifacts          │
                         │ • Models             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │      REST API        │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                 ┌───────────────┐    ┌────────────────┐
                 │ Docker        │    │ Prometheus     │
                 │ Deployment    │    │ Metrics        │
                 └───────────────┘    └───────┬────────┘
                                              │
                                              ▼
                                      ┌────────────────┐
                                      │    Grafana     │
                                      │  Dashboards    │
                                      └────────────────┘
```

---

# 5. Architecture Components

## 5.1 Data Layer

The raw NASA C-MAPSS data is stored under:

```text
Data/CMAPSSData/
```

The preprocessing pipeline transforms the raw engine trajectories into processed datasets suitable for machine-learning experiments.

Processed data is stored under:

```text
Data/processed/<subset>/
```

in Parquet format.

---

## 5.2 Data Processing Layer

The data-processing package contains modules for:

* Data loading
* Schema validation
* RUL generation
* Feature engineering
* Scaling
* Sequence generation
* Dataset splitting
* Validation
* Artifact management

Relevant location:

```text
code/src/data_processing/
```

---

## 5.3 Workflow Orchestration

Apache Airflow is used to orchestrate the preprocessing pipeline.

The Airflow DAG is located under:

```text
code/src/dags/
```

The main preprocessing DAG is:

```text
cmapss_preprocessing
```

---

## 5.4 Model Training Layer

Model training is implemented under:

```text
code/src/training/
```

The current training workflow includes a **Random Forest regression baseline**.

Training performs:

1. Loading processed data
2. Training the model
3. Validation
4. Official test evaluation
5. Logging parameters and metrics
6. Logging the trained model/artifacts to MLflow

---

## 5.5 Model Evaluation

The training package contains dedicated evaluation components for:

* Model metrics
* Test-set evaluation
* Engine-endpoint evaluation

Relevant files include:

```text
code/src/training/model_evaluation_metrics.py
code/src/training/test_evaluation.py
code/src/training/engine_endpoint_evaluation.py
```

---

## 5.6 Model Tracking

MLflow is used to track:

* Training experiments
* Model parameters
* Evaluation metrics
* Preprocessing runs
* Trained model artifacts

The project stores MLflow runs locally under:

```text
mlruns/
```

The preprocessing workflow records runs under the:

```text
cmapss-preprocessing
```

experiment.

Training runs are also recorded and subsequently used by the serving API.

---

# 6. Repository / Folder Structure

```text
NASA-CMAPSS-TurboEngine-RUL/
│
├── .github/
│   └── workflows/
│       └── ci_cd_pipeline.yml
│
├── .dvc/
├── .dvcignore
├── .gitignore
│
├── Data/
│   ├── CMAPSSData/
│   └── processed/
│
├── code/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── api_schemas.py
│   │   └── app.py
│   │
│   ├── src/
│   │   ├── dags/
│   │   │
│   │   ├── data_processing/
│   │   │   ├── artifacts.py
│   │   │   ├── constants.py
│   │   │   ├── data_loader.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── preprocessing.py
│   │   │   ├── scaler_state.py
│   │   │   ├── scaling.py
│   │   │   ├── schema.py
│   │   │   ├── sequences.py
│   │   │   ├── split.py
│   │   │   ├── tabular.py
│   │   │   └── validation.py
│   │   │
│   │   ├── inference/
│   │   │
│   │   ├── tracking/
│   │   │
│   │   ├── training/
│   │   │   ├── evaluation/
│   │   │   ├── engine_endpoint_evaluation.py
│   │   │   ├── model_evaluation_metrics.py
│   │   │   ├── run_training.py
│   │   │   ├── test_evaluation.py
│   │   │   └── train.py
│   │   │
│   │   └── build_spark.py
│   │
│   ├── tests/
│   └── rul_estimation.ipynb
│
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   │
│   ├── prometheus-rules/
│   └── prometheus.yml
│
├── .dockerignore
├── .python-version
├── Dockerfile.serve
├── docker-compose.yml
├── pyproject.toml
├── readme.md
└── uv.lock
```

---

# 7. Setup and Installation

## Prerequisites

The project requires:

* Linux or WSL
* Python 3.12
* Java 17+
* `uv`
* Docker for containerized serving/monitoring

The project uses `uv` for Python environment and dependency management.

## Clone the Repository

```bash
git clone https://github.com/AanchalA/NASA-CMAPSS-TurboEngine-RUL.git
cd NASA-CMAPSS-TurboEngine-RUL
```

Switch to the evaluation branch:

```bash
git checkout time-series-processing-and-modeling-experiments
```

## Install Dependencies

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

Set the Python module path:

```bash
export PYTHONPATH="$PWD/code"
```

For every new terminal session:

```bash
cd nasa_c_mapss
source .venv/bin/activate
export PYTHONPATH="$PWD/code"
```

---

# 8. Dependencies

The main dependencies are managed through `pyproject.toml` and `uv.lock`.

### Core dependencies

* Python >= 3.12
* MLflow
* pandas
* scikit-learn

### Training dependencies

* PyArrow
* PyTorch
* XGBoost

### Serving dependencies

* FastAPI
* Uvicorn
* Prometheus Client

### Data/Offline dependencies

* Apache Airflow
* DVC
* PySpark
* SciPy
* Matplotlib
* Seaborn
* Jupyter/IPython tooling

### Development

* pytest

The exact locked dependency versions should be obtained from:

```text
uv.lock
```

---

# 9. Data and Preprocessing Pipeline

## Step 1 — Obtain the Dataset

Download the NASA C-MAPSS dataset from the NASA PCoE repository.

Extract the downloaded archive into:

```text
Data/CMAPSSData/
```

---

## Step 2 — Configure Airflow

From the repository root:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
```

Initialize the Airflow database:

```bash
airflow db migrate
```

Configure the DAG directory:

```bash
sed -i "s|^dags_folder = .*|dags_folder = $PWD/code/src/dags|" "$AIRFLOW_HOME/airflow.cfg"
```

---

## Step 3 — Start Airflow

```bash
export PYTHONPATH="$PWD/code"
export AIRFLOW_HOME="$PWD/.airflow"

airflow standalone
```

Keep this terminal running.

---

## Step 4 — Trigger the Preprocessing Pipeline

Open a second terminal:

```bash
cd nasa_c_mapss
source .venv/bin/activate

export PYTHONPATH="$PWD/code"
export AIRFLOW_HOME="$PWD/.airflow"
```

Check for DAG import errors:

```bash
airflow dags list-import-errors
```

List available DAGs:

```bash
airflow dags list
```

Unpause the preprocessing DAG:

```bash
airflow dags unpause cmapss_preprocessing
```

Run preprocessing for all four datasets:

```bash
airflow dags trigger --conf '{"subset":"ALL"}' cmapss_preprocessing
```

Check the DAG runs:

```bash
airflow dags list-runs cmapss_preprocessing
```

### Process a Single Dataset

Instead of `ALL`, use:

```text
FD001
FD002
FD003
FD004
```

For example:

```bash
airflow dags trigger \
  --conf '{"subset":"FD001"}' \
  cmapss_preprocessing
```

Processed data is generated under:

```text
Data/processed/<subset>/
```

and preprocessing experiments are recorded in MLflow.

---

# 10. Model Training and Evaluation

Preprocessing must be completed before model training.

From the repository root:

```bash
export PYTHONPATH="$PWD/code"
```

Train the Random Forest baseline:

```bash
python code/src/training/run_training.py --subset-id FD001
```

Replace `FD001` with:

```text
FD002
FD003
FD004
```

as required.

For example:

```bash
python code/src/training/run_training.py --subset-id FD004
```

The training pipeline:

1. Loads processed data
2. Trains the model
3. Evaluates validation performance
4. Evaluates the official test RUL
5. Records metrics in MLflow
6. Stores the trained model/artifacts in MLflow

---

# 11. MLflow Usage

MLflow is used for experiment tracking and model/artifact management.

The local tracking directory is:

```text
mlruns/
```

Preprocessing runs are tracked using:

```text
cmapss-preprocessing
```

Training runs contain the model and associated evaluation information.

## Start MLflow UI

From the repository root:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```text
http://localhost:5000
```

The MLflow tracking directory can also be explicitly configured:

```bash
export MLFLOW_TRACKING_URI="file://$PWD/mlruns"
```

The serving API uses this tracking store to resolve the appropriate trained model.

---

# 12. API Usage

The project exposes a REST API using **FastAPI**.

## Start the API Locally

From the repository root:

```bash
export PYTHONPATH="$PWD/code"
export MLFLOW_TRACKING_URI="file://$PWD/mlruns"

uvicorn api.app:app \
  --app-dir code \
  --host 0.0.0.0 \
  --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Prediction Endpoint

The prediction endpoint is:

```text
POST /predict
```

Example request:

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

The API uses the latest observation cycle when multiple observations for the same engine are supplied.

The response contains the estimated RUL:

```json
{
  "predicted_rul": <VALUE>
}
```

## Interactive API Documentation

When the API is running, FastAPI also provides interactive API documentation at:

```text
http://localhost:8000/docs
```

---

# 13. Docker Execution

Docker is used for model serving.

The repository contains:

```text
Dockerfile.serve
```

## Build the Serving Image

From the repository root:

```bash
docker build \
  -f Dockerfile.serve \
  -t nasa-cmapss-api .
```

## Run the API Container

```bash
docker run --rm \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="file://$PWD/mlruns" \
  -v "$PWD/mlruns:$PWD/mlruns:ro" \
  nasa-cmapss-api
```

The MLflow directory is mounted read-only into the container.

The serving container does **not** require:

* NASA dataset
* Java
* Spark
* Airflow

because the serving stage only requires the previously trained MLflow artifacts.

---

# 14. Monitoring — Prometheus and Grafana

The project includes monitoring using:

* Prometheus
* Grafana
* FastAPI Prometheus metrics
* Prometheus rules

Monitoring configuration is located under:

```text
monitoring/
```

Structure:

```text
monitoring/
├── grafana/
│   ├── dashboards/
│   └── provisioning/
├── prometheus-rules/
└── prometheus.yml
```

## Start Monitoring Stack

Make sure the FastAPI application is running on port `8000`.

Then run:

```bash
docker compose up -d
```

This starts:

```text
Prometheus → http://localhost:9090
Grafana    → http://localhost:3000
```

FastAPI exposes Prometheus metrics at:

```text
http://localhost:8000/metrics
```

## Grafana Login

Default credentials configured by the project:

```text
Username: admin
Password: admin
```

After logging into Grafana, the provisioned dashboards can be used to inspect API/application metrics.

---

# 15. CI/CD

GitHub Actions is used for automated testing.

Workflow:

```text
.github/workflows/ci_cd_pipeline.yml
```

The workflow runs on pushes and pull requests targeting the project's development branches, including:

* `main`
* `data-preprocessing`
* `model-training`
* `model-deployment`
* `actions-logging-monitoring`
* `time-series-processing-and-modeling-experiments`

## CI Pipeline

The CI workflow:

1. Checks out the repository
2. Installs `uv`
3. Sets up Python 3.12
4. Sets up Java 17
5. Installs locked project dependencies
6. Runs pytest
7. Excludes tests requiring the local NASA dataset

Run the same test suite locally with:

```bash
uv run --no-sync pytest -m "not requires_data"
```

Dataset-dependent tests can be run separately after the NASA dataset has been configured.

---

# 16. DVC and Data Versioning

DVC configuration is included in the repository:

```text
.dvc/
.dvcignore
```

DVC is included as an offline/project dependency and is intended for data/model artifact versioning without storing large datasets directly in Git.

The raw NASA dataset should not be committed directly to the repository.

For a reproducible experiment, ensure that the required dataset version and processed artifacts are available before running preprocessing/training.

---

# 17. Reproducibility

To reproduce the project from a clean environment:

## 1. Clone the Repository

```bash
git clone https://github.com/AanchalA/NASA-CMAPSS-TurboEngine-RUL.git
cd NASA-CMAPSS-TurboEngine-RUL
```

## 2. Checkout the Evaluation Branch

```bash
git checkout time-series-processing-and-modeling-experiments
```

## 3. Install the Locked Environment

```bash
uv sync --frozen
```

For the complete offline/training/development environment:

```bash
uv sync --frozen --all-extras --dev
```

## 4. Configure Python Path

```bash
export PYTHONPATH="$PWD/code"
```

## 5. Add NASA Dataset

Place the extracted dataset under:

```text
Data/CMAPSSData/
```

## 6. Run Preprocessing

Use the Airflow preprocessing DAG:

```bash
airflow dags trigger \
  --conf '{"subset":"ALL"}' \
  cmapss_preprocessing
```

## 7. Train the Model

```bash
python code/src/training/run_training.py \
  --subset-id FD001
```

## 8. Verify MLflow Runs

Check the generated:

```text
mlruns/
```

directory.

## 9. Start the API

```bash
export MLFLOW_TRACKING_URI="file://$PWD/mlruns"

uvicorn api.app:app \
  --app-dir code \
  --host 0.0.0.0 \
  --port 8000
```

## 10. Verify the API

```bash
curl http://localhost:8000/health
```

## 11. Start Monitoring

```bash
docker compose up -d
```

---

# 18. End-to-End Execution

For a complete run, follow this sequence:

```text
1. Clone repository
        ↓
2. Checkout evaluation branch
        ↓
3. Install dependencies using uv
        ↓
4. Download NASA C-MAPSS dataset
        ↓
5. Place dataset in Data/CMAPSSData/
        ↓
6. Start Airflow
        ↓
7. Trigger cmapss_preprocessing DAG
        ↓
8. Generate Data/processed/<subset>/
        ↓
9. Train model
        ↓
10. Evaluate model
        ↓
11. Log experiment/model to MLflow
        ↓
12. Start FastAPI
        ↓
13. Send /predict request
        ↓
14. Expose /metrics
        ↓
15. Start Prometheus
        ↓
16. Visualize metrics in Grafana
```

---

# 19. Quick Start

For experienced users, the complete workflow can be summarized as:

```bash
# Clone
git clone https://github.com/AanchalA/NASA-CMAPSS-TurboEngine-RUL.git
cd NASA-CMAPSS-TurboEngine-RUL

# Checkout branch
git checkout time-series-processing-and-modeling-experiments

# Install environment
uv sync --frozen --all-extras --dev

# Activate
source .venv/bin/activate

# Python path
export PYTHONPATH="$PWD/code"

# Dataset
# Extract NASA CMAPSSData into:
# Data/CMAPSSData/

# Configure Airflow
export AIRFLOW_HOME="$PWD/.airflow"
airflow db migrate

# Start Airflow
airflow standalone

# In another terminal:
export PYTHONPATH="$PWD/code"
export AIRFLOW_HOME="$PWD/.airflow"

airflow dags unpause cmapss_preprocessing

airflow dags trigger \
  --conf '{"subset":"FD001"}' \
  cmapss_preprocessing

# Train
python code/src/training/run_training.py \
  --subset-id FD001

# Configure MLflow
export MLFLOW_TRACKING_URI="file://$PWD/mlruns"

# Serve
uvicorn api.app:app \
  --app-dir code \
  --host 0.0.0.0 \
  --port 8000

# Monitoring
docker compose up -d
```

---

# 20. Key Technologies

| Component                | Technology       |
| ------------------------ | ---------------- |
| Programming Language     | Python 3.12      |
| Dependency Management    | uv               |
| Data Processing          | pandas, PyArrow  |
| Distributed Processing   | PySpark          |
| Workflow Orchestration   | Apache Airflow   |
| Machine Learning         | scikit-learn     |
| Additional ML            | PyTorch, XGBoost |
| Experiment Tracking      | MLflow           |
| Data Versioning          | DVC              |
| API                      | FastAPI          |
| API Server               | Uvicorn          |
| Containerization         | Docker           |
| Metrics                  | Prometheus       |
| Visualization/Monitoring | Grafana          |
| Testing                  | pytest           |
| CI/CD                    | GitHub Actions   |

---

# 21. Project Deliverables

The project provides an end-to-end predictive-maintenance pipeline covering:

* [x] NASA C-MAPSS dataset integration
* [x] Automated preprocessing pipeline
* [x] Airflow workflow orchestration
* [x] Time-series data processing
* [x] Feature engineering
* [x] Data validation
* [x] Sequence generation
* [x] Scaling
* [x] Model training
* [x] Model evaluation
* [x] MLflow experiment tracking
* [x] Model artifact management
* [x] FastAPI inference API
* [x] Dockerized model serving
* [x] Prometheus metrics
* [x] Grafana monitoring
* [x] Automated tests
* [x] GitHub Actions CI
* [x] DVC integration

---

# 22. Limitations and Future Work

The current branch provides a complete baseline pipeline and deployment/monitoring infrastructure. Future work can extend the modeling layer with additional RUL estimation approaches, including:

* LSTM/GRU-based sequence models
* Temporal CNNs
* Transformer-based time-series models
* XGBoost-based approaches
* Model ensembles
* Hyperparameter optimization
* Uncertainty estimation
* Online/streaming RUL estimation
* Model drift monitoring
* Automated model retraining
* Production model registry and promotion workflows

---


# 23. References

1. NASA Prognostics Center of Excellence (PCoE), C-MAPSS Turbofan Engine Degradation Dataset.
2. NASA C-MAPSS documentation and dataset repository.
3. MLflow documentation.
4. Apache Airflow documentation.
5. FastAPI documentation.
6. Prometheus documentation.
7. Grafana documentation.
8. DVC documentation.

---

## License

See the repository license for the applicable terms.

