from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from api.api_schemas import PredictRequest, PredictionResponse
from src.build_spark import spark_session_context
from src.data_processing.schema import CMAPSS_SCHEMA
from src.tracking import fetch_run_id
from src.training import predict_rul


@asynccontextmanager
async def lifespan(app: FastAPI):
    with spark_session_context(app_name="cmapss-inference-api") as spark:
        app.state.spark = spark
        yield


app = FastAPI(title="NASA C-MAPSS RUL API", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    spark = getattr(request.app.state, "spark", None)
    return {"status": "ok", "spark_ready": spark is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest, request: Request) -> PredictionResponse:
    
    subset_id = payload.subset_id.upper()
    
    training_run_id = fetch_run_id(subset_id=subset_id,
                                   experiment_type="training",
                                   model_type=payload.model_type)
    
    raw_trajectory = request.app.state.spark.createDataFrame(
        [observation.model_dump() for observation in payload.observations],
        schema=CMAPSS_SCHEMA)
    
    predicted_rul = predict_rul(subset_id, training_run_id, raw_trajectory)

    return PredictionResponse(subset_id=subset_id,
                              model_type=payload.model_type,
                              predicted_rul=predicted_rul)
