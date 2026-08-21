from fastapi import FastAPI

from api.api_schemas import PredictRequest, PredictionResponse
from src.tracking import fetch_run_id
from src.training import predict_rul


app = FastAPI(title="NASA C-MAPSS RUL API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest) -> PredictionResponse:
    
    subset_id = payload.subset_id.upper()
    
    training_run_id = fetch_run_id(subset_id=subset_id,
                                   experiment_type="training",
                                   model_type=payload.model_type)
    
    observations = [observation.model_dump() for observation in payload.observations]
    
    predicted_rul = predict_rul(subset_id, training_run_id, observations)

    return PredictionResponse(subset_id=subset_id,
                              model_type=payload.model_type,
                              predicted_rul=predicted_rul)
