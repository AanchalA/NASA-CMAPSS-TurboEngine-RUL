import logging
from time import perf_counter

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.tracking import fetch_run_id
from src.inference import predict_rul
from api.api_schemas import PredictRequest, PredictionResponse
from src.inference.inference_monitoring import RUL_PREDICTED_VALUE, RUL_PREDICTION_LATENCY, RUL_PREDICTIONS


LOGGER = logging.getLogger(__name__)

app = FastAPI(title="NASA C-MAPSS RUL API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=str(CONTENT_TYPE_LATEST))


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest) -> PredictionResponse:
    subset_id = payload.subset_id.upper()
    labels = {"subset": subset_id, "model_type": payload.model_type}
    started_at = perf_counter()
    status = "error"
    training_run_id = None

    try:
        training_run_id = fetch_run_id(subset_id=subset_id,
                                       experiment_type="training",
                                       model_type=payload.model_type)

        observations = [observation.model_dump() for observation in payload.observations]

        predicted_rul = predict_rul(subset_id, training_run_id, observations)

        RUL_PREDICTED_VALUE.labels(**labels).set(predicted_rul)
        status = "success"

        LOGGER.info("RUL prediction completed status=success subset=%s model_type=%s "
                    "training_run_id=%s observation_count=%d predicted_rul=%.6f "
                    "duration_seconds=%.6f",
                    subset_id,
                    payload.model_type,
                    training_run_id,
                    len(observations),
                    predicted_rul,
                    perf_counter() - started_at)

        return PredictionResponse(subset_id=subset_id,
                                  model_type=payload.model_type,
                                  predicted_rul=predicted_rul)

    except Exception:
        LOGGER.exception("RUL prediction completed status=error subset=%s model_type=%s "
                         "training_run_id=%s observation_count=%d duration_seconds=%.6f",
                         subset_id,
                         payload.model_type,
                         training_run_id,
                         len(payload.observations),
                         perf_counter() - started_at)
        raise

    finally:
        duration = perf_counter() - started_at
        RUL_PREDICTIONS.labels(status=status, **labels).inc()
        RUL_PREDICTION_LATENCY.labels(status=status, **labels).observe(duration)
