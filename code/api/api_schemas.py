from typing import Annotated

from pydantic import BaseModel, Field


class Observation(BaseModel):
    unit_id: Annotated[int, Field(ge=1)]
    cycle: Annotated[int, Field(ge=1)]
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    subset_id: str
    model_type: str
    observations: Annotated[list[Observation], Field(min_length=1)]


class PredictionResponse(BaseModel):
    subset_id: str
    model_type: str
    predicted_rul: float
