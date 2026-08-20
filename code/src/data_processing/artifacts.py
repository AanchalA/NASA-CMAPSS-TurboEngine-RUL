from dataclasses import dataclass

from src.data_processing.scaling import RegimeSensorScaler, SensorStatistics
from src.data_processing.schema import CMAPSS_SENSOR_COLUMNS


@dataclass(frozen=True)
class PreprocessingArtifacts:
    regime_mapping: dict
    dropped_feature_columns: tuple
    retained_sensor_columns: tuple
    regime_sensor_scaler: object = None

    def __post_init__(self):
        retained = self.retained_sensor_columns
        
        if len(retained) != len(set(retained)):
            raise ValueError("retained_sensor_columns must not contain duplicates")

        unknown = [column for column in retained if column not in CMAPSS_SENSOR_COLUMNS]
        if unknown:
            raise ValueError("retained_sensor_columns contains non-canonical sensors: " + ", ".join(unknown))

        canonical = tuple(column for column in CMAPSS_SENSOR_COLUMNS if column in set(retained))
        if retained != canonical:
            raise ValueError("retained_sensor_columns must use canonical sensor order")

        overlap = set(retained).intersection(self.dropped_feature_columns)
        if overlap:
            raise ValueError("retained sensors must not also be dropped: "+ ", ".join(sorted(overlap)))


def preprocessing_artifacts_data(artifacts):
    scaler = artifacts.regime_sensor_scaler
    scaler_data = None
    if scaler is not None:
        scaler_data = [{"regime": regime,
                        "sensors": [{"sensor": sensor, "mean": values.mean, "std": values.std}
                                    for sensor, values in scaler.statistics[regime].items()]}
                       for regime in sorted(scaler.statistics)]

    return {"regime_mapping": [{"setting_1": key[0], "setting_2": key[1], "setting_3": key[2], "regime": regime}
                               for key, regime in sorted(artifacts.regime_mapping.items(), key=lambda item: item[1])],
            "dropped_feature_columns": list(artifacts.dropped_feature_columns),
            "retained_sensor_columns": list(artifacts.retained_sensor_columns),
            "regime_sensor_scaler": scaler_data}


def preprocessing_artifacts_from_data(root):
    regime_records = root["regime_mapping"]
    regime_mapping = {}

    for record in regime_records:
        key = (record["setting_1"], record["setting_2"], record["setting_3"])
        regime = record["regime"]
        regime_mapping[key] = regime

    scaler_records = root["regime_sensor_scaler"]
    regime_sensor_scaler = None

    if scaler_records is not None:
        statistics = {}

        for regime_record in scaler_records:
            regime = regime_record["regime"]
            statistics[regime] = {record["sensor"]: SensorStatistics(mean=record["mean"], std=record["std"]) 
                                  for record in regime_record["sensors"]}
        
        regime_sensor_scaler = RegimeSensorScaler(statistics=statistics)

    dropped_values = root["dropped_feature_columns"]
    retained_sensor_values = root["retained_sensor_columns"]

    artifacts = PreprocessingArtifacts(regime_mapping=regime_mapping,
                                       dropped_feature_columns=tuple(dropped_values),
                                       retained_sensor_columns=tuple(retained_sensor_values),
                                       regime_sensor_scaler=regime_sensor_scaler)

    return artifacts
