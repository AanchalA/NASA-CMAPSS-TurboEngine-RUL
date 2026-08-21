from dataclasses import dataclass


@dataclass(frozen=True)
class SensorStatistics:
    mean: float
    std: float


@dataclass(frozen=True)
class GlobalSensorScaler:
    statistics: dict
    @property
    def sensor_columns(self):
        return tuple(self.statistics)


@dataclass(frozen=True)
class RegimeSensorScaler:
    statistics: dict
    @property
    def sensor_columns(self):
        first_regime = next(iter(self.statistics.values()), {})
        return tuple(first_regime)
