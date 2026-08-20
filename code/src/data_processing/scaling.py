from dataclasses import dataclass

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F


@dataclass(frozen=True)
class SensorStatistics:
    mean: float
    std: float


@dataclass(frozen=True)
class RegimeSensorScaler:
    statistics: dict
    @property
    def sensor_columns(self):
        first_regime = next(iter(self.statistics.values()), {})
        return tuple(first_regime)


def statistics_from_row(row, sensor_columns):
    statistics = {}
    
    for index, sensor in enumerate(sensor_columns):
        mean = row[f"mean_{index}"]
        std = row[f"std_{index}"]
        minimum = row[f"minimum_{index}"]
        maximum = row[f"maximum_{index}"]
        effective_std = 0.0 if minimum == maximum else float(std)
        statistics[sensor] = SensorStatistics(float(mean), effective_std)
    
    return statistics


def aggregation_expressions(sensor_columns):    
    return [expression for index, sensor in enumerate(sensor_columns) 
            for expression in (F.mean(sensor).alias(f"mean_{index}"),
                               F.stddev_pop(sensor).alias(f"std_{index}"),
                               F.min(sensor).alias(f"minimum_{index}"),
                               F.max(sensor).alias(f"maximum_{index}"))]


def fit_regime_sensor_scaler(train_df, sensor_columns):    
    rows = train_df.groupBy("regime").agg(*aggregation_expressions(sensor_columns)).collect()

    statistics = {}   
    for row in rows:
        regime = int(row["regime"])
        regime_statistics = statistics_from_row(row, sensor_columns)
        statistics[regime] = regime_statistics

    return RegimeSensorScaler(statistics=statistics)


def regime_scaled_column(sensor, scaler):
    expression = F.lit(None).cast("double")
    
    for regime, regime_statistics in scaler.statistics.items():
        values = regime_statistics[sensor]
        scaled = (F.lit(0.0) if values.std == 0.0 else (F.col(sensor) - F.lit(values.mean)) / F.lit(values.std))
        expression = F.when(F.col("regime") == F.lit(regime), scaled).otherwise(expression)
    
    return expression.alias(sensor)


def apply_global_sensor_scaler(df, scaler_model: PipelineModel, sensor_columns):
    scaled = scaler_model.transform(df)
    sensor_indexes = {sensor: index for index, sensor in enumerate(sensor_columns)}
    
    scaled_values = vector_to_array(F.col("scaled_sensor_features"))
    
    return scaled.select(*(scaled_values[sensor_indexes[column]].alias(column) 
                           if column in sensor_indexes else F.col(column) 
                           for column in df.columns))


def apply_sensor_scaler(df, scaler_stats):
    scaled = set(scaler_stats.sensor_columns)
    expressions = (regime_scaled_column(column, scaler_stats) 
                   if column in scaled else F.col(column) 
                   for column in df.columns)
    
    return df.select(*expressions)
