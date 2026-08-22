from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from src.data_processing.constants import (CMAPSS_FEATURE_COLUMNS, DEFAULT_EWMA_ALPHA,
                                           DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS,
                                           OPERATING_SETTING_COLUMNS, PROTECTED_COLUMNS)


KEY_COLUMNS = ("regime_setting_1", "regime_setting_2", "regime_setting_3")


# _______________ CONSTANT FEATURE DROPPING _______________
def find_constant_feature_columns(train_df):

    statistics = train_df.agg(*(expression for index, column in enumerate(CMAPSS_FEATURE_COLUMNS)
                                for expression in (F.min(column).alias(f"minimum_{index}"),
                                                   F.max(column).alias(f"maximum_{index}")))).first()
    if statistics is None:
        return []

    return [column for index, column in enumerate(CMAPSS_FEATURE_COLUMNS)
            if statistics[f"minimum_{index}"] is not None
            and statistics[f"minimum_{index}"] == statistics[f"maximum_{index}"]]


def drop_feature_columns(df, columns):
    protected = PROTECTED_COLUMNS.intersection(columns)
    if protected:
        raise ValueError("protected columns cannot be removed: " + ", ".join(sorted(protected)))

    return df.drop(*columns)


# _______________ REGIME SETUP _______________
def row_to_regime_key(row):
    return (float(row[KEY_COLUMNS[0]]), float(row[KEY_COLUMNS[1]]), float(row[KEY_COLUMNS[2]]))


def with_regime_key(df):
    keyed = df
    for setting, decimals, key_column in zip(OPERATING_SETTING_COLUMNS, (0, 2, 0), KEY_COLUMNS, strict=True):
        keyed = keyed.withColumn(key_column, F.bround(F.col(setting), decimals))
    return keyed


def fit_operating_regime_mapping(train_df):
    keys = [row_to_regime_key(row) for row in with_regime_key(train_df).select(*KEY_COLUMNS).distinct().collect()]
    if not keys:
        raise ValueError("cannot fit operating regimes from an empty training DataFrame")
    return {key: regime_id for regime_id, key in enumerate(sorted(keys), start=1)}


def mapping_dataframe(spark, regime_mapping):
    schema = StructType([*(StructField(column, DoubleType(), nullable=False) for column in KEY_COLUMNS),
                         StructField("fitted_regime", IntegerType(), nullable=False)])
    rows = [(*key, regime_id) for key, regime_id in regime_mapping.items()]
    return spark.createDataFrame(rows, schema=schema)


def assign_operating_regimes(df, regime_mapping):
    if "regime" in df.columns:
        raise ValueError("input DataFrame already contains a regime column")
    if not regime_mapping:
        raise ValueError("operating-regime mapping must not be empty")

    keyed = with_regime_key(df)
    mapping_df = mapping_dataframe(df.sparkSession, regime_mapping)
    assigned = keyed.join(F.broadcast(mapping_df), on=list(KEY_COLUMNS), how="left")

    unseen = assigned.filter(F.col("fitted_regime").isNull())
    unseen_keys = [row_to_regime_key(row) for row in unseen.select(*KEY_COLUMNS).distinct().limit(20).collect()]
    if unseen_keys:
        raise ValueError("operating-regime keys unseen in training: " + str(sorted(unseen_keys)))

    output_columns = list(df.columns)
    regime_position = output_columns.index("cycle") + 1 if "cycle" in output_columns else 0
    output_columns.insert(regime_position, "regime")

    return assigned.select(*(F.col("fitted_regime").cast("int").alias("regime")
                             if column == "regime" else F.col(column) for column in output_columns))


# _______________ TEMPORAL FEATURES _______________
def engine_cycle_window():
    return Window.partitionBy("unit_id").orderBy("cycle")


def add_lag_difference_features(df, sensor_columns, lags=DEFAULT_LAGS):

    sensor_columns = tuple(sensor_columns)
    lags = tuple(lags)

    generated_columns = {f"{sensor}_{feature}_{lag}" for sensor in sensor_columns for lag in lags for feature in ("lag", "diff")}

    collisions = generated_columns.intersection(df.columns)
    if collisions:
        raise ValueError("generated columns already exist: " + ", ".join(sorted(collisions)))

    cycle_window = engine_cycle_window()

    lag_expressions = [F.lag(F.col(sensor), lag).over(cycle_window).alias(f"{sensor}_lag_{lag}")
                       for sensor in sensor_columns for lag in lags]
    lagged = df.select("*", *lag_expressions)

    difference_expressions = [(F.col(sensor) - F.col(f"{sensor}_lag_{lag}")).alias(f"{sensor}_diff_{lag}")
                              for sensor in sensor_columns for lag in lags]

    return lagged.select("*", *difference_expressions)


def rolling_slope_and_statistics_features(df, sensor_columns, windows=DEFAULT_ROLLING_WINDOWS):
    sensor_columns = tuple(sensor_columns)
    windows = tuple(windows)

    cycle_window = engine_cycle_window()
    rolling_windows = {window: cycle_window.rowsBetween(-(window - 1), 0) for window in windows}

    rolling_expressions = [expression.over(rolling_windows[window]).alias(f"{sensor}_rolling_{statistic}_{window}")
                           for sensor in sensor_columns for window in windows
                           for statistic, expression in (("mean", F.avg(F.col(sensor))), ("std", F.stddev_samp(F.col(sensor))))]

    slope_expressions = [F.regr_slope(F.col(sensor), F.col("cycle")).over(rolling_windows[window])
                         .alias(f"{sensor}_rolling_slope_{window}") for sensor in sensor_columns for window in windows]

    return df.select("*", *rolling_expressions, *slope_expressions)


def add_ewma_features(df, sensor_columns, alpha=DEFAULT_EWMA_ALPHA):
    sensor_columns = tuple(sensor_columns)

    history_window = engine_cycle_window().rowsBetween(Window.unboundedPreceding, Window.currentRow)
    history_columns = {sensor: f"__{sensor}_ewma_history" for sensor in sensor_columns}

    with_histories = df.select("*", *(F.collect_list(F.col(sensor)).over(history_window).alias(history_columns[sensor])
                                      for sensor in sensor_columns))

    ewma_expressions = [F.aggregate(F.slice(F.col(history_columns[sensor]), 2,F.size(F.col(history_columns[sensor])) - 1,),
                                    F.element_at(F.col(history_columns[sensor]), 1),
                                    lambda previous, current: alpha * current + (1 - alpha) * previous).alias(f"{sensor}_ewma")
                        for sensor in sensor_columns]

    return with_histories.select(*df.columns, *ewma_expressions)
