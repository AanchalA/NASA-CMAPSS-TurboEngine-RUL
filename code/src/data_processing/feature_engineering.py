from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from src.data_processing.schema import (
    CMAPSS_FEATURE_COLUMNS,
    OPERATING_SETTING_COLUMNS,
    PROTECTED_COLUMNS,
)


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
