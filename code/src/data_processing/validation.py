from pyspark.sql import Window, functions as F
from pyspark.sql.types import IntegralType, NumericType

from src.data_processing.constants import CMAPSS_COLUMNS, FINAL_OBSERVED_RUL_COLUMN


def validate_cmapss_data(df):    
    validate_columns_and_types(df)
    validate_missing_values(df)
    validate_finite_values(df)
    validate_cycle_sequences(df)


def validate_columns_and_types(df):
    missing_columns = [column for column in CMAPSS_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"schema validation failed. \nMissing required columns {missing_columns}")

    field_types = {field.name: field.dataType for field in df.schema.fields}
    non_numeric = [column for column in CMAPSS_COLUMNS if not isinstance(field_types[column], NumericType)]
    if non_numeric:
        raise ValueError(f"schema validation failed, required columns must be numeric. non-numeric columns={non_numeric}")


def validate_missing_values(df):
    floating_columns = {field.name for field in df.schema.fields if field.name in CMAPSS_COLUMNS and field.dataType.typeName() in {"float", "double"}}
    
    counts = df.agg(*(F.sum(F.when(F.col(column).isNull()
                    | (F.isnan(F.col(column)) if column in floating_columns else F.lit(False)), 1)
                            .otherwise(0)).alias(column) for column in CMAPSS_COLUMNS)).first()
    
    if counts is None:
        return

    by_column = {column: int(counts[column])for column in CMAPSS_COLUMNS if counts[column]}    
    total = sum(by_column.values())
    if total:
        raise ValueError(f"missing-value validation failed. missing or NaN values={total}, by_column={by_column}")


def validate_finite_values(df):
    measurement_columns = CMAPSS_COLUMNS[2:]
    counts = df.agg(*(F.sum(F.when((F.col(column) == F.lit(float("inf"))) 
                                   | (F.col(column) == F.lit(float("-inf"))), 1,)
                            .otherwise(0)).alias(column) for column in measurement_columns)).first()
    if counts is None:
        return

    by_column = {column: int(counts[column]) for column in measurement_columns if counts[column]}
    total = sum(by_column.values())
    if total:
        raise ValueError(f"finite-value validation failed. non finite values={total}, by_column={by_column}")


def validate_cycle_sequences(df):
    cycle_window = Window.partitionBy("unit_id").orderBy("cycle")
    with_previous = df.select("unit_id", "cycle").withColumn("previous_cycle", F.lag("cycle").over(cycle_window))
    
    invalid_engines = (with_previous.where((F.col("previous_cycle").isNull() & (F.col("cycle") != 1))
                                           | (F.col("previous_cycle").isNotNull() & ((F.col("cycle") - F.col("previous_cycle")) != 1)))
                       .select("unit_id").distinct())
    
    invalid_count = invalid_engines.count()
    
    if invalid_count:
        sample_ids = [row["unit_id"] for row in invalid_engines.limit(10).collect()]        
        raise ValueError(f"cycle-sequence validation failed. Engines with invalid cycle sequences={invalid_count}, sample_unit_ids={sample_ids}")


def validate_test_target_inputs(test_df, final_rul_df):

    if "RUL" in test_df.columns:
        raise ValueError("test RUL construction failed. test data already contains RUL")

    reserved_columns = [column for column in (FINAL_OBSERVED_RUL_COLUMN, "final_observed_cycle") if column in test_df.columns]
    if reserved_columns:
        raise ValueError(f"test RUL construction failed: test data contains reserved columns {reserved_columns}")

    required_target_columns = ("unit_id", FINAL_OBSERVED_RUL_COLUMN)
    
    missing_target_columns = [column for column in required_target_columns if column not in final_rul_df.columns]
    if missing_target_columns:
        raise ValueError(f"test RUL construction failed: missing official target columns {missing_target_columns}")

    target_types = {field.name: field.dataType for field in final_rul_df.schema.fields}
    
    non_integral = [column for column in required_target_columns if not isinstance(target_types[column], IntegralType)]
    if non_integral:
        raise ValueError(f"test RUL construction failed, integer official target columns required. non integer columns={non_integral}")

    engine_rows = test_df.select("unit_id").distinct().orderBy("unit_id").collect()
    if not engine_rows:
        raise ValueError("test RUL construction failed: test data contains no engines")

    target_rows = final_rul_df.select(*required_target_columns).orderBy("unit_id").collect()

    target_ids: list[int] = []
    
    for row in target_rows:
        unit_id = row["unit_id"]
        final_rul = row[FINAL_OBSERVED_RUL_COLUMN]
        
        if unit_id is None or final_rul is None:
            raise ValueError("test RUL construction failed: null official unit_id or terminal RUL")
        
        if final_rul < 0:
            raise ValueError(f"test RUL construction failed. official terminal RUL must be non negative. unit_id={unit_id}, value={final_rul}")
        
        target_ids.append(int(unit_id))

    official_record_count = len(target_rows)
    expected_ids = list(range(1, official_record_count + 1))
    
    if target_ids != expected_ids:
        raise ValueError(f"test RUL construction failed. Official unit_ids must match line order IDs 1..{official_record_count}. actual={target_ids}")

    test_ids = [int(row["unit_id"]) for row in engine_rows]
    
    if test_ids != expected_ids:
        missing_ids = sorted(set(expected_ids) - set(test_ids))
        extra_ids = sorted(set(test_ids) - set(expected_ids))
        raise ValueError(f"test RUL engine mapping mismatch. expected official unit_ids 1..{official_record_count} | missing test unit_ids={missing_ids} | extra test unit_ids={extra_ids}")
