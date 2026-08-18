from pathlib import Path
from os import fspath

from pyspark.sql.types import IntegerType
from pyspark.sql import Window, functions as F

from src.data_processing.schema import CMAPSS_SCHEMA, FINAL_OBSERVED_RUL_COLUMN, TEST_RUL_SCHEMA
from src.data_processing.validation import validate_test_target_inputs



# _______________ TRAIN DATA LOADER _______________ 
def load_cmapss_raw(spark, path):    
    
    tokens = F.split(F.trim(F.col("value")), r"\s+")
    raw_lines = spark.read.text(fspath(path))
    return raw_lines.select(*(tokens[index].cast(field.dataType).alias(field.name)
            for index, field in enumerate(CMAPSS_SCHEMA.fields)))   


def add_rul_target(df):    
    engine_window = Window.partitionBy("unit_id")
    maximum_cycle = F.max("cycle").over(engine_window)

    return df.withColumn("RUL", maximum_cycle - F.col("cycle"))


# _______________ TEST DATA LOADER _______________ 
def load_cmapss_test_rul(spark, path):    
    target_path = Path(fspath(path))
    records: list[tuple[int, int]] = []
            
    lines = target_path.read_text(encoding="utf-8").splitlines()    
    for _, line in enumerate(lines, start=1):
        tokens = line.split()        
        if not tokens:
            continue                      
        
        records.append((len(records) + 1, int(tokens[0])))
        
    if not records:
        raise ValueError("CMAPSS test RUL file contains no target records")

    return spark.createDataFrame(records, schema=TEST_RUL_SCHEMA).orderBy("unit_id")


def add_test_rul_target(test_df, final_rul_df):
    validate_test_target_inputs(test_df, final_rul_df)

    observed_final_cycles = test_df.groupBy("unit_id").agg(F.max("cycle").alias("final_observed_cycle"))
    target_by_engine = observed_final_cycles.join(final_rul_df.select("unit_id", FINAL_OBSERVED_RUL_COLUMN), on="unit_id", how="inner")
    
    result = (test_df.join(target_by_engine, on="unit_id", how="inner").withColumn("RUL", 
                (F.col(FINAL_OBSERVED_RUL_COLUMN) + F.col("final_observed_cycle") - F.col("cycle")).cast(IntegerType()))
              .drop("final_observed_cycle", FINAL_OBSERVED_RUL_COLUMN).select(*test_df.columns, "RUL"))

    return result
