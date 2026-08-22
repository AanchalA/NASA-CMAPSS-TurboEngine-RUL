from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from src.data_processing.constants import CMAPSS_FEATURE_COLUMNS, FINAL_OBSERVED_RUL_COLUMN


CMAPSS_SCHEMA = StructType([StructField("unit_id", IntegerType(), nullable=True),
                            StructField("cycle", IntegerType(), nullable=True),
                            *(StructField(column, DoubleType(), nullable=True) 
                              for column in CMAPSS_FEATURE_COLUMNS)])

TEST_RUL_SCHEMA = StructType([StructField("unit_id", IntegerType(), nullable=False),
                              StructField(FINAL_OBSERVED_RUL_COLUMN, IntegerType(), nullable=False)])
