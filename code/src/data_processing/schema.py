from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType


CMAPSS_COLUMNS = (
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *(f"sensor_{number}" for number in range(1, 22)),
)

CMAPSS_FEATURE_COLUMNS = CMAPSS_COLUMNS[2:]
CMAPSS_SENSOR_COLUMNS = tuple(column for column in CMAPSS_FEATURE_COLUMNS if column.startswith("sensor_"))
PROTECTED_COLUMNS = frozenset(("unit_id", "cycle", "RUL"))
OPERATING_SETTING_COLUMNS = ("setting_1", "setting_2", "setting_3")
FINAL_OBSERVED_RUL_COLUMN = "final_observed_rul"

SUPPORTED_SUBSETS = ("FD001", "FD002", "FD003", "FD004")

CMAPSS_SCHEMA = StructType(
    [
        StructField("unit_id", IntegerType(), nullable=True),
        StructField("cycle", IntegerType(), nullable=True),
        *(
            StructField(column, DoubleType(), nullable=True)
            for column in CMAPSS_FEATURE_COLUMNS
        ),
    ]
)

TEST_RUL_SCHEMA = StructType(
    [
        StructField("unit_id", IntegerType(), nullable=False),
        StructField(FINAL_OBSERVED_RUL_COLUMN, IntegerType(), nullable=False),
    ]
)
