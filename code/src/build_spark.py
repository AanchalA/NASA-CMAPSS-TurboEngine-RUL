from contextlib import contextmanager
from pyspark.sql import SparkSession

@contextmanager
def spark_session_context(app_name, spark_master="local[*]"):

    spark = None
    try:
        spark = (SparkSession.builder.master(spark_master)
                 .appName(app_name)
                 .config("spark.ui.enabled", "false")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("WARN")
        yield spark
    finally:
        if spark is not None:
            spark.stop()