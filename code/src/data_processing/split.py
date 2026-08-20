from math import floor

from pyspark.sql import functions as F


def split_train_validation_by_unit(df, validation_fraction=0.2, seed=42):
    """Splitting training data deterministically while keeping engines intact."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    unit_ids = df.select("unit_id").distinct()
    engine_count = unit_ids.count()
    if engine_count < 2:
        raise ValueError("at least two engines are required for a non-empty split")

    rounded_count = floor(engine_count * validation_fraction + 0.5)
    validation_count = min(max(rounded_count, 1), engine_count - 1)

    validation_ids = (unit_ids.withColumn("split_score", F.xxhash64(F.col("unit_id"), F.lit(seed)))
                      .orderBy(F.col("split_score"), F.col("unit_id"))
                      .limit(validation_count)
                      .select("unit_id"))

    train_df = df.join(validation_ids, on="unit_id", how="left_anti")
    validation_df = df.join(validation_ids, on="unit_id", how="left_semi")
    
    return train_df, validation_df
