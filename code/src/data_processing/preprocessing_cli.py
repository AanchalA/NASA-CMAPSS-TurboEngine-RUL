import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
from pathlib import Path

from pyspark.sql import SparkSession

from src.data_processing.schema import SUPPORTED_SUBSETS
from src.data_processing.preprocessing import run_subset_preprocessing


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, choices=SUPPORTED_SUBSETS)
    parser.add_argument("--raw-data-dir", required=True, type=Path)
    parser.add_argument("--processed-data-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spark-master", default="local[*]")
    return parser

@contextmanager
def spark_session_context(spark_master, subset):
    spark = None
    try:
        spark = (SparkSession.builder.master(spark_master)
                 .appName(f"cmapss-preprocessing-{subset}")
                 .config("spark.ui.enabled", "false")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("WARN")
        yield spark
    finally:
        if spark is not None:
            spark.stop()
  

def main(argv=None):
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with spark_session_context(args.spark_master, args.subset) as spark:
        result = run_subset_preprocessing(
            spark=spark,
            subset=args.subset,
            raw_data_dir=args.raw_data_dir,
            output_dir=args.processed_data_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
