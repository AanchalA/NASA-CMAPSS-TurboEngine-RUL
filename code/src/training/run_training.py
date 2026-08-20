import argparse

from src.build_spark import spark_session_context
from src.tracking.fetch_id import fetch_preprocessing_run_id
from src.training.test_evaluation import evaluate_test_data
from src.training.train import train_random_forest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-id", default="FD001")    
    parser.add_argument("--processed-data-dir", default="Data/processed")
    parser.add_argument("--raw-data-dir", default="Data/CMAPSSData")    
    return parser.parse_args()


def run_random_forest_baseline(spark, subset_id, preprocessing_run_id, 
                               processed_data_dir, raw_data_dir, parameters=None):
    
    training_run_id = train_random_forest(spark=spark, subset_id=subset_id, 
                                          preprocessing_run_id=preprocessing_run_id, 
                                          processed_data_dir=processed_data_dir, 
                                          parameters=parameters)
    
    test_metrics = evaluate_test_data(spark=spark, subset_id=subset_id,
                                      training_run_id=training_run_id,
                                      processed_data_dir=processed_data_dir,
                                      raw_data_dir=raw_data_dir)

    return {"subset_id": subset_id.upper(), 
            "preprocessing_run_id": preprocessing_run_id,
            "training_run_id": training_run_id,
            "test_metrics": test_metrics}


if __name__ == "__main__":
    args = parse_args()
    
    model_parameters = {"n_estimators": 100, 
                        "random_state": 42,
                        "n_jobs": -1}

    with spark_session_context(app_name="cmapss-random-forest-baseline") as spark:
        result = run_random_forest_baseline(
            spark=spark,
            subset_id=args.subset_id,
            preprocessing_run_id=fetch_preprocessing_run_id(args.subset_id),
            processed_data_dir=args.processed_data_dir,
            raw_data_dir=args.raw_data_dir,
            parameters=model_parameters,
        )

    print(result)
