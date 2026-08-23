import argparse

from src.training import evaluate_test_data
from src.tracking.mlflow_run_id import fetch_run_id
from src.training.tree_models import train_random_forest, train_xgboost


MODEL_TYPES = ("random_forest", "xgboost")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--subset-id", default="FD001")
    parser.add_argument("--processed-data-dir", default="Data/processed")
    parser.add_argument("--raw-data-dir", default="Data/CMAPSSData")    
    return parser.parse_args()


def run_training(model_type, subset_id, preprocessing_run_id,
                 processed_data_dir, raw_data_dir, parameters=None):
    
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}. Expected one of {MODEL_TYPES}.")
    
    print(f"[{subset_id.upper()}] starting model training...", flush=True)

    trainer = train_random_forest if model_type == "random_forest" else train_xgboost
    
    training_run_id = trainer(subset_id=subset_id,
                              preprocessing_run_id=preprocessing_run_id,
                              processed_data_dir=processed_data_dir,
                              parameters=parameters)

    print(f"[{subset_id.upper()}] evaluating the test set...", flush=True)
    
    test_metrics = evaluate_test_data(subset_id=subset_id,
                                      training_run_id=training_run_id,
                                      processed_data_dir=processed_data_dir,
                                      raw_data_dir=raw_data_dir)

    return {"subset_id": subset_id.upper(), 
            "preprocessing_run_id": preprocessing_run_id,
            "training_run_id": training_run_id,
            "test_metrics": test_metrics}


if __name__ == "__main__":
    args = parse_args()

    result = run_training(model_type=args.model_type,
                          subset_id=args.subset_id,
                          preprocessing_run_id=fetch_run_id(args.subset_id),
                          processed_data_dir=args.processed_data_dir,
                          raw_data_dir=args.raw_data_dir)

    print(result)
