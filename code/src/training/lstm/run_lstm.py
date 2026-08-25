import argparse

from src.tracking.mlflow_run_id import fetch_run_id
from src.training.lstm import train_lstm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-id", default="FD001")
    parser.add_argument("--preprocessing-run-id")
    parser.add_argument("--processed-data-dir", default="Data/processed")
    parser.add_argument("--raw-data-dir", default="Data/CMAPSSData")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    preprocessing_run_id = args.preprocessing_run_id or fetch_run_id(args.subset_id)
    
    result = train_lstm(subset_id=args.subset_id,
                        preprocessing_run_id=preprocessing_run_id,
                        processed_data_dir=args.processed_data_dir,
                        raw_data_dir=args.raw_data_dir,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate)
    
    print(result)
