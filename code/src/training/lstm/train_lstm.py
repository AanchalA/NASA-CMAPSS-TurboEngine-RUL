import json
from os import getenv
from pathlib import Path

import mlflow
import mlflow.pytorch as mlflow_pytorch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_processing import (CMAPSS_SENSOR_COLUMNS,
                                 build_endpoint_sequences,
                                 build_sliding_window_sequences,
                                 select_tabular_model_inputs)
from src.tracking import configure_mlflow
from src.training.evaluation import (evaluate_predictions,
                                     evaluate_prediction_diagnostics,
                                     prepare_official_test_metadata,
                                     select_pseudo_test_endpoints)


SEQUENCE_LENGTH = 30


class LSTMRegressor(nn.Module):
    def __init__(self, feature_count, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs)
        return self.output(hidden[-1]).squeeze(-1)


def predict_sequences(model, sequences, batch_size=256):
    sequences_by_length = {}
    
    for index, sequence in enumerate(sequences):
        sequences_by_length.setdefault(len(sequence), []).append((index, sequence))

    predictions = np.empty(len(sequences), dtype=np.float32)
    model.eval()
    
    with torch.no_grad():
        for indexed_sequences in sequences_by_length.values():
            indices, same_length_sequences = zip(*indexed_sequences)
            
            loader = DataLoader(TensorDataset(torch.as_tensor(np.stack(same_length_sequences), dtype=torch.float32)),
                                batch_size=batch_size)
            offset = 0
            
            for (features,) in loader:
                batch_predictions = model(features).cpu().numpy()
                batch_indices = indices[offset : offset + len(batch_predictions)]
                predictions[list(batch_indices)] = batch_predictions
                offset += len(batch_predictions)
                
    return predictions


def train_lstm(subset_id, preprocessing_run_id, processed_data_dir, raw_data_dir, 
               epochs=30, batch_size=64, learning_rate=1e-3, seed=42, 
               rul_cap=None, feature_set="base_sequence"):
    
    normalized_subset = subset_id.upper()    

    configure_mlflow()    
    torch.manual_seed(seed)
    
    subset_path = Path(processed_data_dir) / normalized_subset / preprocessing_run_id
    train_dataframe = pd.read_parquet(subset_path / "train")
    validation_dataframe = pd.read_parquet(subset_path / "validation")
    test_dataframe = pd.read_parquet(subset_path / "test")
    
    if feature_set == "temporal_sequence":
        feature_columns = list(select_tabular_model_inputs(train_dataframe)[0].columns)
    else:
        feature_columns = [sensor for sensor in CMAPSS_SENSOR_COLUMNS if sensor in train_dataframe.columns]
        
    for dataframe in (train_dataframe, validation_dataframe, test_dataframe):
        dataframe[feature_columns] = dataframe[feature_columns].fillna(0).astype(np.float32)

    train_sequences, train_targets = build_sliding_window_sequences(train_dataframe, feature_columns, SEQUENCE_LENGTH)
    
    if rul_cap is not None:
        train_targets = np.minimum(train_targets, rul_cap)
        
    validation_metadata = select_pseudo_test_endpoints(validation_dataframe, seed=seed)
    validation_sequences = build_endpoint_sequences(validation_dataframe, validation_metadata, feature_columns, SEQUENCE_LENGTH)    
    validation_targets = validation_metadata["RUL"].to_numpy()

    test_metadata = prepare_official_test_metadata(test_dataframe, normalized_subset, raw_data_dir)
    test_sequences = build_endpoint_sequences(test_dataframe, test_metadata, feature_columns, SEQUENCE_LENGTH)

    dataset = TensorDataset(torch.as_tensor(train_sequences, dtype=torch.float32),
                            torch.as_tensor(train_targets, dtype=torch.float32))
    
    generator = torch.Generator().manual_seed(seed)    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
    model = LSTMRegressor(len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    mlflow.set_experiment(getenv("CMAPSS_MLFLOW_TRAINING_EXPERIMENT", "cmapss-training"))    
    target_name = "raw_rul" if rul_cap is None else f"capped_rul_{rul_cap:g}"
    
    with mlflow.start_run(run_name=f"{normalized_subset}-lstm-{feature_set}-{target_name}") as run:
        print(f"[{normalized_subset}] fitting LSTM on {len(dataset):,} windows, {SEQUENCE_LENGTH} cycles and {len(feature_columns)} features "
              f"with target={target_name}...", flush=True)
        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
        
            for features, targets in loader:
                optimizer.zero_grad()
                loss = loss_function(model(features), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)
                
            epoch_loss = total_loss / len(dataset)
            
            mlflow.log_metric("training_mse", epoch_loss, step=epoch)            
            print(f"[{normalized_subset}] epoch {epoch:02d}/{epochs}: mse={epoch_loss:.4f}", flush=True)

        validation_predictions = predict_sequences(model, validation_sequences)
        validation_metrics = evaluate_predictions(validation_targets, validation_predictions)
        
        test_predictions = predict_sequences(model, test_sequences)
        test_metrics = evaluate_predictions(test_metadata["RUL"], test_predictions)
        
        validation_diagnostics, validation_diagnostic_metrics = evaluate_prediction_diagnostics(validation_metadata, validation_predictions)
        validation_metrics.update(validation_diagnostic_metrics)
        
        test_diagnostics, test_diagnostic_metrics = evaluate_prediction_diagnostics(test_metadata, test_predictions)
        test_metrics.update(test_diagnostic_metrics)

        mlflow.log_params({"model_type": type(model).__name__,
                           "subset": normalized_subset,
                           "preprocessing_run_id": preprocessing_run_id,
                           "feature_set": feature_set,
                           "feature_count": len(feature_columns),
                           "sequence_length": SEQUENCE_LENGTH,
                           "hidden_size": 64,
                           "num_layers": 1,
                           "dropout": 0,
                           "target": target_name,
                           "rul_cap": "none" if rul_cap is None else rul_cap,
                           "epochs": epochs,
                           "batch_size": batch_size,
                           "learning_rate": learning_rate,
                           "random_seed": seed,
                           "validation_protocol": "pseudo_test_endpoint",
                           "validation_cutoff_seed": seed,
                           "validation_minimum_observed_fraction": 0.5,
                           "validation_endpoint_count": len(validation_targets)})
        
        mlflow.log_text(json.dumps(feature_columns), "feature_names.json")
        
        mlflow.log_metrics(validation_metrics)
        mlflow.log_metrics({f"test_{name}": value for name, value in test_metrics.items()})
        
        mlflow.log_text(validation_diagnostics.to_csv(index=False), "validation_endpoint_predictions.csv")
        mlflow.log_text(test_diagnostics.to_csv(index=False), "test_predictions.csv")
        
        mlflow_pytorch.log_model(model, name="model", input_example=train_sequences[:1].astype(np.float32), serialization_format="pickle")

    return {"subset_id": normalized_subset,
            "preprocessing_run_id": preprocessing_run_id,
            "training_run_id": run.info.run_id,
            "training_target": target_name,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics}
