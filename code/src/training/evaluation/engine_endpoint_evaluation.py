import hashlib
import pandas as pd
from math import ceil
from pathlib import Path

from src.training.evaluation import nasa_score_contribution


def deterministic_cutoff(unit_id, first_cycle, final_cycle, seed):
    candidates = final_cycle - first_cycle
    if candidates < 1:
        raise ValueError(f"validation unit {unit_id} needs at least two observed cycles")

    digest = hashlib.sha256(f"{seed}:{unit_id}".encode()).digest()
    offset = int.from_bytes(digest[:8], "big") % candidates
    return first_cycle + offset


def select_pseudo_test_endpoints(validation_dataframe, seed=42, minimum_observed_fraction=0.5):

    endpoints = []
    ordered = validation_dataframe.sort_values(["unit_id", "cycle"])

    for unit_id, trajectory in ordered.groupby("unit_id", sort=True):
        
        final_cycle = int(trajectory["cycle"].max())
        first_cycle = max(int(trajectory["cycle"].min()), ceil(final_cycle * minimum_observed_fraction))
        
        cutoff = deterministic_cutoff(unit_id, first_cycle, final_cycle, seed)
        
        endpoint = trajectory.loc[trajectory["cycle"] == cutoff]
        
        if len(endpoint) != 1:
            raise ValueError(f"validation unit {unit_id} has no unique cutoff cycle {cutoff}")
        
        endpoints.append(endpoint.iloc[0])    

    return pd.DataFrame(endpoints).reset_index(drop=True)


def prepare_pseudo_test_validation(validation_dataframe, feature_columns, seed=42, minimum_observed_fraction=0.5):
    
    endpoint_dataframe = select_pseudo_test_endpoints(validation_dataframe, seed=seed, minimum_observed_fraction=minimum_observed_fraction)
    
    return (endpoint_dataframe.loc[:, feature_columns], 
            endpoint_dataframe.loc[:, "RUL"], 
            endpoint_dataframe.loc[:, ["unit_id", "cycle", "RUL"]])


def evaluate_prediction_diagnostics(metadata, predictions):
    
    diagnostics = metadata.rename(columns={"cycle": "cutoff_cycle", "RUL": "true_RUL"}).copy()
    diagnostics["prediction"] = predictions
    diagnostics["error"] = diagnostics["prediction"] - diagnostics["true_RUL"]
    diagnostics["nasa_contribution"] = diagnostics["error"].map(nasa_score_contribution)
    diagnostics = diagnostics.sort_values("nasa_contribution", ascending=False).reset_index(drop=True)
    errors = diagnostics["error"]
    contributions = diagnostics["nasa_contribution"]

    metrics = {"worst_positive_error": float(max(0.0, errors.max())),
               "worst_negative_error": float(min(0.0, errors.min())),
               "largest_nasa_contribution": float(contributions.max()),
               "top_3_nasa_contribution": float(contributions.nlargest(3).sum())}

    return diagnostics, metrics


def prepare_official_test_metadata(test_dataframe, subset_id, raw_data_dir):
    
    final_cycles = (test_dataframe.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=True).tail(1)[["unit_id", "cycle"]])
    official_rul = pd.read_csv( Path(raw_data_dir) / f"RUL_{subset_id}.txt", sep=r"\s+", header=None, names=["RUL"])
    official_rul.insert(0, "unit_id", range(1, len(official_rul) + 1))
    
    return final_cycles.merge(official_rul, on="unit_id", how="inner", validate="one_to_one")
