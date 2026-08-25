from math import expm1, fsum

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def life_ratio_to_rul(cycle, life_ratio, minimum_life_ratio=1e-6):
    cycles = np.asarray(cycle, dtype=float)
    ratios = np.clip(np.asarray(life_ratio, dtype=float), minimum_life_ratio, 1.0)
    remaining_life = cycles * (1.0 / ratios - 1.0)
    return float(remaining_life) if remaining_life.ndim == 0 else remaining_life


def nasa_score_contribution(error):
    return expm1(-error / 13) if error < 0 else expm1(error / 10)


def evaluate_predictions(y_true, y_pred):
    
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    errors = [prediction - truth for truth, prediction in zip(y_true, y_pred)]
    
    bias = fsum(errors) / len(errors)

    late_prediction_rate = (sum(error > 0 for error in errors) / len(errors))
    
    nasa_score = fsum(nasa_score_contribution(error) for error in errors)

    return {"rmse": float(rmse), "mae": float(mae), "nasa_score": nasa_score,
            "bias": float(bias), "late_prediction_rate": float(late_prediction_rate)}
