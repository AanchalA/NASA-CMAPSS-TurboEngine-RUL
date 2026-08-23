from math import expm1, fsum

from sklearn.metrics import mean_absolute_error, root_mean_squared_error


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
