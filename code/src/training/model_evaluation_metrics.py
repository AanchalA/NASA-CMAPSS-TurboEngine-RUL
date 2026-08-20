from math import expm1, fsum

from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_predictions(y_true, y_pred):
    
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    errors = (prediction - truth for truth, prediction in zip(y_true, y_pred))
    
    nasa_score = fsum(expm1(-error / 13) if error < 0 else expm1(error / 10) for error in errors)

    return {"rmse": float(rmse), "mae": float(mae), "nasa_score": nasa_score}
