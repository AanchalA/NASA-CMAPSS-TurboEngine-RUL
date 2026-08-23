from xgboost import XGBRegressor

from src.training.tree_models.tabular import train_tabular_model


DEFAULT_PARAMETERS = {"objective": "reg:squarederror",
                      "n_estimators": 500,
                      "max_depth": 6,
                      "learning_rate": 0.05,
                      "subsample": 0.8,
                      "colsample_bytree": 0.8,
                      "random_state": 42,
                      "n_jobs": -1}


def train_xgboost(subset_id, preprocessing_run_id, processed_data_dir, parameters=None, rul_cap=125):
    
    model = XGBRegressor(**(parameters or DEFAULT_PARAMETERS))
    
    return train_tabular_model(model=model, model_name="xgboost", subset_id=subset_id, 
                               preprocessing_run_id=preprocessing_run_id, 
                               processed_data_dir=processed_data_dir, rul_cap=rul_cap)
