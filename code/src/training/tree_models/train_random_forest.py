from sklearn.ensemble import RandomForestRegressor

from src.training.tree_models.tabular import train_tabular_model


DEFAULT_PARAMETERS= {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 1,
}


def train_random_forest(subset_id, preprocessing_run_id, processed_data_dir, parameters=None, rul_cap=None):
    
    model = RandomForestRegressor(**(parameters or DEFAULT_PARAMETERS))
    
    return train_tabular_model(model=model, model_name="random-forest", subset_id=subset_id,
                               preprocessing_run_id=preprocessing_run_id,
                               processed_data_dir=processed_data_dir,rul_cap=rul_cap)
