from lightgbm import LGBMRegressor

from src.training.tree_models.tabular import train_tabular_model


DEFAULT_PARAMETERS = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def train_lightgbm(subset_id, preprocessing_run_id, processed_data_dir, parameters=None):
    model = LGBMRegressor(**(parameters or DEFAULT_PARAMETERS))

    return train_tabular_model(model=model, model_name="lightgbm", subset_id=subset_id,
                               preprocessing_run_id=preprocessing_run_id,
                               processed_data_dir=processed_data_dir)
