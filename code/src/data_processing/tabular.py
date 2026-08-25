import pandas as pd

from src.data_processing.constants import DEFAULT_EWMA_ALPHA, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS


TABULAR_METADATA_COLUMNS = frozenset(("unit_id", "cycle", "life_ratio", "operating_regime"))


def add_pandas_temporal_features(dataframe, sensor_columns):
    
    trajectory = dataframe.sort_values("cycle").reset_index(drop=True)
    cycles = trajectory["cycle"]
    temporal_columns = {}

    for sensor in sensor_columns:
        values = trajectory[sensor]
        
        for lag in DEFAULT_LAGS:
            temporal_columns[f"{sensor}_lag_{lag}"] = values.shift(lag)
            temporal_columns[f"{sensor}_diff_{lag}"] = values - values.shift(lag)
        
        for window in DEFAULT_ROLLING_WINDOWS:
            rolling_values = values.rolling(window, min_periods=1)
            rolling_cycles = cycles.rolling(window, min_periods=1)
            
            count = rolling_values.count()
            
            covariance = (values * cycles).rolling(window, min_periods=1).sum() - (rolling_values.sum() * rolling_cycles.sum() / count)
            cycle_variance = (cycles * cycles).rolling(window, min_periods=1).sum() - (rolling_cycles.sum() ** 2 / count)
            
            temporal_columns[f"{sensor}_rolling_mean_{window}"] = rolling_values.mean()
            temporal_columns[f"{sensor}_rolling_std_{window}"] = rolling_values.std()
            temporal_columns[f"{sensor}_rolling_slope_{window}"] = covariance / cycle_variance
            
        temporal_columns[f"{sensor}_ewma"] = values.ewm(alpha=DEFAULT_EWMA_ALPHA, adjust=False).mean()

    return pd.concat((trajectory, pd.DataFrame(temporal_columns)), axis=1)


def select_tabular_model_inputs(dataframe, feature_columns=None):
    if feature_columns is None:
        feature_columns = [column for column in dataframe.columns if column not in TABULAR_METADATA_COLUMNS]
    return dataframe.loc[:, feature_columns], dataframe.loc[:, "life_ratio"]
