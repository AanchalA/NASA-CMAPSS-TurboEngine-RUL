import numpy as np


DEFAULT_SEQUENCE_LENGTH = 30


def build_sliding_window_sequences(dataframe, feature_columns, sequence_length=DEFAULT_SEQUENCE_LENGTH):    
    
    ordered = dataframe.sort_values(["unit_id", "cycle"])
    sequences = []
    targets = []

    for _, engine in ordered.groupby("unit_id", sort=False):
        
        if len(engine) < sequence_length:
            continue

        values = engine.loc[:, feature_columns].to_numpy()
        engine_sequences = np.lib.stride_tricks.sliding_window_view(values, window_shape=sequence_length, axis=0).transpose(0, 2, 1)
        sequences.append(engine_sequences)
        targets.append(engine.loc[:, "life_ratio"].to_numpy()[sequence_length - 1:])

    if not sequences:
        return (np.empty((0, sequence_length, len(feature_columns))),
                np.empty((0,), dtype=dataframe["life_ratio"].dtype))

    return np.concatenate(sequences), np.concatenate(targets)


def build_endpoint_sequences(dataframe, endpoints, feature_columns, sequence_length=DEFAULT_SEQUENCE_LENGTH):        
    ordered = dataframe.sort_values(["unit_id", "cycle"])
    sequences = []

    for endpoint in endpoints.itertuples(index=False):
        trajectory = ordered.loc[(ordered["unit_id"] == endpoint.unit_id) 
                                 & (ordered["cycle"] <= endpoint.cycle)].tail(sequence_length)
        
        sequences.append(trajectory.loc[:, feature_columns].to_numpy())

    return sequences
