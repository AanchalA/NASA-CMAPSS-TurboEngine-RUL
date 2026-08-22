import numpy as np


DEFAULT_SEQUENCE_LENGTH = 30


def build_sliding_window_sequences(dataframe, feature_columns, sequence_length=DEFAULT_SEQUENCE_LENGTH):
    feature_columns = tuple(feature_columns)
    
    ordered = dataframe.sort_values(["unit_id", "cycle"])
    sequences = []
    targets = []

    for _, engine in ordered.groupby("unit_id", sort=False):
        
        if len(engine) < sequence_length:
            continue

        values = engine.loc[:, feature_columns].to_numpy()
        engine_sequences = np.lib.stride_tricks.sliding_window_view(values, window_shape=sequence_length, axis=0).transpose(0, 2, 1)
        sequences.append(engine_sequences)
        targets.append(engine.loc[:, "RUL"].to_numpy()[sequence_length - 1:])

    if not sequences:
        return (np.empty((0, sequence_length, len(feature_columns))),
                np.empty((0,), dtype=dataframe["RUL"].dtype))

    return np.concatenate(sequences), np.concatenate(targets)
