def build_tabular_inputs(dataframe, retained_sensor_columns):
    feature_columns = list(retained_sensor_columns)
    pandas_data = dataframe.select(*feature_columns, "RUL").toPandas()

    X = pandas_data.loc[:, feature_columns]
    y = pandas_data.loc[:, "RUL"]

    return X, y, feature_columns
