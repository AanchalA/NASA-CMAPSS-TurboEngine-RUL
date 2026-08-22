from prometheus_client import Counter, Gauge, Histogram


METRIC_LABELS = ("subset", "model_type", "status")


RUL_PREDICTIONS = Counter("rul_predictions_total",
                          "Total RUL prediction requests processed.",
                          METRIC_LABELS)

RUL_PREDICTION_LATENCY = Histogram("rul_prediction_latency_seconds",
                                   "End-to-end RUL prediction request latency in seconds.",
                                   METRIC_LABELS)

RUL_PREDICTED_VALUE = Gauge("rul_predicted_value",
                            "Latest predicted RUL value.",
                            ("subset", "model_type"))
