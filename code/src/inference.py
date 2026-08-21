from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

# Metrics definitions
INFERENCE_COUNT = Counter('rul_predictions_total', 'Total RUL predictions processed')
LATENCY_HISTOGRAM = Histogram('rul_prediction_latency_seconds', 'Inference latency in seconds')
ENGINE_RUL_GAUGE = Gauge('engine_predicted_rul', 'Predicted Remaining Useful Life', ['engine_id'])

# Expose HTTP metrics endpoint on port 8000
start_http_server(8000)

def predict_engine_rul(engine_id: int, sensor_inputs, model):
    start_time = time.time()
    
    predicted_rul = model.predict(sensor_inputs)[0]
    
    INFERENCE_COUNT.inc()
    LATENCY_HISTOGRAM.observe(time.time() - start_time)
    ENGINE_RUL_GAUGE.labels(engine_id=str(engine_id)).set(predicted_rul)
    
    return predicted_rul
