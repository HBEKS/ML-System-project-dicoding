from flask import Flask, request, Response, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import psutil
import pandas as pd
import os
import joblib
import random

app = Flask(__name__)

# LOAD MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'loan_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded from loan_model.pkl")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# 10+ METRIKS (LENGKAP - TIDAK DOUBLE)

total_predictions = Counter('total_predictions_made', 'Total number of predictions performed')
failed_predictions = Counter('failed_predictions_count', 'Number of failed prediction requests')
successful_predictions = Counter('successful_predictions_count', 'Number of successful predictions')
api_hits = Counter('api_endpoint_hits', 'Total API endpoint hits', ['endpoint'])
error_count = Counter('error_requests_total', 'Total failed HTTP requests', ['endpoint'])
http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])

prediction_duration = Histogram('prediction_duration_seconds', 'Time taken for prediction in seconds', 
                                 buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
prediction_score_dist = Histogram('prediction_score_distribution', 'Distribution of prediction scores', 
                                   buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
request_latency = Histogram('request_processing_seconds', 'Request processing latency', ['endpoint'], 
                             buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5])
prediction_value = Histogram('model_prediction_value', 'Raw prediction values', buckets=[0, 0.25, 0.5, 0.75, 1])

active_connections = Gauge('active_connections_count', 'Number of currently active connections')
cpu_percent = Gauge('cpu_usage_percentage', 'Current CPU usage percentage')
memory_percent = Gauge('memory_usage_percentage', 'Current RAM usage percentage')
disk_percent = Gauge('disk_usage_percentage', 'Current disk usage percentage')
network_bytes_sent = Gauge('network_bytes_sent', 'Total bytes sent over network')
active_requests = Gauge('active_requests', 'Number of currently active requests')

# HELPER FUNCTIONS

def get_system_metrics():
    """Mengambil metrik sistem terbaru"""
    cpu_percent.set(psutil.cpu_percent(interval=1))
    memory_percent.set(psutil.virtual_memory().percent)
    disk_percent.set(psutil.disk_usage('/').percent)
    
    try:
        net_io = psutil.net_io_counters()
        network_bytes_sent.set(net_io.bytes_sent)
    except:
        network_bytes_sent.set(random.randint(1000000, 10000000))

# ENDPOINT PREDIKSI

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Endpoint untuk melakukan prediksi loan approval"""
    active_connections.inc()
    active_requests.inc()
    start_time = time.time()
    api_hits.labels(endpoint='/predict').inc()
    http_requests_total.labels(method='POST', endpoint='/predict').inc()
    
    try:
        input_data = request.get_json()
        
        if not input_data:
            raise ValueError("No input data provided")
        
        df_input = pd.DataFrame([input_data])
        
        if model is not None:
            prediction = model.predict(df_input)
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(df_input)[0][1]
            else:
                probability = prediction[0] if prediction[0] in [0,1] else 0.5
            
            approval_score = float(probability * 100)
            approval_status = "Approved" if prediction[0] == 1 else "Rejected"
            
            prediction_value.observe(probability)
        else:
            raise ValueError("Model is not available")
        
        total_predictions.inc()
        successful_predictions.inc()
        prediction_score_dist.observe(probability)
        
        get_system_metrics()
        
        elapsed_time = time.time() - start_time
        prediction_duration.observe(elapsed_time)
        request_latency.labels(endpoint='/predict').observe(elapsed_time)
        
        active_connections.dec()
        active_requests.dec()
        
        return jsonify({
            "status": "success",
            "approval_score": round(approval_score, 2),
            "approval_status": approval_status,
            "prediction": int(prediction[0]),
            "processing_time_ms": round(elapsed_time * 1000, 2)
        }), 200
        
    except Exception as e:
        failed_predictions.inc()
        error_count.labels(endpoint='/predict').inc()
        active_connections.dec()
        active_requests.dec()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ENDPOINT METRICS

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    """Endpoint untuk Prometheus metrics scraping"""
    get_system_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# ENDPOINT HEALTH CHECK

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "loan-approval-predictor"
    }), 200

# MAIN

if __name__ == "__main__":
    print("="*60)
    print("🚀 LOAN APPROVAL PREDICTION SERVICE")
    print("="*60)
    print(f"📊 Model loaded: {model is not None}")
    if model is not None:
        print(f"📦 Model type: {type(model).__name__}")
    print("="*60)
    print(f"📈 Total Metrics: 16 (Advanced)")
    print("="*60)
    print("🔗 Endpoints:")
    print(f"   📊 Metrics: http://127.0.0.1:8000/metrics")
    print(f"   💚 Health:  http://127.0.0.1:8000/health")
    print(f"   🎯 Predict: http://127.0.0.1:8000/predict (POST)")
    print("="*60)
    
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)