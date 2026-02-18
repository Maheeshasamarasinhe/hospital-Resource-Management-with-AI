from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import numpy as np
import json
import math
import pymysql

app = Flask(__name__)
CORS(app)

# ===== BASE DIR =====
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== MYSQL CONFIG =====
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     3306,          # Change to 3307 if XAMPP uses that port
    "user":     "root",
    "password": "",
    "database": "hospital_db",
}

def get_db():
    return pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)

# Test database connection at startup
def test_db_connection():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        print("✅ Database connected successfully!")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   → Predictions will still work but historical data fetch will be unavailable.")
        return False

# ===== CUSTOM ATTENTION LAYER (Required for model loading) =====
class AttentionLayer(layers.Layer):
    """Attention mechanism to focus on important time steps"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                  shape=(input_shape[-1], input_shape[-1]),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                  shape=(input_shape[-1],),
                                  initializer='zeros',
                                  trainable=True)
        self.u = self.add_weight(name='attention_context',
                                  shape=(input_shape[-1], 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

# ===== LOAD ASSETS =====
try:
    # Register custom layer for model loading
    custom_objects = {'AttentionLayer': AttentionLayer}
    
    # Try loading .keras format first, fallback to .h5
    try:
        model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'hospital_multi_output_model.keras'), 
                                            custom_objects=custom_objects, compile=False)
    except:
        model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'hospital_multi_output_model.h5'), 
                                            custom_objects=custom_objects, compile=False)
    
    scaler_seq    = joblib.load(os.path.join(BASE_DIR, 'scaler_seq.joblib'))
    scaler_static = joblib.load(os.path.join(BASE_DIR, 'scaler_static.joblib'))
    scaler_target = joblib.load(os.path.join(BASE_DIR, 'scaler_target.joblib'))
    
    # Load metadata if available
    try:
        with open(os.path.join(BASE_DIR, 'model_metadata.json'), 'r') as f:
            model_metadata = json.load(f)
        print(f"Model loaded: {model_metadata.get('model_name', 'Unknown')} v{model_metadata.get('version', '?')}")
    except:
        model_metadata = None
    
    print("✅ All assets loaded successfully!")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    model = None

# Test database connection at startup
test_db_connection()


target_cols = ['Dengue', 'Road_Accidents', 'Heart_Patients', 'Hadisi_Anthuru', 'Tuberculosis', 'Cold', 'Fever']

@app.route('/', methods=['GET'])
def health_check():
    """API Health Check Endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": model_metadata if model_metadata else "Not available"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        
        if 'sequential_data' not in data or 'static_data' not in data:
            return jsonify({"error": "Missing input data. Required: sequential_data (12x10), static_data (1x5)"}), 400

        raw_seq = np.array(data['sequential_data'])
        raw_static = np.array(data['static_data']).reshape(1, -1)
        
        # Validate input shapes
        if raw_seq.shape != (12, 10):
            return jsonify({"error": f"Invalid sequential_data shape. Expected (12, 10), got {raw_seq.shape}"}), 400
        if raw_static.shape != (1, 5):
            return jsonify({"error": f"Invalid static_data shape. Expected (1, 5), got {raw_static.shape}"}), 400

        scaled_seq = scaler_seq.transform(raw_seq).reshape(1, 12, 10)
        scaled_static = scaler_static.transform(raw_static)

        prediction_scaled = model.predict([scaled_seq, scaled_static], verbose=0)
        prediction_final = scaler_target.inverse_transform(prediction_scaled)[0]

        results = {disease: max(0, int(round(val))) for disease, val in zip(target_cols, prediction_final)}
        
        # Calculate total expected patients
        total_patients = sum(results.values())
        
        return jsonify({
            "status": "success",
            "predictions": results,
            "total_expected_patients": total_patients,
            "recommendation": get_resource_recommendation(results)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def get_resource_recommendation(predictions):
    """Generate resource allocation recommendations based on predictions"""
    recommendations = []
    
    if predictions.get('Dengue', 0) > 150:
        recommendations.append("🚨 High Dengue Alert: Prepare additional beds in infectious disease ward")
    if predictions.get('Road_Accidents', 0) > 100:
        recommendations.append("🚑 High Accident Alert: Ensure emergency trauma team availability")
    if predictions.get('Heart_Patients', 0) > 100:
        recommendations.append("❤️ High Cardiac Cases: Increase ICU capacity")
    if predictions.get('Cold', 0) + predictions.get('Fever', 0) > 200:
        recommendations.append("🤒 High Respiratory Cases: Stock up on medications")
    
    return recommendations if recommendations else ["✅ Normal patient load expected"]

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information and metadata"""
    return jsonify({
        "target_diseases": target_cols,
        "required_sequential_features": [
            "Dengue", "Road_Accidents", "Heart_Patients", "Hadisi_Anthuru", 
            "Tuberculosis", "Cold", "Fever", "Rainfall", "Avg_Temperature", "Humidity"
        ],
        "required_static_features": [
            "Month_Sin", "Month_Cos", "Festive_Season", "Public_Holidays", 
            "Public_Awareness_Level"
        ],
        "window_size": 12,
        "metadata": model_metadata
    })

@app.route('/history', methods=['GET'])
def history():
    """Return average case counts for the requested month across all years."""
    month = request.args.get('month', type=int)
    if not month or not (1 <= month <= 12):
        return jsonify({"error": "Provide month as integer 1-12"}), 400
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                AVG(dc.dengue)         AS Dengue,
                AVG(dc.road_accidents) AS Road_Accidents,
                AVG(dc.heart_patients) AS Heart_Patients,
                AVG(dc.hadisi_anthuru) AS Hadisi_Anthuru,
                AVG(dc.tuberculosis)   AS Tuberculosis,
                AVG(dc.cold)           AS Cold,
                AVG(dc.fever)          AS Fever
            FROM disease_cases dc
            JOIN time_periods   tp ON tp.id = dc.period_id
            WHERE tp.month = %s
        """, (month,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        # round values
        avg_cases = {k: round(float(v), 1) if v else 0 for k, v in row.items()}
        return jsonify({"month": month, "avg_cases": avg_cases})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict-frontend', methods=['POST'])
def predict_frontend():
    """
    Simplified endpoint for the React frontend.
    Accepts high-level inputs, fetches last 12 months from DB,
    builds the sequential window and runs the model.
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data        = request.get_json()
        month       = int(data['month'])
        humidity    = float(data['humidity'])
        rainfall    = float(data['rainfall'])
        temperature = float(data['temperature'])
        festive     = int(data['festive'])
        awareness   = float(data['awareness'])

        # Cyclical month encoding
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)

        # ----------- Fetch last 12 months from DB -----------
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                dc.dengue, dc.road_accidents, dc.heart_patients,
                dc.hadisi_anthuru, dc.tuberculosis, dc.cold, dc.fever,
                ef.rainfall, ef.avg_temperature, ef.humidity
            FROM disease_cases dc
            JOIN time_periods          tp ON tp.id = dc.period_id
            JOIN environmental_factors ef ON ef.period_id = tp.id
            ORDER BY tp.date DESC
            LIMIT 12
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if len(rows) < 12:
            return jsonify({"error": f"Not enough historical data in DB. Found {len(rows)} rows, need 12."}), 400

        rows.reverse()   # chronological order
        seq_cols_order = ['dengue','road_accidents','heart_patients','hadisi_anthuru',
                          'tuberculosis','cold','fever','rainfall','avg_temperature','humidity']
        raw_seq    = np.array([[row[c] for c in seq_cols_order] for row in rows], dtype=float)
        raw_static = np.array([[month_sin, month_cos, festive, 0, awareness]], dtype=float)
        # note: Public_Holidays not in form → default 0

        scaled_seq    = scaler_seq.transform(raw_seq).reshape(1, 12, 10)
        scaled_static = scaler_static.transform(raw_static)

        pred_scaled = model.predict([scaled_seq, scaled_static], verbose=0)
        pred_final  = scaler_target.inverse_transform(pred_scaled)[0]

        results = {d: max(0, int(round(v))) for d, v in zip(target_cols, pred_final)}
        total   = sum(results.values())

        return jsonify({
            "status":                "success",
            "predictions":           results,
            "total_expected_patients": total,
            "recommendation":        get_resource_recommendation(results),
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    test_db_connection()
    app.run(debug=True, host='0.0.0.0', port=5000)