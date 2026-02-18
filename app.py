from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)

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
        model = tf.keras.models.load_model('hospital_multi_output_model.keras', 
                                            custom_objects=custom_objects, compile=False)
    except:
        model = tf.keras.models.load_model('hospital_multi_output_model.h5', 
                                            custom_objects=custom_objects, compile=False)
    
    scaler_seq = joblib.load('scaler_seq.joblib')
    scaler_static = joblib.load('scaler_static.joblib')
    scaler_target = joblib.load('scaler_target.joblib')
    
    # Load metadata if available
    try:
        with open('model_metadata.json', 'r') as f:
            model_metadata = json.load(f)
        print(f"Model loaded: {model_metadata.get('model_name', 'Unknown')} v{model_metadata.get('version', '?')}")
    except:
        model_metadata = None
    
    print("✅ All assets loaded successfully!")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    model = None


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

if __name__ == '__main__':
   
    app.run(debug=True, port=5000)