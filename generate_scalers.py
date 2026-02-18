"""
Regenerate scaler files and model_metadata.json from the existing CSV and trained model.
Run this once with:  conda run -n tf_env python generate_scalers.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, 'hospital_multi_disease_final.csv'))

target_cols  = ['Dengue', 'Road_Accidents', 'Heart_Patients',
                'Hadisi_Anthuru', 'Tuberculosis', 'Cold', 'Fever']
seq_cols     = target_cols + ['Rainfall', 'Avg_Temperature', 'Humidity']
static_cols  = ['Month_Sin', 'Month_Cos', 'Festive_Season',
                'Public_Holidays', 'Public_Awareness_Level']

# ── Fit scalers ───────────────────────────────────────────────────────────────
scaler_seq    = MinMaxScaler()
scaler_static = MinMaxScaler()
scaler_target = MinMaxScaler()

scaler_seq.fit(df[seq_cols])
scaler_static.fit(df[static_cols])
scaler_target.fit(df[target_cols])

# ── Save scalers ──────────────────────────────────────────────────────────────
joblib.dump(scaler_seq,    os.path.join(BASE_DIR, 'scaler_seq.joblib'))
joblib.dump(scaler_static, os.path.join(BASE_DIR, 'scaler_static.joblib'))
joblib.dump(scaler_target, os.path.join(BASE_DIR, 'scaler_target.joblib'))
print("✅ Scalers saved.")

# ── Save metadata ─────────────────────────────────────────────────────────────
metadata = {
    "model_name": "Hospital Multi-Disease Prediction Model",
    "version": "2.0",
    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "architecture": "Hybrid Bidirectional LSTM + Attention + Dense",
    "input_features": {
        "sequential": seq_cols,
        "static": static_cols
    },
    "output_features": target_cols,
    "window_size": 12,
}
with open(os.path.join(BASE_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)
print("✅ model_metadata.json saved.")
print("\nAll files ready. You can now run:  python app.py")
