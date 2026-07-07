import joblib
import json

brain_fog_model = joblib.load("models/brain_fog.pkl")
digital_addiction_model = joblib.load("models/digital_addiction.pkl")
attention_model = joblib.load("models/attention_fragmentation.pkl")
overstimulation_model = joblib.load("models/digital_overstimulation.pkl")
memory_model = joblib.load("models/memory_retention.pkl")
cognitive_age_model = joblib.load("models/cognitive_age.pkl")

scaler = joblib.load("models/scaler.pkl")

with open("models/scaler_columns.json", "r") as f:
    scaler_cols = json.load(f)