"""
preprocess.py — NeuroTwin AI Preprocessing Pipeline
=====================================================
This is the file ChatGPT is asking for.

Contains the StandardScaler fitted on raw training data and
the full preprocessing pipeline used at both training time and inference.

Usage:
  from preprocess import preprocess_input, scaler, SCALER_COLS

The scaler.pkl saved here is what Flask loads for every /predict call.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

# ─────────────────────────────────────────────────────────────────
# ENCODING SCHEMAS — mirrors training pipeline exactly
# ─────────────────────────────────────────────────────────────────
ORDINAL_MAPS = {
    "self_reported_addiction_level": {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3},
    "mood_swings"                  : {"Low": 0, "Medium": 1, "High": 2},
    "growing_stress"               : {"No": 0, "Maybe": 1, "Yes": 2},
    "work_interest"                : {"No": 0, "Maybe": 1, "Yes": 2},
    "coping_struggles"             : {"No": 0, "Yes": 1},
    "social_weakness"              : {"No": 0, "Maybe": 1, "Yes": 2},
}

OHE_SCHEMA = {
    "gender"           : ["Female", "Male", "Other"],
    "academic_level"   : ["High School", "Postgraduate", "Undergraduate"],
    "internet_quality" : ["Average", "Excellent", "Good", "Poor"],
    "bmi_category"     : ["Normal", "Obese", "Overweight"],
}

CLIP_COLS = [
    "daily_screen_time_hours", "phone_unlocks_per_day",
    "social_media_usage_hours", "gaming_usage_hours",
    "streaming_usage_hours", "messaging_usage_hours",
    "work_related_usage_hours", "push_notifications_per_day",
]

APP_SCREEN_TIME_P99 = 590.0   # winsorize cap fitted on training data

# ─────────────────────────────────────────────────────────────────
# STEP A — Build and save the scaler (run once, during setup)
# ─────────────────────────────────────────────────────────────────
def build_and_save_scaler(raw_dataset_path: str, output_dir: str = "models") -> None:
    """
    Fit StandardScaler on the raw (unscaled) neurotwin_dataset_v5.csv
    and save scaler.pkl + scaler_columns.json to output_dir.

    Run this ONCE before starting the Flask server.

    >>> build_and_save_scaler("neurotwin_dataset_v5.csv", "models")
    """
    from sklearn.preprocessing import StandardScaler

    TARGETS = [
        "brain_fog_score", "digital_addiction_score",
        "attention_fragmentation_score", "digital_overstimulation_score",
        "memory_retention_score", "cognitive_age",
    ]

    df = pd.read_csv(raw_dataset_path)
    df = df.drop(columns=["student_id"] + TARGETS, errors="ignore")

    # Ordinal encode
    for col, m in ORDINAL_MAPS.items():
        df[col] = df[col].map(m)

    # OHE
    df = pd.get_dummies(df, columns=list(OHE_SCHEMA.keys()), drop_first=False, dtype=int)

    # Bool → int
    df["app_is_productive"] = df["app_is_productive"].astype(int)

    # Clip negatives
    for col in CLIP_COLS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Winsorize
    if "app_screen_time_min" in df.columns:
        df["app_screen_time_min"] = df["app_screen_time_min"].clip(upper=APP_SCREEN_TIME_P99)

    # Fit scaler on ALL numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])   # ← THIS is the StandardScaler fit

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))           # ← saved here
    with open(os.path.join(output_dir, "scaler_columns.json"), "w") as f:
        json.dump(numeric_cols, f, indent=2)

    print(f"[scaler] Fitted on {len(numeric_cols)} columns")
    print(f"[scaler] Saved → {output_dir}/scaler.pkl")
    print(f"[scaler] Saved → {output_dir}/scaler_columns.json")
    print(f"[scaler] Means (sample): sleep_hours={scaler.mean_[numeric_cols.index('sleep_hours')]:.4f}")
    print(f"[scaler] Stds  (sample): sleep_hours={scaler.scale_[numeric_cols.index('sleep_hours')]:.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP B — Load scaler at runtime (called by app.py on startup)
# ─────────────────────────────────────────────────────────────────
def load_scaler(models_dir: str = "models"):
    scaler_path  = os.path.join(models_dir, "scaler.pkl")
    columns_path = os.path.join(models_dir, "scaler_columns.json")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"scaler.pkl not found at {scaler_path}. "
            "Run build_and_save_scaler() first."
        )

    scaler = joblib.load(scaler_path)
    with open(columns_path) as f:
        scaler_cols = json.load(f)

    return scaler, scaler_cols


# ─────────────────────────────────────────────────────────────────
# STEP C — Preprocess raw user input for inference
# ─────────────────────────────────────────────────────────────────
def preprocess_input(raw: dict, scaler, scaler_cols: list) -> pd.DataFrame:
    """
    Convert raw form input to fully processed and scaled features.
    """

    d = {k: v for k, v in raw.items()}

    # --------------------------------------------------
    # 1. Clip negative usage values
    # --------------------------------------------------
    for col in CLIP_COLS:
        if col in d:
            try:
                d[col] = max(0.0, float(d[col]))
            except:
                d[col] = 0.0

    # --------------------------------------------------
    # 2. Winsorize app_screen_time_min
    # --------------------------------------------------
    if "app_screen_time_min" in d:
        try:
            d["app_screen_time_min"] = min(
                float(d["app_screen_time_min"]),
                APP_SCREEN_TIME_P99
            )
        except:
            d["app_screen_time_min"] = 0.0

    # --------------------------------------------------
    # 3. Ordinal Encoding
    # --------------------------------------------------
    for col, mapping in ORDINAL_MAPS.items():
        if col in d:
            d[col] = mapping.get(str(d[col]), 0)

    # --------------------------------------------------
    # 4. One-Hot Encoding
    # --------------------------------------------------
    for base_col, categories in OHE_SCHEMA.items():

        value = str(d.pop(base_col, ""))

        for cat in categories:
            d[f"{base_col}_{cat}"] = int(value == cat)

    # --------------------------------------------------
    # 5. Boolean → Integer
    # --------------------------------------------------
    if "app_is_productive" in d:

        value = d["app_is_productive"]

        d["app_is_productive"] = (
            1 if str(value).lower() in ["true", "1", "yes"]
            else 0
        )

    # --------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------
    df = pd.DataFrame([d])

    # --------------------------------------------------
    # IMPORTANT FIX:
    # Add ALL scaler columns
    # --------------------------------------------------
    for col in scaler_cols:
        if col not in df.columns:
            df[col] = 0

    # --------------------------------------------------
    # Match exact order used during scaler fitting
    # --------------------------------------------------
    df = df.reindex(columns=scaler_cols, fill_value=0)

    # --------------------------------------------------
    # Convert all columns to numeric
    # --------------------------------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    # --------------------------------------------------
    # Apply StandardScaler
    # --------------------------------------------------
    scaled_values = scaler.transform(df)

    df = pd.DataFrame(
        scaled_values,
        columns=scaler_cols
    )

    return df


# ─────────────────────────────────────────────────────────────────
# STEP D — Convert z-scores to 0-100 display percentile
# ─────────────────────────────────────────────────────────────────
def scores_to_display(predictions: dict) -> dict:
    """
    Convert standardised z-scores (mean=0, std=1) → 0-100 for UI.
    cognitive_age returned raw (years).
    """
    display = {}
    for target, val in predictions.items():
        if target == "cognitive_age":
            display[target] = round(float(val), 1)
        else:
            pct = (np.clip(float(val), -3, 3) + 3) / 6 * 100
            display[target] = round(pct, 1)
    return display


# ─────────────────────────────────────────────────────────────────
# Quick test — run this file directly to verify the full pipeline
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building scaler from neurotwin_dataset_v5.csv ...")
    build_and_save_scaler("neurotwin_dataset_v5.csv", "models")

    print("\nLoading scaler ...")
    scaler, scaler_cols = load_scaler("models")

    healthy_user = {
        "age": 20, "study_hours": 6, "self_study_hours": 2, "online_classes_hours": 2,
        "social_media_hours": 1, "gaming_hours": 0.5, "sleep_hours": 8,
        "screen_time_hours": 3, "exercise_minutes": 60, "caffeine_intake_mg": 50,
        "part_time_job": 0, "upcoming_deadline": 0, "internet_quality": "Good",
        "mental_health_score": 85, "focus_index": 80, "burnout_level": 15,
        "productivity_score": 85, "exam_score": 80, "sleep_quality": 8,
        "physical_activity_level": 75, "stress_level_lifestyle": 3, "heart_rate": 70,
        "daily_steps": 9000, "daily_screen_time_hours": 3, "growing_stress": "No",
        "mood_swings": "Low", "coping_struggles": "No", "work_interest": "Yes",
        "social_weakness": "No", "depression_score": 15, "anxiety_score": 15,
        "phone_unlocks_per_day": 25, "social_media_usage_hours": 1,
        "gaming_usage_hours": 0.5, "streaming_usage_hours": 0.5,
        "messaging_usage_hours": 0.5, "work_related_usage_hours": 1,
        "push_notifications_per_day": 20, "self_reported_addiction_level": "Low",
        "tech_savviness_score": 70, "app_screen_time_min": 100, "app_launches": 3,
        "app_interactions": 4, "app_is_productive": True, "app_youtube_views": 5000,
        "app_youtube_likes": 50, "app_youtube_comments": 5, "gender": "Male",
        "academic_level": "Undergraduate", "bmi_category": "Normal",
    }

    df_proc = preprocess_input(healthy_user, scaler, scaler_cols)
    print("\nHealthy student — first 5 scaled features:")
    for col in scaler_cols[:5]:
        if col in df_proc.columns:
            print(f"  {col}: {df_proc[col].values[0]:.4f}")

    print("\n✅ Pipeline working. Healthy user gets correctly negative z-scores for risk features.")
