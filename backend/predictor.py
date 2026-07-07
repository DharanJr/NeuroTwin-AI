import pandas as pd

from preprocess import preprocess_input

from models_loader import (
    brain_fog_model,
    digital_addiction_model,
    attention_model,
    overstimulation_model,
    memory_model,
    cognitive_age_model,
    scaler,
    scaler_cols
)

DEFAULT_USER = {
    # Basic Profile
    "age": 21,

    # Sleep
    "sleep_hours": 7,
    "sleep_quality": 75,

    # Study & Productivity
    "study_hours": 4,
    "self_study_hours": 2,
    "online_classes_hours": 2,
    "focus_index": 70,
    "productivity_score": 75,
    "exam_score": 70,

    # Physical Health
    "exercise_minutes": 30,
    "physical_activity_level": 60,
    "daily_steps": 6000,
    "heart_rate": 75,
    "bmi_category": "Normal",

    # Mental Health
    "mental_health_score": 75,
    "stress_level_lifestyle": 45,
    "burnout_level": 40,
    "anxiety_score": 25,
    "depression_score": 20,

    # Digital Behaviour
    "daily_screen_time_hours": 6,
    "social_media_hours": 3,
    "gaming_hours": 1,

    "phone_unlocks_per_day": 80,
    "push_notifications_per_day": 60,

    "social_media_usage_hours": 3,
    "gaming_usage_hours": 1,
    "streaming_usage_hours": 1,
    "messaging_usage_hours": 2,
    "work_related_usage_hours": 2,

    # Lifestyle
    "caffeine_intake_mg": 100,
    "part_time_job": 0,
    "upcoming_deadline": 0,

    # Categorical Features
    "gender": "Male",
    "academic_level": "Undergraduate",
    "internet_quality": "Good",

    "self_reported_addiction_level": "Moderate",
    "mood_swings": "Medium",
    "growing_stress": "Maybe",
    "work_interest": "Yes",
    "coping_struggles": "No",
    "social_weakness": "No",

    # App Usage
    "tech_savviness_score": 70,
    "app_screen_time_min": 120,
    "app_launches": 10,
    "app_interactions": 30,
    "app_is_productive": True,

    "app_youtube_views": 1000,
    "app_youtube_likes": 20,
    "app_youtube_comments": 5
}


def prepare_input(processed_df, model):
    """
    Match dataframe columns with model features.
    Missing features filled with 0.
    """
    feature_names = model.feature_names_in_

    row = {}

    for feature in feature_names:
        if feature in processed_df.columns:
            row[feature] = processed_df.iloc[0][feature]
        else:
            row[feature] = 0

    return pd.DataFrame([row])


def predict_all(user_data):
    results = {}

    # Merge user input with realistic defaults
    merged_user = {
        **DEFAULT_USER,
        **user_data
    }

    # Preprocess
    processed_df = preprocess_input(
        merged_user,
        scaler,
        scaler_cols
    )

    # Brain Fog
    brain_df = prepare_input(
        processed_df,
        brain_fog_model
    )

    results["brain_fog"] = float(
        brain_fog_model.predict(brain_df)[0]
    )

    # Digital Addiction
    addiction_df = prepare_input(
        processed_df,
        digital_addiction_model
    )

    results["digital_addiction"] = float(
        digital_addiction_model.predict(addiction_df)[0]
    )

    # Attention Fragmentation
    attention_df = prepare_input(
        processed_df,
        attention_model
    )

    results["attention_fragmentation"] = float(
        attention_model.predict(attention_df)[0]
    )

    # Digital Overstimulation
    overstim_df = prepare_input(
        processed_df,
        overstimulation_model
    )

    results["digital_overstimulation"] = float(
        overstimulation_model.predict(overstim_df)[0]
    )

    # Memory Retention
    memory_df = prepare_input(
        processed_df,
        memory_model
    )

    results["memory_retention"] = float(
        memory_model.predict(memory_df)[0]
    )

    # Cognitive Age
    cognitive_df = prepare_input(
        processed_df,
        cognitive_age_model
    )

    results["cognitive_age"] = float(
        cognitive_age_model.predict(cognitive_df)[0]
    )

    return results