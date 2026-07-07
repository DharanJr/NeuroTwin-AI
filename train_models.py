import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# LOAD DATASET
# ==========================================================

DATASET_PATH = "data/cognitive_digital_twin.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("COGNITIVE DIGITAL TWIN MODEL TRAINING")
print("=" * 70)

print(f"\nDataset Shape: {df.shape}")

# ==========================================================
# CREATE FOLDERS
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ==========================================================
# TARGETS
# ==========================================================

targets = [
    "brain_fog_score",
    "digital_addiction_score",
    "attention_fragmentation_score",
    "digital_overstimulation_score",
    "memory_retention_score",
    "cognitive_age"
]

# ==========================================================
# LEAKAGE MAP
# ==========================================================

LEAKAGE_MAP = {

    "brain_fog_score": [
        "sleep_hours",
        "focus_index",
        "exam_score",
        "productivity_score"
    ],

    "digital_addiction_score": [
        "daily_screen_time_hours"
    ],

    "attention_fragmentation_score": [
        "phone_unlocks_per_day"
    ],

    "digital_overstimulation_score": [
        "social_media_usage_hours"
    ],

    "memory_retention_score": [
        "exam_score",
        "focus_index",
        "productivity_score"
    ],

    "cognitive_age": []
}

# ==========================================================
# TARGET DISTRIBUTION CHECK
# ==========================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION CHECK")
print("=" * 70)

for target in targets:

    print(f"\n{target}")
    print(df[target].describe())

# ==========================================================
# TRAINING LOOP
# ==========================================================

for target in targets:

    print("\n" + "=" * 70)
    print(f"TRAINING MODEL : {target}")
    print("=" * 70)

    # ======================================================
    # COGNITIVE AGE SPECIAL CASE
    # ======================================================

    if target == "cognitive_age":

        drop_cols = ["cognitive_age"]

    else:

        drop_cols = [target]

        other_targets = [
            t for t in targets
            if t != target
        ]

        drop_cols.extend(other_targets)

        for leak_col in LEAKAGE_MAP[target]:

            if leak_col in df.columns:
                drop_cols.append(leak_col)

    drop_cols = list(set(drop_cols))

    print("\nExcluded Columns:")

    for col in drop_cols:
        print(f"  - {col}")

    X = df.drop(columns=drop_cols)

    y = df[target]

    print(f"\nFeature Count : {X.shape[1]}")
    print(f"Rows          : {X.shape[0]}")

    # ======================================================
    # SPLIT
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Train Samples : {len(X_train)}")
    print(f"Test Samples  : {len(X_test)}")

    # ======================================================
    # MODEL
    # ======================================================

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ======================================================
    # PREDICTIONS
    # ======================================================

    train_predictions = model.predict(X_train)

    test_predictions = model.predict(X_test)

    # ======================================================
    # METRICS
    # ======================================================

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    mse = mean_squared_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(mse)

    print("\nMODEL PERFORMANCE")
    print("-" * 40)

    print(f"Train R² : {train_r2:.4f}")
    print(f"Test  R² : {test_r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    report_path = (
        f"reports/{target}_feature_importance.csv"
    )

    importance_df.to_csv(
        report_path,
        index=False
    )

    print("\nTOP 15 FEATURES")
    print(importance_df.head(15))

    # ======================================================
    # SAVE MODEL
    # ======================================================

    model_name = target.replace(
        "_score",
        ""
    )

    model_path = f"models/{model_name}.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(f"\nSaved Model : {model_path}")
    print(f"Saved Report: {report_path}")

# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 70)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Models:")

for target in targets:

    model_name = target.replace(
        "_score",
        ""
    )

    print(f"models/{model_name}.pkl")

print("\nTraining Completed.")