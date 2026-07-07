import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("data/cognitive_digital_twin.csv")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("Dataset Shape:", df.shape)

# =====================================
# TARGET VARIABLE
# =====================================

y = df["brain_fog_score"]

# =====================================
# REMOVE TARGETS / LEAKAGE FEATURES
# =====================================

leakage_columns = [
    "brain_fog_score",
    "digital_addiction_score",
    "attention_fragmentation_score",
    "digital_overstimulation_score",
    "memory_retention_score",
    "cognitive_age"
]

X = df.drop(columns=leakage_columns)

print("\nFeatures Shape:", X.shape)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================
# RANDOM FOREST MODEL
# =====================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest Model...")

model.fit(X_train, y_train)

print("Model Training Completed Successfully")

# =====================================
# PREDICTIONS
# =====================================

predictions = model.predict(X_test)

# =====================================
# EVALUATION METRICS
# =====================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

# =====================================
# RESULTS
# =====================================

print("\n" + "=" * 50)
print("BRAIN FOG MODEL RESULTS")
print("=" * 50)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =====================================
# FEATURE IMPORTANCE
# =====================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 50)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 50)

print(feature_importance.head(20))

# =====================================
# SAVE FEATURE IMPORTANCE
# =====================================

feature_importance.to_csv(
    "reports/brain_fog_feature_importance_v2.csv",
    index=False
)

print("\nFeature importance saved to:")
print("reports/brain_fog_feature_importance_v2.csv")

# =====================================
# SAVE MODEL
# =====================================

try:
    import joblib

    joblib.dump(
        model,
        "models/brain_fog_random_forest.pkl"
    )

    print("\nModel saved successfully:")
    print("models/brain_fog_random_forest.pkl")

except Exception as e:
    print("\nModel save skipped.")
    print(e)