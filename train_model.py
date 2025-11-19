import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from xgboost import XGBClassifier

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_PATH = "delhi_metro_cleaned_final.csv"
SAVE_DIR = "saved_models"
TARGET = "Platform Crowd Level at Boarding Station"

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

df = df[df[TARGET].notna()].reset_index(drop=True)

num_cols = [
    "Journey Time (mins)",
    "Wait Time (mins)",
    "Number of Line Changes",
    "official_ridership_scaled"
]

drop_cols = [
    "Train Crowd Level When You Boarded",
    "Crowd Level at Destination Station",
    "Overall Journey Satisfaction"
]

features = [c for c in df.columns if c not in drop_cols + [TARGET]]
cat_cols = [c for c in features if c not in num_cols]

# Clean numerics
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
df[cat_cols] = df[cat_cols].fillna("MISSING").astype(str)

# --------------------------------------------------
# LABEL FIX (1–5 → 0–4)
# --------------------------------------------------
df["target_shifted"] = df[TARGET].astype(int) - 1
y = df["target_shifted"]
X = df[features]

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# ENCODING + SCALING
# --------------------------------------------------
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(X_train[cat_cols])

scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train_final = np.hstack([
    scaler.transform(X_train[num_cols]),
    encoder.transform(X_train[cat_cols])
])

X_test_final = np.hstack([
    scaler.transform(X_test[num_cols]),
    encoder.transform(X_test[cat_cols])
])

sample_weights = compute_sample_weight("balanced", y_train)

# --------------------------------------------------
# TRAIN XGBOOST
# --------------------------------------------------
print("\n[INFO] Training XGBoost Model...")

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [6, 10],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

search = RandomizedSearchCV(
    xgb, param_grid, n_iter=10, scoring="balanced_accuracy",
    cv=3, random_state=42, n_jobs=2
)

search.fit(X_train_final, y_train, sample_weight=sample_weights)
best_model = search.best_estimator_

# --------------------------------------------------
# TESTING
# --------------------------------------------------
pred_shifted = best_model.predict(X_test_final)
pred = pred_shifted + 1
y_true = y_test + 1

print("\nAccuracy:", accuracy_score(y_true, pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_true, pred))

# --------------------------------------------------
# SAVE MODEL + PREPROCESSORS
# --------------------------------------------------
pickle.dump(best_model, open(f"{SAVE_DIR}/xgb_platform_crowd.pkl", "wb"))
pickle.dump(encoder, open(f"{SAVE_DIR}/platform_encoder.pkl", "wb"))
pickle.dump(scaler, open(f"{SAVE_DIR}/platform_scaler.pkl", "wb"))

print("\n✅ Model and preprocessors saved in:", SAVE_DIR)
