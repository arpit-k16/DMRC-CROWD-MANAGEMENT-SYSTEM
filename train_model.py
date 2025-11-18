# ================================================================
# FINAL CLEAN TRAINING SCRIPT (CSV ONLY + FIXED LABEL ISSUE)
# Trains XGBoost and saves model, encoder, scaler.
# Target: Platform Crowd Level at Boarding Station (1–5)
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import random, os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_PATH = "delhi_metro_cleaned_no_unknown.csv"
SAVE_DIR = "saved_models"
TARGET = "Platform Crowd Level at Boarding Station"

os.makedirs(SAVE_DIR, exist_ok=True)

RND = 42
random.seed(RND)
np.random.seed(RND)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' missing in dataset")

df = df[df[TARGET].notna()].reset_index(drop=True)

# Columns to skip (leakage)
drop_cols = [
    "Train Crowd Level When You Boarded",
    "Crowd Level at Destination Station",
    "Overall Journey Satisfaction"
]

features = [c for c in df.columns if c not in drop_cols + [TARGET]]

# Numeric columns
num_cols = [
    "Journey Time (mins)",
    "Wait Time (mins)",
    "Number of Line Changes",
    "official_ridership_scaled"
]

# Categorical columns
cat_cols = [c for c in features if c not in num_cols]

# Clean data
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
df[cat_cols] = df[cat_cols].fillna("MISSING").astype(str)

# ------------------------------------------------------------
# FIXED: Convert target 1–5 → 0–4 FOR XGBOOST
# ------------------------------------------------------------
df["target_shifted"] = df[TARGET].astype(int) - 1
y = df["target_shifted"]

X = df[features]

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RND
)

# ------------------------------------------------------------
# ENCODING + SCALING
# ------------------------------------------------------------
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

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# ------------------------------------------------------------
# XGBOOST MODEL
# ------------------------------------------------------------
print("\n[INFO] Training XGBoost...")

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=5,  # Because classes = 0,1,2,3,4
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=RND,
    n_jobs=4
)

xgb_params = {
    "n_estimators": [200, 400],
    "max_depth": [6, 10],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb, xgb_params, n_iter=10, scoring="balanced_accuracy",
    cv=3, random_state=RND, n_jobs=2
)

xgb_search.fit(X_train_final, y_train, sample_weight=sample_weights)
best_xgb = xgb_search.best_estimator_

pred_xgb_shifted = best_xgb.predict(X_test_final)
pred_xgb = pred_xgb_shifted + 1
true_xgb = y_test + 1

print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(true_xgb, pred_xgb))
print("Balanced Accuracy:", balanced_accuracy_score(true_xgb, pred_xgb))

pickle.dump(best_xgb, open(f"{SAVE_DIR}/xgb_platform_crowd.pkl", "wb"))

# ------------------------------------------------------------
# SAVE PREPROCESSORS
# ------------------------------------------------------------
pickle.dump(encoder, open(f"{SAVE_DIR}/platform_encoder.pkl", "wb"))
pickle.dump(scaler, open(f"{SAVE_DIR}/platform_scaler.pkl", "wb"))

print("\n✅ ALL MODELS SAVED IN:", SAVE_DIR)

# ------------------------------------------------------------
# UNIVERSAL PREDICTION FUNCTION
# ------------------------------------------------------------
def predict_crowd(input_sample):
    df = pd.DataFrame([input_sample])

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[cat_cols] = df[cat_cols].fillna("MISSING").astype(str)

    X_num = scaler.transform(df[num_cols])
    X_cat = encoder.transform(df[cat_cols])
    X_final = np.hstack([X_num, X_cat])

    shifted_pred = best_xgb.predict(X_final)[0]
    return int(shifted_pred + 1)   # Convert 0–4 → 1–5

# ------------------------------------------------------------
# SAMPLE PREDICTION
# ------------------------------------------------------------
sample = {
    "Source Station": "Rajiv Chowk",
    "Destination Station": "Huda City Centre",
    "Metro Line Used (Primary)": "Yellow Line",
    "Boarding Time Category": "Morning Peak",
    "Day-Type": "Weekday",
    "Weather Condition": "Clear",
    "Purpose of Travel": "Work",
    "Age Group": "25-34",
    "Frequency of Metro Usage": "Daily",
    "Journey Time (mins)": 32,
    "Wait Time (mins)": 3,
    "Number of Line Changes": 0,
    "official_ridership_scaled": 0.80,
    "Could You Get a Seat Immediately?": "No",
    "Peak Crowd Point During Journey": "Boarding Station",
    "Special Event Nearby": "No"
}

print("\nSample XGB Prediction:", predict_crowd(sample))