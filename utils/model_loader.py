
# utils/model_loader.py
import pickle
import numpy as np
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=1)
def load_models():
    xgb = pickle.load(open("saved_models/xgb_platform_crowd.pkl","rb"))
    enc = pickle.load(open("saved_models/platform_encoder.pkl","rb"))
    scaler = pickle.load(open("saved_models/platform_scaler.pkl","rb"))
    return xgb, enc, scaler

def preprocess_sample(sample, encoder, scaler):
    df = pd.DataFrame([sample])
    num_cols = ["Journey Time (mins)","Wait Time (mins)","Number of Line Changes","official_ridership_scaled"]
    cat_cols = [c for c in df.columns if c not in num_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[cat_cols] = df[cat_cols].fillna("MISSING").astype(str)
    X = np.hstack([scaler.transform(df[num_cols]), encoder.transform(df[cat_cols])])
    return X

def predict_models(X, xgb):
    return {"xgb": int(xgb.predict(X)[0])+1}
