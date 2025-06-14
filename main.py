from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import joblib, pandas as pd
from datetime import datetime, timedelta
from typing import Dict
import numpy as np

# Load the models
preprocessor = joblib.load('rf_scaler.pkl')
harvest_model = joblib.load('rf_model.pkl')

app = FastAPI()

# Model Input Data
class SensorData(BaseModel):
    date: str          # 'YYYY-MM-DD'
    temperature: float
    humidity: float
    tds: float
    ph: float

# Helper function for feature engineering
def create_features(
    df: pd.DataFrame,
    date_col: str = 'Date',
    phase_col: str = 'Phase'
) -> pd.DataFrame:
    """
    Build expanding & phase stats for a single time-series DataFrame,
    with dynamic Phase assignment based on Growth Days thresholds.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']

    # 0) Assign phase based on Growth Days if available
    if 'Growth Days' in df.columns:
        gd = df['Growth Days']
        df[phase_col] = np.where(
            gd < 15, 0,
            np.where(gd < 30, 1, 2)
        ).astype(int)
    else:
        df[phase_col] = 0

    # 1) Expanding statistics for each feature
    for feat in features:
        exp = df[feat].expanding(min_periods=1)
        df[f"{feat} Expanding Mean"] = exp.mean()
        df[f"{feat} Expanding Std"] = exp.std()
        df[f"{feat} Expanding Min"] = exp.min()
        df[f"{feat} Expanding Max"] = exp.max()
        df[f"{feat} Expanding Median"] = exp.median()

    # 2) Phase-based summary statistics
    agg_funcs = ['mean', 'min', 'max', 'median', 'std']
    phase_stats = (
        df
        .groupby(phase_col)[features]
        .agg(agg_funcs)
        .reset_index()
    )
    # Flatten multi-index columns
    phase_stats.columns = (
        [phase_col] +
        [f"{feat} Phase {stat.capitalize()}"
         for feat, stat in phase_stats.columns
         if feat != phase_col]
    )

    # 3) Merge back phase summaries
    df = df.merge(phase_stats, on=phase_col, how='left')
    return df


@app.post("/predict-harvest")
def predict_harvest(window: List[SensorData]):
    if not window:
        raise HTTPException(status_code=400, detail="Payload cannot be empty")

    # 1) Build & sort DataFrame (no early exit, no padding)
    df = pd.DataFrame([{
        'Date':        datetime.strptime(r.date, "%Y-%m-%d"),
        'Temperature': r.temperature,
        'Humidity':    r.humidity,
        'TDS Value':   r.tds,
        'pH Level':    r.ph,
    } for r in window]).sort_values('Date').reset_index(drop=True)

    # 3) Pad backwards to ensure 7 days

    # 4) Feature engineering
    df = create_features(df)
    print("\n=== Final 7-Day DataFrame ===")
    print(df)

    # 5) Prepare model input
    expected = list(preprocessor.feature_names_in_)
    last_row = df.iloc[[-1]].reindex(columns=expected, fill_value=0)
    X = preprocessor.transform(last_row)

    print("\n=== Model Input After Preprocessing ===")
    print(pd.DataFrame(X, columns=preprocessor.get_feature_names_out()))

    # 6) Predict
    y = harvest_model.predict(X)
    print("\n=== Harvest Day Prediction ===")
    print(int(y[0]))
    return {"predicted_harvest_day": int(y[0])}
