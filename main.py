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
    if 'Growth Days' not in df.columns:
        df['Growth Days'] = (df[date_col] - df[date_col].min()).dt.days

    # Determine Phase
    df[phase_col] = np.where(
        df['Growth Days'] < 15, 0,
        np.where(df['Growth Days'] < 30, 1, 2)
    ).astype(int)

    out = []

    # Treat entire DataFrame as one plant
    plant_df = df.sort_values(date_col).reset_index(drop=True)

    # Expanding stats for all features
    for feat in features:
        exp = plant_df[feat].expanding(min_periods=1)
        plant_df[f"{feat} Expanding Mean"] = exp.mean()
        plant_df[f"{feat} Expanding Std"] = exp.std()
        plant_df[f"{feat} Expanding Min"] = exp.min()
        plant_df[f"{feat} Expanding Max"] = exp.max()
        plant_df[f"{feat} Expanding Median"] = exp.median()

    # Phase-based expanding stats
    for feat in features:
        grp = plant_df.groupby(phase_col)[feat]
        plant_df[f"{feat} Phase Mean"] = grp.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        plant_df[f"{feat} Phase Std"] = grp.expanding(min_periods=1).std().reset_index(level=0, drop=True)
        plant_df[f"{feat} Phase Min"] = grp.expanding(min_periods=1).min().reset_index(level=0, drop=True)
        plant_df[f"{feat} Phase Max"] = grp.expanding(min_periods=1).max().reset_index(level=0, drop=True)
        plant_df[f"{feat} Phase Median"] = grp.expanding(min_periods=1).median().reset_index(level=0, drop=True)

    return plant_df


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
        'Plant_ID':    1,  # simulate single plant ID
    } for r in window])
    
    # 1) Average multiple readings per day (like in local model)
    sensor_cols = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    daily_avg = (
        df.groupby(['Plant_ID', 'Date'])[sensor_cols]
          .mean()
          .reset_index()
          .sort_values(['Plant_ID', 'Date'])
    )
    
    # 2) Add Growth Days
    daily_avg['Growth Days'] = (
        daily_avg.groupby('Plant_ID')['Date']
                 .transform(lambda x: (x - x.min()).dt.days)
    )
    # 3) Feature engineering
    feats = create_features(daily_avg, date_col='Date')
    print("\n=== Final 7-Day DataFrame ===")
    print(feats)

    X = feats.drop(columns=['Plant_ID', 'Date', 'Growth Days'], errors='ignore')

    # 5) Prepare model input
    expected = list(preprocessor.feature_names_in_)
    X_latest = X.reindex(columns=expected)
    X_scaled = preprocessor.transform(X_latest)

    print("\n=== Model Input After Preprocessing ===")
    print(pd.DataFrame(X, columns=preprocessor.get_feature_names_out()))

    # 6) Predict
    preds = harvest_model.predict(X_scaled)

    print("\n=========== Harvest Day Prediction ===========")
    print(preds)  # Display the prediction with decimals
    predicted_day = float(preds[-1])  # Get the last prediction
    return {"predicted_harvest_day": predicted_day}
