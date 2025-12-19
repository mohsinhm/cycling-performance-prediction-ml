import joblib
import pandas as pd
from pathlib import Path
import sys

# Ensure project root is on sys.path so `src` package can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import add_features

MODEL_PATH = Path("models/model.joblib")

if not MODEL_PATH.exists():
    raise SystemExit(f"Model file not found at {MODEL_PATH.resolve()}")

model = joblib.load(MODEL_PATH)

input_df = pd.DataFrame([{
    "distance_km": 30.0,
    "elevation_gain_m": 200.0,
    "ride_time_min": 90.0,
    "temperature_c": 28.0,
    "route_type": "flat",
}])

# Apply same feature engineering as training
input_df = add_features(input_df)

pred = model.predict(input_df)
print("Sample prediction:", pred)
