import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Example feature: elevation per km
    if {"elevation_gain_m", "distance_km"}.issubset(df.columns):
        df["elevation_per_km"] = df["elevation_gain_m"] / df["distance_km"].clip(lower=0.1)
    # Example feature: speed from distance and time (if not the target)
    if {"distance_km", "ride_time_min"}.issubset(df.columns) and "computed_speed_kmph" not in df.columns:
        hours = df["ride_time_min"].clip(lower=1e-3) / 60.0
        df["computed_speed_kmph"] = df["distance_km"] / hours
    return df
