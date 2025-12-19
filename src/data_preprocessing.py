import pandas as pd
from config import DATA_RAW_PATH, DATA_PROCESSED_PATH

def load_raw_data(path: str = DATA_RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing values for simplicity
    df = df.dropna()
    # Ensure correct data types
    if "route_type" in df.columns:
        df["route_type"] = df["route_type"].astype("category")
    return df

def save_processed(df: pd.DataFrame, path: str = DATA_PROCESSED_PATH) -> None:
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_raw_data()
    df_clean = basic_cleaning(df)
    save_processed(df_clean)
    print(f"Processed data saved to {DATA_PROCESSED_PATH}")
