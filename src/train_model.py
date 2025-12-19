import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from config import DATA_PROCESSED_PATH, MODEL_PATH, TARGET_COL, RANDOM_STATE
from features import add_features

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED_PATH)
    df = add_features(df)
    return df

if __name__ == "__main__":
    df = load_data()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in processed data.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
