import pandas as pd
from typing import Optional, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------
# 📥 DATA PREPROCESSING
# -------------------------
def preprocess_data(
    path: str,
    training: bool = True,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    df = pd.read_csv(path)

    # -------------------------
    # 🎯 TARGET CLEANING (FIXED)
    # -------------------------
    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' not found in dataset")

    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()

    valid_map = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0
    }

    df["Churn"] = df["Churn"].map(valid_map)

    # Debug prints (important for CI/CD logs)
    print("Unique values after mapping:", df["Churn"].unique())

    # 🚨 Safety check
    if df["Churn"].isnull().all():
        raise ValueError("Churn mapping failed — all values became NaN")

    # Remove invalid rows
    df = df.dropna(subset=["Churn"])

    print("Dataset shape after cleaning:", df.shape)

    # -------------------------
    # 🎯 TRAIN / INFERENCE SPLIT
    # -------------------------
    if training:
        if target_col is None:
            target_col = "Churn"

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)

        return X, y

    else:
        return df, None


# -------------------------
# 🔥 PREPROCESSOR
# -------------------------
def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    # Automatically detect features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    return preprocess
