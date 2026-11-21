from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load dataset from CSV or Excel using pathlib for cross-platform safety."""
    path = Path(filepath)
    if not path.exists():
        # Automatically fallback to .xlsx if .csv was configured (or vice versa)
        alt_path = path.with_suffix(".xlsx" if path.suffix.lower() == ".csv" else ".csv")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df.columns = [col.strip() for col in df.columns]
    return df


def clean_patient_data(df: pd.DataFrame, config: dict, is_training: bool = True) -> pd.DataFrame:
    """Clean anomalies, fix typos, deduplicate records, and engineer BMI."""
    df = df.copy()
    id_col = config["data"]["id_col"]
    target_col = config["data"]["target_col"]
    typo_maps = config["data"].get("typo_maps", {})

    # 1. Deduplicate patient records based on ID
    if id_col in df.columns:
        df = df.drop_duplicates(subset=[id_col], keep="first")

    # 2. Strip whitespace across all string/object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # 3. Standardize target column ('Survive') using typo map
    if target_col in df.columns:
        survive_map = typo_maps.get(target_col, {})
        cleaned_target = (
            df[target_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(survive_map)
        )
        df.loc[cleaned_target.index, target_col] = cleaned_target
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        if is_training:
            df = df.dropna(subset=[target_col])
            df[target_col] = df[target_col].astype(int)

    # 4. Standardize categorical columns using typo maps
    categorical_cols = config["data"]["binary_cols"] + config["data"]["ordinal_cols"]
    for col in categorical_cols:
        if col in df.columns:
            col_map = typo_maps.get(col, {})
            non_null = df[col].dropna().astype(str).str.strip().str.lower().replace(col_map)
            df.loc[non_null.index, col] = non_null

    # 5. Fix negative Age values (e.g., -60 -> 60) and impossible physical bounds
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").abs()
        df.loc[(df["Age"] < 1) | (df["Age"] > 120), "Age"] = np.nan

    for col in config["data"]["numeric_cols"]:
        if col != "Age" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= 0, col] = np.nan

    # 6. Feature Engineering: Body Mass Index (BMI)
    if {"Height", "Weight"}.issubset(df.columns):
        height_m = df["Height"] / 100.0
        df["BMI"] = df["Weight"] / (height_m ** 2)

    # 7. Drop non-predictive columns (ID, Favourite color)
    cols_to_drop = [c for c in config["data"]["drop_cols"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Build leak-free preprocessing pipeline with scaling, binary encoding, and ordinal encoding."""
    numeric_features: List[str] = list(config["data"]["numeric_cols"]) + ["BMI"]
    binary_features: List[str] = list(config["data"]["binary_cols"])
    ordinal_features: List[str] = list(config["data"]["ordinal_cols"])
    encoding_cats: dict = config["data"]["encoding_categories"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_categories = [encoding_cats[col] for col in binary_features]
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("binary_encoder", OrdinalEncoder(
            categories=binary_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    ordinal_categories = [encoding_cats[col] for col in ordinal_features]
    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal_encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", binary_transformer, binary_features),
            ("ord", ordinal_transformer, ordinal_features),
        ],
        remainder="drop",
    )
    return preprocessor


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
