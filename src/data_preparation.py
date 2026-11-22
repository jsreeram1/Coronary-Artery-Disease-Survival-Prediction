import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


def validate_config(config: dict) -> None:
    """Validate that required top-level and nested configuration keys exist.

    Args:
        config: Parsed YAML configuration dictionary.

    Raises:
        ValueError: If required configuration sections or keys are missing.
    """
    required_sections = ["paths", "data", "training", "models"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    required_data_keys = ["target_col", "id_col", "numeric_cols", "binary_cols", "ordinal_cols", "encoding_categories"]
    for key in required_data_keys:
        if key not in config["data"]:
            raise ValueError(f"Missing required key '{key}' under config['data']")


def validate_schema(df: pd.DataFrame, config: dict, is_training: bool = True) -> None:
    """Verify that expected columns exist in the input DataFrame before preprocessing.

    Args:
        df: Raw loaded DataFrame.
        config: Configuration dictionary specifying required feature groups.
        is_training: Whether target column must be present.

    Raises:
        ValueError: If expected feature or target columns are missing.
    """
    expected_features = (
        list(config["data"]["numeric_cols"])
        + list(config["data"]["binary_cols"])
        + list(config["data"]["ordinal_cols"])
    )
    if is_training:
        expected_features.append(config["data"]["target_col"])

    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataset is missing required schema columns: {missing_cols}")


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load dataset from CSV or Excel using pathlib for cross-platform safety.

    Args:
        filepath: Path to the raw CSV or Excel dataset.

    Returns:
        Loaded pandas DataFrame with stripped column headers.

    Raises:
        FileNotFoundError: If neither the specified file nor alternate extension exists.
        ValueError: If the file extension is unsupported.
    """
    path = Path(filepath)
    if not path.exists():
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
    logger.info("Loaded raw dataset '%s' with shape %s", path.name, df.shape)
    return df


def clean_patient_data(
    df: pd.DataFrame,
    config: dict,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Clean anomalies, fix typos, deduplicate records, and engineer features.

    Args:
        df: Raw patient DataFrame.
        config: Project configuration dictionary.
        is_training: If True, drops rows with missing/unmappable target labels.

    Returns:
        Tuple of (cleaned feature/target DataFrame, preserved patient ID Series or None).
    """
    validate_schema(df, config, is_training=is_training)
    df = df.copy()
    initial_count = len(df)
    id_col = config["data"]["id_col"]
    target_col = config["data"]["target_col"]
    typo_maps = config["data"].get("typo_maps", {})

    # 1. Single-source deduplication by patient ID
    patient_ids: Optional[pd.Series] = None
    if id_col in df.columns:
        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
        patient_ids = df[id_col].copy()
        logger.info("Deduplicated patient records: %d -> %d rows", initial_count, len(df))

    # 2. Strip whitespace across all string/object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # 3. Standardize target column ('Survive') using .map()
    if target_col in df.columns:
        survive_map = typo_maps.get(target_col, {})
        cleaned_target = (
            df[target_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: survive_map.get(x, x))
        )
        df.loc[cleaned_target.index, target_col] = cleaned_target
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        if is_training:
            valid_mask = df[target_col].notna()
            df = df.loc[valid_mask].reset_index(drop=True)
            if patient_ids is not None:
                patient_ids = patient_ids.loc[valid_mask].reset_index(drop=True)
            df[target_col] = df[target_col].astype(int)
            class_counts = df[target_col].value_counts().to_dict()
            logger.info("Cleaned target distribution: %s (total=%d)", class_counts, len(df))

    # 4. Standardize categorical columns using typo maps
    categorical_cols = config["data"]["binary_cols"] + config["data"]["ordinal_cols"]
    for col in categorical_cols:
        if col in df.columns:
            col_map = typo_maps.get(col, {})
            non_null = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .map(lambda x: col_map.get(x, x))
            )
            df.loc[non_null.index, col] = non_null

    # 5. Fix negative Age values and impossible physical bounds
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").abs()
        df.loc[(df["Age"] < 1) | (df["Age"] > 120), "Age"] = np.nan

    for col in config["data"]["numeric_cols"]:
        if col != "Age" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= 0, col] = np.nan

    # 6. Feature Engineering: BMI & Age-Creatinine Clinical Interaction
    if {"Height", "Weight"}.issubset(df.columns):
        df["BMI"] = df["Weight"] / ((df["Height"] / 100.0) ** 2)
    if {"Age", "Creatinine"}.issubset(df.columns):
        df["Age_x_Creatinine"] = df["Age"] * df["Creatinine"]

    # 7. Drop non-predictive columns
    cols_to_drop = [c for c in config["data"]["drop_cols"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df, patient_ids


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Build leak-free preprocessing pipeline with scaling, binary encoding, and ordinal encoding.

    Args:
        config: Project configuration dictionary.

    Returns:
        Unfitted scikit-learn ColumnTransformer.
    """
    numeric_features: List[str] = list(config["data"]["numeric_cols"]) + ["BMI", "Age_x_Creatinine"]
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", binary_transformer, binary_features),
            ("ord", ordinal_transformer, ordinal_features),
        ],
        remainder="drop",
    )


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y.

    Args:
        df: Cleaned DataFrame containing features and target.
        target_col: Name of the target column.

    Returns:
        Tuple of (X DataFrame, y Series).

    Raises:
        ValueError: If target_col is not present in df.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    return df.drop(columns=[target_col]), df[target_col]
