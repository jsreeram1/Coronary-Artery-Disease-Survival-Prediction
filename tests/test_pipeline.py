import pandas as pd
import pytest
from main import load_config
from src.data_preparation import build_preprocessor, clean_patient_data, split_features_target


@pytest.fixture
def project_config():
    return load_config("config.yaml")


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "ID": ["TIG1GE", "VXUCV9", "ZK6PAC", "TIG1GE"],
        "Survive": ["0", "yes", "No", "0"],
        "Gender": ["male", "Female", "M", "male"],
        "Smoke": ["Yes", "no", "N", "Yes"],
        "Diabetes": ["Normal", "prediabetes", "diabetes", "Normal"],
        "Age": list((50, 75, -60, 50)),
        "Ejection Fraction": ["low", "Normal", "high", "low"],
        "Sodium": list((137, 130, 140, 137)),
        "Creatinine": list((0.7, 2.5, 0.9, 0.7)),
        "Platelets": list((265000, 150000, 310000, 265000)),
        "Creatine phosphokinase": list((582, 120, 250, 582)),
        "Blood Pressure": list((120, 145, 115, 120)),
        "Hemoglobin": list((12.3, 13.0, 15.3, 12.3)),
        "Height": list((170, 165, 175, 170)),
        "Weight": list((70, 80, 68, 70)),
        "Favourite color": ["green", "black", "yellow", "green"],
    })


def test_clean_patient_data_handles_typos_and_anomalies(sample_raw_df, project_config):
    cleaned = clean_patient_data(sample_raw_df, project_config, is_training=True)

    # Deduplication (4 rows -> 3 unique IDs)
    assert len(cleaned) == 3

    # Target standardized to integer 0/1
    assert cleaned["Survive"].tolist() == list((0, 1, 0))

    # Typo standardization checks
    assert cleaned["Gender"].tolist() == ["Male", "Female", "Male"]
    assert cleaned["Smoke"].tolist() == ["Yes", "No", "No"]
    assert cleaned["Diabetes"].tolist() == ["Normal", "Pre-diabetes", "Diabetes"]
    assert cleaned["Ejection Fraction"].tolist() == ["Low", "Normal", "High"]

    # Negative age fixed (-60 -> 60)
    assert (cleaned["Age"] > 0).all()
    assert 60 in cleaned["Age"].values

    # Noise columns dropped & BMI created
    assert "ID" not in cleaned.columns
    assert "Favourite color" not in cleaned.columns
    assert "BMI" in cleaned.columns


def test_preprocessor_binary_and_ordinal_encoding(sample_raw_df, project_config):
    cleaned = clean_patient_data(sample_raw_df, project_config, is_training=True)
    X, y = split_features_target(cleaned, "Survive")
    preprocessor = build_preprocessor(project_config)
    X_transformed = preprocessor.fit_transform(X)

    # 10 numeric (9 raw + BMI) + 2 binary + 2 ordinal = 14 total features
    assert X_transformed.shape == (3, 14)
    assert not pd.isna(X_transformed).any()

    # Verify ordinal encoding order for Ejection Fraction (index 12: Low=0, Normal=1, High=2)
    assert X_transformed[:, 12].tolist() == list((0.0, 1.0, 2.0))

