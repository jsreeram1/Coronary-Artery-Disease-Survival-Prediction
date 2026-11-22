import copy
import pandas as pd
import pytest
from main import load_config, run_predict, run_train
from src.data_preparation import (
    build_preprocessor,
    clean_patient_data,
    load_raw_data,
    split_features_target,
)


@pytest.fixture
def project_config():
    """Load base project configuration."""
    return load_config("config.yaml")


@pytest.fixture
def sample_raw_df():
    """Create a realistic synthetic dataset covering typos, duplicates, and anomalies."""
    rows = []
    patterns = [
        ("ID_1", "0", "male", "Yes", "Normal", 50, "low", 137, 0.7, 265000, 582, 120, 12.3, 170, 70, "green"),
        ("ID_2", "yes", "Female", "no", "prediabetes", 75, "Normal", 130, 2.5, 150000, 120, 145, 13.0, 165, 80, "black"),
        ("ID_3", "No", "M", "N", "diabetes", -60, "high", 140, 0.9, 310000, 250, 115, 15.3, 175, 68, "yellow"),
        ("ID_1", "0", "male", "Yes", "Normal", 50, "low", 137, 0.7, 265000, 582, 120, 12.3, 170, 70, "green"),
    ]
    for batch in range(5):
        for idx, p in enumerate(patterns):
            base_id = next(iter(p))
            unique_id = f"{base_id}_{batch}" if not (batch == 0 and idx == 3) else "ID_1_0"
            row = [unique_id] + list(p[1:])
            rows.append(row)

    columns = [
        "ID", "Survive", "Gender", "Smoke", "Diabetes", "Age", "Ejection Fraction",
        "Sodium", "Creatinine", "Platelets", "Creatine phosphokinase",
        "Blood Pressure", "Hemoglobin", "Height", "Weight", "Favourite color"
    ]
    return pd.DataFrame(rows, columns=columns)


def test_clean_patient_data_handles_typos_and_anomalies(sample_raw_df, project_config):
    """Verify deduplication, typo normalization, negative age fix, and engineered features."""
    cleaned, patient_ids = clean_patient_data(sample_raw_df, project_config, is_training=True)

    assert len(cleaned) == 15
    assert len(patient_ids) == 15
    assert set(cleaned["Survive"].unique()) == {0, 1}
    assert set(cleaned["Gender"].unique()) == {"Male", "Female"}
    assert set(cleaned["Smoke"].unique()) == {"Yes", "No"}
    assert set(cleaned["Diabetes"].unique()) == {"Normal", "Pre-diabetes", "Diabetes"}
    assert set(cleaned["Ejection Fraction"].unique()) == {"Low", "Normal", "High"}
    assert (cleaned["Age"] > 0).all()
    assert "ID" not in cleaned.columns
    assert "Favourite color" not in cleaned.columns
    assert {"BMI", "Age_x_Creatinine"}.issubset(cleaned.columns)


def test_preprocessor_binary_and_ordinal_encoding(sample_raw_df, project_config):
    """Verify leak-free preprocessor output dimensionality and ordinal category ordering."""
    cleaned, _ = clean_patient_data(sample_raw_df, project_config, is_training=True)
    X, _ = split_features_target(cleaned, "Survive")
    preprocessor = build_preprocessor(project_config)
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape == (15, 15)
    assert not pd.isna(X_transformed).any()
    ef_col = X_transformed[:, 13].tolist()
    assert ef_col[:3] == [0.0, 1.0, 2.0]


def test_negative_cases_missing_files_and_bad_schema(tmp_path, sample_raw_df, project_config):
    """Verify defensive exceptions for missing files, unsupported formats, malformed YAML, and missing columns."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nonexistent.csv")

    bad_ext = tmp_path / "data.txt"
    bad_ext.write_text("dummy", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_raw_data(bad_ext)

    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text("paths: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed YAML"):
        load_config(bad_yaml)

    broken_df = sample_raw_df.drop(columns=["Creatinine"])
    with pytest.raises(ValueError, match="missing required schema columns"):
        clean_patient_data(broken_df, project_config, is_training=True)


def test_end_to_end_train_and_predict_workflow(tmp_path, sample_raw_df, project_config):
    """Run a fast end-to-end train -> predict integration test in an isolated temporary directory."""
    raw_csv = tmp_path / "raw_patients.csv"
    sample_raw_df.to_csv(raw_csv, index=False)

    cfg = copy.deepcopy(project_config)
    cfg["paths"]["raw_data"] = str(raw_csv)
    cfg["paths"]["processed_data"] = str(tmp_path / "processed.csv")
    cfg["paths"]["model_output"] = str(tmp_path / "model.joblib")
    cfg["paths"]["preprocessor_output"] = str(tmp_path / "preprocessor.joblib")
    cfg["paths"]["metrics_output"] = str(tmp_path / "metrics.json")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    cfg["training"]["cv_folds"] = 2
    for m_name in cfg["models"]:
        cfg["models"][m_name]["enabled"] = m_name in {"DummyBaseline", "LogisticRegression"}

    summary = run_train(cfg)
    assert summary["selected_best_model"] == "LogisticRegression"
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "reports" / "latest_roc_pr_curves.png").exists()

    unlabeled_csv = tmp_path / "new_patients.csv"
    sample_raw_df.drop(columns=["Survive"]).to_csv(unlabeled_csv, index=False)
    pred_csv = tmp_path / "predictions.csv"

    preds_df = run_predict(cfg, unlabeled_csv, pred_csv)
    assert pred_csv.exists()
    assert len(preds_df) == 15
    assert {"ID", "Predicted_Survive", "Survival_Probability", "Risk_Flag"}.issubset(preds_df.columns)
