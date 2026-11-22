import argparse
import logging
import platform
from pathlib import Path
import random
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from src.data_preparation import (
    clean_patient_data,
    load_raw_data,
    split_features_target,
    validate_config,
)
from src.model_training import train_and_evaluate


def configure_logging() -> None:
    """Configure structured console logging with timestamps and severity levels."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_global_seeds(seed: int) -> None:
    """Set global random seeds for Python and NumPy reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load and validate YAML configuration from disk.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If YAML is empty, malformed, or missing required keys.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file '{path}' must define a YAML mapping.")

    validate_config(config)
    return config


def run_train(config: dict) -> dict:
    """Execute full training, model comparison, and evaluation report generation.

    Args:
        config: Validated project configuration dictionary.

    Returns:
        Evaluation summary dictionary.
    """
    logger = logging.getLogger(__name__)
    seed = int(config["training"]["random_state"])
    set_global_seeds(seed)
    logger.info(
        "Environment | Python=%s | pandas=%s | numpy=%s | scikit-learn=%s | seed=%d",
        platform.python_version(),
        pd.__version__,
        np.__version__,
        sklearn.__version__,
        seed,
    )

    raw_df = load_raw_data(config["paths"]["raw_data"])
    cleaned_df, _ = clean_patient_data(raw_df, config, is_training=True)

    processed_path = Path(config["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)

    X, y = split_features_target(cleaned_df, config["data"]["target_col"])
    logger.info("Starting model training on %d records (%d raw/engineered features)", len(X), len(X.columns))
    summary = train_and_evaluate(X, y, config)

    logger.info(
        "Selected Best Model: %s (Test ROC-AUC: %.4f)",
        summary["selected_best_model"],
        summary["best_test_roc_auc"],
    )
    for model_name, metrics in summary["model_comparisons"].items():
        logger.info(
            "Model %-22s | CV AUC: %.4f | Test AUC: %.4f | PR AUC: %.4f | Acc: %.4f | Class-0 Recall: %.4f",
            model_name,
            metrics["cv_roc_auc"],
            metrics["test_roc_auc"],
            metrics["test_pr_auc"],
            metrics["test_accuracy"],
            metrics["test_recall_non_survival_class_0"],
        )
    return summary


def run_predict(config: dict, input_file: str | Path, output_file: str | Path) -> pd.DataFrame:
    """Run batch survival predictions on new patient records using the saved pipeline.

    Args:
        config: Validated project configuration dictionary.
        input_file: Path to input patient records (.csv or .xlsx).
        output_file: Destination path for output predictions CSV.

    Returns:
        DataFrame containing predictions, probabilities, and clinical risk flags.
    """
    logger = logging.getLogger(__name__)
    model_path = Path(config["paths"]["model_output"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run --mode train first.")

    model = joblib.load(model_path)
    raw_df = load_raw_data(input_file)
    id_col = config["data"]["id_col"]
    threshold = float(config["training"].get("classification_threshold", 0.50))

    cleaned_df, patient_ids = clean_patient_data(raw_df, config, is_training=False)
    if config["data"]["target_col"] in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=[config["data"]["target_col"]])

    probs = model.predict_proba(cleaned_df)[:, 1]
    preds = (probs >= threshold).astype(int)

    out_df = pd.DataFrame({
        "Predicted_Survive": preds,
        "Survival_Probability": probs.round(4),
        "Risk_Flag": ["High Risk (Preemptive Care)" if p == 0 else "Standard Risk" for p in preds],
    })
    if patient_ids is not None:
        out_df.insert(0, id_col, patient_ids.values)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(out_df), out_path.resolve())
    return out_df


def main() -> int:
    """CLI entry point returning 0 on success and 1 on error."""
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Coronary Disease Survival Prediction Pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Execution mode")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--input", default="data/new_patients.csv", help="Input file for prediction mode")
    parser.add_argument("--output", default="data/predictions.csv", help="Output file for prediction mode")

    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        if args.mode == "train":
            run_train(cfg)
        elif args.mode == "predict":
            run_predict(cfg, args.input, args.output)
        return 0
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
