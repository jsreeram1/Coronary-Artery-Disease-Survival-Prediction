import argparse
from pathlib import Path
import joblib
import pandas as pd
import yaml

from src.data_preparation import clean_patient_data, load_raw_data, split_features_target
from src.model_training import train_and_evaluate


def load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(Path(config_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_train(config: dict) -> None:
    print(f"Loading raw data from {config['paths']['raw_data']}...")
    raw_df = load_raw_data(config["paths"]["raw_data"])
    cleaned_df = clean_patient_data(raw_df, config, is_training=True)

    processed_path = Path(config["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)

    X, y = split_features_target(cleaned_df, config["data"]["target_col"])
    print(f"Training models on {len(X)} cleaned patient records ({X.shape[1]} features)...")
    summary = train_and_evaluate(X, y, config)

    print("\n=== Model Evaluation Summary ===")
    print(f"Selected Best Model: {summary['selected_best_model']} (Test ROC-AUC: {summary['best_test_roc_auc']})")
    for model_name, metrics in summary["model_comparisons"].items():
        print(
            f"- {model_name:22s} | CV AUC: {metrics['cv_roc_auc']:.4f} | "
            f"Test AUC: {metrics['test_roc_auc']:.4f} | "
            f"Acc: {metrics['test_accuracy']:.4f} | "
            f"Non-Survival Recall (Class 0): {metrics['test_recall_non_survival_class_0']:.4f}"
        )


def run_predict(config: dict, input_file: str, output_file: str) -> None:
    model_path = Path(config["paths"]["model_output"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run --mode train first.")

    model = joblib.load(model_path)
    raw_df = load_raw_data(input_file)
    id_col = config["data"]["id_col"]

    # Deduplicate raw_df IDs in sync with clean_patient_data
    if id_col in raw_df.columns:
        raw_df = raw_df.drop_duplicates(subset=[id_col], keep="first")
        patient_ids = raw_df[id_col].reset_index(drop=True)
    else:
        patient_ids = None

    cleaned_df = clean_patient_data(raw_df, config, is_training=False)
    if config["data"]["target_col"] in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=[config["data"]["target_col"]])

    preds = model.predict(cleaned_df)
    probs = model.predict_proba(cleaned_df)[:, 1]

    out_df = pd.DataFrame({
        "Predicted_Survive": preds,
        "Survival_Probability": probs.round(4),
        "Risk_Flag": ["High Risk (Preemptive Care)" if p == 0 else "Standard Risk" for p in preds],
    })
    if patient_ids is not None:
        out_df.insert(0, id_col, patient_ids)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coronary Disease Survival Prediction Pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Execution mode")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--input", default="data/new_patients.csv", help="Input file for prediction mode")
    parser.add_argument("--output", default="data/predictions.csv", help="Output file for prediction mode")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "predict":
        run_predict(cfg, args.input, args.output)
