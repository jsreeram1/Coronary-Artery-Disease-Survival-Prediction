from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import yaml

from src.data_preparation import build_preprocessor

logger = logging.getLogger(__name__)


def build_estimator_registry(random_state: int) -> Dict[str, Any]:
    """Return mapping of supported model names to instantiated scikit-learn estimators."""
    return {
        "DummyBaseline": DummyClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "SupportVectorMachine": SVC(probability=True, class_weight="balanced", random_state=random_state),
    }


def save_evaluation_plots(
    y_test: pd.Series,
    model_curves: Dict[str, tuple],
    best_model_name: str,
    best_y_pred: np.ndarray,
    reports_dir: Path,
    timestamp: str,
) -> None:
    """Generate and save ROC Curve, Precision-Recall Curve, and Best-Model Confusion Matrix."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))
    for name, (y_prob, roc_auc, pr_auc) in model_curves.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        ax_pr.plot(rec, prec, label=f"{name} (AP = {pr_auc:.3f})")

    diag = [0.0, 1.0]
    ax_roc.plot(diag, diag, "k--", alpha=0.5)
    ax_roc.set_title("Holdout ROC Curves Across Models")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", fontsize=8)

    ax_pr.set_title("Holdout Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(reports_dir / f"roc_pr_curves_{timestamp}.png", dpi=150)
    plt.savefig(reports_dir / "latest_roc_pr_curves.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_test, best_y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: Non-Survival (0)", "Pred: Survival (1)"],
        yticklabels=["Actual: Non-Survival (0)", "Actual: Survival (1)"],
        ax=ax_cm,
    )
    ax_cm.set_title(f"Confusion Matrix — {best_model_name}")
    plt.tight_layout()
    plt.savefig(reports_dir / f"confusion_matrix_{timestamp}.png", dpi=150)
    plt.savefig(reports_dir / "latest_confusion_matrix.png", dpi=150)
    plt.close(fig_cm)


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
    """Split data, tune config-defined models via Stratified CV, and save artifacts & reports."""
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    cv_folds = config["training"]["cv_folds"]
    primary_metric = config["training"]["primary_metric"]
    threshold = float(config["training"].get("classification_threshold", 0.50))

    imbalance_ratio = float(y.value_counts().max() / y.value_counts().min())
    logger.info("Training target imbalance ratio (majority:minority) = %.2f : 1", imbalance_ratio)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_preprocessor = build_preprocessor(config)
    registry = build_estimator_registry(random_state)

    results = {}
    model_curves = {}
    best_model_name = None
    best_pipeline = None
    best_y_pred = None
    best_score = -np.inf

    for name, model_cfg in config["models"].items():
        if not model_cfg.get("enabled", True):
            continue
        if name not in registry:
            raise ValueError(f"Unsupported model '{name}' in config['models'].")

        logger.info("Tuning model: %s", name)
        try:
            pipeline = Pipeline(steps=[
                ("preprocessor", clone(base_preprocessor)),
                ("classifier", registry[name]),
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=model_cfg.get("params", {}),
                scoring=primary_metric,
                cv=cv,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
        except Exception as exc:
            logger.error("Model training failed for '%s': %s", name, exc)
            raise RuntimeError(f"Training crashed on model '{name}': {exc}") from exc

        best_est = grid.best_estimator_
        y_prob = best_est.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        test_roc_auc = float(roc_auc_score(y_test, y_prob))
        test_pr_auc = float(average_precision_score(y_test, y_prob))
        test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
        test_acc = float(accuracy_score(y_test, y_pred))
        non_survival_recall = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))

        model_curves[name] = (y_prob, test_roc_auc, test_pr_auc)
        results[name] = {
            "best_params": grid.best_params_,
            "cv_roc_auc": round(float(grid.best_score_), 4),
            "test_roc_auc": round(test_roc_auc, 4),
            "test_pr_auc": round(test_pr_auc, 4),
            "test_accuracy": round(test_acc, 4),
            "test_f1_survival": round(test_f1, 4),
            "test_precision_survival": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "test_recall_non_survival_class_0": round(non_survival_recall, 4),
            "classification_threshold": threshold,
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }

        if name != "DummyBaseline" and test_roc_auc > best_score:
            best_score = test_roc_auc
            best_model_name = name
            best_pipeline = best_est
            best_y_pred = y_pred

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = Path(config["paths"].get("reports_dir", "reports"))
    save_evaluation_plots(y_test, model_curves, best_model_name, best_y_pred, reports_dir, timestamp)

    model_path = Path(config["paths"]["model_output"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    joblib.dump(best_pipeline, model_path.with_name(f"{model_path.stem}_{timestamp}.joblib"))

    if "preprocessor_output" in config["paths"]:
        prep_path = Path(config["paths"]["preprocessor_output"])
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline.named_steps["preprocessor"], prep_path)

    with open(reports_dir / f"config_snapshot_{timestamp}.yaml", "w", encoding="utf-8") as cfg_out:
        yaml.safe_dump(config, cfg_out, sort_keys=False)

    summary = {
        "run_timestamp_utc": timestamp,
        "selected_best_model": best_model_name,
        "best_test_roc_auc": round(float(best_score), 4),
        "classification_threshold": threshold,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "split_sizes": {"train_rows": len(X_train), "test_rows": len(X_test)},
        "model_comparisons": results,
    }

    metrics_path = Path(config["paths"]["metrics_output"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
