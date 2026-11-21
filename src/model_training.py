import json
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data_preparation import build_preprocessor


def get_candidate_models(random_state: int) -> Dict[str, Dict[str, Any]]:
    """Define 4 candidate classifiers and hyperparameter grids for survival prediction."""
    return {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
            "params": {
                "classifier__C": list((0.01, 0.1, 1.0, 10.0)),
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(class_weight="balanced", random_state=random_state),
            "params": {
                "classifier__n_estimators": list((100, 200)),
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": list((2, 5)),
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "params": {
                "classifier__n_estimators": list((100, 200)),
                "classifier__learning_rate": list((0.05, 0.1)),
                "classifier__max_depth": list((3, 5)),
            },
        },
        "SupportVectorMachine": {
            "estimator": SVC(probability=True, class_weight="balanced", random_state=random_state),
            "params": {
                "classifier__C": list((0.5, 1.0, 5.0)),
                "classifier__kernel": ["rbf", "linear"],
            },
        },
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
    """Split data, tune candidate models via Stratified CV, and evaluate on holdout test set."""
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    cv_folds = config["training"]["cv_folds"]
    primary_metric = config["training"]["primary_metric"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    candidates = get_candidate_models(random_state)

    results = {}
    best_model_name = None
    best_pipeline = None
    best_score = -np.inf

    for name, spec in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor(config)),
            ("classifier", spec["estimator"]),
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            scoring=primary_metric,
            cv=cv,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_est = grid.best_estimator_
        y_pred = best_est.predict(X_test)
        y_prob = best_est.predict_proba(X_test)[:, 1]

        test_roc_auc = roc_auc_score(y_test, y_prob)
        test_f1 = f1_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        non_survival_recall = recall_score(y_test, y_pred, pos_label=0)

        results[name] = {
            "best_params": grid.best_params_,
            "cv_roc_auc": round(float(grid.best_score_), 4),
            "test_roc_auc": round(float(test_roc_auc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1_survival": round(float(test_f1), 4),
            "test_precision_survival": round(float(precision_score(y_test, y_pred)), 4),
            "test_recall_non_survival_class_0": round(float(non_survival_recall), 4),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        if test_roc_auc > best_score:
            best_score = test_roc_auc
            best_model_name = name
            best_pipeline = best_est

    model_path = Path(config["paths"]["model_output"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    summary = {
        "selected_best_model": best_model_name,
        "best_test_roc_auc": round(float(best_score), 4),
        "model_comparisons": results,
    }

    metrics_path = Path(config["paths"]["metrics_output"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
