# Coronary Artery Disease Survival Prediction

## 1. Project Overview & Clinical Objective
Cardiovascular diseases—particularly **coronary artery disease (CAD)**—are among the leading causes of mortality globally. Early risk stratification enables clinicians to identify high-risk patients (`Survive = 0`) and initiate timely, preemptive medical interventions.

This repository provides a modular, reproducible, configuration-driven machine learning pipeline that:
1. Audits, validates, and cleans noisy clinical records (`15,000` raw records $\rightarrow$ `14,042` unique validated patient records).
2. Engineers domain-informed physiological and interaction features (`BMI` and `Age_x_Creatinine`).
3. Benchmarks four distinct classifiers against a naive `DummyBaseline` using **5-Fold Stratified Cross-Validation**.
4. Generates automated visual diagnostics (ROC curves, Precision-Recall curves, Confusion Matrix heatmaps), timestamped model snapshots, and a CLI for batch patient survival inference.

---

## 2. Repository Structure
```text
coronary_survival_prediction/
├── config.yaml                          # Centralized paths, typo maps, encoding scales, & hyperparameter grids
├── main.py                              # CLI entry point supporting --mode train and --mode predict
├── requirements.txt                     # Pinned Python package dependencies
├── README.md                            # Project documentation, methodology, schema, & clinical discussion
├── data/
│   ├── patients_med_data.csv            # Raw patient dataset (.csv or .xlsx supported)
│   └── processed_patients.csv           # Deduplicated, cleaned, & feature-engineered dataset
├── models/
│   ├── best_survival_model.joblib       # Latest serialized best-performing scikit-learn pipeline
│   ├── fitted_preprocessor.joblib       # Standalone fitted ColumnTransformer artifact
│   └── evaluation_metrics.json          # Cross-validation, holdout test metrics, & split metadata
├── reports/
│   ├── latest_roc_pr_curves.png         # Comparative ROC & Precision-Recall curves across all models
│   ├── latest_confusion_matrix.png      # Holdout confusion matrix heatmap for the selected best model
│   └── config_snapshot_<UTC>.yaml       # Exact configuration snapshot saved per training run
├── notebooks/
│   └── eda.ipynb                        # Exploratory Data Analysis: IQR outlier audit, VIF, & RF importance
├── src/
│   ├── __init__.py
│   ├── data_preparation.py              # Schema validation, typo cleaning, feature engineering, & encoding
│   └── model_training.py                # Config-driven GridSearchCV training, evaluation, & plot generation
└── tests/
    └── test_pipeline.py                 # Automated pytest suite (unit, schema/negative, & end-to-end tests)
```

---

## 3. Expected Input Data Schema
The pipeline enforces strict schema validation upon loading (`validate_schema()`):

| Column Name | Type | Description / Allowed Values |
| :--- | :--- | :--- |
| `ID` | String | Unique patient identifier (used for deduplication and prediction tracking; excluded from model features). |
| `Survive` | Binary (`0`/`1`) | **Target variable** (required for `--mode train`, optional for `--mode predict`): `1` = Survived, `0` = Non-Survival. |
| `Age` | Numeric | Patient age in years (negative sign-entry errors corrected via `abs()`). |
| `Gender` | Categorical | Biological sex (`Female`, `Male`). |
| `Smoke` | Categorical | Smoking status (`No`, `Yes`). |
| `Diabetes` | Ordinal | Glycemic status (`Normal` < `Pre-diabetes` < `Diabetes`). |
| `Ejection Fraction` | Ordinal | Cardiac pumping capacity (`Low` < `Normal` < `High`). |
| `Sodium` | Numeric | Serum sodium level (mEq/L). |
| `Creatinine` | Numeric | Serum creatinine level (mg/dL). |
| `Platelets` | Numeric | Platelet count (kiloplatelets/mL). |
| `Creatine phosphokinase` | Numeric | CPK enzyme level (mcg/L). |
| `Blood Pressure` | Numeric | Systolic/arterial blood pressure (mmHg). |
| `Hemoglobin` | Numeric | Hemoglobin concentration (g/dL). |
| `Height` | Numeric | Patient height (cm). |
| `Weight` | Numeric | Patient weight (kg). |
| `Favourite color` | String | Non-clinical noise column (automatically dropped). |

---

## 4. Exploratory Data Analysis & Preprocessing Decisions

Key findings discovered in `notebooks/eda.ipynb` and implemented in `src/data_preparation.py`:

1. **Deduplication & Target Standardization (`15,000` $\rightarrow$ `14,042` Records)**:
   * Removed **958 duplicate patient `ID` rows** (`keep="first"`) to prevent identical patient records from leaking across training and holdout test splits.
   * Standardized inconsistent string labels across `Survive`, `Gender`, `Smoke`, `Diabetes`, and `Ejection Fraction` using configuration-driven lookup dictionaries.
2. **Target Class Imbalance (`2.11 : 1` Ratio)**:
   * Cleaned dataset contains **`9,529` Non-Survival (`0`, `67.86%`)** and **`4,513` Survival (`1`, `32.14%`)** records (`2.11 : 1` ratio).
   * *Analytical Implication*: Because predicting the majority class (`0`) alone yields `67.85%` baseline accuracy, accuracy is insufficient on its own. We use **Stratified splitting**, **`class_weight="balanced"`**, **ROC-AUC**, **PR-AUC**, and **Class-0 Recall** to evaluate models fairly.
3. **Anomaly Correction & Outlier Retention Strategy**:
   * **Sign Errors**: Converted negative `Age` entries (e.g., `-60` $\rightarrow$ `60`) via `abs()` and set physiologically impossible `<= 0` lab measurements to `NaN`.
   * **Clinical Outliers (IQR Audit)**: Right-skewed biomarkers (`Creatine phosphokinase`, `Creatinine`, `Platelets`) contain extreme clinical values outside standard `1.5 * IQR` fences. Rather than deleting or clipping valid extreme lab results—which represent acute cardiac or renal distress—we **retain** them and rely on median imputation and tree-based ensembles (`RandomForest`, `GradientBoosting`) that naturally handle skewed tails.
4. **Missing Value Strategy (`499` Missing `Creatinine` Records)**:
   * All `499` rows missing `Creatinine` (~3.5% of cleaned records) are **retained** and imputed using **`SimpleImputer(strategy="median")`** inside the cross-validated `ColumnTransformer`. This preserves 14 other valid clinical features per patient, avoids hospital selection bias, resists right-skew distortion, and ensures `--mode predict` can score new patients who arrive with pending lab results.
5. **Domain Feature Engineering & Leak-Free Encoding (`15` Model Features)**:
   * **Engineered Features**:
     * `BMI` = $\text{Weight (kg)} / (\text{Height (m)})^2$
     * `Age_x_Creatinine` = $\text{Age} \times \text{Creatinine}$ (captures compounding cardio-renal risk in older patients)
   * **Preprocessing (`ColumnTransformer` fit strictly on training folds)**:
     * *11 Numeric Features* (`Age`, `Sodium`, `Creatinine`, `Platelets`, `Creatine phosphokinase`, `Blood Pressure`, `Hemoglobin`, `Height`, `Weight`, `BMI`, `Age_x_Creatinine`): Median imputation + `StandardScaler`.
     * *2 Binary Features* (`Gender`: `Female=0, Male=1`; `Smoke`: `No=0, Yes=1`): Mode imputation + `OrdinalEncoder`.
     * *2 Ordinal Features* (`Ejection Fraction`: `Low=0 < Normal=1 < High=2`; `Diabetes`: `Normal=0 < Pre-diabetes=1 < Diabetes=2`): Mode imputation + ordered `OrdinalEncoder`.

---

## 5. Model Comparison & Clinical Interpretation

All models were evaluated using an **80/20 stratified train-test split** (`11,233` train / `2,809` test records) and **5-Fold Stratified Cross-Validation** (`GridSearchCV`, `random_state=42`, operating threshold = `0.50`):

| Model | 5-Fold CV ROC-AUC | Holdout Test ROC-AUC | Holdout Test PR-AUC | Test Accuracy | Non-Survival Recall (Class `0`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Gradient Boosting** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Support Vector Machine (SVC)** | 0.9993 | 0.9993 | 0.9958 | 0.9925 | 0.9932 |
| **Logistic Regression** | 0.8770 | 0.8747 | 0.7706 | 0.7903 | 0.7865 |
| **Dummy Baseline (`most_frequent`)** | 0.5000 | 0.5000 | 0.3215 | 0.6785 | 1.0000 |

### Key Analytical Takeaways
1. **Substantial Lift Over Naive Baseline**: All trained classifiers significantly outperform `DummyBaseline` (`0.5000` ROC-AUC, `0.3215` PR-AUC).
2. **Non-Linear Clinical Interactions**: Tree-based ensembles (**Random Forest** and **Gradient Boosting**) and **RBF-kernel SVM** (`0.9993` Test AUC) substantially outperform linear **Logistic Regression** (`0.8747` Test AUC). This confirms strong non-linear decision boundaries and multi-variable interactions among **Ejection Fraction**, **Serum Creatinine**, **Age**, and **Hemoglobin**.
3. **Clinical Safety & Threshold Configuration**: **Random Forest** achieves `1.0000` **Non-Survival Recall (Class `0`)** and `1.0000` ROC-AUC across both 5-fold cross-validation and holdout evaluation, and is saved as the default production artifact (`models/best_survival_model.joblib`). For deployment scenarios requiring even more conservative screening on unseen cohorts, `classification_threshold` can be tuned directly in `config.yaml`.

---

## 6. Setup, Reproducibility & Usage

### Environment Requirements
* **Python**: `3.11+` (tested on `Python 3.11.17`)
* Global random seeds (`random` and `numpy`) and estimator seeds are pinned via `training.random_state: 42` in `config.yaml`.

### Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows PowerShell: .\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### Run Automated Tests
```bash
python -m pytest -v
```
