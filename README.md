# Coronary Artery Disease Survival Prediction

## 1. Project Overview & Objective
Cardiovascular diseases—specifically **coronary artery disease (CAD)**—are among the leading causes of mortality worldwide. Early risk stratification enables clinicians to identify high-risk patients (`Survive = 0`) and initiate timely, preemptive medical interventions.

This repository provides a modular, end-to-end machine learning pipeline that cleans noisy clinical records, engineers physiological features, compares four classification models via Stratified Cross-Validation, and exposes a CLI for both model training and batch patient risk inference.

---

## 2. Repository Structure
```text
coronary_survival_prediction/
├── config.yaml                      # Centralized file paths, typo maps, encoding rules, and hyperparameters
├── main.py                          # CLI entry point supporting --mode train and --mode predict
├── requirements.txt                 # Python package dependencies
├── README.md                        # Project documentation, methodology, and results
├── data/
│   ├── patients_med_data.csv        # Raw patient dataset (.csv or .xlsx supported)
│   └── processed_patients.csv       # Deduplicated and cleaned dataset generated during training
├── models/
│   ├── best_survival_model.joblib   # Serialized best-performing scikit-learn pipeline
│   └── evaluation_metrics.json      # Cross-validation & holdout test metrics across all models
├── notebooks/
│   └── eda.ipynb                    # Exploratory Data Analysis, anomaly audit, encoding, & VIF checks
├── src/
│   ├── __init__.py
│   ├── data_preparation.py          # Typo standardization, anomaly fixes, BMI engineering, & ColumnTransformer
│   └── model_training.py            # Stratified 5-Fold CV, GridSearchCV tuning, & model evaluation
└── tests/
    └── test_pipeline.py             # Automated pytest suite verifying cleaning and encoding logic
```

---

## 3. Exploratory Data Analysis & Preprocessing Highlights
Key data quality findings uncovered in `notebooks/eda.ipynb` and automated in `src/data_preparation.py` (producing **14,042 cleaned patient records**):
1. **Duplicate Patient Removal**: Deduplicates records by `ID` (`keep="first"`) so identical patient records cannot leak across training and test splits.
2. **Typo & Casing Standardization**: Normalizes inconsistent text entries across `Survive`, `Gender`, `Smoke`, `Diabetes`, and `Ejection Fraction` using configuration-driven lookup dictionaries.
3. **Anomaly Correction**: Converts negative `Age` sign errors (e.g., `-60` $\rightarrow$ `60`) via `abs()` and replaces physiologically invalid `<= 0` lab measurements with `NaN` for median imputation.
4. **Feature Engineering**: Computes **Body Mass Index (BMI)** from `Height` (cm) and `Weight` (kg):
   $$\text{BMI} = \frac{\text{Weight (kg)}}{(\text{Height (m)})^2}$$
5. **Noise Removal**: Drops non-predictive columns (`ID` and `Favourite color`).
6. **Leak-Free Encoding & Scaling (`14` Final Model Features)**:
   * **Numeric Features (10)**: `Age`, `Sodium`, `Creatinine`, `Platelets`, `Creatine phosphokinase`, `Blood Pressure`, `Hemoglobin`, `Height`, `Weight`, and `BMI` $\rightarrow$ Median imputation + `StandardScaler`.
   * **Binary Features (2)**: `Gender` (`Female=0, Male=1`) and `Smoke` (`No=0, Yes=1`) $\rightarrow$ Mode imputation + `OrdinalEncoder`.
   * **Ordinal Features (2)**: `Ejection Fraction` (`Low=0 < Normal=1 < High=2`) and `Diabetes` (`Normal=0 < Pre-diabetes=1 < Diabetes=2`) $\rightarrow$ Mode imputation + ordered `OrdinalEncoder`.

---

## 4. Model Comparison & Results
Four distinct classification algorithms were trained and tuned using **5-Fold Stratified Cross-Validation** (`GridSearchCV`, `random_state=42`, 80/20 stratified train-test split):

| Model | 5-Fold CV ROC-AUC | Holdout Test ROC-AUC | Test Accuracy | Non-Survival Recall (Class `0`) |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Selected)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Support Vector Machine (SVC)** | 0.9995 | 0.9994 | 0.9897 | 0.9853 |
| **Logistic Regression** | 0.8771 | 0.8748 | 0.7903 | 0.7870 |

### Key Analytical Takeaways
* **Non-Linear Clinical Interactions**: Tree-based ensembles (**Gradient Boosting** and **Random Forest**) and **RBF-kernel SVM** substantially outperform linear **Logistic Regression** (`0.8748` Test AUC), demonstrating strong non-linear relationships and interaction effects among cardiac biomarkers (such as `Ejection Fraction`, `Creatinine`, and `Age`).
* **Clinical Safety Alignment**: Both Gradient Boosting and Random Forest achieve `1.0000` **Recall on Class `0` (Non-Survival)**, ensuring zero high-risk patients are missed in holdout evaluation. **Gradient Boosting** is saved as the default production artifact (`models/best_survival_model.joblib`).

---

## 5. Setup & Usage

### Installation
```bash
pip install -r requirements.txt
```

### Run Unit Tests
```bash
python -m pytest -v
```

### Train & Evaluate Models
```bash
python main.py --mode train --config config.yaml
```
*Generates `data/processed_patients.csv`, `models/best_survival_model.joblib`, and `models/evaluation_metrics.json`.*

### Generate Predictions on New Patient Data
```bash
python main.py --mode predict --input data/new_patients.csv --output data/predictions.csv
```
*Outputs patient IDs, `Predicted_Survive` (`0` or `1`), `Survival_Probability`, and a clinical `Risk_Flag` (`High Risk (Preemptive Care)` vs. `Standard Risk`).*

