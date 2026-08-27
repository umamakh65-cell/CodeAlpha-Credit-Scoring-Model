
"""
CodeAlpha Task 1 — Credit Scoring Model

This project predicts whether a person is a good or bad credit risk using
the UCI Statlog (German Credit Data) dataset.

Run:
    python credit_scoring.py

The script downloads the dataset automatically, preprocesses mixed
categorical/numerical features, trains Logistic Regression, Decision Tree,
and Random Forest models, compares Precision/Recall/F1/ROC-AUC, and saves
the best model plus evaluation plots.
"""

from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.inspection import permutation_importance
import joblib

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_FILE = DATA_DIR / "german.data"

COLUMN_NAMES = [
    "checking_account", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "dependents", "telephone",
    "foreign_worker", "credit_risk"
]

def download_data():
    if not DATA_FILE.exists():
        print("Downloading UCI German Credit dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES)

def prepare_target(df):
    # UCI target: 1 = good credit, 2 = bad credit.
    # Convert to 1 = good, 0 = bad for easier binary classification.
    df = df.copy()
    df["credit_risk"] = (df["credit_risk"] == 1).astype(int)
    return df

def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numerical),
        ("categorical", categorical_pipe, categorical)
    ])

def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probability)
    }

    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probability)
    plt.title(f"{name} — ROC Curve")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name.lower().replace(' ', '_')}_roc_curve.png", dpi=200)
    plt.close()

    return metrics

def main():
    df = prepare_target(download_data())

    print("\nDataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df["credit_risk"].value_counts())

    X = df.drop(columns="credit_risk")
    y = df["credit_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
    }

    results = {}
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(name, pipeline, X_test, y_test)
        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results.values()).sort_values(
        "ROC-AUC", ascending=False
    )
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, RESULTS_DIR / "best_credit_scoring_model.joblib")

    # A simple feature-importance analysis using permutation importance.
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=5,
        random_state=42, scoring="roc_auc"
    )
    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": perm.importances_mean
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(RESULTS_DIR / "permutation_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    top = importance_df.head(10).sort_values("Importance")
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Mean decrease in ROC-AUC")
    plt.title(f"Top Features — {best_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_feature_importance.png", dpi=200)
    plt.close()

    print(f"\nBest model by ROC-AUC: {best_name}")
    print("Saved all results in the results/ folder.")

if __name__ == "__main__":
    main()
