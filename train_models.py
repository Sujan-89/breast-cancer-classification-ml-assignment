"""Train the five classifiers and write the reproducible experiment outputs.

Run this script locally before starting or deploying the Streamlit application.
The application only reads the artifacts produced here.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_utils import FEATURE_NAMES, MODEL_FILES, ROOT, TARGET_COLUMN, evaluate_model, positive_scores


RANDOM_STATE = 42
TEST_SIZE = 0.20
SOURCE_URL = "https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def class_counts(target: pd.Series) -> dict[str, int]:
    return {str(label): int((target == label).sum()) for label in [0, 1]}


def build_models() -> dict[str, Pipeline]:
    """Choose parameters before looking at the held-out test results."""
    classifiers = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE
        ),
    }
    return {
        name: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", "passthrough" if name == "Decision Tree" else StandardScaler()),
            ("classifier", classifier),
        ])
        for name, classifier in classifiers.items()
    }


def write_feature_schema() -> None:
    rows = []
    for name in FEATURE_NAMES:
        if name.startswith("mean_"):
            description = "Mean measurement of " + name.removeprefix("mean_").replace("_", " ")
        elif name.endswith("_error"):
            description = "Standard error of " + name.removesuffix("_error").replace("_", " ")
        else:
            description = "Worst measurement of " + name.removeprefix("worst_").replace("_", " ")
        rows.append({"column": name, "role": "input", "type": "numeric", "description": description})
    rows.append({
        "column": TARGET_COLUMN, "role": "target", "type": "integer", "description": "0 = benign; 1 = malignant"
    })
    pd.DataFrame(rows).to_csv(ROOT / "data" / "feature_schema.csv", index=False)


def main() -> None:
    for folder in ["data", "model", "reports"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.rename(columns=lambda name: name.replace(" ", "_"))
    if list(X.columns) != FEATURE_NAMES:
        raise RuntimeError("The dataset feature names do not match the project schema.")

    # The sklearn copy encodes malignant as 0. Make malignant the positive class.
    y = (1 - dataset.target).astype(int).rename(TARGET_COLUMN)
    complete_data = X.assign(**{TARGET_COLUMN: y})
    train_indices, test_indices = train_test_split(
        np.arange(len(X)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    complete_data.to_csv(ROOT / "data" / "breast_cancer.csv", index=False, float_format="%.17g")
    complete_data.iloc[train_indices].to_csv(
        ROOT / "data" / "train_data.csv", index=False, float_format="%.17g"
    )
    complete_data.iloc[test_indices].to_csv(ROOT / "test_data.csv", index=False, float_format="%.17g")
    write_feature_schema()
    write_json(ROOT / "reports" / "split_indices.json", {
        "index_definition": "Zero-based row positions in data/breast_cancer.csv and the sklearn dataset",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    })

    results = []
    details = {}
    model_metadata = {}
    predictions_table = pd.DataFrame({"source_row": test_indices, "actual_target": y_test.to_numpy()})

    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = positive_scores(pipeline, X_test)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results.append({"ML Model Name": name, **metrics})
        details[name] = {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y_test, predictions, labels=[0, 1], target_names=["Benign", "Malignant"],
                output_dict=True, zero_division=0,
            ),
        }
        filename = MODEL_FILES[name]
        joblib.dump(pipeline, ROOT / "model" / filename, compress=3)
        model_metadata[name] = {
            "file": filename,
            "classifier": type(pipeline.named_steps["classifier"]).__name__,
            "parameters": pipeline.named_steps["classifier"].get_params(deep=False),
            "preprocessing": ["training-median imputation"] + (
                [] if name == "Decision Tree" else ["training-fitted StandardScaler"]
            ),
        }
        slug = Path(filename).stem
        predictions_table[f"{slug}_prediction"] = predictions
        predictions_table[f"{slug}_malignant_probability"] = scores

    comparison = pd.DataFrame(results)
    comparison.to_csv(ROOT / "reports" / "metrics.csv", index=False, float_format="%.17g")
    predictions_table.to_csv(ROOT / "reports" / "test_predictions.csv", index=False, float_format="%.17g")
    write_json(ROOT / "reports" / "evaluation_details.json", details)
    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "source_url": SOURCE_URL,
        "dataset_doi": "https://doi.org/10.24432/C5DW2B",
        "dataset_license": "CC BY 4.0",
        "loader": "sklearn.datasets.load_breast_cancer(as_frame=True)",
        "feature_names": FEATURE_NAMES,
        "target_column": TARGET_COLUMN,
        "target_mapping": {"0": "Benign", "1": "Malignant"},
        "original_sklearn_target_mapping": {"0": "Malignant", "1": "Benign"},
        "target_conversion": "project_target = 1 - sklearn_target",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_samples": len(X),
        "n_features": len(FEATURE_NAMES),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_counts": class_counts(y),
        "train_class_counts": class_counts(y_train),
        "test_class_counts": class_counts(y_test),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "joblib_version": joblib.__version__,
        "models": model_metadata,
    }
    write_json(ROOT / "model" / "metadata.json", metadata)
    write_json(ROOT / "reports" / "experiment_summary.json", {
        "dataset": metadata["dataset"],
        "n_samples": len(X),
        "n_features": len(FEATURE_NAMES),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_counts": class_counts(y),
        "train_class_counts": class_counts(y_train),
        "test_class_counts": class_counts(y_test),
        "split": "80/20 stratified holdout; random_state=42",
        "positive_class": "1 = Malignant",
        "precision_recall_f1_average": "binary",
        "auc_input": "predict_proba column for class 1",
        "undefined_precision_recall_f1": "zero_division=0",
        "single_class_auc": "Not defined; the app displays N/A",
        "winner_criterion": ["F1", "MCC", "AUC"],
        "test_set_used_for_parameter_tuning": False,
        "preprocessing_fit_on": "training rows only",
    })

    print(f"Dataset: {len(X)} rows, {len(FEATURE_NAMES)} features")
    print(f"Training: {len(X_train)} rows; testing: {len(X_test)} rows")
    print("Target: 0 = benign; 1 = malignant")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("Saved models, data splits, metrics and predictions.")


if __name__ == "__main__":
    main()
