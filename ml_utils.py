"""Shared data checks and evaluation for the saved classification models."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
TARGET_COLUMN = "target"
MAX_ROWS = 10_000
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

FEATURE_NAMES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error",
    "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def _read_csv_bytes(source: Any) -> bytes:
    """Read a small CSV without consuming an uploaded file's current position."""
    if isinstance(source, (bytes, bytearray)):
        content = bytes(source)
    elif isinstance(source, (str, Path)):
        with Path(source).open("rb") as stream:
            content = stream.read(MAX_UPLOAD_BYTES + 1)
    elif hasattr(source, "read"):
        position = None
        try:
            if hasattr(source, "seek"):
                position = source.tell()
                source.seek(0)
            content = source.read(MAX_UPLOAD_BYTES + 1)
        finally:
            if position is not None:
                source.seek(position)
        if isinstance(content, str):
            content = content.encode("utf-8")
    else:
        raise ValueError("Provide a CSV file, file path, or CSV bytes.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The CSV must be no larger than 2 MiB. Upload test rows only.")
    if not content.strip():
        raise ValueError("The CSV is empty. Upload a header and at least one test row.")
    return content


def read_test_csv(source: Any) -> pd.DataFrame:
    """Validate a labelled test CSV and put its columns in the training order.

    The target is 0 for benign and 1 for malignant. Individual missing feature
    values are allowed: the saved pipeline fills them with training medians.
    A feature that is missing in every uploaded row is rejected.
    """
    content = _read_csv_bytes(source)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the file as a UTF-8, comma-separated CSV.") from exc

    try:
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        duplicate_columns = [name for name, count in Counter(header).items() if count > 1]
        if duplicate_columns:
            raise ValueError("Duplicate columns: " + ", ".join(duplicate_columns))

        expected = FEATURE_NAMES + [TARGET_COLUMN]
        missing = [name for name in expected if name not in header]
        extra = [name for name in header if name not in expected]
        if missing or extra:
            problems = []
            if missing:
                problems.append("Missing columns: " + ", ".join(missing))
            if extra:
                problems.append("Unexpected columns: " + ", ".join(extra))
            raise ValueError(". ".join(problems) + ". Use the supplied test_data.csv as a template.")

        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"CSV row {line_number} has {len(row)} fields; expected {len(header)}."
                )
            row_count += 1
            if row_count > MAX_ROWS:
                raise ValueError(f"Upload at most {MAX_ROWS:,} test rows.")
        if row_count == 0:
            raise ValueError("The CSV contains a header but no test rows.")
        frame = pd.read_csv(io.StringIO(text), float_precision="round_trip")
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("The CSV could not be read. Check its header, commas, and quoting.") from exc

    frame = frame.loc[:, FEATURE_NAMES + [TARGET_COLUMN]].copy()
    for name in FEATURE_NAMES:
        original = frame[name]
        if pd.api.types.is_bool_dtype(original):
            raise ValueError(f"Feature '{name}' must contain numbers, not true/false values.")
        numeric = pd.to_numeric(original, errors="coerce")
        if (numeric.isna() & original.notna()).any():
            raise ValueError(f"Feature '{name}' contains a non-numeric value.")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"Feature '{name}' contains an infinite value.")
        if numeric.isna().all():
            raise ValueError(f"Feature '{name}' is missing in every row.")
        frame[name] = numeric.astype(float)

    if pd.api.types.is_bool_dtype(frame[TARGET_COLUMN]):
        raise ValueError("The target must use 0 for benign and 1 for malignant.")
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("The target must contain only 0 (benign) and 1 (malignant), with no blanks.")
    frame[TARGET_COLUMN] = target.astype(int)
    return frame


def positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return the probability for malignant=1, regardless of class-column order."""
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("The model must contain the positive class 1 (malignant).")
    return np.asarray(model.predict_proba(X))[:, int(positive_columns[0])]


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calculate the six assignment metrics without fitting the model."""
    predictions = model.predict(X)
    scores = positive_scores(model, X)
    auc = roc_auc_score(y, scores) if np.unique(y).size == 2 else float("nan")
    return {
        "Accuracy": float(accuracy_score(y, predictions)),
        "AUC": float(auc),
        "Precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, predictions)),
    }


def load_models() -> dict[str, Pipeline]:
    """Load the five trusted pipelines committed with this project."""
    metadata_path = ROOT / "model" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("model/metadata.json is missing. Run python train_models.py locally first.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError("The model metadata cannot be read. Restore the model folder or retrain locally.") from exc

    trained_version = metadata.get("sklearn_version")
    if trained_version != sklearn.__version__:
        raise RuntimeError(
            f"These models require scikit-learn {trained_version}; this environment has "
            f"{sklearn.__version__}. Install requirements.txt or retrain locally with your chosen version."
        )
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("The saved model schema does not match this app's feature names.")
    if metadata.get("target_column") != TARGET_COLUMN or metadata.get("target_mapping") != {
        "0": "Benign", "1": "Malignant"
    }:
        raise RuntimeError("The saved model target mapping does not match this app.")

    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = ROOT / "model" / filename
        if not path.is_file():
            raise FileNotFoundError(f"model/{filename} is missing. Include all five saved models.")
        try:
            model = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Could not load {filename}. Restore it or rerun training locally.") from exc
        if not isinstance(model, Pipeline):
            raise RuntimeError(f"{filename} must contain the complete preprocessing and model pipeline.")
        if list(getattr(model, "feature_names_in_", [])) != FEATURE_NAMES:
            raise RuntimeError(f"The feature order in {filename} does not match the app.")
        if list(getattr(model, "classes_", [])) != [0, 1]:
            raise RuntimeError(f"The class labels in {filename} do not match the app.")
        loaded[name] = model
    return loaded
