"""Checks for the data contract, saved pipelines and reported experiment."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

from ml_utils import (
    FEATURE_NAMES, MAX_ROWS, MAX_UPLOAD_BYTES, MODEL_FILES, ROOT, TARGET_COLUMN,
    evaluate_model, load_models, positive_scores, read_test_csv,
)


class CsvValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = read_test_csv(ROOT / "test_data.csv")

    def csv_bytes(self, frame):
        return frame.to_csv(index=False).encode("utf-8")

    def test_reorders_columns_and_keeps_target_mapping(self):
        frame = self.test_data.iloc[:6, ::-1]
        result = read_test_csv(self.csv_bytes(frame))
        self.assertEqual(list(result.columns), FEATURE_NAMES + [TARGET_COLUMN])
        self.assertTrue(result[TARGET_COLUMN].isin([0, 1]).all())

    def test_accepts_utf8_bom_and_file_like_input(self):
        content = b"\xef\xbb\xbf" + self.csv_bytes(self.test_data.iloc[:3])
        source = io.BytesIO(content)
        source.seek(4)
        result = read_test_csv(source)
        self.assertEqual(len(result), 3)
        self.assertEqual(source.tell(), 4)

    def test_rejects_missing_and_extra_columns(self):
        for frame in [
            self.test_data.drop(columns=[FEATURE_NAMES[0]]),
            self.test_data.assign(unexpected_id=1),
            self.test_data.drop(columns=[TARGET_COLUMN]),
        ]:
            with self.subTest(columns=frame.columns.tolist()), self.assertRaises(ValueError):
                read_test_csv(self.csv_bytes(frame))

    def test_rejects_duplicate_headers(self):
        content = self.csv_bytes(self.test_data.iloc[:2]).decode("utf-8")
        content = content.replace("mean_texture", "mean_radius", 1)
        with self.assertRaisesRegex(ValueError, "Duplicate columns"):
            read_test_csv(content.encode("utf-8"))

    def test_rejects_bad_feature_values(self):
        for value in ["not-a-number", float("inf"), -float("inf")]:
            frame = self.test_data.iloc[:3].copy()
            frame[FEATURE_NAMES[0]] = frame[FEATURE_NAMES[0]].astype(object)
            frame.loc[frame.index[0], FEATURE_NAMES[0]] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_test_csv(self.csv_bytes(frame))

    def test_allows_partial_missing_but_rejects_empty_feature(self):
        frame = self.test_data.iloc[:3].copy()
        frame.loc[frame.index[0], FEATURE_NAMES[0]] = np.nan
        result = read_test_csv(self.csv_bytes(frame))
        self.assertTrue(pd.isna(result.loc[0, FEATURE_NAMES[0]]))
        frame[FEATURE_NAMES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "missing in every row"):
            read_test_csv(self.csv_bytes(frame))

    def test_rejects_invalid_or_missing_target(self):
        for value in [2, -1, 0.5, "malignant", np.nan]:
            frame = self.test_data.iloc[:3].copy()
            frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(object)
            frame.loc[frame.index[0], TARGET_COLUMN] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_test_csv(self.csv_bytes(frame))

    def test_rejects_empty_malformed_and_oversized_files(self):
        header = ",".join(FEATURE_NAMES + [TARGET_COLUMN])
        for content in [b"", (header + "\n").encode(), (header + "\n1,2\n").encode(), b"x" * (MAX_UPLOAD_BYTES + 1)]:
            with self.subTest(length=len(content)), self.assertRaises(ValueError):
                read_test_csv(content)

    def test_rejects_more_than_row_limit(self):
        header = ",".join(FEATURE_NAMES + [TARGET_COLUMN])
        row = ",".join(["1"] * len(FEATURE_NAMES) + ["0"])
        content = (header + "\n" + (row + "\n") * (MAX_ROWS + 1)).encode()
        with self.assertRaisesRegex(ValueError, "at most"):
            read_test_csv(content)


class SavedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = load_models()
        cls.frame = read_test_csv(ROOT / "test_data.csv")
        cls.X = cls.frame[FEATURE_NAMES]
        cls.y = cls.frame[TARGET_COLUMN]
        cls.metadata = json.loads((ROOT / "model" / "metadata.json").read_text())

    def test_target_is_malignant_positive(self):
        full = pd.read_csv(ROOT / "data" / "breast_cancer.csv", float_precision="round_trip")
        original = load_breast_cancer()
        np.testing.assert_array_equal(full[TARGET_COLUMN], 1 - original.target)
        self.assertEqual(int(full[TARGET_COLUMN].sum()), 212)

    def test_training_and_test_rows_are_separate_and_complete(self):
        split = json.loads((ROOT / "reports" / "split_indices.json").read_text())
        train_indices = split["train_indices"]
        test_indices = split["test_indices"]
        self.assertEqual(len(train_indices), 455)
        self.assertEqual(len(test_indices), 114)
        self.assertFalse(set(train_indices) & set(test_indices))
        self.assertEqual(set(train_indices) | set(test_indices), set(range(569)))
        full = pd.read_csv(ROOT / "data" / "breast_cancer.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(
            self.frame.reset_index(drop=True), full.iloc[test_indices].reset_index(drop=True), check_dtype=False
        )

    def test_preprocessing_uses_only_training_rows(self):
        training = pd.read_csv(ROOT / "data" / "train_data.csv", float_precision="round_trip")
        expected_medians = training[FEATURE_NAMES].median().to_numpy()
        for name, pipeline in self.models.items():
            with self.subTest(model=name):
                np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, expected_medians)
                scaler = pipeline.named_steps["scaler"]
                if scaler != "passthrough":
                    np.testing.assert_allclose(scaler.mean_, training[FEATURE_NAMES].mean().to_numpy())
                    self.assertEqual(scaler.n_samples_seen_, 455)

    def test_loaded_models_reproduce_reported_metrics_without_fitting(self):
        recorded = pd.read_csv(ROOT / "reports" / "metrics.csv").set_index("ML Model Name")
        self.assertEqual(list(self.models), list(MODEL_FILES))
        for name, model in self.models.items():
            with self.subTest(model=name), patch.object(model, "fit", side_effect=AssertionError("Evaluation must not fit")):
                actual = evaluate_model(model, self.X, self.y)
                for metric, value in actual.items():
                    self.assertAlmostEqual(value, recorded.loc[name, metric], places=12)
                self.assertAlmostEqual(actual["AUC"], roc_auc_score(self.y, positive_scores(model, self.X)), places=12)

    def test_missing_feature_is_imputed_by_saved_pipeline(self):
        X = self.X.iloc[:5].copy()
        X.iloc[0, 0] = np.nan
        for name, model in self.models.items():
            with self.subTest(model=name):
                self.assertTrue(np.isfinite(positive_scores(model, X)).all())

    def test_single_class_auc_is_undefined(self):
        frame = self.frame.loc[self.frame[TARGET_COLUMN] == 1]
        result = evaluate_model(self.models["Logistic Regression"], frame[FEATURE_NAMES], frame[TARGET_COLUMN])
        self.assertTrue(np.isnan(result["AUC"]))
        self.assertTrue(all(np.isfinite(value) for key, value in result.items() if key != "AUC"))

    def test_positive_probability_follows_class_label(self):
        class ReversedClassModel:
            classes_ = np.array([1, 0])

            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.3, 0.7]])

        np.testing.assert_allclose(positive_scores(ReversedClassModel(), self.X.iloc[:2]), [0.9, 0.3])

    def test_version_mismatch_has_clear_error(self):
        with patch("ml_utils.sklearn.__version__", "0.0.0"), self.assertRaisesRegex(RuntimeError, "require scikit-learn"):
            load_models()


if __name__ == "__main__":
    unittest.main()
