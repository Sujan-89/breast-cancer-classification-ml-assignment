"""Exercise the frontend with bundled and uploaded data."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.metrics import confusion_matrix

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from ml_utils import FEATURE_NAMES, MODEL_FILES, ROOT, TARGET_COLUMN, evaluate_model, load_models


class UploadedCSV(io.BytesIO):
    def __init__(self, content: bytes, name: str = "uploaded_test.csv"):
        super().__init__(content)
        self.name = name
        self.size = len(content)


@unittest.skipIf(AppTest is None, "Install requirements.txt to run the frontend tests.")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_frame = pd.read_csv(ROOT / "test_data.csv")
        cls.saved_metrics = pd.read_csv(ROOT / "reports" / "metrics.csv")

    def open_app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        return app

    def find_comparison(self, app):
        for table in app.dataframe:
            if "ML Model Name" in table.value.columns:
                return table.value
        self.fail("The all-model comparison was not displayed.")

    def test_bundled_data_matches_saved_experiment_without_training(self):
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("The app must not train models.")):
            app = self.open_app()
        self.assertEqual(len(app.error), 0)
        actual = self.find_comparison(app)
        self.assertEqual(actual["ML Model Name"].tolist(), list(MODEL_FILES))
        pd.testing.assert_frame_equal(actual, self.saved_metrics, check_exact=False, rtol=1e-12, atol=1e-12)
        self.assertTrue(any("Confusion matrix" == tab.label for tab in app.tabs))
        self.assertTrue(any("Classification report" == tab.label for tab in app.tabs))

    def test_each_model_selection_updates_its_metrics(self):
        app = self.open_app()
        for name in MODEL_FILES:
            with self.subTest(model=name):
                app.selectbox(key="selected_model").set_value(name).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                actual = {item.label: item.value for item in app.metric}
                expected = self.saved_metrics.set_index("ML Model Name").loc[name]
                for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
                    self.assertEqual(actual[metric], f"{expected[metric]:.4f}")

    def test_error_filters_match_confusion_counts_and_keep_original_rows(self):
        app = self.open_app()
        target = self.test_frame[TARGET_COLUMN].to_numpy()
        predicted = load_models()["Logistic Regression"].predict(self.test_frame[FEATURE_NAMES])
        _, false_positives, false_negatives, _ = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
        cases = [
            ("Incorrect only", false_positives + false_negatives, target != predicted),
            ("False negatives", false_negatives, (target == 1) & (predicted == 0)),
            ("False positives", false_positives, (target == 0) & (predicted == 1)),
        ]
        for label, expected_count, mask in cases:
            with self.subTest(filter=label):
                app.radio(key="prediction_filter").set_value(label).run()
                self.assertEqual(len(app.exception), 0)
                filtered = next(table.value for table in app.dataframe if "Malignant probability" in table.value.columns)
                self.assertEqual(len(filtered), expected_count)
                original_rows = pd.Series(range(1, len(target) + 1))[mask].tolist()
                self.assertEqual(filtered["Row"].tolist(), original_rows)
                count_text = f"Matching rows: {expected_count} of {len(target)}."
                self.assertTrue(any(count_text in item.value for item in app.caption))

    def test_upload_mode_waits_for_a_file(self):
        app = self.open_app()
        app.radio(key="data_source").set_value("Upload a CSV").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(any("Upload a test CSV" in message.value for message in app.info))
        self.assertEqual(len(app.dataframe), 0)

    def run_uploaded(self, frame):
        app = self.open_app()
        app.radio(key="data_source").set_value("Upload a CSV")
        uploaded = UploadedCSV(frame.to_csv(index=False).encode("utf-8"))
        with patch("streamlit.file_uploader", return_value=uploaded):
            app.run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        return app

    def test_uploaded_subset_recomputes_all_five_models(self):
        subset = self.test_frame.iloc[:25].copy()
        app = self.run_uploaded(subset)
        self.assertEqual(len(app.error), 0)
        actual = self.find_comparison(app).set_index("ML Model Name")
        for name, model in load_models().items():
            expected = evaluate_model(model, subset[FEATURE_NAMES], subset[TARGET_COLUMN])
            for metric, value in expected.items():
                self.assertAlmostEqual(actual.loc[name, metric], value, places=12)
        self.assertTrue(any("Uploaded CSV" in item.value for item in app.caption))

    def test_missing_feature_upload_has_a_readable_error(self):
        app = self.run_uploaded(self.test_frame.drop(columns=[FEATURE_NAMES[0]]))
        self.assertGreater(len(app.error), 0)
        self.assertTrue(any(FEATURE_NAMES[0] in item.value for item in app.error))
        self.assertEqual(len(app.dataframe), 0)

    def test_non_numeric_feature_upload_has_a_readable_error(self):
        invalid = self.test_frame.copy()
        invalid[FEATURE_NAMES[0]] = invalid[FEATURE_NAMES[0]].astype(object)
        invalid.loc[invalid.index[0], FEATURE_NAMES[0]] = "not a number"
        app = self.run_uploaded(invalid)
        self.assertGreater(len(app.error), 0)
        self.assertEqual(len(app.dataframe), 0)

    def test_single_class_upload_shows_auc_as_not_available(self):
        only_benign = self.test_frame[self.test_frame[TARGET_COLUMN] == 0].iloc[:15]
        app = self.run_uploaded(only_benign)
        self.assertEqual(len(app.error), 0)
        comparison = self.find_comparison(app)
        self.assertTrue(comparison["AUC"].isna().all())
        auc_card = next(item for item in app.metric if item.label == "AUC")
        self.assertEqual(auc_card.value, "N/A")
        self.assertTrue(any("one target class" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
