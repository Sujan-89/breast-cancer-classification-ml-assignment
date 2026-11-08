"""Explore saved classification models using labelled test data."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from ml_utils import (
    FEATURE_NAMES,
    MAX_ROWS,
    MAX_UPLOAD_BYTES,
    MODEL_FILES,
    ROOT,
    TARGET_COLUMN,
    evaluate_model,
    load_models,
    positive_scores,
    read_test_csv,
)


METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
CLASS_NAMES = {0: "Benign", 1: "Malignant"}


def artifact_signature() -> tuple:
    """Refresh the model cache when the deployed artifacts change."""
    paths = [ROOT / "model" / name for name in MODEL_FILES.values()]
    paths.append(ROOT / "model" / "metadata.json")
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


@st.cache_resource(show_spinner=False)
def cached_models(signature: tuple) -> dict:
    """Keep the saved pipelines in memory for repeated evaluations."""
    return load_models()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def metric_format(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def input_controls() -> tuple[str, bytes | None, str]:
    """Collect a model choice and either the bundled or uploaded test CSV."""
    with st.sidebar:
        st.header("Evaluation setup")
        selected_model = st.selectbox("Model", list(MODEL_FILES), key="selected_model")
        source = st.radio(
            "Test data",
            ["Bundled test data", "Upload a CSV"],
            key="data_source",
        )

        bundled_file = ROOT / "test_data.csv"
        if bundled_file.is_file():
            st.download_button(
                "Download test_data.csv",
                data=bundled_file.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
                key="download_test_data",
                width="stretch",
            )

        uploaded = None
        if source == "Upload a CSV":
            uploaded = st.file_uploader(
                "Upload labelled test data",
                type=["csv"],
                key="test_csv",
                help="Include all 30 input features and the target column. Use held-out test rows.",
            )

        st.caption(
            f"CSV limit: {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB and "
            f"{MAX_ROWS:,} rows. Models are already trained."
        )
        st.markdown("**Target values**  \n`0` = Benign  \n`1` = Malignant")

        with st.expander("CSV requirements"):
            st.write(
                "Use the exact column names below. Column order can change. "
                "All features must be numeric; target must contain 0 or 1. "
                "Additional columns, including an exported index, are not accepted."
            )
            st.write(
                "Individual missing feature values are filled using the medians "
                "learned during training. A completely empty feature column is rejected."
            )
            st.code("\n".join(FEATURE_NAMES + [TARGET_COLUMN]), language="text")

    if source == "Upload a CSV":
        if uploaded is None:
            return selected_model, None, "Uploaded CSV"
        return selected_model, uploaded.getvalue(), f"Uploaded CSV: {uploaded.name}"

    return selected_model, bundled_file.read_bytes(), "Bundled holdout: test_data.csv"


def comparison_table(models: dict, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rows.append({"ML Model Name": name, **evaluate_model(model, features, target)})
    return pd.DataFrame(rows, columns=["ML Model Name", *METRICS])


def show_comparison(comparison: pd.DataFrame, has_both_classes: bool) -> None:
    st.subheader("Compare all five models")
    st.write("Every row below is calculated from the same current test CSV.")
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={name: st.column_config.NumberColumn(name, format="%.4f") for name in METRICS},
    )
    st.download_button(
        "Download comparison CSV",
        data=csv_bytes(comparison),
        file_name="model_comparison.csv",
        mime="text/csv",
        key="download_comparison",
    )

    if has_both_classes:
        ranked = comparison.sort_values(["F1", "MCC", "AUC"], ascending=False, kind="stable")
        best = ranked.iloc[0]
        tied = comparison[
            np.isclose(comparison["F1"], best["F1"], rtol=0, atol=1e-12)
            & np.isclose(comparison["MCC"], best["MCC"], rtol=0, atol=1e-12)
            & np.isclose(comparison["AUC"], best["AUC"], rtol=0, atol=1e-12)
        ]
        leader = ", ".join(tied["ML Model Name"])
        st.caption(
            f"Leading model{'s' if len(tied) > 1 else ''} on this data: {leader}. "
            "Ranking uses F1, then MCC, then AUC. This comparison applies to the current test data."
        )


def show_confusion_matrix(target: pd.Series, predicted: np.ndarray) -> None:
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    axis.imshow(matrix, cmap="Purples", vmin=0)
    axis.set_xticks([0, 1], ["Benign (0)", "Malignant (1)"])
    axis.set_yticks([0, 1], ["Benign (0)", "Malignant (1)"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    for row in range(2):
        for column in range(2):
            colour = "white" if matrix[row, column] > matrix.max() / 2 else "#2b2340"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=colour, fontsize=18)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    tn, fp, fn, tp = matrix.ravel()
    st.caption(f"True negatives: {tn} | False positives: {fp} | False negatives: {fn} | True positives: {tp}")


def show_classification_report(target: pd.Series, predicted: np.ndarray) -> None:
    report = classification_report(
        target,
        predicted,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    report_rows = {name: values for name, values in report.items() if isinstance(values, dict)}
    frame = pd.DataFrame.from_dict(report_rows, orient="index").rename_axis("Class").reset_index()
    frame = frame.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support"})
    frame["Support"] = frame["Support"].astype(int)
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            **{name: st.column_config.NumberColumn(name, format="%.4f") for name in ["Precision", "Recall", "F1"]},
            "Support": st.column_config.NumberColumn("Support", format="%d"),
        },
    )
    st.caption("Support is the number of actual samples in each class. Undefined precision, recall or F1 is reported as 0.")


def show_roc_curves(models: dict, features: pd.DataFrame, target: pd.Series) -> None:
    if target.nunique() < 2:
        st.info("ROC curves and AUC require both target classes. Include benign and malignant test rows to calculate them.")
        return

    figure, axis = plt.subplots(figsize=(6.2, 4.5))
    colours = ["#6d28d9", "#0f766e", "#d97706", "#2563eb", "#be185d"]
    for (name, model), colour in zip(models.items(), colours):
        probabilities = positive_scores(model, features)
        fpr, tpr, _ = roc_curve(target, probabilities, pos_label=1)
        axis.plot(fpr, tpr, label=name, color=colour, linewidth=1.8)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.15)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption("All five ROC curves use malignant (1) as the positive class and probability scores from the saved models.")


def show_predictions(dataset: pd.DataFrame, model: object, selected_model: str) -> None:
    features = dataset[FEATURE_NAMES]
    actual = dataset[TARGET_COLUMN].to_numpy()
    predicted = model.predict(features).astype(int)
    probabilities = positive_scores(model, features)
    results = pd.DataFrame(
        {
            "Row": np.arange(1, len(dataset) + 1),
            "Actual": dataset[TARGET_COLUMN].map(CLASS_NAMES),
            "Predicted": pd.Series(predicted, index=dataset.index).map(CLASS_NAMES),
            "Malignant probability": probabilities,
            "Correct": actual == predicted,
        }
    )
    error_filter = st.radio(
        "Inspect prediction errors",
        ["All", "Incorrect only", "False negatives", "False positives"],
        horizontal=True,
        key="prediction_filter",
        help=(
            "False negatives are malignant samples predicted as benign. "
            "False positives are benign samples predicted as malignant."
        ),
    )
    masks = {
        "All": np.ones(len(results), dtype=bool),
        "Incorrect only": actual != predicted,
        "False negatives": (actual == 1) & (predicted == 0),
        "False positives": (actual == 0) & (predicted == 1),
    }
    filtered = results.loc[masks[error_filter]]
    st.caption(
        f"Matching rows: {len(filtered):,} of {len(results):,}. "
        "Row numbers refer to the original CSV data rows, excluding the header."
    )
    st.dataframe(
        filtered.head(100),
        hide_index=True,
        width="stretch",
        column_config={"Malignant probability": st.column_config.NumberColumn(format="%.4f")},
    )
    if len(filtered) > 100:
        st.caption(f"Showing the first 100 of {len(filtered):,} matching rows.")
    if filtered.empty:
        st.info("There are no predictions in this category for the selected model and current test data.")

    export = dataset.copy()
    export.insert(0, "csv_row", np.arange(1, len(dataset) + 1))
    export["predicted_target"] = predicted
    export["predicted_label"] = pd.Series(predicted, index=dataset.index).map(CLASS_NAMES)
    export["malignant_probability"] = probabilities
    export["correct"] = dataset[TARGET_COLUMN].to_numpy() == predicted
    name = MODEL_FILES[selected_model].replace(".joblib", "_predictions.csv")
    st.download_button(
        "Download all predictions for selected model",
        data=csv_bytes(export),
        file_name=name,
        mime="text/csv",
        key="download_predictions",
    )
    st.caption(
        "The download contains every input row, regardless of the filter, plus original row numbers and results. "
        "Use test_data.csv as the input template; prediction exports include extra columns."
    )


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Model Comparison", page_icon="📊", layout="wide")
    st.title("Breast Cancer Model Comparison")
    st.markdown("**Breast Cancer Wisconsin (Diagnostic)**  \nEvaluate five saved classifiers on labelled test data.")

    try:
        selected_model, data, source_label = input_controls()
    except OSError:
        st.error("The bundled test_data.csv file is missing or cannot be read. Add it to the app folder and restart the app.")
        st.stop()

    if data is None:
        st.info("Upload a test CSV in the sidebar to view model results. The downloadable test_data.csv is ready to use.")
        st.stop()

    try:
        dataset = read_test_csv(io.BytesIO(data))
    except (ValueError, OSError) as error:
        st.error(f"The test CSV could not be evaluated: {error}")
        st.info("Check the CSV requirements in the sidebar or start with the supplied test_data.csv.")
        st.stop()

    try:
        models = cached_models(artifact_signature())
    except (OSError, RuntimeError, ValueError) as error:
        st.error("The saved models could not be loaded.")
        with st.expander("Details for the app owner", expanded=True):
            st.write(str(error))
            st.write("Check that model/ contains all five trained pipelines and metadata.json, and that requirements.txt matches the training environment.")
        st.stop()

    features = dataset[FEATURE_NAMES]
    target = dataset[TARGET_COLUMN].astype(int)
    has_both_classes = target.nunique() == 2
    st.caption(f"Data source: {source_label}")
    summary_columns = st.columns(4)
    for column, label, value in zip(
        summary_columns,
        ["Test rows", "Input features", "Benign rows", "Malignant rows"],
        [len(dataset), len(FEATURE_NAMES), int((target == 0).sum()), int((target == 1).sum())],
    ):
        column.metric(label, value)

    if source_label.startswith("Uploaded"):
        st.caption("Use held-out test rows. The app evaluates the data supplied and does not retrain the models.")

    missing_values = int(features.isna().sum().sum())
    if missing_values:
        st.info(f"{missing_values:,} missing feature values will be filled using the training medians saved with each model.")
    if not has_both_classes:
        st.info("This CSV contains one target class. AUC is N/A; the other metrics are shown with undefined precision, recall or F1 set to 0.")

    try:
        comparison = comparison_table(models, features, target)
        selected = models[selected_model]
        predicted = selected.predict(features)
    except (ValueError, RuntimeError) as error:
        st.error(f"The saved models could not evaluate this CSV: {error}")
        st.stop()

    show_comparison(comparison, has_both_classes)
    st.divider()
    st.subheader(selected_model)
    selected_metrics = comparison.set_index("ML Model Name").loc[selected_model]
    for column, metric in zip(st.columns(6), METRICS):
        column.metric(metric, metric_format(selected_metrics[metric]))
    st.caption("Precision, recall and F1 refer to malignant (1). AUC uses class-1 probability scores; predicted labels come from each classifier's predict method.")

    matrix_tab, report_tab, roc_tab, predictions_tab = st.tabs(
        ["Confusion matrix", "Classification report", "ROC curves", "Predictions"]
    )
    with matrix_tab:
        plot_column, explanation_column = st.columns([1, 1])
        with plot_column:
            show_confusion_matrix(target, predicted)
        with explanation_column:
            st.markdown(
                "**Read the matrix**\n\n"
                "Rows show the actual class; columns show the predicted class. "
                "Correct predictions lie on the diagonal.\n\n"
                "A false negative is a malignant sample predicted as benign. "
                "A false positive is a benign sample predicted as malignant."
            )
    with report_tab:
        show_classification_report(target, predicted)
    with roc_tab:
        plot_column, _ = st.columns([2, 1])
        with plot_column:
            show_roc_curves(models, features, target)
    with predictions_tab:
        show_predictions(dataset, selected, selected_model)

    with st.expander("Preview the current test data"):
        st.dataframe(dataset.head(10), hide_index=True, width="stretch")
        st.caption("First 10 rows, before saved preprocessing is applied.")

    with st.expander("About this experiment"):
        st.write(
            "The source dataset contains 569 records and 30 numeric features. "
            "A stratified 80/20 split with random_state=42 creates 455 training rows "
            "and 114 held-out test rows. All five models use the same split. "
            "Preprocessing is learned only from the training rows and saved with each model."
        )
        st.write(
            "The target was recoded so 0 means benign and 1 means malignant. "
            "The app loads the saved pipelines and calculates results; it does not train on uploaded files. "
            "A single holdout comparison is evidence about this experiment, not proof that one model will always perform best."
        )
        st.markdown(
            "Dataset: [UCI Breast Cancer Wisconsin (Diagnostic)]"
            "(https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic). "
            "Available under CC BY 4.0. Educational classification project."
        )


if __name__ == "__main__":
    main()
