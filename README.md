# Breast Cancer Classification: Five-Model Comparison

Author: Sujan Mukherjee

## a. Problem statement

The aim of this project is to classify a breast mass as benign or malignant using numerical measurements of cell nuclei. Five classification algorithms are trained on the same dataset and compared using Accuracy, ROC AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC).

The Streamlit application lets a user upload labelled test data, compare all five saved models, and inspect individual predictions. Preprocessing and model training happen in a separate Python script. The application loads the fitted pipelines and evaluates the supplied records without retraining.

## b. Dataset description

| Item | Description |
| --- | --- |
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Public repository | UCI Machine Learning Repository |
| Dataset link | [UCI dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic) |
| Number of records | 569 |
| Number of input features | 30 numerical features |
| Classification task | Binary classification: benign or malignant |
| Class distribution | 357 benign and 212 malignant |
| Target column | `target`: 0 = Benign, 1 = Malignant |
| Missing values in the original dataset | None |
| Training set | 455 records: 285 benign and 170 malignant |
| Test set | 114 records: 72 benign and 42 malignant |
| Split method | Stratified 80/20 split with `random_state=42` |
| Dataset licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |

The features describe ten characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each has a mean, a standard-error measurement and a worst-value measurement, giving 30 input features.

The dataset is loaded through `sklearn.datasets.load_breast_cancer`, which provides a bundled copy of the UCI dataset. The code changes spaces in the feature names to underscores and reverses scikit-learn's original target encoding using `target = 1 - original_target`. Malignant is therefore the positive class in this project. The target and any identifiers are excluded from the input features.

The full dataset meets the assignment minimum of 500 instances and 12 features. The smaller `test_data.csv` contains only the held-out records used for evaluation.

## c. GitHub Repository Link

GitHub repository: **Add your public repository URL here after creating it.**

Live Streamlit app: **Add your deployed application URL here after deployment.**

Edit these two lines directly in this README. Copy the actual URLs from your browser and check both links in a private browser window before submission.

## d. Models used

### Training and evaluation method

All five models use exactly the same training and test rows. The split is created before preprocessing. Each saved file contains a scikit-learn `Pipeline` with a median imputer, any required scaling, and the classifier. The imputer and scaler are fitted only on the training set.

Logistic Regression, KNN, Gaussian Naive Bayes and SVM use `StandardScaler`. The Decision Tree uses the original feature scale after imputation. There are no missing values in the source dataset; the imputer supports occasional blank feature cells in uploaded test data.

The following parameters were fixed for the experiment. No feature selection, hyperparameter search or decision-threshold tuning was performed on the test set. SVM uses training-only internal probability calibration for its probability output.

| ML Model Name | Main configuration |
| --- | --- |
| Logistic Regression | `C=1.0`, `max_iter=5000`, `random_state=42` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=2`, `random_state=42` |
| KNN | `n_neighbors=5`, `weights="uniform"` |
| Naive Bayes | `GaussianNB`, default `var_smoothing=1e-9` |
| Support Vector Machine | RBF kernel, `C=1.0`, `gamma="scale"`, `probability=True`, `random_state=42` |

### Comparison table

The table below records the measured results on the supplied 114-row test set. Values are rounded to four decimal places. Full-precision values are available in `reports/metrics.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9123 | 0.9444 | 0.9444 | 0.8095 | 0.8718 | 0.8113 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

Precision, recall and F1 use binary averaging with malignant (`1`) as the positive class. Accuracy and MCC use predicted class labels. ROC AUC uses the predicted probability for class `1`. The class labels come from each estimator's `predict` method; the code does not replace them with a new probability threshold.

### Observations on model performance

These observations were written after reviewing the measured metrics and confusion matrices. They are maintained directly in this README; running the training script does not generate or overwrite them.

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Logistic Regression achieved 96.49% accuracy and the highest AUC, 0.9960. It correctly identified 39 of 42 malignant records and misclassified one benign record as malignant. Its F1 of 0.9512 was second to SVM. The result shows that a scaled linear classifier is a strong baseline for this split. A high AUC describes good ranking across thresholds; it does not require the best accuracy at the model's chosen decision rule. |
| Decision Tree | The tree had the lowest accuracy, 91.23%, and F1, 0.8718. It missed eight malignant records and labelled two benign records as malignant. Its recall of 0.8095 was the lowest of the five models. The chosen depth and leaf-size limits make the tree relatively simple. Further training-only validation would be needed to establish whether different tree settings improve its generalization. |
| KNN | KNN reached 95.61% accuracy with F1 0.9383. It missed four malignant records and produced one false positive. This was a competitive result, though slightly behind Logistic Regression and SVM. Scaling makes the feature distances comparable. With five uniformly weighted neighbours, its probabilities take only a few values, which can create ties when calculating AUC. |
| Naive Bayes | Gaussian Naive Bayes achieved 92.11% accuracy, 0.8889 F1 and 0.9891 AUC. It missed six malignant records and produced three false positives. Its AUC was strong even though its label-based scores were lower. Correlated measurements such as radius, perimeter and area may challenge the conditional-independence assumption, but this experiment alone does not establish that as the cause of the errors. |
| Support Vector Machine | SVM had the highest accuracy, 97.37%, F1, 0.9630, and MCC, 0.9442. It detected 39 of 42 malignant records and produced no false positives, giving precision 1.0000. Its recall matched Logistic Regression at 0.9286. Its AUC of 0.9947 was slightly below Logistic Regression's. The three false negatives show that perfect precision did not mean every malignant record was detected. |
| Overall Winner for this dataset | SVM is the overall winner for this fixed test split when prioritizing malignant-class F1, with MCC and AUC as tie-breakers. It also leads on accuracy and precision. Logistic Regression is a close alternative and has the best AUC. These results describe one 114-record holdout; a different split or dataset could change the ranking. |

### Confusion-matrix counts

| Model | True negatives | False positives | False negatives | True positives |
| --- | --- | --- | --- | --- |
| Logistic Regression | 71 | 1 | 3 | 39 |
| Decision Tree | 70 | 2 | 8 | 34 |
| KNN | 71 | 1 | 4 | 38 |
| Naive Bayes | 69 | 3 | 6 | 36 |
| Support Vector Machine | 72 | 0 | 3 | 39 |

A false negative is an actual malignant record predicted as benign. A false positive is an actual benign record predicted as malignant.

## Application customization

The frontend is designed around this experiment, with a dataset-specific title, a teal theme and a dedicated evaluation sidebar. Its features include:

- A choice between the bundled held-out dataset and an uploaded CSV, with a download of the original test file.
- Dataset summary cards showing test rows, feature count and both class counts.
- A live five-model comparison on the current CSV, with a downloadable comparison table.
- Six metric cards for the selected model and four diagnostic tabs: confusion matrix, classification report, ROC curves and predictions.
- Class-labelled confusion matrices with an explanation of false positives and false negatives.
- An overlay of all five ROC curves using the same test records.
- A prediction explorer for reviewing incorrect predictions, false negatives and false positives, plus a download of all predictions.
- Validation messages for incorrect CSV headers, invalid targets, non-numeric values and oversized uploads.
- An experiment explanation covering the split, label mapping and saved preprocessing.

## Repository files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Customized Streamlit frontend |
| `ml_utils.py` | CSV checks, saved-model loading and evaluation functions |
| `train_models.py` | Offline training and numerical experiment outputs |
| `requirements.txt` | Pinned application and machine-learning dependencies |
| `README.md` | Project description, manually maintained results and observations |
| `test_data.csv` | The 114 held-out test records with all 30 features and target |
| `model/` | Five fitted `.joblib` pipelines and `metadata.json` |
| `data/` | Full dataset, training data and feature schema |
| `reports/` | Numerical metrics, predictions, classification reports and split records |
| `tests/` | Input-validation, saved-model and frontend checks |
| `.streamlit/config.toml` | Theme, upload limit and application settings |
| `DATA_SOURCE.md` | Dataset attribution and transformation notes |

## Run locally in VS Code on Windows

Use Python 3.12. Python 3.12.10 is suitable. Open the folder containing `app.py` in VS Code and select **Terminal > New Terminal**. The following commands are for PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If a Python 3.12 virtual environment already exists, reuse it. Calling its executable directly avoids a separate activation step.

Reproduce the experiment and verify the project:

```powershell
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The five trained models are included, so training is optional when only demonstrating the supplied app. Stop the local server with `Ctrl+C`.

After changing the training settings, rerun training and review `reports/metrics.csv` and `reports/evaluation_details.json`. Update the comparison table and observations in this README manually. No script edits this file.

## CSV requirements

- Use the supplied `test_data.csv` first. Its schema is also listed in `data/feature_schema.csv`.
- Include all 30 input features and the `target` column. Column order may change; names must match exactly.
- Keep the measurements in their original units. The saved pipelines apply preprocessing.
- Use `0` for benign and `1` for malignant. The target cannot contain blanks or other labels.
- Use a comma-separated UTF-8 CSV. Do not add an ID, saved row index or prediction columns.
- Individual missing feature cells are imputed with the saved training-set medians. Entirely missing feature columns, infinite values and non-numeric text are rejected.
- Upload test data only, with a maximum of 2 MiB and 10,000 rows.
- Both target classes are needed for AUC. For a single-class upload, AUC is displayed as N/A. Undefined precision, recall or F1 uses `zero_division=0`.

The app checks the CSV structure but cannot prove that an arbitrary upload contains previously unseen records. The supplied test set is excluded from training, and its source row indices are recorded in `reports/split_indices.json`.

## Manual deployment

1. Create a public GitHub repository and upload the individual project files and folders. Keep `app.py`, `requirements.txt`, `README.md` and `test_data.csv` at the repository root.
2. Confirm that `model/` contains all five `.joblib` files and `metadata.json`. Include `.streamlit/config.toml`.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
4. Choose **Create app**, select the repository and branch, and enter `app.py` as the main file path.
5. Select Python **3.12** in **Advanced settings**, leave secrets empty and click **Deploy**.
6. Check the deployed app with `test_data.csv`, then copy its actual `https://...streamlit.app/` URL.
7. Edit section c of this README directly to include the real repository and app links, then commit that edit manually.
8. Check both links in a private browser window before submitting them.

The repository does not need PDF files, a documentation generator or publishing scripts. Guidance documents are supplied separately. The app loads the saved pipelines during deployment and does not run the training script.

## Reproducibility and limitations

The experiment uses Python 3.12 and the exact package versions in `requirements.txt`. Keep the scikit-learn version used for loading the models consistent with the version used for training. If dependencies are deliberately changed, retrain the models and review the results before updating the saved artifacts.

The comparison uses one stratified holdout and no tuning study. Differences of a few predictions should be interpreted cautiously. The data is a historical benchmark, and this project evaluates classification algorithms for an academic assignment.

## References

- Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5DW2B).
- [scikit-learn dataset loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
- [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
