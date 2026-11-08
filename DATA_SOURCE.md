# Dataset attribution

Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.

- Dataset page: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- DOI: https://doi.org/10.24432/C5DW2B
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Licence text: https://creativecommons.org/licenses/by/4.0/
- Bundled loader: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

This project uses scikit-learn's bundled copy of the UCI dataset. It changes
feature-name spaces to underscores, reverses the target encoding so that
0 means benign and 1 means malignant, and exports deterministic training and
test splits. The original identifier is not used as an input feature.
No source measurements are synthetically generated or altered.

The complete converted dataset is data/breast_cancer.csv. The training subset
is data/train_data.csv. The held-out subset is test_data.csv. Zero-based row
positions linking the split files to the complete dataset are recorded in
reports/split_indices.json.
