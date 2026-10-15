# YuvaIntern Week 3 — Unsupervised Learning & Model Evaluation

**Intern:** Sahil Sant  
**Program:** AI Pioneers Internship (Machine Learning Internship)  
**Task:** Week 3 — Unsupervised Learning & Model Evaluation

## Project Overview

This project demonstrates both unsupervised exploration and a separate supervised-evaluation workflow using the Scikit-learn Breast Cancer Wisconsin Diagnostic benchmark dataset (569 observations, 30 numeric predictors). The benchmark is used for educational experiments only and is **not intended for medical diagnosis or clinical decision-making**.

### Part A — Unsupervised Learning
- Standardize numerical predictors before distance-based clustering.
- Compare K-Means and Agglomerative (hierarchical) clustering for k = 2 through 6.
- Evaluate cluster compactness/separation using the silhouette score.
- Use Adjusted Rand Index (ARI) only as a **post-hoc external comparison** against known labels; labels are never supplied to clustering `.fit()`.
- Apply PCA to visualize the 30-dimensional scaled feature space in two dimensions and report explained variance.

### Part B — Supervised Evaluation
- Use a stratified 80/20 train/test split with `random_state=42`.
- Tune Logistic Regression and Random Forest using 5-fold `StratifiedKFold` cross-validation and `GridSearchCV`, optimizing ROC-AUC.
- Evaluate the held-out test set using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Save the best fitted pipelines, CSV result tables, and plots.

## Measured Results (reproducible run)

- K-Means selected **k = 2** using the highest silhouette score in the tested range; silhouette = **0.3450**.
- K-Means ARI against benchmark diagnosis labels, calculated after clustering only = **0.6707**.
- The first two PCA components explain **63.24%** of the standardized feature variance.
- Logistic Regression test accuracy = **96.49%**, malignant-class recall = **92.86%**, F1 = **95.12%**, ROC-AUC = **0.9960**.
- Random Forest test accuracy = **96.49%**, malignant-class recall = **90.48%**, F1 = **95.00%**, ROC-AUC = **0.9940**.

These are measurements for this specific benchmark, random seed, split, preprocessing, and parameter grid—not guarantees of future or clinical performance.

## Repository Structure

```text
.
├── data/
│   └── breast_cancer.csv
├── figures/
│   ├── best_model_confusion_matrix.png
│   ├── clustering_silhouette_by_k.png
│   ├── kmeans_cluster_sizes.png
│   ├── kmeans_clusters_pca_projection.png
│   ├── roc_curve_comparison.png
│   └── supervised_model_metrics.png
├── models/
│   ├── best_kmeans.joblib
│   ├── clustering_scaler.joblib
│   ├── pca_2_components.joblib
│   ├── tuned_logistic_regression.joblib
│   └── tuned_random_forest.joblib
├── notebooks/
│   └── week3_unsupervised_and_evaluation.ipynb
├── report/
│   └── YuvaIntern_Week3_Unsupervised_Model_Evaluation_Report.docx
├── results/
│   ├── best_clustering_models.csv
│   ├── best_models_cv_fold_scores.csv
│   ├── cluster_profile.csv
│   ├── clustering_metrics.csv
│   ├── confusion_matrices.csv
│   ├── pca_explained_variance.csv
│   ├── pca_cluster_projection.csv
│   ├── supervised_model_metrics.csv
│   └── run_summary.json
├── src/
│   └── run_analysis.py
├── submission_description.txt
└── requirements.txt
```

Additional grid-search and per-class metric CSVs are also saved under `results/` after execution.

## Run on Windows / VS Code

1. Extract the project ZIP and open its folder in VS Code.
2. Open Terminal → New Terminal.
3. Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

4. Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

5. Run the entire analysis:

```powershell
python src/run_analysis.py
```

6. Open `notebooks/week3_unsupervised_and_evaluation.ipynb` and run cells from top to bottom if you want to explore the steps interactively.

## Key Files

- `report/YuvaIntern_Week3_Unsupervised_Model_Evaluation_Report.docx` — Word report with methodology, measured results, code examples, tables, figures, and interpretation.
- `results/supervised_model_metrics.csv` — held-out test metrics and selected hyperparameters.
- `results/clustering_metrics.csv` — silhouette and post-hoc ARI for k = 2 through 6.
- `results/run_summary.json` — concise machine-readable summary of the run.

## Methodological Notes

- Diagnosis labels are excluded from clustering features and never passed into K-Means or Agglomerative Clustering. ARI and cluster diagnosis proportions are only calculated after cluster assignments have been created.
- Scaling is necessary here because clustering uses distances and input variables have different ranges.
- PCA's two-dimensional projection is for visualization; clustering itself is fitted in the full standardized 30-feature space.
- The supervised split is stratified, and cross-validation/tuning is performed on training data only. The held-out test set is used for final reporting.
- This educational benchmark is not a medical device and cannot be used to make health decisions.
