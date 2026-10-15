"""YuvaIntern Week 3: Unsupervised Learning & Model Evaluation.

This script regenerates clustering evaluations, PCA projections, supervised model
searches, held-out metrics, figures, and saved models using a fixed random seed.
The built-in breast-cancer benchmark is for education only, not clinical use.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, auc, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, silhouette_score,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for folder in (DATA, RESULTS, FIGURES, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def prepare_dataset() -> pd.DataFrame:
    """Write a self-contained CSV from sklearn's local benchmark dataset."""
    source = load_breast_cancer(as_frame=True)
    frame = source.data.copy()
    # Make the positive class intuitive for the report: 1 = malignant, 0 = benign.
    frame["diagnosis_malignant"] = (source.target.to_numpy() == 0).astype(int)
    frame.to_csv(DATA / "breast_cancer.csv", index=False)
    return frame


def run_clustering(df: pd.DataFrame) -> dict:
    """Fit clusters on scaled predictors only; labels are used only after fitting."""
    feature_cols = [c for c in df.columns if c != "diagnosis_malignant"]
    X = df[feature_cols].to_numpy()
    y = df["diagnosis_malignant"].to_numpy()  # evaluation only, never passed to fit()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS / "clustering_scaler.joblib")

    k_values = range(2, 7)
    rows: list[dict] = []
    fitted: dict[str, tuple[int, np.ndarray, object]] = {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        km_labels = km.fit_predict(X_scaled)
        rows.append({
            "algorithm": "K-Means", "n_clusters": k,
            "silhouette_score": silhouette_score(X_scaled, km_labels),
            "adjusted_rand_index_vs_diagnosis": adjusted_rand_score(y, km_labels),
            "inertia": km.inertia_,
        })
        fitted[f"K-Means_{k}"] = (k, km_labels, km)

        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        hc_labels = hc.fit_predict(X_scaled)
        rows.append({
            "algorithm": "Agglomerative (Hierarchical)", "n_clusters": k,
            "silhouette_score": silhouette_score(X_scaled, hc_labels),
            "adjusted_rand_index_vs_diagnosis": adjusted_rand_score(y, hc_labels),
            "inertia": np.nan,
        })
        fitted[f"Agglomerative_{k}"] = (k, hc_labels, hc)

    scores = pd.DataFrame(rows)
    scores.to_csv(RESULTS / "clustering_metrics.csv", index=False)

    # Select k by silhouette score only; ARI is reported as an external comparison,
    # not used to choose the number of clusters.
    best_rows = scores.loc[scores.groupby("algorithm")["silhouette_score"].idxmax()].copy()
    best_rows = best_rows.sort_values("algorithm")
    best_rows.to_csv(RESULTS / "best_clustering_models.csv", index=False)

    best_kmeans_row = scores[scores["algorithm"] == "K-Means"].sort_values(
        "silhouette_score", ascending=False
    ).iloc[0]
    best_k = int(best_kmeans_row["n_clusters"])
    best_model = fitted[f"K-Means_{best_k}"][2]
    best_labels = fitted[f"K-Means_{best_k}"][1]
    joblib.dump(best_model, MODELS / "best_kmeans.joblib")

    # PCA is used to project the scaled 30-feature space into 2 dimensions for display.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)
    joblib.dump(pca, MODELS / "pca_2_components.joblib")
    pd.DataFrame({
        "principal_component_1": pcs[:, 0],
        "principal_component_2": pcs[:, 1],
        "kmeans_cluster": best_labels,
        "diagnosis_malignant_for_posthoc_comparison_only": y,
    }).to_csv(RESULTS / "pca_cluster_projection.csv", index=False)

    explained = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(RESULTS / "pca_explained_variance.csv", index=False)

    profile = df[feature_cols].copy()
    profile["cluster"] = best_labels
    profile["diagnosis_malignant_for_posthoc_comparison_only"] = y
    summaries = []
    for cluster_id, group in profile.groupby("cluster"):
        summaries.append({
            "cluster": int(cluster_id),
            "observations": int(len(group)),
            "malignant_fraction_posthoc": float(group["diagnosis_malignant_for_posthoc_comparison_only"].mean()),
            "mean_radius": float(group["mean radius"].mean()),
            "mean_texture": float(group["mean texture"].mean()),
            "mean_perimeter": float(group["mean perimeter"].mean()),
        })
    cluster_profile = pd.DataFrame(summaries).sort_values("cluster")
    cluster_profile.to_csv(RESULTS / "cluster_profile.csv", index=False)

    # Plot clustering metrics by k.
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for algorithm, group in scores.groupby("algorithm"):
        ax.plot(group["n_clusters"], group["silhouette_score"], marker="o", label=algorithm)
    ax.set_title("Silhouette Score by Number of Clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (higher is better)")
    ax.set_xticks(list(k_values))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "clustering_silhouette_by_k.png", dpi=180)
    plt.close(fig)

    # Projection colours are K-Means clusters fitted in standardized 30-D space.
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=best_labels, s=24, alpha=0.78)
    ax.set_title(f"K-Means Clusters Projected onto PCA (k={best_k})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.grid(True, alpha=0.2)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [f"Cluster {i}" for i in sorted(np.unique(best_labels))], title="Cluster")
    fig.tight_layout()
    fig.savefig(FIGURES / "kmeans_clusters_pca_projection.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(cluster_profile["cluster"].astype(str), cluster_profile["observations"])
    ax.set_title(f"Observation Count by K-Means Cluster (k={best_k})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of observations")
    ax.bar_label(bars)
    fig.tight_layout()
    fig.savefig(FIGURES / "kmeans_cluster_sizes.png", dpi=180)
    plt.close(fig)

    return {
        "features": feature_cols,
        "X_scaled": X_scaled,
        "y": y,
        "scores": scores,
        "best_rows": best_rows,
        "best_kmeans_k": best_k,
        "best_kmeans_silhouette": float(best_kmeans_row["silhouette_score"]),
        "best_kmeans_ari": float(best_kmeans_row["adjusted_rand_index_vs_diagnosis"]),
        "pca_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_pc2": float(pca.explained_variance_ratio_[1]),
        "pca_cumulative": float(pca.explained_variance_ratio_.sum()),
        "cluster_profile": cluster_profile,
    }


def run_supervised(df: pd.DataFrame) -> dict:
    feature_cols = [c for c in df.columns if c != "diagnosis_malignant"]
    X = df[feature_cols]
    y = df["diagnosis_malignant"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    logreg_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    logreg_grid = {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}
    rf_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8],
        "model__min_samples_leaf": [1, 3],
    }
    searches: dict[str, GridSearchCV] = {
        "Logistic Regression": GridSearchCV(
            logreg_pipe, logreg_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
            return_train_score=True,
        ),
        "Random Forest": GridSearchCV(
            rf_pipe, rf_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
            return_train_score=True,
        ),
    }

    result_rows = []
    fold_rows = []
    confusion_rows = []
    curves = {}
    for model_name, search in searches.items():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        row = {
            "model": model_name,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision_malignant": precision_score(y_test, y_pred, zero_division=0),
            "test_recall_malignant": recall_score(y_test, y_pred, zero_division=0),
            "test_f1_malignant": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_prob),
            "cv_best_mean_roc_auc": search.best_score_,
            "cv_to_test_auc_gap": search.best_score_ - roc_auc_score(y_test, y_prob),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            "true_benign_pred_benign": int(tn),
            "benign_pred_malignant_false_positive": int(fp),
            "malignant_pred_benign_false_negative": int(fn),
            "true_malignant_pred_malignant": int(tp),
        }
        result_rows.append(row)

        for split_i in range(1, 6):
            fold_rows.append({
                "model": model_name,
                "fold": split_i,
                "validation_roc_auc": float(search.cv_results_[f"split{split_i-1}_test_score"][search.best_index_]),
            })
        pd.DataFrame(search.cv_results_).to_csv(
            RESULTS / ("logistic_regression_grid_search.csv" if model_name == "Logistic Regression" else "random_forest_grid_search.csv"),
            index=False,
        )
        for actual_label in [0, 1]:
            for predicted_label in [0, 1]:
                confusion_rows.append({
                    "model": model_name,
                    "actual_diagnosis": "Malignant" if actual_label == 1 else "Benign",
                    "predicted_diagnosis": "Malignant" if predicted_label == 1 else "Benign",
                    "count": int(cm[actual_label, predicted_label]),
                })
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        curves[model_name] = (fpr, tpr, auc(fpr, tpr))
        joblib.dump(best_model, MODELS / (
            "tuned_logistic_regression.joblib" if model_name == "Logistic Regression" else "tuned_random_forest.joblib"
        ))

        # Classification report per class and fixed-threshold prediction output.
        report_dict = classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malignant"], output_dict=True, zero_division=0
        )
        pd.DataFrame(report_dict).transpose().to_csv(
            RESULTS / ("logistic_regression_classification_report.csv" if model_name == "Logistic Regression" else "random_forest_classification_report.csv")
        )

    metrics = pd.DataFrame(result_rows).sort_values("test_roc_auc", ascending=False)
    metrics.to_csv(RESULTS / "supervised_model_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "best_models_cv_fold_scores.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(RESULTS / "confusion_matrices.csv", index=False)

    # ROC curve comparison
    fig, ax = plt.subplots(figsize=(7.7, 5.7))
    for model_name, (fpr, tpr, model_auc) in curves.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={model_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance baseline")
    ax.set_title("ROC Curves on Held-Out Test Set")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curve_comparison.png", dpi=180)
    plt.close(fig)

    best_name = str(metrics.iloc[0]["model"])
    best_search = searches[best_name]
    y_pred_best = best_search.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["Benign (0)", "Malignant (1)"]
    ax.set_xticks([0, 1], labels=classes, rotation=18, ha="right")
    ax.set_yticks([0, 1], labels=classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "best_model_confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Plot held-out metrics, allowing results to show trade-offs not just one score.
    plot_metrics = metrics.set_index("model")[[
        "test_accuracy", "test_precision_malignant", "test_recall_malignant", "test_f1_malignant", "test_roc_auc"
    ]]
    ax = plot_metrics.plot(kind="bar", figsize=(10, 5.7), ylim=(0, 1), width=0.78)
    ax.set_title("Supervised Models: Held-Out Test Metrics")
    ax.set_ylabel("Score (0 to 1)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "supervised_model_metrics.png", dpi=180)
    plt.close()

    return {
        "X_train_shape": tuple(X_train.shape),
        "X_test_shape": tuple(X_test.shape),
        "train_malignant": int(y_train.sum()),
        "test_malignant": int(y_test.sum()),
        "metrics": metrics,
        "folds": pd.DataFrame(fold_rows),
        "best_model_name": best_name,
        "best_model_cv_auc": float(metrics.iloc[0]["cv_best_mean_roc_auc"]),
        "best_model_test_auc": float(metrics.iloc[0]["test_roc_auc"]),
        "best_model_confusion_matrix": cm.tolist(),
    }


def write_run_summary(df: pd.DataFrame, clustering: dict, supervised: dict) -> None:
    summary = {
        "dataset": "Scikit-learn Breast Cancer Wisconsin Diagnostic (built-in benchmark)",
        "rows": int(df.shape[0]),
        "predictor_count": int(df.shape[1] - 1),
        "benign_count": int((df["diagnosis_malignant"] == 0).sum()),
        "malignant_count": int((df["diagnosis_malignant"] == 1).sum()),
        "clustering_best_kmeans_k_by_silhouette": clustering["best_kmeans_k"],
        "clustering_best_kmeans_silhouette": clustering["best_kmeans_silhouette"],
        "clustering_best_kmeans_ari_posthoc": clustering["best_kmeans_ari"],
        "pca_pc1_variance_ratio": clustering["pca_pc1"],
        "pca_pc2_variance_ratio": clustering["pca_pc2"],
        "pca_two_component_cumulative_variance_ratio": clustering["pca_cumulative"],
        "supervised_train_shape": list(supervised["X_train_shape"]),
        "supervised_test_shape": list(supervised["X_test_shape"]),
        "best_supervised_model_by_test_roc_auc": supervised["best_model_name"],
        "best_supervised_model_cv_auc": supervised["best_model_cv_auc"],
        "best_supervised_model_test_auc": supervised["best_model_test_auc"],
        "best_supervised_model_confusion_matrix_labels_order": ["Benign (0)", "Malignant (1)"],
        "best_supervised_model_confusion_matrix": supervised["best_model_confusion_matrix"],
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()
    df = prepare_dataset()
    clustering = run_clustering(df)
    supervised = run_supervised(df)
    write_run_summary(df, clustering, supervised)
    print("Week 3 analysis completed.")
    print(f"Dataset: {len(df)} rows, {df.shape[1]-1} predictors")
    print("\nClustering models selected by silhouette:")
    print(clustering["best_rows"][["algorithm", "n_clusters", "silhouette_score", "adjusted_rand_index_vs_diagnosis"]].to_string(index=False))
    print("\nPCA cumulative variance (2 components): {:.4f}".format(clustering["pca_cumulative"]))
    print("\nSupervised model results:")
    print(supervised["metrics"][["model", "test_accuracy", "test_precision_malignant", "test_recall_malignant", "test_f1_malignant", "test_roc_auc", "cv_best_mean_roc_auc"]].to_string(index=False))
    print("\nAll generated artifacts are saved under data/, results/, figures/, and models/.")


if __name__ == "__main__":
    main()
