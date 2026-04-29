"""
scripts/train_xgboost.py — Light XGBoost classifier for T/B-cell reactivity.

A dependency-light alternative to the full scTRP contrastive-learning pipeline.
Requires only gene expression from .h5ad files; no GPU, scGPT, or ESM2 needed.

The default gene panel (30 genes) is derived from GRN analysis of the training
cohort. Provide --gene_list to override with your own CSV of gene names.

EXAMPLE COMMAND:
    python scripts/train_xgboost.py \\
        --train_data_path /path/to/train.h5ad \\
        --val_data_path   /path/to/val.h5ad \\
        --test_data_path  /path/to/test.h5ad \\
        --output_dir      /path/to/output/
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.cluster import adjusted_rand_score

warnings.filterwarnings("ignore")

# Default 30-gene panel (reactive and non-reactive markers)
DEFAULT_GENES = [
    "FGFBP2", "GNLY", "GZMB", "HLA-DRA", "IL7R", "IRF4", "LAIR2",
    "TNFRSF18", "XCL2", "ZNF683", "CD83", "FASLG", "GZMA",
    "CCL20", "DNAJB1", "FOS", "FTL", "HMOX1", "HSPB1", "LEF1",
    "LGALS1", "DUSP1", "HSPA1A", "HSPA1B", "ANXA1", "CD69", "DDIT3",
    "CCR7", "KLRG1", "TIMP1",
]


def extract_features(adata, genes, split_name="split"):
    available = [g for g in genes if g in adata.var_names]
    missing = set(genes) - set(available)
    if missing:
        print(f"  [{split_name}] {len(missing)} genes not found: {sorted(missing)}")
    X = adata[:, available].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = pd.DataFrame(X, columns=available, index=adata.obs_names)
    y = adata.obs["reactivity"].astype(int).values if "reactivity" in adata.obs else None
    if y is not None:
        print(f"  [{split_name}] shape={X.shape}  label 1: {y.sum()}  label 0: {(y == 0).sum()}")
    else:
        print(f"  [{split_name}] shape={X.shape}  (no reactivity label)")
    return X, y


def get_preds(model, X):
    return model.predict(X), model.predict_proba(X)[:, 1]


def output_metrics(predictions_list, function_names_list, test_labels, output_file_path=None):
    results = []
    for preds, score in predictions_list:
        precision         = precision_score(test_labels, preds, average="weighted")
        recall            = recall_score(test_labels, preds, average="weighted")
        f1                = f1_score(test_labels, preds, average="weighted")
        roc               = roc_auc_score(test_labels, score)
        accuracy          = accuracy_score(test_labels, preds)
        balanced_accuracy = balanced_accuracy_score(test_labels, preds)
        ari               = adjusted_rand_score(test_labels, preds)
        mcc               = matthews_corrcoef(test_labels, preds)
        tn, fp, fn, tp    = confusion_matrix(test_labels, preds).ravel()
        sen               = tp / (tp + fn)
        spe               = tn / (tn + fp)
        gmean             = (sen * spe) ** 0.5
        results.append([precision, recall, f1, roc, accuracy, balanced_accuracy, ari, mcc, gmean])

    metrics_names = [
        "Precision", "Recall", "F1", "ROC_AUC", "Accuracy",
        "Balanced_accuracy", "ARI", "MCC", "Gmean",
    ]

    if output_file_path is not None:
        with open(output_file_path, "w") as f:
            f.write("Classifying_Function," + ",".join(metrics_names) + "\n")
            for name, result in zip(function_names_list, results):
                f.write(f"{name}," + ",".join(map(str, result)) + "\n")

    print(f"\n{'Metric':<22}" + "  ".join(f"{n:>16}" for n in function_names_list))
    for i, name in enumerate(metrics_names):
        vals = "  ".join(f"{r[i]:>16.4f}" for r in results)
        print(f"  {name:<20} {vals}")

    return results


def save_predictions(model, X, adata, split_name, output_dir):
    """Save per-cell predicted label and reactive score to CSV."""
    preds, scores = get_preds(model, X)
    df = pd.DataFrame({
        "cell_id":          X.index,
        "predicted_label":  preds,
        "reactive_score":   scores,
    })
    if "reactivity" in adata.obs:
        df["true_label"] = adata.obs["reactivity"].astype(int).values
    out_path = output_dir / f"predictions_{split_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Light XGBoost classifier for T/B-cell reactivity prediction."
    )
    parser.add_argument("--train_data_path", required=True,
                        help="Training split (.h5ad)")
    parser.add_argument("--val_data_path",   required=True,
                        help="Validation split (.h5ad)")
    parser.add_argument("--test_data_path",  required=True,
                        help="Test split (.h5ad)")
    parser.add_argument("--output_dir",      required=True,
                        help="Directory to save model, importance CSV, predictions, and plots")
    parser.add_argument("--gene_list", default=None,
                        help="CSV file with gene names (first column used). "
                             "Defaults to the built-in 30-gene panel.")
    # XGBoost hyperparameters (defaults match the notebook)
    parser.add_argument("--n_estimators",          type=int,   default=500)
    parser.add_argument("--learning_rate",         type=float, default=0.05)
    parser.add_argument("--max_depth",             type=int,   default=4)
    parser.add_argument("--min_child_weight",      type=int,   default=5)
    parser.add_argument("--subsample",             type=float, default=0.8)
    parser.add_argument("--colsample_bytree",      type=float, default=0.8)
    parser.add_argument("--gamma",                 type=float, default=1.0)
    parser.add_argument("--reg_alpha",             type=float, default=0.1)
    parser.add_argument("--reg_lambda",            type=float, default=1.0)
    parser.add_argument("--early_stopping_rounds", type=int,   default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gene list
    if args.gene_list is not None:
        gene_df = pd.read_csv(args.gene_list)
        genes = gene_df.iloc[:, 0].tolist()
        print(f"Loaded {len(genes)} genes from {args.gene_list}")
    else:
        genes = DEFAULT_GENES
        print(f"Using default {len(genes)}-gene panel")

    # Load data
    print("\nLoading h5ad files...")
    adata_train = sc.read_h5ad(args.train_data_path)
    adata_val   = sc.read_h5ad(args.val_data_path)
    adata_test  = sc.read_h5ad(args.test_data_path)

    print("\nExtracting features...")
    X_train, y_train = extract_features(adata_train, genes, "train")
    X_val,   y_val   = extract_features(adata_val,   genes, "val")
    X_test,  y_test  = extract_features(adata_test,  genes, "test")

    # Handle class imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nClass balance → pos: {pos}, neg: {neg}, scale_pos_weight: {scale_pos_weight:.3f}")

    # Build model
    model = xgb.XGBClassifier(
        objective             = "binary:logistic",
        eval_metric           = ["logloss", "auc"],
        n_estimators          = args.n_estimators,
        learning_rate         = args.learning_rate,
        max_depth             = args.max_depth,
        min_child_weight      = args.min_child_weight,
        subsample             = args.subsample,
        colsample_bytree      = args.colsample_bytree,
        gamma                 = args.gamma,
        reg_alpha             = args.reg_alpha,
        reg_lambda            = args.reg_lambda,
        scale_pos_weight      = scale_pos_weight,
        early_stopping_rounds = args.early_stopping_rounds,
        random_state          = 42,
        n_jobs                = -1,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )
    print(f"\nBest iteration: {model.best_iteration}")

    # Evaluate
    print("\n── Val metrics ──")
    output_metrics([get_preds(model, X_val)], ["Val"], y_val)

    print("\n── Test metrics ──")
    output_metrics(
        [get_preds(model, X_test)], ["Test"], y_test,
        output_file_path=str(output_dir / "metrics_test.csv"),
    )

    # Per-cell predictions → CSV
    print("\nSaving per-cell predictions...")
    save_predictions(model, X_train, adata_train, "train", output_dir)
    save_predictions(model, X_val,   adata_val,   "val",   output_dir)
    save_predictions(model, X_test,  adata_test,  "test",  output_dir)

    # Feature importance
    importance_df = pd.DataFrame({
        "gene":       list(X_train.columns),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 most important genes:")
    print(importance_df.head(20).to_string(index=False))

    top_n  = min(30, len(importance_df))
    top_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(7, top_n * 0.25 + 1))
    ax.barh(top_df["gene"][::-1], top_df["importance"][::-1], color="#3498db")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top Gene Importances — XGBoost Reactivity Classifier")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    print(f"\nSaved: {output_dir}/feature_importance.png")

    # Save model artifacts
    model.save_model(str(output_dir / "xgboost_reactivity.json"))
    importance_df.to_csv(output_dir / "gene_importance.csv", index=False)
    joblib.dump(model, str(output_dir / "xgboost_reactivity.pkl"))
    print(f"Saved: xgboost_reactivity.json / .pkl, gene_importance.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
