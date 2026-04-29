"""
Inference utilities for scTRP.

Mirrors classifier/infer_supcon_functions.py — all public functions are
re-exported from scTRP.inference for convenience.
"""

import os
import sys
from pathlib import Path

# ---- scGPT source path ----
# Priority: SCGPT_PATH env var  →  known fallback paths below.
# Set the env var or pass --scgpt_path to any script to override.
# Example:  export SCGPT_PATH=/your/path/to/scGPT/
_KNOWN_SCGPT_SOURCE = [
    "/fs/ess/PCON0022/lsxgf/tools_related/scGPT/",
    "/cluster/pixstor/xudong-lab/suli/tools_related/scGPT/",
]
_env_scgpt = os.environ.get("SCGPT_PATH")
if _env_scgpt:
    sys.path.insert(0, _env_scgpt)
else:
    for _p in _KNOWN_SCGPT_SOURCE:
        if Path(_p).exists():
            sys.path.insert(0, _p)
            break

import argparse
import copy
import gc
import json
import time
import traceback
import warnings
import random
import pickle

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections.abc import Sequence
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import Parallel, delayed
import multiprocessing
import jenkspy
import ot

from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

import flash_attn
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"


# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #

def return_count_data(data, cfg):
    all_counts = (
        data.layers[cfg.input_layer_key].A
        if issparse(data.layers[cfg.input_layer_key])
        else data.layers[cfg.input_layer_key]
    )
    return all_counts


def get_project_emb(cfg, mdl, adata, genes, max_len, vb, pad_tok, pad_val, include_zero, dev):
    mdl.eval()
    all_counts = return_count_data(adata, cfg)
    tokenized_test = tokenize_and_pad_batch(
        data=all_counts,
        gene_ids=genes,
        max_len=max_len,
        vocab=vb,
        pad_token=pad_tok,
        pad_value=pad_val,
        append_cls=True,
        include_zero_gene=include_zero,
    )
    all_gene_ids = tokenized_test["genes"].to(dev)
    all_values = tokenized_test["values"].to(dev)
    src_key_padding_mask = all_gene_ids.eq(vb[pad_tok])
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.amp):
        cell_embeddings = mdl.encode_batch(
            all_gene_ids,
            all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=cfg.batch_size,
            batch_labels=None,
            time_step=0,
            return_np=True,
        )
        pro_cell_embeddings = mdl.cls_decoder(torch.from_numpy(cell_embeddings).to(dev))
    adata.obsm["X_scGPT_trained"] = cell_embeddings
    adata.obsm["X_scGPT_prj"] = pro_cell_embeddings.cpu().numpy()
    del cell_embeddings, pro_cell_embeddings
    sc.pp.neighbors(adata, use_rep="X_scGPT_prj")
    sc.tl.umap(adata, min_dist=0.5)
    sc.tl.tsne(adata, use_rep="X_scGPT_prj")
    return adata


# ------------------------------------------------------------------ #
# Cluster center helpers
# ------------------------------------------------------------------ #

def get_cluster_center(adata, obsm_key, norm_or_not):
    center_embeddings = {}
    unique_labels = np.unique(adata.obs["reactivity"].tolist())
    labels_np = adata.obs["reactivity"].values
    if norm_or_not:
        ori_emb = adata.obsm[obsm_key]
        input_emb = ori_emb / np.linalg.norm(ori_emb, axis=1, keepdims=True)
    else:
        input_emb = adata.obsm[obsm_key]
    for label in unique_labels:
        label_embeddings = input_emb[labels_np == label]
        center_embeddings[label] = label_embeddings.mean(axis=0)
    return center_embeddings


# ------------------------------------------------------------------ #
# Classifiers
# ------------------------------------------------------------------ #

def knn_classifier(N, train_embeddings, train_labels, val_embeddings):
    knn = KNeighborsClassifier(n_neighbors=N)
    knn.fit(train_embeddings, train_labels)
    knn_predictions = knn.predict(val_embeddings)
    probabilities = knn.predict_proba(val_embeddings)[:, 1]
    return (knn_predictions, probabilities)


def nearest_center(val_embeddings, class_centers_dic):
    predictions = []
    nearest_center_dis_score = []
    nearest_center_dis_score_toP = []
    for i in range(val_embeddings.shape[0]):
        min_distance = float("inf")
        nearest_center_label = None
        test_embedding = val_embeddings[i]
        for label, center in class_centers_dic.items():
            distance = np.linalg.norm(test_embedding - center)
            if distance < min_distance:
                min_distance = distance
                nearest_center_label = label
        predictions.append(nearest_center_label)
        if i == 0:
            print(f"nearest_center prediction data type is {type(nearest_center_label)}")
        distance_pos = np.linalg.norm(test_embedding - class_centers_dic["1"])
        distance_neg = np.linalg.norm(test_embedding - class_centers_dic["0"])
        nearest_center_dis_score.append(distance_neg - distance_pos)
        nearest_center_dis_score_toP.append(-distance_pos)
    predictions = np.array(predictions, dtype=object)
    nearest_center_dis_score = np.array(nearest_center_dis_score)
    nearest_center_dis_score_toP = np.array(nearest_center_dis_score_toP)
    return (predictions, nearest_center_dis_score, nearest_center_dis_score_toP)


def cosine_similarity_classifier(train_embeddings, test_embeddings, train_labels, uni_labels):
    train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    test_embeddings_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    class_centers_norm = {
        label: train_embeddings_norm[train_labels == label].mean(axis=0)
        for label in uni_labels
    }
    predictions = []
    pos_simi_score = []
    for i in range(test_embeddings_norm.shape[0]):
        test_embedding = test_embeddings_norm[i]
        max_similarity = -1
        nearest_center_label = None
        for label, center in class_centers_norm.items():
            similarity = np.dot(test_embedding, center)
            if similarity >= max_similarity:
                max_similarity = similarity
                nearest_center_label = label
        predictions.append(nearest_center_label)
        pos_simi_score.append(np.dot(test_embedding, class_centers_norm["1"]))
    predictions = np.array(predictions, dtype=object)
    pos_simi_score = np.array(pos_simi_score)
    return (predictions, pos_simi_score)


def oneclass_svm(train_adata_, val_adata, test_adata, obsm_key):
    X_train_normal = train_adata_[train_adata_.obs["reactivity"] == "1"].obsm[obsm_key]
    del train_adata_
    param_grid = {
        "gamma": [0.0001, 0.0078125, 0.009, 0.01, 0.011, 0.012, 0.015],
        "nu": [0.0009, 0.001, 0.0019, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.009],
        "degree": [2, 3, 4, 5, 6, 7, 8],
    }
    ocsvm = OneClassSVM(kernel="poly")
    grid_search = GridSearchCV(ocsvm, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1, refit=True)
    grid_search.fit(X_train_normal, np.ones(X_train_normal.shape[0]))
    print(f"Best parameters found: {grid_search.best_params_}")
    decision_scores = grid_search.best_estimator_.decision_function(val_adata.obsm[obsm_key])
    fpr, tpr, _ = roc_curve(val_adata.obs["reactivity"].values, decision_scores, pos_label="1")
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = _[optimal_idx]
    test_decision_scores = grid_search.best_estimator_.decision_function(test_adata.obsm[obsm_key])
    test_pred_custom_threshold = np.where(test_decision_scores > optimal_threshold, "1", "0").astype("object")
    return (test_pred_custom_threshold, test_decision_scores)


def distance_based_prediction(
    val_embeddings_np, class_centers_dic, val_labels_np, actual_val_emb_np, actual_val_labels_np
):
    distance_prediction_score = []
    input_emb = actual_val_emb_np
    for i in range(input_emb.shape[0]):
        test_embedding = input_emb[i]
        distance = np.linalg.norm(test_embedding - class_centers_dic["1"])
        score = 1 - distance / 2
        distance_prediction_score.append(score)
    distance_prediction_score = np.array(distance_prediction_score)
    fpr, tpr, thresholds = roc_curve(actual_val_labels_np, distance_prediction_score, pos_label="1")
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    distance_prediction_score = []
    input_emb = val_embeddings_np
    for i in range(input_emb.shape[0]):
        test_embedding = input_emb[i]
        distance = np.linalg.norm(test_embedding - class_centers_dic["1"])
        score = 1 - distance / 2
        distance_prediction_score.append(score)
    distance_prediction_score = np.array(distance_prediction_score)
    print(np.max(distance_prediction_score), np.min(distance_prediction_score))
    predictions_Youdensj = np.where(distance_prediction_score > optimal_threshold, "1", "0").astype("object")
    df_tem = pd.DataFrame({"reactivities_labels": val_labels_np, "pred_scores": distance_prediction_score})
    breaks = jenkspy.jenks_breaks(df_tem["pred_scores"], n_classes=2)
    predictions_jenks = np.where(distance_prediction_score > breaks[1], "1", "0").astype("object")
    return optimal_threshold, predictions_Youdensj, breaks[1], predictions_jenks, distance_prediction_score


# ------------------------------------------------------------------ #
# Optimal Transport helpers
# ------------------------------------------------------------------ #

def compute_ot_from_precomputed(M_row, weights_pca_kde):
    a = np.array([1.0])
    M = M_row.reshape(1, -1)
    return np.sqrt(ot.emd2(a, weights_pca_kde, M))


def batch_ot_from_M_all(M_all, weights_pca_kde, n_jobs: int = 8):
    return Parallel(n_jobs=n_jobs)(
        delayed(compute_ot_from_precomputed)(M_all[i], weights_pca_kde)
        for i in range(M_all.shape[0])
    )


def compute_kNN_weights(distances, k: int):
    avg_k_dist = np.mean(distances[:, 1 : k + 1], axis=1)
    densities_knn = 1.0 / (avg_k_dist + 1e-8)
    weights_knn = densities_knn / np.sum(densities_knn)
    return weights_knn


def compute_auc(k, M_all, val_labels_np, distances, batch_ot_from_M_all):
    weights_knn = compute_kNN_weights(distances, k)
    ot_distances = batch_ot_from_M_all(M_all, weights_knn)
    auc_val = roc_auc_score(val_labels_np, -np.array(ot_distances), average="weighted")
    return k, auc_val


def compute_normalized_deltarho(distances, indices, k: int):
    D_NB = distances[:, 1 : k + 1]
    ID_NB = indices[:, 1 : k + 1]
    num_cell = distances.shape[0]
    rho = 1.0 / np.sum(D_NB, axis=1)
    delta = np.zeros(num_cell)
    for ii in range(num_cell):
        temp = np.where(rho > rho[ii])[0]
        inter_temp = np.intersect1d(temp, ID_NB[ii, :])
        if len(inter_temp) == 0:
            delta[ii] = np.max(D_NB[ii, :])
        else:
            ib = np.where(np.isin(ID_NB[ii], inter_temp))[0]
            delta[ii] = np.min(D_NB[ii, ib])
    if rho.ndim != 1 or delta.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if rho.shape[0] != delta.shape[0]:
        raise ValueError("Inputs 'rho' and 'delta' must have the same length.")
    if np.any(rho < 0) or np.any(delta < 0):
        raise ValueError("Inputs 'rho' and 'delta' cannot contain negative values.")
    if np.var(rho) == 0 or np.var(delta) == 0:
        raise ValueError("Inputs 'rho' and 'delta' must have non-zero variance.")
    rho = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))
    delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))
    deltarho = delta * rho
    weights_deltarho = deltarho / np.sum(deltarho)
    return weights_deltarho


def compute_auc_deltarho(k, M_all, val_labels_np, distances, indices, batch_ot_from_M_all):
    weights_knn = compute_normalized_deltarho(distances, indices, k)
    ot_distances = batch_ot_from_M_all(M_all, weights_knn)
    auc_val = roc_auc_score(val_labels_np, -np.array(ot_distances), average="weighted")
    return k, auc_val


def OT_based_prediction(train_emb_all, test_emb, train_labels_np):
    mask = train_labels_np == "1"
    train_emb = train_emb_all[mask]
    nbrs = NearestNeighbors(n_neighbors=31, algorithm="auto").fit(train_emb)
    distances, indices = nbrs.kneighbors(train_emb)
    k_of_max_auc = 30
    weights_knn_1 = compute_kNN_weights(distances, k_of_max_auc)
    weights_knn_2 = compute_normalized_deltarho(distances, indices, k_of_max_auc)
    predictions = []
    function_names = []
    for suffix, weights_knn in zip(
        ["KNN_Weights_30best", "deltarho_30best"], [weights_knn_1, weights_knn_2]
    ):
        M_all = ot.dist(test_emb, train_emb)
        ot_distances = batch_ot_from_M_all(M_all, weights_knn)
        ot_array = np.array(ot_distances)
        y_score = -ot_array
        threshold = np.percentile(y_score, 90)
        y_pred = np.where(y_score >= threshold, "1", "0").astype("object")
        predictions.append((y_pred, y_score))
        df_tem = pd.DataFrame({"pred_scores": y_score})
        breaks = jenkspy.jenks_breaks(df_tem["pred_scores"], n_classes=2)
        predictions_jenks = np.where(y_score > breaks[1], "1", "0").astype("object")
        predictions.append((predictions_jenks, y_score))
        function_names += [f"top10_OT_{suffix}", f"jenks_OT_{suffix}"]
    return predictions, function_names


# ------------------------------------------------------------------ #
# Metrics output
# ------------------------------------------------------------------ #

def output_metrics(output_file_path, predictions_list, function_names_list, test_labels):
    results = []
    for preds_score_tu in predictions_list:
        preds = preds_score_tu[0].astype("object")
        score = preds_score_tu[1]
        precision = precision_score(test_labels, preds, average="weighted")
        recall = recall_score(test_labels, preds, average="weighted")
        f1 = f1_score(test_labels, preds, average="weighted")
        roc = roc_auc_score(test_labels, score, average="weighted")
        accuracy = accuracy_score(test_labels, preds)
        balanced_accuracy = balanced_accuracy_score(test_labels, preds)
        ari = adjusted_rand_score(test_labels, preds)
        mcc = matthews_corrcoef(test_labels, preds)
        tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        gmean = (sen * spe) ** 0.5
        results.append([precision, recall, f1, roc, accuracy, balanced_accuracy, ari, mcc, gmean])
    metrics_names = ["Precision", "Recall", "F1", "ROC_AUC", "Accuracy", "Balanced_accuracy", "ARI", "MCC", "Gmean"]
    with open(output_file_path, "w") as f:
        f.write("Classifying_Function," + ",".join(metrics_names) + "\n")
        for function_name, result in zip(function_names_list, results):
            result_str = ",".join(map(str, result))
            f.write(f"{function_name},{result_str}\n")
    return results
