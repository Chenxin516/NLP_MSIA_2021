from typing import List

import pandas as pd
import numpy as np
from sklearn import metrics


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(
    df: pd.DataFrame, subgroup: str, label_col: str, y_proba_col: str
) -> float:
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label_col], subgroup_examples[y_proba_col])


def compute_bpsn_auc(df, subgroup, label, y_proba_col):
    """
    Computes the AUC of the within-subgroup negative examples
    and the background positive examples.
    """
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[y_proba_col])


def compute_bnsp_auc(df, subgroup, label, y_proba_col):
    """
    Computes the AUC of the within-subgroup positive examples and the
    background negative examples.
    """
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[y_proba_col])


def compute_bias_metrics_for_model(
    dataset: pd.DataFrame, subgroups: List[str], y_proba_col: str, label_col: str
):
    """Computes per-subgroup metrics for all subgroups"""
    records = []
    for subgroup in subgroups:
        record = {
            "subgroup": subgroup,
            "subgroup_size": len(dataset[dataset[subgroup]]),
        }
        record["subgroup_auc"] = compute_subgroup_auc(
            dataset, subgroup, label_col, y_proba_col
        )
        # background positive, subgroup negative
        record["bpsn_auc"] = compute_bpsn_auc(dataset, subgroup, label_col, y_proba_col)
        # background negative, subgroup positive
        record["bnsp_auc"] = compute_bnsp_auc(dataset, subgroup, label_col, y_proba_col)
        records.append(record)
    return pd.DataFrame(records).sort_values("subgroup_auc", ascending=True)


def calculate_overall_auc(df, model_name):
    true_labels = df["label"]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average(
        [
            power_mean(bias_df["subgroup_auc"], POWER),
            power_mean(bias_df["bpsn_auc"], POWER),
            power_mean(bias_df["bnsp_auc"], POWER),
        ]
    )
    return (OVERALL_MODEL_WEIGHT * overall_auc) + (
        (1 - OVERALL_MODEL_WEIGHT) * bias_score
    )


def evaluate_model(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    y_true = df[label_col].values
    y_pred = df["y_pred"].values
    y_proba = df["y_pred_proba"].values

    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    auc_roc = metrics.roc_auc_score(y_true, y_proba)

    df_result = pd.DataFrame(
        {"metrics": ["accuracy", "f1", "auc_roc"], "value": [acc, f1, auc_roc]}
    )
    return df_result
