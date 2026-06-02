import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def print_metrics(title, metrics):
    auc = "nan" if np.isnan(metrics["AUC"]) else f"{metrics['AUC']:.4f}"
    print(f"    [{title}]")
    print(
        f"       Accuracy: {metrics['Accuracy']:.4f} | Precision: {metrics['Precision']:.4f} "
        f"| Recall: {metrics['Recall']:.4f} | Specificity: {metrics['Specificity']:.4f}"
    )
    print(f"       F1: {metrics['F1']:.4f} | ROC-AUC: {auc}")
    print(
        f"       Confusion Matrix -> TN: {metrics['TN']:<3} FP: {metrics['FP']:<3} "
        f"FN: {metrics['FN']:<3} TP: {metrics['TP']:<3}"
    )

