import numpy as np
from typing import Dict, Any, List, Optional

class MetricsCalculator:
    """
    Unified metrics calculation for pipeline and plotting scripts.
    Computes confusion matrix, F1, FPR, precision, recall, etc.
    """
    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or ["Benign", "Malicious"]

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, int]]:
        metrics = {}
        for label in self.labels:
            tp = int(np.sum((y_pred == label) & (y_true == label)))
            fp = int(np.sum((y_pred == label) & (y_true != label)))
            fn = int(np.sum((y_pred != label) & (y_true == label)))
            tn = int(np.sum((y_pred != label) & (y_true != label)))
            metrics[label] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        return metrics

    def binary_metrics(self, counts: Dict[str, int]) -> Dict[str, float]:
        tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
        }

    def aggregate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        per_class = self.confusion_matrix(y_true, y_pred)
        metrics = {}
        for label, counts in per_class.items():
            label_metrics = self.binary_metrics(counts)
            for k, v in label_metrics.items():
                metrics[f"{label.lower()}_{k}"] = v
        # Optionally, add macro/micro metrics
        return metrics

    def overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # For binary classification, use 'Malicious' as positive
        counts = self.confusion_matrix(y_true, y_pred)["Malicious"]
        return self.binary_metrics(counts)
