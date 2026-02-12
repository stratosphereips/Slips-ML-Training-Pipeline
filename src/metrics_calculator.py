import ast
import math
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _safe_literal_eval(raw: str):
    """Safely evaluate dict-like strings coming from logs."""
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return ast.literal_eval(raw.replace("'", '"'))
        except Exception:
            raise


class MetricsCalculator:
    """Unified metrics calculation for pipeline and plotting scripts."""

    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or ["Benign", "Malicious"]

    def confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict[str, int]]:
        metrics: Dict[str, Dict[str, int]] = {}
        for label in self.labels:
            tp = int(np.sum((y_pred == label) & (y_true == label)))
            fp = int(np.sum((y_pred == label) & (y_true != label)))
            fn = int(np.sum((y_pred != label) & (y_true == label)))
            tn = int(np.sum((y_pred != label) & (y_true != label)))
            metrics[label] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        return metrics

    def binary_metrics(self, counts: Dict[str, int]) -> Dict[str, float]:
        tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "accuracy": accuracy,
        }

    def aggregate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        per_class = self.confusion_matrix(y_true, y_pred)
        metrics: Dict[str, Any] = {}
        for label, counts in per_class.items():
            label_metrics = self.binary_metrics(counts)
            for key, value in label_metrics.items():
                metrics[f"{label.lower()}_{key}"] = value
        return metrics

    def overall_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        counts = self.confusion_matrix(y_true, y_pred)["Malicious"]
        return self.binary_metrics(counts)


_GLOBAL_CALCULATOR = MetricsCalculator(labels=["Benign", "Malicious"])


def compute_binary_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    """Convenience wrapper to compute binary metrics for TP/FP/TN/FN counts."""
    return _GLOBAL_CALCULATOR.binary_metrics(counts)


def _compute_mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    numerator = tp * tn - fp * fn
    denom = math.sqrt(
        max(tp + fp, 0) * max(tp + fn, 0) * max(tn + fp, 0) * max(tn + fn, 0)
    )
    return numerator / denom if denom > 0 else 0.0


def compute_multi_metrics(
    per_class: Dict[str, Dict[str, int]]
) -> Dict[str, float]:
    """Aggregate per-class confusion dictionaries into macro/micro metrics."""
    tp_total = fp_total = tn_total = fn_total = 0
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for cls_counts in per_class.values():
        tp = int(cls_counts.get("TP", 0))
        fp = int(cls_counts.get("FP", 0))
        tn = int(cls_counts.get("TN", 0))
        fn = int(cls_counts.get("FN", 0))
        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    total = tp_total + fp_total + tn_total + fn_total
    accuracy = (tp_total + tn_total) / total if total > 0 else 0.0

    micro_precision = (
        tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    )
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    mcc = _compute_mcc(tp_total, fp_total, tn_total, fn_total)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "MCC": mcc,
    }


class LogMetricsParser:
    """Encapsulates all log parsing and ingestion utilities."""

    _BACKGROUND_TOKENS = ("background", "bg")

    def _strip_background(
        self, per_class: Optional[Dict[str, Dict[str, int]]]
    ) -> Dict[str, Dict[str, int]]:
        if not per_class:
            return {}
        return {
            cls: counts
            for cls, counts in per_class.items()
            if cls.lower() not in self._BACKGROUND_TOKENS
        }

    def parse_training_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single training/validation log line into structured counts."""
        out: Dict[str, Any] = {}
        try:
            text = line.strip()

            match_total = re.search(
                r"Total labels\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE
            )
            if match_total:
                raw_val = match_total.group(1)
                out["total_labels"] = (
                    float(raw_val) if "." in raw_val else int(raw_val)
                )

            match_val_size = re.search(
                r"(?:Validation|Testing) size\s*:\s*(\d+)", text, re.IGNORECASE
            )
            if match_val_size:
                out["testing_size"] = int(match_val_size.group(1))

            match_train_size = re.search(
                r"Training size\s*:\s*(\d+)", text, re.IGNORECASE
            )
            if match_train_size:
                out["training_size"] = int(match_train_size.group(1))

            match_seen = re.search(
                r"(?:Validation|Testing) seen labels\s*:\s*(\{.*?\})", text
            )
            if match_seen:
                out["seen"] = _safe_literal_eval(match_seen.group(1))

            match_pred = re.search(
                r"(?:Validation|Testing) predicted labels\s*:\s*(\{.*?\})",
                text,
            )
            if match_pred:
                out["predicted"] = _safe_literal_eval(match_pred.group(1))

            match_metrics = re.search(
                r"(?:Validation|Testing) metrics\s*:\s*(\{.*?\})", text
            )
            if match_metrics:
                metrics = _safe_literal_eval(match_metrics.group(1))
                tp = int(metrics.get("TP", 0))
                fp = int(metrics.get("FP", 0))
                fn = int(metrics.get("FN", 0))
                tn = int(metrics.get("TN", 0))
                out["per_class"] = {
                    "Malicious": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
                    "Benign": {"TP": tn, "FP": fn, "TN": tp, "FN": fp},
                }

            match_seen_tr = re.search(
                r"Training seen labels\s*:\s*(\{.*?\})", text
            )
            if match_seen_tr:
                out["training_seen"] = _safe_literal_eval(match_seen_tr.group(1))

            match_pred_tr = re.search(
                r"Training predicted labels\s*:\s*(\{.*?\})", text
            )
            if match_pred_tr:
                out["training_predicted"] = _safe_literal_eval(
                    match_pred_tr.group(1)
                )

            match_metrics_tr = re.search(
                r"Training metrics\s*:\s*(\{.*?\})", text
            )
            if match_metrics_tr:
                metrics = _safe_literal_eval(match_metrics_tr.group(1))
                tp = int(metrics.get("TP", 0))
                fp = int(metrics.get("FP", 0))
                fn = int(metrics.get("FN", 0))
                tn = int(metrics.get("TN", 0))
                out["training_per_class"] = {
                    "Malicious": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
                    "Benign": {"TP": tn, "FP": fn, "TN": tp, "FN": fp},
                }

            if "per_class" not in out and "seen" in out and "predicted" in out:
                per_class = {
                    label: {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                    for label in out["seen"].keys()
                }
                if per_class:
                    out["per_class"] = per_class

            if "per_class" in out:
                out["per_class"] = self._strip_background(out["per_class"])
            if "training_per_class" in out:
                out["training_per_class"] = self._strip_background(
                    out["training_per_class"]
                )

            return out
        except Exception as exc:
            print("[WARN] parse_training_log_line failed:", exc)
            traceback.print_exc()
            return None

    def parse_testing_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse testing log lines."""
        out: Dict[str, Any] = {}
        try:
            text = line.strip()
            total_match = re.search(
                r"Total flows\s*:\s*(\d+)", text, re.IGNORECASE
            )
            if total_match:
                out["total_flows"] = int(total_match.group(1))

            match_seen = re.search(r"Seen labels\s*:\s*(\{.*?\})", text)
            if match_seen:
                out["seen"] = _safe_literal_eval(match_seen.group(1))

            match_pred = re.search(r"Predicted labels\s*:\s*(\{.*?\})", text)
            if match_pred:
                out["predicted"] = _safe_literal_eval(match_pred.group(1))

            match_metrics = re.search(
                r"Malware metrics(?:\s*\(.*?\))?\s*[:=]\s*(\{.*?\})",
                text,
                re.IGNORECASE,
            )
            if match_metrics:
                metrics = _safe_literal_eval(match_metrics.group(1))
                tp = int(metrics.get("TP", 0))
                fp = int(metrics.get("FP", 0))
                tn = int(metrics.get("TN", 0))
                fn = int(metrics.get("FN", 0))
                out["per_class"] = {
                    "Malicious": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
                    "Benign": {"TP": tn, "FP": fn, "TN": tp, "FN": fp},
                }
                out["binary_summary"] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

            if "per_class" in out:
                out["per_class"] = self._strip_background(out["per_class"])
            return out
        except Exception as exc:
            print("[WARN] parse_testing_log_line failed:", exc)
            traceback.print_exc()
            return None

    def read_training_batches(self, logfile: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        print(f"[INFO] Reading training logfile: {logfile}")
        with open(logfile, "r") as handle:
            for idx, raw_line in enumerate(handle):
                line = raw_line.strip()
                if not line:
                    continue
                parsed = self.parse_training_log_line(line)
                if parsed is None:
                    print(f"[WARN] Skipping unparsable line {idx}: {line[:200]}")
                    continue
                entries.append(parsed)
        return entries

    def read_testing_snapshots(self, logfile: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        print(f"[INFO] Reading testing logfile: {logfile}")
        with open(logfile, "r") as handle:
            for idx, raw_line in enumerate(handle):
                line = raw_line.strip()
                if not line:
                    continue
                parsed = self.parse_testing_log_line(line)
                if parsed is None:
                    print(
                        f"[WARN] Skipping unparsable testing line {idx}: {line[:200]}"
                    )
                    continue
                entries.append(parsed)
        return entries

    def detect_validation_split(self, entries: List[Dict[str, Any]]) -> bool:
        return any(
            ("per_class" in entry and bool(entry["per_class"]))
            or (
                "validation_per_class" in entry
                and bool(entry["validation_per_class"])
            )
            or (entry.get("testing_size", 0) > 0)
            for entry in entries
        )


_GLOBAL_PARSER = LogMetricsParser()


def parse_training_log_line(line: str) -> Optional[Dict[str, Any]]:
    return _GLOBAL_PARSER.parse_training_log_line(line)


def parse_testing_log_line(line: str) -> Optional[Dict[str, Any]]:
    return _GLOBAL_PARSER.parse_testing_log_line(line)


def read_training_batches(logfile: str) -> List[Dict[str, Any]]:
    return _GLOBAL_PARSER.read_training_batches(logfile)


def read_testing_snapshots(logfile: str) -> List[Dict[str, Any]]:
    return _GLOBAL_PARSER.read_testing_snapshots(logfile)


def detect_validation_split(entries: List[Dict[str, Any]]) -> bool:
    return _GLOBAL_PARSER.detect_validation_split(entries)


def compute_malware_metrics(
    per_class: Dict[str, Dict[str, int]]
) -> Dict[str, float]:
    malware_key = next(
        (name for name in per_class.keys() if name.lower() in ("malware", "malicious")),
        None,
    )
    if not malware_key:
        return {
            "fpr": 0.0,
            "fnr": 0.0,
            "fp_over_predicted": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error_rate": 0.0,
        }

    counts = per_class[malware_key]
    binary = compute_binary_metrics(counts)
    tp = counts.get("TP", 0)
    fp = counts.get("FP", 0)
    malware_metrics = {
        "fpr": binary["fpr"],
        "fnr": binary["fnr"],
        "precision": binary["precision"],
        "recall": binary["recall"],
        "f1": binary["f1"],
        "error_rate": 1.0 - binary["accuracy"],
    }
    malware_metrics["fp_over_predicted"] = (
        (fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    )
    return malware_metrics


def process_batch_metrics(
    per_class: Dict[str, Dict[str, int]], class_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    batch_metrics_per_class: Dict[str, Dict[str, float]] = {}
    for cls in class_names:
        bin_metrics = compute_binary_metrics(per_class[cls])
        bin_metrics.update(per_class[cls])
        batch_metrics_per_class[cls] = bin_metrics

    batch_multi = compute_multi_metrics(per_class)
    batch_multi.update(compute_malware_metrics(per_class))
    return batch_metrics_per_class, batch_multi


def process_cumulative_metrics(
    cumul_class_counters: Dict[str, Dict[str, int]], class_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    cumul_metrics_per_class: Dict[str, Dict[str, float]] = {}
    for cls in class_names:
        bin_metrics = compute_binary_metrics(cumul_class_counters[cls])
        bin_metrics.update(cumul_class_counters[cls])
        cumul_metrics_per_class[cls] = bin_metrics

    cumul_multi = compute_multi_metrics(cumul_class_counters)
    cumul_multi.update(compute_malware_metrics(cumul_class_counters))
    return cumul_metrics_per_class, cumul_multi


def accumulate_training_metrics(
    entries: List[Dict[str, Any]], has_validation_data: bool
) -> Tuple[List, ...]:
    print("[INFO] Accumulating batch and cumulative metrics via extractor...")

    batch_metrics_per_class_train: List[Dict[str, Any]] = []
    batch_metrics_multi_train: List[Dict[str, float]] = []
    cumul_metrics_multi_train: List[Dict[str, float]] = []
    cumul_metrics_per_class_train: List[Dict[str, Any]] = []

    if has_validation_data:
        batch_metrics_per_class_val: List[Dict[str, Any]] = []
        batch_metrics_multi_val: List[Dict[str, float]] = []
        cumul_metrics_multi_val: List[Dict[str, float]] = []
        cumul_metrics_per_class_val: List[Dict[str, Any]] = []

    if not entries:
        if has_validation_data:
            return (
                [],
                [],
                [],
                [],
                batch_metrics_per_class_train,
                batch_metrics_multi_train,
                cumul_metrics_multi_train,
                cumul_metrics_per_class_train,
            )
        return (
            batch_metrics_per_class_train,
            batch_metrics_multi_train,
            cumul_metrics_multi_train,
            cumul_metrics_per_class_train,
        )

    first = entries[0]
    class_name_sets: List[set] = []
    if isinstance(first.get("training_predicted"), dict):
        class_name_sets.append(set(first["training_predicted"].keys()))
    if isinstance(first.get("training_per_class"), dict):
        class_name_sets.append(set(first["training_per_class"].keys()))
    if isinstance(first.get("per_class"), dict):
        class_name_sets.append(set(first["per_class"].keys()))
    if isinstance(first.get("validation_per_class"), dict):
        class_name_sets.append(set(first["validation_per_class"].keys()))

    class_names = sorted(set().union(*class_name_sets)) if class_name_sets else []

    cumul_class_counters_train = {
        cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for cls in class_names
    }
    if has_validation_data:
        cumul_class_counters_val = {
            cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
            for cls in class_names
        }

    for data in entries:
        if has_validation_data:
            validation_per_class = data.get(
                "per_class",
                {cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for cls in class_names},
            )
            batch_per_class_val, batch_multi_val = process_batch_metrics(
                validation_per_class, class_names
            )
            batch_metrics_per_class_val.append(batch_per_class_val)
            batch_metrics_multi_val.append(batch_multi_val)

            for cls in class_names:
                for key in ("TP", "FP", "TN", "FN"):
                    cumul_class_counters_val[cls][key] += int(
                        validation_per_class.get(cls, {}).get(key, 0)
                    )

            cumul_per_class_val, cumul_multi_val = process_cumulative_metrics(
                cumul_class_counters_val, class_names
            )
            cumul_metrics_per_class_val.append(cumul_per_class_val)
            cumul_metrics_multi_val.append(cumul_multi_val)

        training_per_class = data.get(
            "training_per_class",
            {cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for cls in class_names},
        )
        batch_per_class_train, batch_multi_train = process_batch_metrics(
            training_per_class, class_names
        )
        batch_metrics_per_class_train.append(batch_per_class_train)
        batch_metrics_multi_train.append(batch_multi_train)

        for cls in class_names:
            for key in ("TP", "FP", "TN", "FN"):
                cumul_class_counters_train[cls][key] += int(
                    training_per_class.get(cls, {}).get(key, 0)
                )

        cumul_per_class_train, cumul_multi_train = process_cumulative_metrics(
            cumul_class_counters_train, class_names
        )
        cumul_metrics_per_class_train.append(cumul_per_class_train)
        cumul_metrics_multi_train.append(cumul_multi_train)

    if has_validation_data:
        return (
            batch_metrics_per_class_val,
            batch_metrics_multi_val,
            cumul_metrics_multi_val,
            cumul_metrics_per_class_val,
            batch_metrics_per_class_train,
            batch_metrics_multi_train,
            cumul_metrics_multi_train,
            cumul_metrics_per_class_train,
        )
    return (
        batch_metrics_per_class_train,
        batch_metrics_multi_train,
        cumul_metrics_multi_train,
        cumul_metrics_per_class_train,
    )


def accumulate_testing_metrics(entries: List[Dict[str, Any]]):
    if not entries:
        return [], [], [], [], []

    class_names = (
        list(entries[0].get("per_class", {}).keys()) or ["Malicious", "Benign"]
    )
    cumul_per_class_series: List[Dict[str, Dict[str, float]]] = []
    cumul_multi_series: List[Dict[str, float]] = []
    cumul_binary_series: List[Dict[str, float]] = []
    cumul_class_counts_series: List[Dict[str, int]] = []
    cumulative_total_flows: List[int] = []

    for data in entries:
        pcm = data.get("per_class", {})

        per_class_metrics_now: Dict[str, Dict[str, float]] = {}
        for cls in class_names:
            counts = {
                key: int(pcm.get(cls, {}).get(key, 0))
                for key in ("TP", "FP", "TN", "FN")
            }
            bin_metrics = compute_binary_metrics(counts)
            bin_metrics.update(counts)
            per_class_metrics_now[cls] = bin_metrics
        cumul_per_class_series.append(per_class_metrics_now)

        snapshot_counts = {
            cls: {
                key: int(pcm.get(cls, {}).get(key, 0))
                for key in ("TP", "FP", "TN", "FN")
            }
            for cls in class_names
        }
        multi_now = compute_multi_metrics(snapshot_counts)
        multi_now.update(compute_malware_metrics(snapshot_counts))
        cumul_multi_series.append(multi_now)

        if "binary_summary" in data:
            bm_counts = {
                key: int(data["binary_summary"].get(key, 0))
                for key in ("TP", "FP", "TN", "FN")
            }
        else:
            mal = pcm.get("Malicious", {})
            tp = int(mal.get("TP", 0))
            fp = int(mal.get("FP", 0))
            fn = int(mal.get("FN", 0))
            tn = sum(
                int(pcm.get(cls, {}).get("TN", 0))
                for cls in class_names
                if cls.lower() not in ("malware", "malicious")
            )
            bm_counts = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        cumul_binary_series.append(compute_binary_metrics(bm_counts))

        counts_dict = {
            cls: int(pcm.get(cls, {}).get("TP", 0) + pcm.get(cls, {}).get("FN", 0))
            for cls in class_names
        }
        cumul_class_counts_series.append(counts_dict)

        cumulative_total_flows.append(int(data.get("total_flows", 0)))

    return (
        cumul_per_class_series,
        cumul_multi_series,
        cumul_binary_series,
        cumul_class_counts_series,
        cumulative_total_flows,
    )


def _format_summary_section(title: str, metrics_data: Dict[str, float]) -> List[str]:
    lines = [f"\n=== {title} ==="]
    lines.append(f"Accuracy:             {metrics_data.get('accuracy', 0):.4f}")
    lines.append(f"F1:                   {metrics_data.get('f1', 0):.4f}")
    lines.append(f"FPR:                  {metrics_data.get('fpr', 0):.4f}")
    lines.append(f"FNR:                  {metrics_data.get('fnr', 0):.4f}")
    lines.append(f"Macro F1:             {metrics_data.get('macro_f1', 0):.4f}")
    lines.append(f"Precision:            {metrics_data.get('precision', 0):.4f}")
    lines.append(f"Recall:               {metrics_data.get('recall', 0):.4f}")
    lines.append(f"MCC:                  {metrics_data.get('MCC', 0):.4f}")
    return lines


def _format_per_class_table(
    title: str, cum_metrics_per_class: Dict[str, Dict[str, float]]
) -> List[str]:
    lines = [f"\n=== {title} ==="]
    lines.append(
        f"{'Class':<15} {'TP':>8} {'TN':>8} {'FP':>8} {'FN':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    )
    if "Malicious" in cum_metrics_per_class:
        m = cum_metrics_per_class["Malicious"]
        lines.append(
            f"{'Malicious':<15} {int(m.get('TP', 0)):8d} {int(m.get('TN', 0)):8d} {int(m.get('FP', 0)):8d} {int(m.get('FN', 0)):8d} "
            f"{m.get('accuracy', 0.0):8.4f} {m.get('precision', 0.0):8.4f} {m.get('recall', 0.0):8.4f} {m.get('f1', 0.0):8.4f}"
        )
    return lines


def build_training_summary(
    exp_name: str,
    batch_count: int,
    has_validation_data: bool,
    cumul_metrics_multi,
    cumul_metrics_per_class,
    cumul_metrics_multi_training=None,
    cumul_metrics_per_class_training=None,
) -> str:
    lines: List[str] = []

    if has_validation_data and cumul_metrics_multi:
        lines.extend(
            _format_summary_section(
                "VALIDATION Multi-class (Aggregated)", cumul_metrics_multi[-1]
            )
        )
    if has_validation_data and cumul_metrics_multi_training:
        lines.extend(
            _format_summary_section(
                "TRAINING Multi-class (Aggregated)",
                cumul_metrics_multi_training[-1],
            )
        )
    if not has_validation_data and cumul_metrics_multi:
        lines.extend(
            _format_summary_section(
                "TRAINING Multi-class (Aggregated)", cumul_metrics_multi[-1]
            )
        )

    if has_validation_data and cumul_metrics_per_class:
        lines.extend(
            _format_per_class_table(
                "Per-class metrics (Aggregated) - VALIDATION",
                cumul_metrics_per_class[-1],
            )
        )
    if has_validation_data and cumul_metrics_per_class_training:
        lines.extend(
            _format_per_class_table(
                "Per-class metrics (Aggregated) - TRAINING",
                cumul_metrics_per_class_training[-1],
            )
        )
    if not has_validation_data and cumul_metrics_per_class:
        lines.extend(
            _format_per_class_table(
                "Per-class metrics (Aggregated) - TRAINING",
                cumul_metrics_per_class[-1],
            )
        )

    lines.append(f"\nSummary for Experiment {exp_name}:")
    lines.append(f"Total batches processed: {batch_count}")
    lines.append(
        "Data type: Training/Validation split"
        if has_validation_data
        else "Data type: Training only"
    )
    return "\n".join(lines)


def build_testing_summary(exp_name: str, cumul_multi_series, cumul_per_class_series) -> str:
    if not cumul_multi_series:
        return f"No testing data parsed for experiment {exp_name}."
    lines: List[str] = []
    lines.extend(
        _format_summary_section(
            f"TESTING Multi-class (Aggregated) — {exp_name}",
            cumul_multi_series[-1],
        )
    )
    lines.extend(
        _format_per_class_table(
            "Per-class metrics (Aggregated) - TESTING",
            cumul_per_class_series[-1],
        )
    )
    return "\n".join(lines)


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract metrics from training/testing logs without plotting."
    )
    parser.add_argument("--train-log", help="Path to training log file", default=None)
    parser.add_argument("--test-log", help="Path to testing log file", default=None)
    parser.add_argument("--experiment", help="Experiment identifier", default="N/A")
    args = parser.parse_args()

    if args.train_log:
        entries = read_training_batches(args.train_log)
        if not entries:
            print("[ERROR] No training entries parsed; aborting summary.")
        else:
            has_validation = detect_validation_split(entries)
            result = accumulate_training_metrics(entries, has_validation)
            if has_validation:
                (
                    _,
                    _,
                    cumul_metrics_multi,
                    cumul_metrics_per_class,
                    _,
                    _,
                    cumul_metrics_multi_training,
                    cumul_metrics_per_class_training,
                ) = result
            else:
                (
                    _,
                    _,
                    cumul_metrics_multi,
                    cumul_metrics_per_class,
                ) = result
                cumul_metrics_multi_training = None
                cumul_metrics_per_class_training = None

            summary = build_training_summary(
                args.experiment,
                len(entries),
                has_validation,
                cumul_metrics_multi,
                cumul_metrics_per_class,
                cumul_metrics_multi_training,
                cumul_metrics_per_class_training,
            )
            print(summary)

    if args.test_log:
        entries = read_testing_snapshots(args.test_log)
        (
            cumul_per_class_series,
            cumul_multi_series,
            _,
            _,
            _,
        ) = accumulate_testing_metrics(entries)
        summary = build_testing_summary(
            args.experiment, cumul_multi_series, cumul_per_class_series
        )
        print(summary)


if __name__ == "__main__":
    _cli()
