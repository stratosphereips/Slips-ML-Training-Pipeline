# base_utils.py
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Ensure src/ is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# METRIC DISPLAY CONFIGURATIONS
# Single source of truth for all plotting - change once, applies everywhere
# ============================================================================

# Metrics to show in malware-focused plots (with FPR, FNR, F1, error rate)
MALWARE_PLOT_METRICS = {
    "FPR": "fpr",
    "FNR": "fnr",
    "F1": "f1",
    "Accuracy": "accuracy",  # This IS benign-malicious accuracy
    "Total Error Rate": "error_rate",
}

# Metrics for accuracy-only plots
ACCURACY_PLOT_METRICS = {
    "Accuracy": "accuracy",
}

# Metrics for train/val comparison plots
COMPARISON_PLOT_METRICS = [
    ("accuracy", "Accuracy", "train_val_accuracy.png"),
    ("f1", "F1", "train_val_malware_f1.png"),
    ("MCC", "MCC", "train_val_mcc.png"),
]

# Metrics for FN/FP rate comparison plots
FN_RATE_METRIC = ("fnr", "FN Rate")
FP_RATE_METRIC = ("fp_over_predicted", "FP Rate")


# ============================================================================
# METRIC EXTRACTION FUNCTIONS
# ============================================================================


def extract_metrics_for_plot(
    metrics_dict: Dict[str, float], display_mapping: Dict[str, str]
) -> Dict[str, float]:
    """
    Generic extractor: maps display names to metric keys.

    Args:
        metrics_dict: Dict with computed metrics (e.g., from accumulate_metrics)
        display_mapping: Dict mapping display_name -> metric_key

    Returns:
        Dict with display names as keys
    """
    return {
        display_name: metrics_dict.get(metric_key, 0.0)
        for display_name, metric_key in display_mapping.items()
    }


def extract_comparison_for_plot(
    val_metric: float,
    train_metric: float,
    val_label: str = "Validation",
    train_label: str = "Training",
) -> Dict[str, float]:
    """
    Build comparison dict for train vs val plots.
    """
    return {val_label: val_metric, train_label: train_metric}


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, return the normalized path.
    """
    p = os.path.abspath(path)
    os.makedirs(p, exist_ok=True)
    return p
# ------------------------
# Plotting helpers
# ------------------------
def plot_major_metrics_together(
    series: List[Dict[str, float]],
    outpath: str,
    title: str = "Metrics over tests",
    xvals: Optional[List] = None,
    xlabel: str = "Index",
):
    if series is None or len(series) == 0:
        print(f"[INFO] plot_major_metrics_together: no data for {outpath}")
        return

    outdir = os.path.dirname(os.path.abspath(outpath))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    metric_names = []
    first_keys = list(series[0].keys())
    for k in first_keys:
        if k not in metric_names:
            metric_names.append(k)
    for entry in series[1:]:
        for k in entry.keys():
            if k not in metric_names:
                metric_names.append(k)

    metric_values = {m: [] for m in metric_names}
    for entry in series:
        for m in metric_names:
            metric_values[m].append(entry.get(m, 0.0))

    n = len(next(iter(metric_values.values())))
    if xvals is None:
        x_axis = list(range(1, n + 1))
    else:
        try:
            if len(xvals) == n:
                x_axis = xvals
            else:
                x_axis = list(range(1, n + 1))
        except Exception:
            x_axis = list(range(1, n + 1))

        plt.figure(figsize=(8, 4.5))
    for m in metric_names:
        vals = metric_values[m]
        # show a marker when there's only one point so single-point series are visible
        marker = "o" if len(vals) == 1 else None
        plt.plot(x_axis, vals, label=m, linewidth=1.5, marker=marker)
        # Plot scatter points at the vertices (start and end)
        plt.scatter([x_axis[0], x_axis[-1]], [vals[0], vals[-1]],
                    color=plt.gca().lines[-1].get_color(), s=30, zorder=3, label=None)


    plt.xlabel(xlabel)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)

    all_vals = [v for vals in metric_values.values() for v in vals]
    finite_vals = [float(x) for x in all_vals if np.isfinite(x)]

    if finite_vals:
        min_val = min(finite_vals)
        max_val = max(finite_vals)
        value_range = max_val - min_val

        # Check if values look like probabilities/rates (0-1 range)
        if 0 <= min_val and max_val <= 1:
            # If the range is very small (< 0.05), we have high accuracy scenario
            if value_range < 0.05:
                # Show it's a zoomed view by using a tighter range
                # but DON'T make it look like the full scale
                margin = max(0.002, value_range * 0.2)
                lower = max(0, min_val - margin)
                upper = min(1, max_val + margin)
                plt.ylim(lower, upper)
            else:
                # Normal range - show full 0 to 1
                plt.ylim(0, 1.05)
        else:
            # Not probability metrics - use natural range
            margin = 0.05 * value_range if value_range > 0 else 0.05
            plt.ylim(min_val - margin, max_val + margin)

    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
