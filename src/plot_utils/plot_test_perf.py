#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_utils import (
    MALWARE_PLOT_METRICS,
    ensure_dir,
    extract_metrics_for_plot,
    plot_major_metrics_together,
)
from metrics_calculator import (
    accumulate_testing_metrics,
    build_testing_summary,
    read_testing_snapshots,
)


def _choose_sparse_xticks(batch_count, labels):
    """
    Always return numeric positions for xticks (0..batch_count-1) as the first
    element. The second element is a list of labels where only a limited set
    of positions contain text (sparse labels); other positions are "".

    This prevents accidental use of string labels as x coordinates.
    """
    # full numeric positions for plotting (monotonic)
    positions = list(range(batch_count))

    if batch_count <= 20:
        # keep all labels for small series
        return positions, labels

    max_labels = 15
    step = max(1, batch_count // max_labels)
    indices = list(range(0, batch_count, step))
    if indices[-1] != batch_count - 1:
        indices.append(batch_count - 1)

    sparse_labels = [""] * batch_count
    for i in indices:
        # guard: labels might be shorter than batch_count
        if i < len(labels):
            sparse_labels[i] = labels[i]
        else:
            sparse_labels[i] = str(i)

    # NOTE: first element is the full numeric positions (not the sparse indices)
    return positions, sparse_labels


def plot_counts_series(
    series_of_dicts, outpath, title, xlabels=None, xlabel="Index"
):
    if series_of_dicts is None or not series_of_dicts:
        # print("[INFO] No data to plot for", title)
        return
    classes = list(next(iter(series_of_dicts)).keys())
    values_per_class = {
        c: [entry.get(c, 0) for entry in series_of_dicts] for c in classes
    }
    n = len(series_of_dicts)
    x_positions = list(range(n))
    plt.figure(figsize=(9, 4))

    # when there's just one batch/point, show a marker so the point is visible
    marker = "o" if n == 1 else None
    for cls in classes:
        plt.plot(x_positions, values_per_class[cls], label=cls, linewidth=1, marker=marker)

    if xlabels is None:
        labels = [str(i) for i in range(n)]
    else:
        labels = list(xlabels)
    if n >= 10:
        cleaned = []
        for lab in labels:
            if "\n" in lab:
                cleaned.append(lab.split("\n", 1)[0])
            else:
                cleaned.append(lab)
        labels = cleaned
    idxs, sparse_labels = _choose_sparse_xticks(n, labels)
    plt.xticks(idxs, [sparse_labels[i] for i in idxs], rotation=45, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    max_val = max(v for values in values_per_class.values() for v in values)
    top = max_val * 1.05 if max_val > 0 else 1
    plt.ylim(0, top)
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath)
    plt.close()
    # print(f"[SAVED] {outpath}")


def plot_confusion_matrix_from_final(final_per_class, outpath):
    mal = final_per_class.get("Malicious", {})
    tp = int(mal.get("TP", 0))
    fn = int(mal.get("FN", 0))
    fp = int(mal.get("FP", 0))
    tn = int(mal.get("TN", 0))
    cm = np.array([[tp, fn], [fp, tn]])
    labels = np.array([[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", f"TN\n{tn}"]])
    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Pred Malicious", "Pred Benign"], rotation=45)
    plt.yticks([0, 1], ["True Malicious", "True Benign"])
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, labels[i, j], ha="center", va="center", color="black"
            )
    plt.title("Confusion matrix (final snapshot)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    # print(f"[SAVED] {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot testing performance metrics."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to testing log file or directory",
    )
    parser.add_argument(
        "-e", "--exp", required=True, help="Experiment identifier"
    )
    parser.add_argument(
        "--save_folder", required=False, help="Output folder", default=None
    )
    args = parser.parse_args()

    save_folder = args.save_folder
    if save_folder is not None:
        if not os.path.isdir(save_folder):
            raise NotADirectoryError(
                f"Output folder does not exist: {save_folder}"
            )
        base_dir = ensure_dir(save_folder)
    else:
        base_dir = ensure_dir("performance_metrics")

    file_path = args.file
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, "testing.log")
        print(f"[INFO] -f is a directory, using: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    testing_dir = ensure_dir(os.path.join(base_dir, "testing", args.exp))
    print(f"[INFO] Output folder: {testing_dir}")

    entries = read_testing_snapshots(file_path)
    if not entries:
        print("[ERROR] No testing entries parsed; exiting.")
        return

    (
        cumul_per_class_series,
        cumul_multi_series,
        _,
        cumul_class_counts_series,
        cumulative_total_flows,
    ) = accumulate_testing_metrics(entries)
    n = len(cumul_multi_series)
    # print(f"[INFO] Building plots for {n} snapshots")

    # aggregated class counts
    xlabels = (
        [str(x) for x in cumulative_total_flows]
        if any(cumulative_total_flows)
        else [str(i) for i in range(n)]
    )
    out_counts = os.path.join(
        testing_dir, "class_counts_aggregated_testing.png"
    )
    print(
        "[INFO] Plotting aggregated class counts (TP+FN per class so-far)..."
    )
    plot_counts_series(
        cumul_class_counts_series,
        out_counts,
        title="Aggregated class counts\n(Total labeled samples seen so-far)",
        xlabels=xlabels,
        xlabel="Cumulative flows seen",
    )

    # malware metrics
    print(
        "[INFO] Plotting main metrics (FPR, FNR, F1, Accuracy) over snapshots..."
    )
    malware_metrics_data = [
        extract_metrics_for_plot(m, MALWARE_PLOT_METRICS)
        for m in cumul_multi_series
    ]
    out_malware = os.path.join(
        testing_dir, "malicious_metrics_aggregated_testing.png"
    )
    xvals = (
        cumulative_total_flows
        if any(cumulative_total_flows)
        else list(range(n))
    )
    plot_major_metrics_together(
        malware_metrics_data,
        out_malware,
        title="Malicious metrics (Aggregated)\n(testing set: so-far)",
        xvals=xvals,
        xlabel="Total flows seen",
    )

    # FPR/FNR only
    print("[INFO] Saving FPR/FNR-only plot...")
    fpr_fnr_series = [
        {"FPR": m.get("fpr", 0), "FNR": m.get("fnr", 0)}
        for m in cumul_multi_series
    ]
    out_fprfnr = os.path.join(testing_dir, "malicious_fpr_fnr_over_time.png")
    plot_major_metrics_together(
        fpr_fnr_series,
        out_fprfnr,
        title="Malicious FPR & FNR over time\n(testing snapshots)",
        xvals=xvals,
        xlabel="Total flows seen",
    )

    # predicted vs seen
    print(
        "[INFO] Plotting predicted vs seen counts (per-snapshot) for Malicious & Benign..."
    )
    pred_seen_series = []
    for e in entries:
        seen = e.get("seen", {})
        pred = e.get("predicted", {})
        pred_seen_series.append(
            {
                "Seen Malicious": int(seen.get("Malicious", 0)),
                "Pred Malicious": int(pred.get("Malicious", 0)),
                "Seen Benign": int(seen.get("Benign", 0)),
                "Pred Benign": int(pred.get("Benign", 0)),
            }
        )
    out_predseen = os.path.join(
        testing_dir, "predicted_vs_seen_per_snapshot.png"
    )
    plot_counts_series(
        pred_seen_series,
        out_predseen,
        title="Predicted vs Seen counts per snapshot",
        xlabels=xlabels,
        xlabel="Snapshot / cumulative flows seen",
    )

    # confusion matrix (final snapshot)
    print("[INFO] Plotting final confusion matrix (final snapshot)...")
    final_per_class = cumul_per_class_series[-1]
    out_cm = os.path.join(testing_dir, "confusion_matrix_final.png")
    plot_confusion_matrix_from_final(final_per_class, out_cm)

    summary_text = build_testing_summary(
        args.exp, cumul_multi_series, cumul_per_class_series
    )
    summary_text += f"\nTotal test lines processed: {len(entries)}"
    summary_path = os.path.join(testing_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    # print(f"[SAVED] {summary_path}")
    print(summary_text)


if __name__ == "__main__":
    main()
