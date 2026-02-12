#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt

# Ensure src/ is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_utils import ensure_dir, plot_major_metrics_together
from metrics_calculator import (
    accumulate_training_metrics,
    build_training_summary,
    compute_binary_metrics,
    compute_malware_metrics,
    compute_multi_metrics,
    detect_validation_split,
    read_training_batches,
)


def calculate_class_counts(entries, data_key, class_names):
    batch_class_counts = []
    for entry in entries:
        if data_key in entry and entry[data_key]:
            counts = {
                cls: int(
                    entry[data_key][cls].get("TP", 0)
                    + entry[data_key][cls].get("FN", 0)
                )
                for cls in class_names
            }
        else:
            counts = {cls: 0 for cls in class_names}
        batch_class_counts.append(counts)

    cumul_class_counts = {cls: 0 for cls in class_names}
    cumul_class_counts_per_batch = []
    for counts in batch_class_counts:
        for cls in class_names:
            cumul_class_counts[cls] += counts[cls]
        cumul_class_counts_per_batch.append(cumul_class_counts.copy())

    return batch_class_counts, cumul_class_counts_per_batch


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
    series_of_dicts, outpath, title, xlabels=None, xlabel="Batch"
):
    if series_of_dicts is None or not series_of_dicts:
        print("[INFO] No data to plot for", title)
        return

    classes = list(next(iter(series_of_dicts)).keys())
    values_per_class = {
        c: [entry.get(c, 0) for entry in series_of_dicts] for c in classes
    }
    batch_count = len(series_of_dicts)
    x_positions = list(range(batch_count))

    plt.figure(figsize=(9, 4))
    marker = "o" if batch_count == 1 else None
    for cls in classes:
        plt.plot(x_positions, values_per_class[cls], label=cls, linewidth=1, marker=marker)

    if xlabels is None:
        labels = [str(i) for i in range(batch_count)]
    else:
        labels = list(xlabels)

    if batch_count >= 10:
        cleaned = []
        for lab in labels:
            if "\n" in lab:
                cleaned.append(lab.split("\n", 1)[0])
            else:
                cleaned.append(lab)
        labels = cleaned

    idxs, sparse_labels = _choose_sparse_xticks(batch_count, labels)
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


def get_stepping_sizes(entries, batch_count, size_key):
    labels = []
    if batch_count < 10:
        for i, entry in enumerate(entries):
            labels.append(f"{i}\n{entry.get(size_key, 0)}")
        return labels

    if batch_count <= 20:
        return [str(i) for i in range(batch_count)]
    else:
        max_labels = 15
        step = max(1, batch_count // max_labels)
        labels = []
        for i, entry in enumerate(entries):
            if i % step == 0 or i == batch_count - 1:
                labels.append(str(i))
            else:
                labels.append("")
        return labels


def sliding_window_aggregated(
    batch_metrics_per_class, class_names, k, trim_to_full_window=True
):
    n = len(batch_metrics_per_class)
    series_per_class = []
    series_multi = []

    for i in range(n):
        start = max(0, i - k + 1)
        agg = {
            cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for cls in class_names
        }
        for j in range(start, i + 1):
            per_cls = batch_metrics_per_class[j]
            for cls in class_names:
                agg[cls]["TP"] += int(per_cls[cls].get("TP", 0))
                agg[cls]["FP"] += int(per_cls[cls].get("FP", 0))
                agg[cls]["TN"] += int(per_cls[cls].get("TN", 0))
                agg[cls]["FN"] += int(per_cls[cls].get("FN", 0))

        per_class_metrics = {}
        for cls in class_names:
            bin_metrics = compute_binary_metrics(agg[cls])
            bin_metrics.update(agg[cls])
            per_class_metrics[cls] = bin_metrics

        multi = compute_multi_metrics(agg)
        multi.update(compute_malware_metrics(agg))

        series_per_class.append(per_class_metrics)
        series_multi.append(multi)

    if trim_to_full_window:
        if n < k:
            return [], [], None
        start_index = k - 1
        return (
            series_per_class[start_index:],
            series_multi[start_index:],
            start_index,
        )
    else:
        return series_per_class, series_multi, 0


def plot_malware_metrics(metrics_data, output_path, title, xvals, xlabel):
    from base_utils import MALWARE_PLOT_METRICS, extract_metrics_for_plot

    plot_data = [
        extract_metrics_for_plot(entry, MALWARE_PLOT_METRICS)
        for entry in metrics_data
    ]
    plot_major_metrics_together(
        plot_data, output_path, title=title, xvals=xvals, xlabel=xlabel
    )


def plot_accuracy_metrics(metrics_data, output_path, title, xvals, xlabel):
    accuracy_data = []
    for entry in metrics_data:
        accuracy_data.append(
            {"Benign-Malicious Acc": entry.get("accuracy", 0)}
        )
    plot_major_metrics_together(
        accuracy_data, output_path, title=title, xvals=xvals, xlabel=xlabel
    )


def plot_comparison_metrics(
    batch_metrics_multi,
    cumul_metrics_multi,
    batch_metrics_multi_training,
    cumul_metrics_multi_training,
    base_dir,
    stepping_total_sizes,
    cumulative_total_sizes,
    batch_count,
):
    from base_utils import COMPARISON_PLOT_METRICS, extract_comparison_for_plot

    agg_dir = ensure_dir(os.path.join(base_dir, "aggregated"))
    batch_dir = ensure_dir(os.path.join(base_dir, "per_batch"))

    # Aggregated plots
    for metric_key, short_title, filename in COMPARISON_PLOT_METRICS:
        combined = [
            extract_comparison_for_plot(
                cumul_metrics_multi[i].get(metric_key, 0),
                cumul_metrics_multi_training[i].get(metric_key, 0),
            )
            for i in range(batch_count)
        ]
        out = os.path.join(agg_dir, filename)

        # Safe x-labels
        labels = [str(v) for v in cumulative_total_sizes]
        if len(labels) < batch_count:
            labels.extend([str(i) for i in range(len(labels), batch_count)])

        positions, sparse_labels = _choose_sparse_xticks(batch_count, labels)

        plot_major_metrics_together(
            combined,
            out,
            title=f"{short_title}\n(Validation vs Training — Aggregated)",
            xvals=positions,
            xlabel="Aggregated samples",
        )

    # Per-batch plots
    for metric_key, short_title, filename in COMPARISON_PLOT_METRICS:
        combined = [
            extract_comparison_for_plot(
                batch_metrics_multi[i].get(metric_key, 0),
                batch_metrics_multi_training[i].get(metric_key, 0),
            )
            for i in range(batch_count)
        ]
        out = os.path.join(batch_dir, filename.replace(".png", "_batch.png"))

        # Safe x-labels
        labels = [str(v) for v in stepping_total_sizes]
        if len(labels) < batch_count:
            labels.extend([str(i) for i in range(len(labels), batch_count)])

        positions, sparse_labels = _choose_sparse_xticks(batch_count, labels)

        plot_major_metrics_together(
            combined,
            out,
            title=f"{short_title}\n(Validation vs Training — Per-batch)",
            xvals=positions,
            xlabel="Batch",
        )


def plot_comparison_metrics_for_series(
    series_val,
    series_train,
    base_dir,
    xvals,
    start_index,
    batch_count,
    name_prefix,
):
    from base_utils import (
        COMPARISON_PLOT_METRICS,
        FN_RATE_METRIC,
        FP_RATE_METRIC,
        extract_comparison_for_plot,
    )

    if start_index is None:
        print(f"Skipping comparison {name_prefix}: not enough batches")
        return

    ensure_dir(base_dir)
    length = len(series_val)

    # Main comparison metrics
    for metric_key, short_title, base_filename in COMPARISON_PLOT_METRICS:
        filename = base_filename.replace(".png", f"_{name_prefix}.png")
        combined = [
            extract_comparison_for_plot(
                series_val[i].get(metric_key, 0),
                series_train[i].get(metric_key, 0),
            )
            for i in range(length)
        ]
        out = os.path.join(base_dir, filename)

        # Safe x-labels
        labels = [str(v) for v in xvals]
        if len(labels) < length:
            labels.extend([str(i) for i in range(len(labels), length)])
        positions, sparse_labels = _choose_sparse_xticks(length, labels)

        plot_major_metrics_together(
            combined,
            out,
            title=f"{short_title}\n(Validation vs Training — {name_prefix})",
            xvals=positions,
            xlabel="Batch",
        )

    # FN Rate
    fn_metric_key, fn_title = FN_RATE_METRIC
    fn_data = [
        extract_comparison_for_plot(
            series_val[i].get(fn_metric_key, 0),
            series_train[i].get(fn_metric_key, 0),
            f"Validation {fn_title}",
            f"Training {fn_title}",
        )
        for i in range(length)
    ]
    out1 = os.path.join(base_dir, f"train_val_fn_rate_{name_prefix}.png")

    labels = [str(v) for v in xvals]
    if len(labels) < length:
        labels.extend([str(i) for i in range(len(labels), length)])
    positions, sparse_labels = _choose_sparse_xticks(length, labels)

    plot_major_metrics_together(
        fn_data,
        out1,
        title=f"{fn_title}\n(Validation vs Training — {name_prefix})",
        xvals=positions,
        xlabel="Batch",
    )

    # FP Rate
    fp_metric_key, fp_title = FP_RATE_METRIC
    fp_data = [
        extract_comparison_for_plot(
            series_val[i].get(fp_metric_key, 0),
            series_train[i].get(fp_metric_key, 0),
            f"Validation {fp_title}",
            f"Training {fp_title}",
        )
        for i in range(length)
    ]
    out2 = os.path.join(base_dir, f"train_val_fp_rate_{name_prefix}.png")

    labels = [str(v) for v in xvals]
    if len(labels) < length:
        labels.extend([str(i) for i in range(len(labels), length)])
    positions, sparse_labels = _choose_sparse_xticks(length, labels)

    plot_major_metrics_together(
        fp_data,
        out2,
        title=f"{fp_title}\n(Validation vs Training — {name_prefix})",
        xvals=positions,
        xlabel="Batch",
    )


def plot_malware_fn_rate_comparison(
    cumul_metrics_multi,
    cumul_metrics_multi_training,
    base_dir,
    cumulative_total_sizes,
    batch_count,
):
    from base_utils import FN_RATE_METRIC, extract_comparison_for_plot

    agg_dir = ensure_dir(os.path.join(base_dir, "aggregated"))
    fn_metric_key, fn_title = FN_RATE_METRIC

    fn_rate_data = [
        extract_comparison_for_plot(
            cumul_metrics_multi[i].get(fn_metric_key, 0),
            cumul_metrics_multi_training[i].get(fn_metric_key, 0),
            f"Validation {fn_title}",
            f"Training {fn_title}",
        )
        for i in range(batch_count)
    ]
    out = os.path.join(agg_dir, "train_val_fn_rate.png")
    plot_major_metrics_together(
        fn_rate_data,
        out,
        title=f"{fn_title}\n(Validation vs Training — Aggregated)",
        xvals=cumulative_total_sizes,
        xlabel="Aggregated samples",
    )


def plot_malware_fp_over_predicted_comparison(
    cumul_metrics_multi,
    cumul_metrics_multi_training,
    base_dir,
    cumulative_total_sizes,
    batch_count,
):
    from base_utils import FP_RATE_METRIC, extract_comparison_for_plot

    agg_dir = ensure_dir(os.path.join(base_dir, "aggregated"))
    fp_metric_key, fp_title = FP_RATE_METRIC

    fp_rate_data = [
        extract_comparison_for_plot(
            cumul_metrics_multi[i].get(fp_metric_key, 0),
            cumul_metrics_multi_training[i].get(fp_metric_key, 0),
            f"Validation {fp_title}",
            f"Training {fp_title}",
        )
        for i in range(batch_count)
    ]
    out = os.path.join(agg_dir, "train_val_fp_rate.png")
    plot_major_metrics_together(
        fp_rate_data,
        out,
        title=f"{fp_title}\n(Validation vs Training — Aggregated)",
        xvals=cumulative_total_sizes,
        xlabel="Aggregated samples",
    )


def ensure_plot_subdirs(base_dir):
    subs = {}
    for name in ["per_batch", "aggregated", "last5", "last10", "last20"]:
        p = ensure_dir(os.path.join(base_dir, name))
        subs[name] = p
    return subs


def _plot_lastk_class_counts(series_per_class_k, outpath, title, xlabels=None):
    if not series_per_class_k:
        print(f"No class-counts to plot for {outpath}")
        return
    counts_series = []
    for entry in series_per_class_k:
        counts_series.append(
            {
                cls: int(entry[cls].get("TP", 0) + entry[cls].get("FN", 0))
                for cls in entry.keys()
            }
        )
    plot_counts_series(
        counts_series, outpath, title=title, xlabels=xlabels, xlabel="Batch"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot training performance metrics."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to training log file or directory",
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
        file_path = os.path.join(file_path, "training.log")
        print(f"[INFO] -f is a directory, using: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    folder_dir = ensure_dir(os.path.join(base_dir, "training", args.exp))
    # print(f"[INFO] Output folder: {folder_dir}")

    entries = read_training_batches(file_path)
    if not entries:
        print("[ERROR] No entries parsed; exiting.")
        return

    has_validation_data = detect_validation_split(entries)

    if has_validation_data:
        (
            batch_metrics_per_class,
            batch_metrics_multi,
            cumul_metrics_multi,
            cumul_metrics_per_class,
            batch_metrics_per_class_training,
            batch_metrics_multi_training,
            cumul_metrics_multi_training,
            cumul_metrics_per_class_training,
        ) = accumulate_training_metrics(entries, has_validation_data)
    else:
        (
            batch_metrics_per_class,
            batch_metrics_multi,
            cumul_metrics_multi,
            cumul_metrics_per_class,
        ) = accumulate_training_metrics(entries, has_validation_data)

    if has_validation_data:
        validation_dir = ensure_dir(os.path.join(folder_dir, "validation"))
        train_dir = ensure_dir(os.path.join(folder_dir, "training"))
        comparison_dir = ensure_dir(os.path.join(folder_dir, "comparison"))
        print(f"Validation plots will be saved to: {validation_dir}")
        print(f"Training plots will be saved to: {train_dir}")
        print(f"Comparison plots will be saved to: {comparison_dir}")
    else:
        train_dir = ensure_dir(os.path.join(folder_dir, "training"))
        validation_dir = None
        comparison_dir = None
        print(f"Training plots will be saved to: {train_dir}")

    def get_dir(dt):
        if dt == "validation" and validation_dir is not None:
            return validation_dir
        elif dt == "training":
            return train_dir
        else:
            return train_dir

    class_names = list(entries[0]["training_predicted"].keys())
    batch_count = len(entries)

    # ensure subdirs
    if validation_dir:
        ensure_plot_subdirs(validation_dir)
    ensure_plot_subdirs(train_dir)
    if comparison_dir:
        ensure_plot_subdirs(comparison_dir)

    # class counts
    batch_class_counts, cumul_class_counts_per_batch = calculate_class_counts(
        entries, "per_class", class_names
    )
    plot_counts_series(
        batch_class_counts,
        os.path.join(
            get_dir("validation" if has_validation_data else "training"),
            "per_batch",
            f"class_counts_batch_{'validation' if has_validation_data else 'training'}.png",
        ),
        title="Per-batch class counts\n(Number of samples per batch)",
        xlabels=[str(i) for i in range(batch_count)],
        xlabel="Batch",
    )
    plot_counts_series(
        cumul_class_counts_per_batch,
        os.path.join(
            get_dir("validation" if has_validation_data else "training"),
            "aggregated",
            f"class_counts_aggregated_{'validation' if has_validation_data else 'training'}.png",
        ),
        title="Aggregated class counts\n(Total samples seen so far)",
        xlabels=[str(i) for i in range(batch_count)],
        xlabel="Batch",
    )

    if has_validation_data:
        batch_class_counts_training, cumul_class_counts_training_per_batch = (
            calculate_class_counts(entries, "training_per_class", class_names)
        )
        plot_counts_series(
            batch_class_counts_training,
            os.path.join(
                get_dir("training"),
                "per_batch",
                "class_counts_batch_training.png",
            ),
            title="Per-batch class counts (Training)",
            xlabels=[str(i) for i in range(batch_count)],
            xlabel="Batch",
        )
        plot_counts_series(
            cumul_class_counts_training_per_batch,
            os.path.join(
                get_dir("training"),
                "aggregated",
                "class_counts_aggregated_training.png",
            ),
            title="Aggregated class counts (Training)",
            xlabels=[str(i) for i in range(batch_count)],
            xlabel="Batch",
        )

    # stepping sizes
    if has_validation_data:
        cumulative_sizes = []
        total = 0
        for entry in entries:
            size = entry.get("testing_size", 0)
            total += size
            cumulative_sizes.append(total)
        stepping_sizes = get_stepping_sizes(
            entries, batch_count, "testing_size"
        )
    else:
        cumulative_sizes = []
        total = 0
        for entry in entries:
            size = entry.get("training_size", entry.get("testing_size", 0))
            total += size
            cumulative_sizes.append(total)
        stepping_sizes = get_stepping_sizes(
            entries, batch_count, "training_size"
        )

    # malware & accuracy plots
    plot_malware_metrics(
        batch_metrics_multi,
        os.path.join(
            get_dir("validation" if has_validation_data else "training"),
            "per_batch",
            f"malicious_metrics_batch_{'validation' if has_validation_data else 'training'}.png",
        ),
        f"Malicious metrics (per-batch)\n({'Validation' if has_validation_data else 'Training'})",
        stepping_sizes,
        "Batch",
    )
    plot_malware_metrics(
        cumul_metrics_multi,
        os.path.join(
            get_dir("validation" if has_validation_data else "training"),
            "aggregated",
            f"malicious_metrics_aggregated_{'validation' if has_validation_data else 'training'}.png",
        ),
        f"Malicious metrics (Aggregated)\n({'Validation' if has_validation_data else 'Training'})",
        xvals=cumulative_sizes,
        xlabel="Aggregated samples",
    )
    plot_accuracy_metrics(
        batch_metrics_multi,
        os.path.join(
            get_dir("validation" if has_validation_data else "training"),
            "per_batch",
            f"accuracy_batch_{'validation' if has_validation_data else 'training'}.png",
        ),
        f"Benign-Malicious Acc (per-batch)\n({'Validation' if has_validation_data else 'Training'})",
        stepping_sizes,
        "Batch",
    )
    plot_accuracy_metrics(
        cumul_metrics_multi,
        os.path.join(
            get_dir("validation" if has_validation_data else "training"),
            "aggregated",
            f"accuracy_aggregated_{'validation' if has_validation_data else 'training'}.png",
        ),
        f"Benign-Malicious Acc (Aggregated)\n({'Validation' if has_validation_data else 'Training'})",
        xvals=cumulative_sizes,
        xlabel="Aggregated samples",
    )

    # sliding windows and last-k plots
    def make_lastk_and_plot(
        k, per_class_batch, multi_batch, base_dir, stepping_sizes_all, label
    ):
        series_per_class_k, series_multi_k, start_idx = (
            sliding_window_aggregated(
                per_class_batch, class_names, k, trim_to_full_window=True
            )
        )
        if start_idx is None or len(series_multi_k) == 0:
            print(
                f"[INFO] Not enough batches for last-{k} ({label}), skipping."
            )
            return None, None, None
        n = len(per_class_batch)
        xvals = list(range(start_idx, n))
        folder = os.path.join(base_dir, f"last{k}")
        ensure_dir(folder)
        plot_malware_metrics(
            series_multi_k,
            os.path.join(folder, f"malicious_metrics_last{k}_{label}.png"),
            f"Malicious metrics (last-{k})\n({label})",
            xvals,
            "Batch",
        )
        plot_accuracy_metrics(
            series_multi_k,
            os.path.join(folder, f"accuracy_last{k}_{label}.png"),
            f"Benign-Malicious Acc (last-{k})\n({label})",
            xvals,
            "Batch",
        )
        _plot_lastk_class_counts(
            series_per_class_k,
            os.path.join(folder, f"class_counts_last{k}_{label}.png"),
            title=f"Aggregated class counts (last-{k})\n({label})",
            xlabels=[str(i) for i in xvals],
        )
        return series_per_class_k, series_multi_k, start_idx

    label_val = "validation" if has_validation_data else "training"
    last5_per_class_val, last5_multi_val, last5_start = make_lastk_and_plot(
        5,
        batch_metrics_per_class,
        batch_metrics_multi,
        get_dir("validation" if has_validation_data else "training"),
        stepping_sizes,
        label_val,
    )
    last10_per_class_val, last10_multi_val, last10_start = make_lastk_and_plot(
        10,
        batch_metrics_per_class,
        batch_metrics_multi,
        get_dir("validation" if has_validation_data else "training"),
        stepping_sizes,
        label_val,
    )
    last20_per_class_val, last20_multi_val, last20_start = make_lastk_and_plot(
        20,
        batch_metrics_per_class,
        batch_metrics_multi,
        get_dir("validation" if has_validation_data else "training"),
        stepping_sizes,
        label_val,
    )

    # training-specific plots
    if has_validation_data:
        cumulative_training_sizes = []
        total_training = 0
        for entry in entries:
            size = entry.get("training_size", 0)
            total_training += size
            cumulative_training_sizes.append(total_training)
        stepping_training_sizes = get_stepping_sizes(
            entries, batch_count, "training_size"
        )

        plot_malware_metrics(
            batch_metrics_multi_training,
            os.path.join(
                get_dir("training"),
                "per_batch",
                "malicious_metrics_batch_training.png",
            ),
            "Malicious metrics (per-batch)\n(Training)",
            stepping_training_sizes,
            "Batch",
        )
        plot_malware_metrics(
            cumul_metrics_multi_training,
            os.path.join(
                get_dir("training"),
                "aggregated",
                "malicious_metrics_aggregated_training.png",
            ),
            "Malicious metrics (Aggregated)\n(Training)",
            xvals=cumulative_training_sizes,
            xlabel="Aggregated samples",
        )
        plot_accuracy_metrics(
            batch_metrics_multi_training,
            os.path.join(
                get_dir("training"), "per_batch", "accuracy_batch_training.png"
            ),
            "Benign-Malicious Acc (per-batch)\n(Training)",
            stepping_training_sizes,
            "Batch",
        )
        plot_accuracy_metrics(
            cumul_metrics_multi_training,
            os.path.join(
                get_dir("training"),
                "aggregated",
                "accuracy_aggregated_training.png",
            ),
            "Benign-Malicious Acc (Aggregated)\n(Training)",
            xvals=cumulative_training_sizes,
            xlabel="Aggregated samples",
        )

        last5_per_class_train, last5_multi_train, last5_start_train = (
            make_lastk_and_plot(
                5,
                batch_metrics_per_class_training,
                batch_metrics_multi_training,
                get_dir("training"),
                stepping_training_sizes,
                "training",
            )
        )
        last10_per_class_train, last10_multi_train, last10_start_train = (
            make_lastk_and_plot(
                10,
                batch_metrics_per_class_training,
                batch_metrics_multi_training,
                get_dir("training"),
                stepping_training_sizes,
                "training",
            )
        )
        last20_per_class_train, last20_multi_train, last20_start_train = (
            make_lastk_and_plot(
                20,
                batch_metrics_per_class_training,
                batch_metrics_multi_training,
                get_dir("training"),
                stepping_training_sizes,
                "training",
            )
        )

        # comparison x-axis
        batch_total_sizes = [
            entry.get("training_size", 0) + entry.get("testing_size", 0)
            for entry in entries
        ]
        if batch_count < 10:
            stepping_total_sizes = [
                f"{i}\n{size}" for i, size in enumerate(batch_total_sizes)
            ]
        else:
            stepping_total_sizes = get_stepping_sizes(
                [{"dummy": 0}] * batch_count, batch_count, "dummy"
            )

        cumulative_total_sizes = []
        total_so_far = 0
        for size in batch_total_sizes:
            total_so_far += size
            cumulative_total_sizes.append(total_so_far)

        plot_comparison_metrics(
            batch_metrics_multi,
            cumul_metrics_multi,
            batch_metrics_multi_training,
            cumul_metrics_multi_training,
            comparison_dir,
            stepping_total_sizes,
            cumulative_total_sizes,
            batch_count,
        )
        plot_malware_fn_rate_comparison(
            cumul_metrics_multi,
            cumul_metrics_multi_training,
            comparison_dir,
            cumulative_total_sizes,
            batch_count,
        )
        plot_malware_fp_over_predicted_comparison(
            cumul_metrics_multi,
            cumul_metrics_multi_training,
            comparison_dir,
            cumulative_total_sizes,
            batch_count,
        )

        # comparison for last-k
        def maybe_plot_compare(
            last_multi_val,
            last_start_val,
            last_multi_train,
            last_start_train,
            kname,
        ):
            if (
                last_multi_val
                and last_start_val is not None
                and last_multi_train
                and last_start_train is not None
            ):
                start = max(last_start_val, last_start_train)
                offset_val = start - last_start_val
                offset_train = start - last_start_train
                len_val = len(last_multi_val) - offset_val
                len_train = len(last_multi_train) - offset_train
                common_len = min(len_val, len_train)
                if common_len <= 0:
                    print(f"No overlapping region for {kname} comparison.")
                    return
                slice_val = last_multi_val[
                    offset_val : offset_val + common_len
                ]
                slice_train = last_multi_train[
                    offset_train : offset_train + common_len
                ]
                xvals = list(range(start, start + common_len))
                base = os.path.join(comparison_dir, kname)
                plot_comparison_metrics_for_series(
                    slice_val,
                    slice_train,
                    base,
                    xvals,
                    start,
                    common_len,
                    kname,
                )

        maybe_plot_compare(
            last5_multi_val,
            last5_start,
            last5_multi_train,
            last5_start_train,
            "last5",
        )
        maybe_plot_compare(
            last10_multi_val,
            last10_start,
            last10_multi_train,
            last10_start_train,
            "last10",
        )
        maybe_plot_compare(
            last20_multi_val,
            last20_start,
            last20_multi_train,
            last20_start_train,
            "last20",
        )

    summary_txt = build_training_summary(
        args.exp,
        batch_count,
        has_validation_data,
        cumul_metrics_multi,
        cumul_metrics_per_class,
        cumul_metrics_multi_training if has_validation_data else None,
        cumul_metrics_per_class_training if has_validation_data else None,
    )
    summary_path = os.path.join(folder_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_txt)
    # print(f"[SAVED] {summary_path}")
    print(summary_txt)


if __name__ == "__main__":
    main()
