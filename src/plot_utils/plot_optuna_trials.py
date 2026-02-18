#!/usr/bin/env python3
"""Scatter plots for Optuna trials saved under experiments/<exp>/optuna."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D

try:  # Prefer package-relative import when available
    from .base_utils import ensure_dir  # type: ignore
except ImportError:  # Fallback for standalone execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from base_utils import ensure_dir  # type: ignore  # noqa: E402

MetricPair = Tuple[Optional[float], Optional[float]]
DEFAULT_METRICS = ("f1", "fpr")


@dataclass
class TrialFiles:
    """Paths associated with a single Optuna trial."""

    number: int
    metrics_path: Path
    context_path: Optional[Path] = None
    overrides_path: Optional[Path] = None


@dataclass
class TrialRecord:
    """Resolved metadata + metrics for plotting."""

    number: int
    classifier: str
    inner_classifier: Optional[str]
    train: MetricPair
    test: MetricPair
    metrics_path: Path


_TRIAL_DIR_RE = re.compile(r"^trial_(\d+)$")


def _read_json(path: Path) -> Dict:
    with path.open("r") as handle:
        return json.load(handle)


def _read_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _normalize_metric_name(name: str) -> str:
    return str(name or "").strip().lower()


def _strip_metric_prefix(metric_key: str) -> str:
    text = _normalize_metric_name(metric_key)
    for prefix in ("train_", "test_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _metric_candidates(name: str) -> List[str]:
    normalized = _normalize_metric_name(name)
    collapsed = normalized.replace("_", "")
    return [name, normalized, collapsed]


def _resolve_metric_value(payload: Dict, metric_name: str, prefix: Optional[str]) -> Optional[float]:
    key = f"{prefix}_{metric_name}" if prefix else metric_name
    value = payload.get(key)
    if value is None and prefix == "test":
        value = payload.get(metric_name)
    return float(value) if _is_number(value) else None


def _first_existing(directory: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _first_existing(directory: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    return None


def _discover_trial_files(optuna_dir: Path) -> List[TrialFiles]:
    records: List[TrialFiles] = []
    for child in sorted(optuna_dir.iterdir()):
        if not child.is_dir():
            continue
        match = _TRIAL_DIR_RE.match(child.name)
        if not match:
            continue
        number = int(match.group(1))
        metrics_path = _first_existing(
            child,
            (
                "metrics.json",
                "results.json",
                "result.json",
                f"{child.name}_result.json",
            ),
        )
        if not metrics_path:
            continue
        context_path = child / "context.yaml"
        if not context_path.exists():
            context_path = None
        overrides_path = child / "overrides.yaml"
        if not overrides_path.exists():
            overrides_path = None
        records.append(
            TrialFiles(
                number=number,
                metrics_path=metrics_path,
                context_path=context_path,
                overrides_path=overrides_path,
            )
        )
    return records


def _load_summary(optuna_path: Path) -> Tuple[Dict, bool]:
    summary_path = optuna_path / "optuna_summary.json"
    if summary_path.is_file():
        with open(summary_path, "r") as fh:
            return json.load(fh), True
    return {}, False


def _infer_metric_names_from_trials(trial_files: Sequence[TrialFiles]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for files in trial_files:
        payload = _read_json(files.metrics_path)
        for key, value in payload.items():
            if not _is_number(value):
                continue
            base_name = _strip_metric_prefix(key)
            if not base_name or base_name in seen:
                continue
            seen.add(base_name)
            ordered.append(base_name)
    return ordered


def _extract_inner_classifier(params: Dict) -> Optional[str]:
    if not isinstance(params, dict):
        return None
    nested = params.get("model")
    if isinstance(nested, dict):
        nested_type = nested.get("type") or nested.get("classifier_type")
        if isinstance(nested_type, str):
            return nested_type
        deeper = nested.get("model")
        if isinstance(deeper, dict):
            nested_type = deeper.get("type")
            if isinstance(nested_type, str):
                return nested_type
    return None


def _extract_classifier_info(context_path: Optional[Path], overrides_path: Optional[Path]) -> Tuple[str, Optional[str]]:
    classifier = "UnknownClassifier"
    inner_classifier = None
    context = _read_yaml(context_path) if context_path else {}
    overrides = _read_yaml(overrides_path) if overrides_path else {}

    model_block = context.get("model") if isinstance(context, dict) else None
    if isinstance(model_block, dict):
        classifier = model_block.get("classifier_type", classifier)
        params = model_block.get("classifier_params", {})
        inner_classifier = _extract_inner_classifier(params)

    if (not classifier or classifier == "UnknownClassifier") and isinstance(overrides, dict):
        candidate = overrides.get("model.classifier_type")
        if isinstance(candidate, str):
            classifier = candidate

    if inner_classifier is None and isinstance(overrides, dict):
        for key, value in overrides.items():
            if isinstance(key, str) and key.endswith(".model.type") and isinstance(value, str):
                inner_classifier = value
                break
    return classifier or "UnknownClassifier", inner_classifier


def _build_trial_record(files: TrialFiles, metric_names: Sequence[str]) -> Optional[TrialRecord]:
    payload = _read_json(files.metrics_path)
    metric_a = metric_names[0]
    metric_b = metric_names[1]

    train_x = _resolve_metric_value(payload, metric_a, prefix="train")
    train_y = _resolve_metric_value(payload, metric_b, prefix="train")
    test_x = _resolve_metric_value(payload, metric_a, prefix="test")
    test_y = _resolve_metric_value(payload, metric_b, prefix="test")

    if train_x is None and train_y is None and test_x is None and test_y is None:
        return None

    classifier, inner_classifier = _extract_classifier_info(files.context_path, files.overrides_path)

    return TrialRecord(
        number=files.number,
        classifier=classifier,
        inner_classifier=inner_classifier,
        train=(train_x, train_y),
        test=(test_x, test_y),
        metrics_path=files.metrics_path,
    )


def _short_label(value: Optional[str]) -> str:
    if not value:
        return "(none)"
    if "." in value:
        return value.split(".")[-1]
    return value


def _build_color_map(labels: Iterable[str]) -> Dict[str, str]:
    unique = list(dict.fromkeys(labels))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not palette:
        palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B"]
    colors = {}
    for idx, label in enumerate(unique):
        colors[label] = palette[idx % len(palette)]
    return colors


def _build_marker_map(inner_labels: Iterable[Optional[str]]) -> Dict[Optional[str], str]:
    unique = list(dict.fromkeys(inner_labels))
    marker_cycle = [
        "o",
        "s",
        "^",
        "D",
        "P",
        "X",
        "v",
        "<",
        ">",
        "*",
        "h",
        "H",
    ]
    markers: Dict[Optional[str], str] = {}
    for idx, label in enumerate(unique):
        markers[label] = marker_cycle[idx % len(marker_cycle)]
    return markers


def _scatter_points(
    records: Sequence[TrialRecord],
    phase: str,
    metric_names: Sequence[str],
    colors: Dict[str, str],
    markers: Dict[Optional[str], str],
    out_path: Path,
    annotate: bool = False,
) -> bool:
    phase_metrics: List[Tuple[TrialRecord, float, float]] = []
    for record in records:
        pair = record.train if phase == "train" else record.test
        if pair[0] is None or pair[1] is None:
            continue
        phase_metrics.append((record, pair[0], pair[1]))

    if not phase_metrics:
        print(f"[WARN] No {phase} metrics available; skipping plot {out_path.name}")
        return False

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for record, x_val, y_val in phase_metrics:
        ax.scatter(
            x_val,
            y_val,
            c=[colors.get(record.classifier, "#333333")],
            marker=markers.get(record.inner_classifier),
            s=70,
            edgecolor="#1f1f1f",
            linewidth=0.5,
            alpha=0.85,
            label=f"Trial {record.number}",
        )
        if annotate:
            ax.annotate(
                str(record.number),
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=8,
            )

    ax.set_xlabel(f"{phase.title()} {metric_names[0]}")
    ax.set_ylabel(f"{phase.title()} {metric_names[1]}")
    ax.set_title(f"Optuna trials ({phase} metrics)")
    ax.grid(True, linestyle=":", linewidth=0.6)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label=_short_label(label),
            markerfacecolor=colors[label],
            markeredgecolor="#1f1f1f",
            markersize=8,
        )
        for label in colors
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[label],
            color="#4a4a4a",
            label=_short_label(label),
            markerfacecolor="#d9d9d9",
            markeredgecolor="#4a4a4a",
            markersize=8,
        )
        for label in markers
    ]
    legend1 = ax.legend(
        handles=color_handles,
        title="Classifier",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=8,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=marker_handles,
        title="Inner classifier",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        fontsize=8,
    )

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path}")
    return True


def _scatter_deltas(
    records: Sequence[TrialRecord],
    metric_names: Sequence[str],
    colors: Dict[str, str],
    markers: Dict[Optional[str], str],
    out_path: Path,
    annotate: bool = False,
) -> bool:
    deltas: List[Tuple[TrialRecord, float, float]] = []
    for record in records:
        train_pair = record.train
        test_pair = record.test
        if None in (*train_pair, *test_pair):
            continue
        dx = train_pair[0] - test_pair[0]
        dy = train_pair[1] - test_pair[1]
        deltas.append((record, dx, dy))

    if not deltas:
        print(f"[WARN] No train/test pairs available; skipping plot {out_path.name}")
        return False

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for record, dx, dy in deltas:
        ax.scatter(
            dx,
            dy,
            c=[colors.get(record.classifier, "#333333")],
            marker=markers.get(record.inner_classifier),
            s=70,
            edgecolor="#1f1f1f",
            linewidth=0.5,
            alpha=0.85,
        )
        if annotate:
            ax.annotate(
                str(record.number),
                (dx, dy),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=8,
            )

    ax.axvline(0.0, color="#9c9c9c", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#9c9c9c", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Train - Test {metric_names[0]}")
    ax.set_ylabel(f"Train - Test {metric_names[1]}")
    ax.set_title("Optuna trials (train minus test deltas)")
    ax.grid(True, linestyle=":", linewidth=0.6)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label=_short_label(label),
            markerfacecolor=colors[label],
            markeredgecolor="#1f1f1f",
            markersize=8,
        )
        for label in colors
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[label],
            color="#4a4a4a",
            label=_short_label(label),
            markerfacecolor="#d9d9d9",
            markeredgecolor="#4a4a4a",
            markersize=8,
        )
        for label in markers
    ]
    legend1 = ax.legend(
        handles=color_handles,
        title="Classifier",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=8,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=marker_handles,
        title="Inner classifier",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        fontsize=8,
    )

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path}")
    return True


def generate_optuna_trial_plots(optuna_dir: str, annotate: bool = False) -> List[Path]:
    optuna_path = Path(optuna_dir).expanduser().resolve()
    if optuna_path.name != "optuna" and (optuna_path / "optuna").is_dir():
        optuna_path = optuna_path / "optuna"
        print(f"[INFO] Using optuna folder: {optuna_path}")
    trial_files = _discover_trial_files(optuna_path)
    if not trial_files:
        raise RuntimeError(f"No trial metrics found in {optuna_path}")

    summary, summary_present = _load_summary(optuna_path)
    metric_names = []
    for raw in summary.get("metric_names", []):
        text = str(raw).strip()
        if not text:
            continue
        normalized = _normalize_metric_name(text)
        if normalized not in metric_names:
            metric_names.append(normalized)

    if not summary_present:
        print("[WARN] optuna_summary.json missing; treating this as an incomplete run and inferring metrics from finished trials.")

    if len(metric_names) < 2:
        inferred = _infer_metric_names_from_trials(trial_files)
        for name in inferred:
            normalized = _normalize_metric_name(name)
            if normalized and normalized not in metric_names:
                metric_names.append(normalized)

    if len(metric_names) < 2:
        for fallback in DEFAULT_METRICS:
            if fallback not in metric_names:
                metric_names.append(fallback)
            if len(metric_names) >= 2:
                break

    if len(metric_names) < 2:
        raise ValueError("Unable to determine at least two metric names for plotting")

    metric_names = metric_names[:2]

    records: List[TrialRecord] = []
    for files in trial_files:
        record = _build_trial_record(files, metric_names)
        if record:
            records.append(record)
    if not records:
        raise RuntimeError("No usable trial metrics were parsed")

    colors = _build_color_map(record.classifier for record in records)
    markers = _build_marker_map(
        record.inner_classifier for record in records
    )

    outputs: List[Path] = []
    train_plot = optuna_path / "optuna_trials_training.png"
    if _scatter_points(records, "train", metric_names, colors, markers, train_plot, annotate=annotate):
        outputs.append(train_plot)

    test_plot = optuna_path / "optuna_trials_testing.png"
    if _scatter_points(records, "test", metric_names, colors, markers, test_plot, annotate=annotate):
        outputs.append(test_plot)

    delta_plot = optuna_path / "optuna_trials_train_test_delta.png"
    if _scatter_deltas(records, metric_names, colors, markers, delta_plot, annotate=annotate):
        outputs.append(delta_plot)

    return outputs


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Optuna trial scatter charts.")
    parser.add_argument(
        "optuna_dir",
        help="Path to the optuna/ directory inside an experiment",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with trial numbers",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    generate_optuna_trial_plots(args.optuna_dir, annotate=args.annotate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
