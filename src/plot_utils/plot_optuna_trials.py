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

# Ensure src/ is in sys.path for imports shared with the pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from base_utils import ensure_dir  # noqa: E402

MetricPair = Tuple[Optional[float], Optional[float]]


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
_RESULT_FILE_RE = re.compile(r"^trial[_-]?(\d+)_result\.json$")
_CONTEXT_FILE_RE = re.compile(r"^trial[_-]?(\d+)_context\.ya?ml$")
_OVERRIDES_FILE_RE = re.compile(r"^trial[_-]?(\d+)_(?:conf|overrides)\.ya?ml$")


def _read_json(path: Path) -> Dict:
    with path.open("r") as handle:
        return json.load(handle)


def _read_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _normalize_metric_name(name: str) -> str:
    cleaned = name.lower().replace(" ", "_").replace("-", "_")
    return cleaned.replace("__", "_")


def _metric_candidates(name: str) -> List[str]:
    normalized = _normalize_metric_name(name)
    collapsed = normalized.replace("_", "")
    return [name, normalized, collapsed]


def _resolve_metric_value(payload: Dict, metric_name: str, prefix: Optional[str]) -> Optional[float]:
    for candidate in _metric_candidates(metric_name):
        keys = [candidate]
        if prefix:
            keys.insert(0, f"{prefix}_{candidate}")
        for key in keys:
            if key in payload:
                value = payload[key]
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
    return None


def _first_existing(directory: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _discover_trial_files(optuna_dir: Path) -> List[TrialFiles]:
    entries: Dict[int, Dict[str, Optional[Path]]] = {}

    def ensure_entry(number: int) -> Dict[str, Optional[Path]]:
        if number not in entries:
            entries[number] = {
                "metrics_path": None,
                "context_path": None,
                "overrides_path": None,
            }
        return entries[number]

    for child in sorted(p for p in optuna_dir.iterdir() if p.is_dir()):
        match = _TRIAL_DIR_RE.match(child.name)
        if not match:
            continue
        number = int(match.group(1))
        entry = ensure_entry(number)
        if entry["metrics_path"] is None:
            entry["metrics_path"] = _first_existing(
                child,
                (
                    "metrics.json",
                    "results.json",
                    "result.json",
                    f"trial_{number:04d}_result.json",
                ),
            )
        entry["context_path"] = entry["context_path"] or (child / "context.yaml" if (child / "context.yaml").exists() else None)
        entry["overrides_path"] = entry["overrides_path"] or (child / "overrides.yaml" if (child / "overrides.yaml").exists() else None)

    for child in sorted(p for p in optuna_dir.iterdir() if p.is_file()):
        name = child.name
        metrics_match = _RESULT_FILE_RE.match(name)
        if metrics_match:
            number = int(metrics_match.group(1))
            entry = ensure_entry(number)
            entry["metrics_path"] = entry["metrics_path"] or child
            continue
        context_match = _CONTEXT_FILE_RE.match(name)
        if context_match:
            number = int(context_match.group(1))
            entry = ensure_entry(number)
            entry["context_path"] = entry["context_path"] or child
            continue
        overrides_match = _OVERRIDES_FILE_RE.match(name)
        if overrides_match:
            number = int(overrides_match.group(1))
            entry = ensure_entry(number)
            entry["overrides_path"] = entry["overrides_path"] or child

    records: List[TrialFiles] = []
    for number in sorted(entries):
        data = entries[number]
        metrics_path = data["metrics_path"]
        if metrics_path is None:
            continue
        records.append(
            TrialFiles(
                number=number,
                metrics_path=metrics_path,
                context_path=data["context_path"],
                overrides_path=data["overrides_path"],
            )
        )
    return records


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
        loc="best",
        frameon=True,
        fontsize=8,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=marker_handles,
        title="Inner classifier",
        loc="lower right",
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
        loc="best",
        frameon=True,
        fontsize=8,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=marker_handles,
        title="Inner classifier",
        loc="lower right",
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
    summary_path = optuna_path / "optuna_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing optuna_summary.json in {optuna_path}")

    summary = _read_json(summary_path)
    metric_names = summary.get("metric_names") or []
    if len(metric_names) < 2:
        raise ValueError("optuna_summary.json must list at least two metric_names")

    trial_files = _discover_trial_files(optuna_path)
    if not trial_files:
        raise RuntimeError(f"No trial metrics found in {optuna_path}")

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
