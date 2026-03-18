#!/usr/bin/env python3
"""Scatter plots for Optuna trials saved under experiments/<exp>/optuna."""

from __future__ import annotations

import ast
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
import yaml

try:  # Prefer package-relative import when available
    from .base_utils import ensure_dir  # type: ignore
except ImportError:  # Fallback for standalone execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from base_utils import ensure_dir  # type: ignore  # noqa: E402

MetricPair = Tuple[Optional[float], Optional[float]]
DEFAULT_METRICS = ("f1", "fpr")
DEFAULT_DIRECTIONS = ("maximize", "minimize")


@dataclass
class TrialFiles:
    """Paths associated with a single Optuna trial."""

    number: int
    directory: Path
    metrics_path: Optional[Path] = None
    context_path: Optional[Path] = None
    overrides_path: Optional[Path] = None


@dataclass
class CommandSpec:
    key: str
    name: str
    command_type: str
    prefixes: List[str]
    folder_name: str
    display_name: str


@dataclass
class TrialRecord:
    """Resolved metadata + metrics for plotting."""

    number: int
    classifier: str
    inner_classifier: Optional[str]
    scaler: Optional[str]
    pca_value: Optional[float]
    train: MetricPair
    test: MetricPair
    metrics_path: Optional[Path]
    per_command: Dict[str, MetricPair]


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


def _normalize_direction(value: str) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"max", "maximize", "maximization"}:
        return "maximize"
    if text in {"min", "minimize", "minimization"}:
        return "minimize"
    return None


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


def _first_existing(directory: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    return None


def _slugify(text: str) -> str:
    base = re.sub(r"[^0-9a-zA-Z]+", "_", (text or "").strip())
    base = base.strip("_") or "unnamed"
    return base.lower()


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
        context_path = child / "context.yaml"
        if not context_path.exists():
            context_path = None
        overrides_path = child / "overrides.yaml"
        if not overrides_path.exists():
            overrides_path = None
        records.append(
            TrialFiles(
                number=number,
                directory=child,
                metrics_path=metrics_path,
                context_path=context_path,
                overrides_path=overrides_path,
            )
        )
    return records


def _resolve_command_pair(
    payload: Dict,
    metric_names: Sequence[str],
    prefixes: Sequence[str],
    command_name: Optional[str] = None,
) -> MetricPair:
    if len(metric_names) < 2:
        return (None, None)
    metric_a, metric_b = metric_names[:2]
    # First, look inside command_metrics by name/alias when present; prefer this for multi-test setups.
    cmd_metrics = payload.get("command_metrics")
    if isinstance(cmd_metrics, dict):
        for key in [command_name, *(prefixes or [])]:
            if not key:
                continue
            block = cmd_metrics.get(key)
            if not isinstance(block, dict):
                continue
            x_val = block.get(metric_a)
            y_val = block.get(metric_b)
            if _is_number(x_val) or _is_number(y_val):
                return (
                    float(x_val) if _is_number(x_val) else None,
                    float(y_val) if _is_number(y_val) else None,
                )

    # Then fall back to legacy top-level keys like test_f1/test_fpr, train_f1/train_fpr
    for prefix in prefixes:
        x_val = _resolve_metric_value(payload, metric_a, prefix=prefix)
        y_val = _resolve_metric_value(payload, metric_b, prefix=prefix)
        if x_val is not None or y_val is not None:
            return (x_val, y_val)
    return (None, None)


def _load_summary(optuna_path: Path) -> Tuple[Dict, bool]:
    summary_path = optuna_path / "optuna_summary.json"
    if summary_path.is_file():
        with open(summary_path, "r") as fh:
            return json.load(fh), True
    return {}, False


def _parse_log_metrics(optuna_path: Path) -> Dict[int, Dict]:
    log_path = optuna_path / "optuna_trials.log"
    if not log_path.is_file():
        return {}
    pattern = re.compile(r"Finished trial (\d+)[^|]*\| metrics:\s*(\{.*\})\s*$")
    results: Dict[int, Dict] = {}
    for line in log_path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        trial_num = int(match.group(1))
        payload_text = match.group(2).strip()
        try:
            payload = ast.literal_eval(payload_text)
        except Exception:
            continue
        if isinstance(payload, dict):
            results[trial_num] = payload
    return results


def _ensure_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _load_config_metrics(optuna_path: Path) -> Tuple[List[str], Dict[str, str], Optional[Dict]]:
    experiment_root = optuna_path.parent
    if experiment_root == optuna_path:
        return [], {}, None
    candidates = (
        "config_effective.yaml",
        "config_effective.yml",
        "config_effective.json",
        "config.yaml",
        "config.yml",
        "config.json",
        "default_config.yaml",
        "default_config.yml",
        "default_config.json",
    )
    for name in candidates:
        cfg_path = experiment_root / name
        if not cfg_path.is_file():
            continue
        try:
            if cfg_path.suffix in {".yaml", ".yml"}:
                data = _read_yaml(cfg_path)
            elif cfg_path.suffix == ".json":
                data = _read_json(cfg_path)
            else:
                continue
        except Exception:
            continue
        optuna_block = data.get("optuna") if isinstance(data, dict) else None
        if not isinstance(optuna_block, dict):
            continue
        metrics_raw = _ensure_list(optuna_block.get("metric") or optuna_block.get("metrics"))
        directions_raw = _ensure_list(optuna_block.get("directions"))
        metrics: List[str] = []
        dir_map: Dict[str, str] = {}
        for idx, raw_metric in enumerate(metrics_raw):
            normalized = _normalize_metric_name(raw_metric)
            if not normalized:
                continue
            if normalized not in metrics:
                metrics.append(normalized)
            if idx < len(directions_raw):
                direction = _normalize_direction(directions_raw[idx])
                if direction:
                    dir_map[normalized] = direction
        return metrics, dir_map, data
    return [], {}, None


def _default_command_specs() -> List[CommandSpec]:
    return [
        CommandSpec(
            key="train",
            name="train",
            command_type="train",
            prefixes=["train"],
            folder_name="train",
            display_name="Train",
        ),
        CommandSpec(
            key="test",
            name="test",
            command_type="test",
            prefixes=["test"],
            folder_name="test",
            display_name="Test",
        ),
    ]


def _load_commands_spec(
    trial_files: Sequence[TrialFiles],
    fallback_config: Optional[Dict],
) -> List[CommandSpec]:
    def extract_commands(cfg: Optional[Dict]) -> Optional[List[Dict]]:
        if not isinstance(cfg, dict):
            return None
        commands = cfg.get("commands")
        return commands if isinstance(commands, list) else None

    commands_block: Optional[List[Dict]] = None
    for files in trial_files:
        if files.context_path and files.context_path.exists():
            cfg = _read_yaml(files.context_path)
            commands_block = extract_commands(cfg)
            if commands_block:
                break
    if not commands_block and fallback_config:
        commands_block = extract_commands(fallback_config)

    if not commands_block:
        return _default_command_specs()

    specs: List[CommandSpec] = []
    used_keys: Set[str] = set()
    train_count = 0
    test_count = 0
    other_counts: Dict[str, int] = {}

    for idx, cmd in enumerate(commands_block):
        name = cmd.get("name") if isinstance(cmd, dict) else None
        if not isinstance(name, str) or not name.strip():
            name = f"command_{idx}"
        display_name = name
        slug = _slugify(name)
        cmd_type = "custom"
        if isinstance(cmd, dict):
            cmd_type = str(cmd.get("command") or "custom").strip().lower() or "custom"

        if cmd_type == "train":
            train_count += 1
            key = "train" if train_count == 1 else f"train{train_count}"
        elif cmd_type == "test":
            test_count += 1
            key = "test" if test_count == 1 else f"test{test_count}"
        else:
            other_counts.setdefault(cmd_type, 0)
            other_counts[cmd_type] += 1
            key = slug or f"cmd{idx}"
            if other_counts[cmd_type] > 1:
                key = f"{key}{other_counts[cmd_type]}"

        base_key = key
        suffix = 2
        while key in used_keys:
            key = f"{base_key}{suffix}"
            suffix += 1
        used_keys.add(key)

        prefixes = [key]
        if slug and slug not in prefixes:
            prefixes.append(slug)
        if cmd_type == "train" and "train" not in prefixes:
            prefixes.append("train")
        if cmd_type == "test" and "test" not in prefixes:
            prefixes.append("test")
        prefixes = [p for p in prefixes if p]

        specs.append(
            CommandSpec(
                key=key,
                name=name,
                command_type=cmd_type,
                prefixes=prefixes,
                folder_name=slug or key,
                display_name=display_name,
            )
        )

    return specs or _default_command_specs()


def _load_trial_payload(
    files: TrialFiles,
    log_metrics: Dict[int, Dict],
    cache: Dict[int, Dict],
) -> Optional[Dict]:
    if files.number in cache:
        return cache[files.number]
    payload: Optional[Dict] = None
    path = files.metrics_path
    if path and path.exists():
        try:
            payload = _read_json(path)
        except Exception:
            payload = None
    if payload is None:
        payload = log_metrics.get(files.number)
    if payload is not None:
        cache[files.number] = payload
    return payload


def _infer_metric_names_from_trials(
    trial_files: Sequence[TrialFiles],
    log_metrics: Dict[int, Dict],
    payload_cache: Dict[int, Dict],
) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for files in trial_files:
        payload = _load_trial_payload(files, log_metrics, payload_cache)
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


def _extract_scaler_choice(context_path: Optional[Path], overrides_path: Optional[Path]) -> Optional[str]:
    """Determine which scaler/preprocessing type was used in the trial.

    Looks at preprocessing.steps[].type in the saved context, falling back to overrides.
    Picks the first step whose type name contains 'scaler' (case-insensitive).
    """

    def pick_from_steps(steps):
        if not isinstance(steps, list):
            return None
        for step in steps:
            if not isinstance(step, dict):
                continue
            stype = step.get("type")
            if isinstance(stype, str) and "scaler" in stype.lower():
                return stype
        return None

    context = _read_yaml(context_path) if context_path else {}
    overrides = _read_yaml(overrides_path) if overrides_path else {}

    if isinstance(context, dict):
        prep = context.get("preprocessing")
        if isinstance(prep, dict):
            scaler = pick_from_steps(prep.get("steps"))
            if scaler:
                return scaler

    # Look for explicit override keys like preprocessing.steps[0].type
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if not isinstance(key, str):
                continue
            if key.lower().endswith(".type") and "preprocessing.steps" in key:
                if isinstance(value, str) and "scaler" in value.lower():
                    return value

    return None


def _extract_pca_value(context_path: Optional[Path], overrides_path: Optional[Path]) -> Optional[float]:
    """Extract PCA n_components (or similar) from preprocessing steps.

    Looks for a preprocessing step whose type contains 'pca' (case-insensitive) and returns its
    n_components if available. Falls back to overrides when present.
    """

    def pick_pca_from_steps(steps):
        if not isinstance(steps, list):
            return None
        for step in steps:
            if not isinstance(step, dict):
                continue
            stype = step.get("type")
            if not isinstance(stype, str) or "pca" not in stype.lower():
                continue
            params = step.get("params") or {}
            if isinstance(params, dict) and "n_components" in params:
                val = params.get("n_components")
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    context = _read_yaml(context_path) if context_path else {}
    overrides = _read_yaml(overrides_path) if overrides_path else {}

    def _to_float(value: object) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Trials store the tuned value here; prefer it when present.
    if isinstance(overrides, dict):
        # Most trials store this as a flattened override key.
        raw = overrides.get("preprocessing.pca.n_components")
        val = _to_float(raw)
        if val is not None:
            return val

        # Backward compatibility with an older nested-style key.
        raw = overrides.get("preprocessing.pca.params.n_components")
        val = _to_float(raw)
        if val is not None:
            return val

    if isinstance(context, dict):
        prep = context.get("preprocessing")
        if isinstance(prep, dict):
            pca_block = prep.get("pca") if isinstance(prep.get("pca"), dict) else None
            if pca_block:
                # Current context schema: preprocessing.pca.n_components
                val = _to_float(pca_block.get("n_components"))
                if val is not None:
                    return val

                # Backward compatibility with nested params style.
                params = pca_block.get("params") if isinstance(pca_block.get("params"), dict) else {}
                val = _to_float(params.get("n_components"))
                if val is not None:
                    return val
            val = pick_pca_from_steps(prep.get("steps"))
            if val is not None:
                return val

    return None


def _build_trial_record(
    files: TrialFiles,
    metric_names: Sequence[str],
    payload: Dict,
    command_specs: Sequence[CommandSpec],
    train_key: str,
    test_key: str,
) -> Optional[TrialRecord]:
    if not isinstance(payload, dict):
        return None

    status = payload.get("status")
    if isinstance(status, str) and status.strip().lower() == "failed":
        return None

    per_command: Dict[str, MetricPair] = {}
    for spec in command_specs:
        per_command[spec.key] = _resolve_command_pair(
            payload, metric_names, spec.prefixes, command_name=spec.name
        )

    has_any_metrics = any(
        (pair[0] is not None or pair[1] is not None) for pair in per_command.values()
    )
    if not has_any_metrics:
        return None

    classifier, inner_classifier = _extract_classifier_info(files.context_path, files.overrides_path)
    scaler = _extract_scaler_choice(files.context_path, files.overrides_path)
    pca_value = _extract_pca_value(files.context_path, files.overrides_path)

    return TrialRecord(
        number=files.number,
        classifier=classifier,
        inner_classifier=inner_classifier,
        scaler=scaler,
        pca_value=pca_value,
        train=per_command.get(train_key, (None, None)),
        test=per_command.get(test_key, (None, None)),
        metrics_path=files.metrics_path,
        per_command=per_command,
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


def _collect_phase_metrics(
    records: Sequence[TrialRecord],
    command_key: str,
) -> List[Tuple[TrialRecord, float, float]]:
    metrics: List[Tuple[TrialRecord, float, float]] = []
    for record in records:
        pair = record.per_command.get(command_key)
        if not pair:
            continue
        if pair[0] is None or pair[1] is None:
            continue
        metrics.append((record, pair[0], pair[1]))
    return metrics


def _dominates(
    lhs: Tuple[float, float],
    rhs: Tuple[float, float],
    directions: Sequence[str],
) -> bool:
    better_or_equal = True
    strictly_better = False
    for idx, direction in enumerate(directions):
        lv = lhs[idx]
        rv = rhs[idx]
        if direction == "maximize":
            if lv < rv:
                better_or_equal = False
                break
            if lv > rv:
                strictly_better = True
        else:  # minimize
            if lv > rv:
                better_or_equal = False
                break
            if lv < rv:
                strictly_better = True
    return better_or_equal and strictly_better


def _pareto_front(
    metrics: Sequence[Tuple[TrialRecord, float, float]],
    directions: Sequence[str],
) -> List[Tuple[TrialRecord, float, float]]:
    front: List[Tuple[TrialRecord, float, float]] = []
    for item in metrics:
        candidate = (item[1], item[2])
        dominated = False
        for incumbent in list(front):
            current = (incumbent[1], incumbent[2])
            if _dominates(current, candidate, directions):
                dominated = True
                break
            if _dominates(candidate, current, directions):
                front.remove(incumbent)
        if not dominated:
            front.append(item)
    return front


def _scatter_points(
    records: Sequence[TrialRecord],
    command_key: str,
    command_label: str,
    metric_names: Sequence[str],
    colors: Dict[str, str],
    markers: Dict[Optional[str], str],
    out_path: Path,
    annotate: bool = False,
    title: Optional[str] = None,
    highlight_trials: Optional[Set[int]] = None,
) -> bool:
    phase_metrics = _collect_phase_metrics(records, command_key)

    if not phase_metrics:
        print(f"[WARN] No {command_label} metrics available; skipping plot {out_path.name}")
        return False

    # Compute axis limits with [0, 1] baseline and small padding
    xs = [x for _, x, _ in phase_metrics if x is not None]
    ys = [y for _, _, y in phase_metrics if y is not None]
    def _bounds(vals):
        lo = min([0.0] + vals) if vals else 0.0
        hi = max([1.0] + vals) if vals else 1.0
        pad = max((hi - lo) * 0.05, 0.02)
        return (lo - pad, hi + pad)
    xlim = _bounds(xs)
    ylim = _bounds(ys)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    highlight_trials = set(highlight_trials or [])

    for record, x_val, y_val in phase_metrics:
        is_highlight = record.number in highlight_trials
        ax.scatter(
            x_val,
            y_val,
            c=[colors.get(record.classifier, "#333333")],
            marker=markers.get(record.inner_classifier),
            s=90 if is_highlight else 70,
            edgecolor="#000000" if is_highlight else "#1f1f1f",
            linewidth=1.0 if is_highlight else 0.5,
            alpha=0.85,
            label=f"Trial {record.number}",
        )
        if annotate or is_highlight:
            ax.annotate(
                str(record.number),
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=9 if is_highlight else 8,
                fontweight="bold" if is_highlight else "normal",
            )

    ax.set_xlabel(f"{command_label} {metric_names[0]}")
    ax.set_ylabel(f"{command_label} {metric_names[1]}")
    default_title = f"Optuna trials ({command_label} metrics)"
    ax.set_title(title or default_title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", linewidth=0.6)

    plotted_records = [item[0] for item in phase_metrics]
    used_color_labels: List[str] = []
    for record in plotted_records:
        label = record.classifier
        if label in colors and label not in used_color_labels:
            used_color_labels.append(label)
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
        for label in used_color_labels
    ]

    used_marker_labels: List[Optional[str]] = []
    for record in plotted_records:
        label = record.inner_classifier
        if label in markers and label not in used_marker_labels:
            used_marker_labels.append(label)
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
        for label in used_marker_labels
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
    if marker_handles:
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


def _scatter_points_colormap(
    records: Sequence[TrialRecord],
    command_key: str,
    command_label: str,
    metric_names: Sequence[str],
    values_map: Dict[int, float],
    cmap_name: str,
    out_path: Path,
    annotate: bool = False,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
) -> bool:
    metrics: List[Tuple[TrialRecord, float, float]] = []
    colors: List[float] = []

    for record in records:
        pair = record.per_command.get(command_key)
        if not pair or pair[0] is None or pair[1] is None:
            continue
        if record.number not in values_map:
            continue
        metrics.append((record, pair[0], pair[1]))
        colors.append(values_map[record.number])

    if not metrics:
        print(f"[WARN] No {command_label} metrics with colormap values; skipping plot {out_path.name}")
        return False

    xs = [x for _, x, _ in metrics]
    ys = [y for _, _, y in metrics]

    def _bounds(vals):
        lo = min([0.0] + vals) if vals else 0.0
        hi = max([1.0] + vals) if vals else 1.0
        pad = max((hi - lo) * 0.05, 0.02)
        return (lo - pad, hi + pad)

    xlim = _bounds(xs)
    ylim = _bounds(ys)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Build a discrete colormap so integer PCA components are distinct but still ordered.
    max_val = max(colors)
    min_val = min(colors)
    # Default to a reasonable upper bound (common PCA max in this project is 11)
    upper = max(max_val, 11)
    lower = min(min_val, 1)

    # Create integer boundaries and center ticks on integers
    bounds = list(range(int(lower), int(upper) + 2))
    base_cmap = cm.get_cmap(cmap_name, len(bounds))
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=base_cmap.N, clip=True)

    sc = ax.scatter(
        xs,
        ys,
        c=colors,
        cmap=base_cmap,
        norm=norm,
        s=70,
        edgecolor="#1f1f1f",
        linewidth=0.5,
        alpha=0.88,
    )

    if annotate:
        for rec, x_val, y_val in metrics:
            ax.annotate(
                str(rec.number),
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=8,
            )

    cbar = plt.colorbar(sc, ax=ax, boundaries=bounds, ticks=bounds[:-1])
    cbar.set_label(colorbar_label or "PCA n_components")

    ax.set_xlabel(f"{command_label} {metric_names[0]}")
    ax.set_ylabel(f"{command_label} {metric_names[1]}")
    ax.set_title(title or f"Optuna trials ({command_label} metrics)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", linewidth=0.6)

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
    command_a_key: str,
    command_b_key: str,
    command_a_label: str,
    command_b_label: str,
    annotate: bool = False,
    title: Optional[str] = None,
) -> bool:
    deltas: List[Tuple[TrialRecord, float, float]] = []
    for record in records:
        pair_a = record.per_command.get(command_a_key)
        pair_b = record.per_command.get(command_b_key)
        if not pair_a or not pair_b:
            continue
        if None in (*pair_a, *pair_b):
            continue
        dx = pair_a[0] - pair_b[0]
        dy = pair_a[1] - pair_b[1]
        deltas.append((record, dx, dy))

    if not deltas:
        print(f"[WARN] No {command_a_label}/{command_b_label} pairs available; skipping plot {out_path.name}")
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
    axis_label = f"{command_a_label} - {command_b_label}"
    ax.set_xlabel(f"{axis_label} {metric_names[0]}")
    ax.set_ylabel(f"{axis_label} {metric_names[1]}")
    default_title = f"Optuna trials ({axis_label} deltas)"
    ax.set_title(title or default_title)
    ax.grid(True, linestyle=":", linewidth=0.6)

    used_color_labels: List[str] = []
    for record, _, _ in deltas:
        label = record.classifier
        if label in colors and label not in used_color_labels:
            used_color_labels.append(label)
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
        for label in used_color_labels
    ]

    used_marker_labels: List[Optional[str]] = []
    for record, _, _ in deltas:
        label = record.inner_classifier
        if label in markers and label not in used_marker_labels:
            used_marker_labels.append(label)
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
        for label in used_marker_labels
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
    if marker_handles:
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


def generate_optuna_trial_plots(
    optuna_dir: str,
    annotate: bool = False,
) -> List[Path]:
    optuna_path = Path(optuna_dir).expanduser().resolve()
    if optuna_path.name != "optuna" and (optuna_path / "optuna").is_dir():
        optuna_path = optuna_path / "optuna"
        print(f"[INFO] Using optuna folder: {optuna_path}")
    trial_files = _discover_trial_files(optuna_path)
    log_metrics = _parse_log_metrics(optuna_path)
    if not trial_files and not log_metrics:
        raise RuntimeError(f"No trial metrics found in {optuna_path}")
    if not trial_files and log_metrics:
        for number in sorted(log_metrics):
            trial_dir = optuna_path / f"trial_{number:04d}"
            trial_files.append(
                TrialFiles(
                    number=number,
                    directory=trial_dir,
                    metrics_path=None,
                    context_path=None,
                    overrides_path=None,
                )
            )

    summary, summary_present = _load_summary(optuna_path)
    config_metrics, config_dir_map, fallback_config = _load_config_metrics(optuna_path)
    payload_cache: Dict[int, Dict] = {}
    metric_names: List[str] = []
    for raw in summary.get("metric_names", []):
        text = str(raw).strip()
        if not text:
            continue
        normalized = _normalize_metric_name(text)
        if normalized not in metric_names:
            metric_names.append(normalized)

    if not summary_present:
        print("[WARN] optuna_summary.json missing; treating this as an incomplete run and inferring metrics from finished trials.")

    for name in config_metrics:
        normalized = _normalize_metric_name(name)
        if normalized and normalized not in metric_names:
            metric_names.append(normalized)

    if len(metric_names) < 2:
        inferred = _infer_metric_names_from_trials(
            trial_files,
            log_metrics,
            payload_cache,
        )
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

    command_specs = _load_commands_spec(trial_files, fallback_config) or _default_command_specs()
    pareto_trials: Set[int] = set()

    def pick_spec(command_type: str) -> Optional[CommandSpec]:
        for spec in command_specs:
            if spec.command_type == command_type:
                return spec
        return None

    if not command_specs:
        raise RuntimeError("Unable to determine any commands for plotting")

    train_spec = pick_spec("train") or command_specs[0]
    test_spec = pick_spec("test")
    if test_spec is None and len(command_specs) > 1:
        for candidate in command_specs:
            if candidate.key != train_spec.key:
                test_spec = candidate
                break
    pareto_source_spec = test_spec or train_spec

    summary_dirs = [
        dir_name
        for dir_name in (
            _normalize_direction(raw) for raw in summary.get("directions", [])
        )
        if dir_name
    ]
    directions: List[str] = []
    minimize_hints = {"fpr", "fnr", "fdr", "far", "loss", "error", "cost"}
    for idx, metric in enumerate(metric_names):
        if idx < len(summary_dirs):
            directions.append(summary_dirs[idx])
            continue
        config_direction = config_dir_map.get(metric)
        if config_direction:
            directions.append(config_direction)
            continue
        if metric in minimize_hints:
            directions.append("minimize")
        elif DEFAULT_DIRECTIONS:
            directions.append(DEFAULT_DIRECTIONS[idx % len(DEFAULT_DIRECTIONS)])
        else:
            directions.append("maximize")

    records: List[TrialRecord] = []
    for files in trial_files:
        payload = _load_trial_payload(files, log_metrics, payload_cache)
        if payload is None:
            continue
        record = _build_trial_record(
            files,
            metric_names,
            payload,
            command_specs,
            train_spec.key,
            test_spec.key if test_spec else train_spec.key,
        )
        if record:
            records.append(record)
    if not records:
        print(
            f"[WARN] No usable trial metrics were parsed in {optuna_path}; skipping plots."
        )
        return []

    # Compute Pareto set from the primary (pareto_source_spec) command before plotting so we can highlight across plots
    pareto_key = pareto_source_spec.key
    pareto_label = pareto_source_spec.display_name
    pareto_metrics = _collect_phase_metrics(records, pareto_key)
    if pareto_metrics:
        front = _pareto_front(pareto_metrics, directions)
        pareto_trials = {record.number for record, _, _ in front}
        print(
            f"[INFO] Identified {len(pareto_trials)} Pareto-optimal trials from {pareto_label} metrics"
        )
    else:
        print(
            f"[WARN] Unable to compute Pareto front because no {pareto_label} metrics were available"
        )

    front_records = [record for record in records if record.number in pareto_trials]

    colors = _build_color_map(record.classifier for record in records)
    markers = _build_marker_map(record.inner_classifier for record in records)

    visuals_dir = optuna_path / "visuals"
    commands_dir = visuals_dir / "commands"
    outputs: List[Path] = []

    for spec in command_specs:
        command_plot = commands_dir / spec.folder_name / f"{spec.key}_metrics.png"
        if _scatter_points(
            records,
            spec.key,
            spec.display_name,
            metric_names,
            colors,
            markers,
            command_plot,
            annotate=annotate,
            highlight_trials=pareto_trials,
        ):
            outputs.append(command_plot)

    classifier_dir = visuals_dir / "classifiers"
    classifier_records: Dict[str, List[TrialRecord]] = {}
    for record in records:
        classifier_records.setdefault(record.classifier, []).append(record)

    for classifier_name in sorted(classifier_records):
        subset = classifier_records[classifier_name]
        classifier_slug = _slugify(classifier_name)
        classifier_label = _short_label(classifier_name)
        for spec in command_specs:
            classifier_plot = classifier_dir / classifier_slug / f"{spec.key}_metrics.png"
            if _scatter_points(
                subset,
                spec.key,
                f"{spec.display_name} ({classifier_label})",
                metric_names,
                colors,
                markers,
                classifier_plot,
                annotate=annotate,
                highlight_trials=pareto_trials,
            ):
                outputs.append(classifier_plot)

    # Grouped plots per scaler choice (use same colors/markers; subset by scaler)
    scaler_dir = visuals_dir / "scalers"
    scaler_records: Dict[str, List[TrialRecord]] = {}
    for record in records:
        scaler_label = record.scaler or "(none)"
        scaler_records.setdefault(scaler_label, []).append(record)

    # Only plot scaler variants when more than one distinct scaler is present
    if len(scaler_records) > 1:
        for scaler_name in sorted(scaler_records):
            subset = scaler_records[scaler_name]
            scaler_slug = _slugify(scaler_name)
            scaler_label = _short_label(scaler_name)
            for spec in command_specs:
                scaler_plot = (
                    scaler_dir
                    / scaler_slug
                    / f"{spec.key}_metrics.png"
                )
                if _scatter_points(
                    subset,
                    spec.key,
                    f"{spec.display_name} ({scaler_label})",
                    metric_names,
                    colors,
                    markers,
                    scaler_plot,
                    annotate=annotate,
                    highlight_trials=pareto_trials,
                ):
                    outputs.append(scaler_plot)

    # PCA colormap plots (only if PCA values are present in any trial)
    pca_values = {record.number: record.pca_value for record in records if record.pca_value is not None}
    if pca_values:
        pca_dir = visuals_dir / "pca"
        for spec in command_specs:
            pca_plot = pca_dir / spec.folder_name / f"{spec.key}_metrics.png"
            if _scatter_points_colormap(
                records,
                spec.key,
                spec.display_name,
                metric_names,
                values_map=pca_values,
                cmap_name="viridis",
                out_path=pca_plot,
                annotate=annotate,
                title=f"Optuna trials ({spec.display_name} metrics) — PCA",
                colorbar_label="PCA n_components",
            ):
                outputs.append(pca_plot)

    delta_outputs_generated = False
    if test_spec and train_spec.key != test_spec.key:
        delta_dir = visuals_dir / "deltas"
        delta_plot = (
            delta_dir
            / f"{train_spec.folder_name}_minus_{test_spec.folder_name}_delta.png"
        )
        if _scatter_deltas(
            records,
            metric_names,
            colors,
            markers,
            delta_plot,
            command_a_key=train_spec.key,
            command_b_key=test_spec.key,
            command_a_label=train_spec.display_name,
            command_b_label=test_spec.display_name,
            annotate=annotate,
        ):
            outputs.append(delta_plot)
            delta_outputs_generated = True
    else:
        print("[WARN] Skipping delta plot because a complete train/test pair was not detected")

    if front_records:
        pareto_dir = visuals_dir / "pareto"
        train_front = pareto_dir / f"{train_spec.folder_name}_pareto.png"
        if _scatter_points(
            front_records,
            train_spec.key,
            train_spec.display_name,
            metric_names,
            colors,
            markers,
            train_front,
            annotate=True,
            title=f"Pareto front ({train_spec.display_name} metrics)",
        ):
            outputs.append(train_front)

        target_spec = pareto_source_spec or train_spec
        if target_spec.key != train_spec.key:
            target_front = pareto_dir / f"{target_spec.folder_name}_pareto.png"
            if _scatter_points(
                front_records,
                target_spec.key,
                target_spec.display_name,
                metric_names,
                colors,
                markers,
                target_front,
                annotate=True,
                title=f"Pareto front ({target_spec.display_name} metrics)",
            ):
                outputs.append(target_front)

        # If there is another test-like command distinct from the Pareto source, also plot its Pareto
        if test_spec and target_spec and test_spec.key != target_spec.key:
            alt_pareto_metrics = _collect_phase_metrics(records, test_spec.key)
            if alt_pareto_metrics:
                alt_front = _pareto_front(alt_pareto_metrics, directions)
                alt_trials = {record.number for record, _, _ in alt_front}
                alt_front_records = [r for r in records if r.number in alt_trials]
                if alt_front_records:
                    alt_front_path = pareto_dir / f"{test_spec.folder_name}_pareto.png"
                    if _scatter_points(
                        alt_front_records,
                        test_spec.key,
                        test_spec.display_name,
                        metric_names,
                        colors,
                        markers,
                        alt_front_path,
                        annotate=True,
                        title=f"Pareto front ({test_spec.display_name} metrics)",
                    ):
                        outputs.append(alt_front_path)

        if delta_outputs_generated and test_spec:
            delta_front = (
                pareto_dir
                / f"{train_spec.folder_name}_minus_{test_spec.folder_name}_delta_pareto.png"
            )
            if _scatter_deltas(
                front_records,
                metric_names,
                colors,
                markers,
                delta_front,
                command_a_key=train_spec.key,
                command_b_key=test_spec.key,
                command_a_label=train_spec.display_name,
                command_b_label=test_spec.display_name,
                annotate=True,
                title=f"Pareto front ({train_spec.display_name} minus {test_spec.display_name} deltas)",
            ):
                outputs.append(delta_front)
    else:
        print("[WARN] No Pareto-optimal trials were identified; skipping Pareto-only plots")

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
    generate_optuna_trial_plots(
        args.optuna_dir,
        annotate=args.annotate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
