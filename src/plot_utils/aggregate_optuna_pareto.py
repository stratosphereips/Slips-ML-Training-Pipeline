#!/usr/bin/env python3
"""Aggregate Optuna trials across experiments and plot per-test-set Pareto fronts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import yaml

    def _yaml_load(path: Path) -> Dict:
        with path.open("r") as handle:
            data = yaml.safe_load(handle)
        return data or {}

except ImportError:
    from ruamel.yaml import YAML

    _YAML = YAML(typ="safe")

    def _yaml_load(path: Path) -> Dict:
        with path.open("r") as handle:
            data = _YAML.load(handle)
        return data or {}

try:
    from .base_utils import ensure_dir  # type: ignore
    from . import plot_optuna_trials as pot  # type: ignore
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from base_utils import ensure_dir  # type: ignore  # noqa: E402
    import plot_utils.plot_optuna_trials as pot  # type: ignore  # noqa: E402


@dataclass
class ExperimentBundle:
    experiment_name: str
    experiment_dir: Path
    optuna_dir: Path
    metric_names: List[str]
    directions: List[str]
    test_commands: List[Dict]
    records: List[pot.TrialRecord]


@dataclass
class TrialEnvelope:
    bundle: ExperimentBundle
    record: pot.TrialRecord
    trial_dir: Path
    command_key: str
    command_label: str
    command_signature_payload: Dict


def _read_yaml(path: Path) -> Dict:
    return _yaml_load(path)


def _normalize_dataset_values(values: Sequence) -> List[str]:
    return [str(v) for v in values]


def _extract_commands_block(trial_files: Sequence[pot.TrialFiles], fallback_config: Optional[Dict]) -> Optional[List[Dict]]:
    for files in trial_files:
        if not files.context_path or not files.context_path.exists():
            continue
        try:
            ctx = _read_yaml(files.context_path)
        except Exception:
            continue
        commands = ctx.get("commands") if isinstance(ctx.get("commands"), list) else None
        if commands:
            return commands

    if isinstance(fallback_config, dict):
        commands = fallback_config.get("commands") if isinstance(fallback_config.get("commands"), list) else None
        if commands:
            return commands

    return None


def _build_single_test_signature(command: Dict, fallback_name: str) -> Tuple[str, Dict]:
    mixer = command.get("mixer") if isinstance(command.get("mixer"), dict) else {}
    datasets = mixer.get("datasets") if isinstance(mixer.get("datasets"), list) else []
    payload = {
        "name": str(command.get("name") or fallback_name),
        "command": "test",
        "mixer_type": str(mixer.get("type") or ""),
        "datasets": _normalize_dataset_values(datasets),
    }
    signature = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return signature, payload


def _extract_test_commands(
    command_specs: Sequence[pot.CommandSpec],
    trial_files: Sequence[pot.TrialFiles],
    fallback_config: Optional[Dict],
) -> List[Dict]:
    commands_block = _extract_commands_block(trial_files, fallback_config) or []
    available_tests = [
        cmd
        for cmd in commands_block
        if isinstance(cmd, dict) and str(cmd.get("command") or "").strip().lower() == "test"
    ]
    used_idx: set = set()
    test_commands: List[Dict] = []

    for spec in command_specs:
        if spec.command_type != "test":
            continue

        matched_idx: Optional[int] = None
        for idx, cmd in enumerate(available_tests):
            if idx in used_idx:
                continue
            if str(cmd.get("name") or "").strip() == spec.name:
                matched_idx = idx
                break
        if matched_idx is None:
            for idx, _ in enumerate(available_tests):
                if idx not in used_idx:
                    matched_idx = idx
                    break

        if matched_idx is not None:
            used_idx.add(matched_idx)
            command = available_tests[matched_idx]
        else:
            command = {"name": spec.name, "command": "test", "mixer": {"type": "", "datasets": []}}

        signature, payload = _build_single_test_signature(command, fallback_name=spec.name)
        test_commands.append(
            {
                "key": spec.key,
                "label": spec.display_name,
                "signature": signature,
                "signature_payload": payload,
            }
        )

    return test_commands


def _resolve_metric_names_and_directions(
    optuna_path: Path,
    trial_files: Sequence[pot.TrialFiles],
    log_metrics: Dict[int, Dict],
    payload_cache: Dict[int, Dict],
) -> Tuple[List[str], List[str], Optional[Dict]]:
    summary, summary_present = pot._load_summary(optuna_path)
    config_metrics, config_dir_map, fallback_config = pot._load_config_metrics(optuna_path)

    metric_names: List[str] = []
    for raw in summary.get("metric_names", []):
        normalized = pot._normalize_metric_name(raw)
        if normalized and normalized not in metric_names:
            metric_names.append(normalized)

    if not summary_present:
        print(f"[WARN] Missing optuna_summary.json in {optuna_path}; inferring metrics from trials")

    for raw in config_metrics:
        normalized = pot._normalize_metric_name(raw)
        if normalized and normalized not in metric_names:
            metric_names.append(normalized)

    if len(metric_names) < 2:
        inferred = pot._infer_metric_names_from_trials(trial_files, log_metrics, payload_cache)
        for raw in inferred:
            normalized = pot._normalize_metric_name(raw)
            if normalized and normalized not in metric_names:
                metric_names.append(normalized)

    if len(metric_names) < 2:
        for fallback in pot.DEFAULT_METRICS:
            if fallback not in metric_names:
                metric_names.append(fallback)
            if len(metric_names) >= 2:
                break

    metric_names = metric_names[:2]
    summary_dirs = [
        direction
        for direction in (pot._normalize_direction(raw) for raw in summary.get("directions", []))
        if direction
    ]

    minimize_hints = {"fpr", "fnr", "fdr", "far", "loss", "error", "cost"}
    directions: List[str] = []
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
        else:
            directions.append(pot.DEFAULT_DIRECTIONS[idx % len(pot.DEFAULT_DIRECTIONS)])

    return metric_names, directions, fallback_config


def _pick_specs(command_specs: Sequence[pot.CommandSpec]) -> Tuple[pot.CommandSpec, Optional[pot.CommandSpec], pot.CommandSpec]:
    def pick_spec(command_type: str) -> Optional[pot.CommandSpec]:
        for spec in command_specs:
            if spec.command_type == command_type:
                return spec
        return None

    train_spec = pick_spec("train") or command_specs[0]
    test_spec = pick_spec("test")
    if test_spec is None and len(command_specs) > 1:
        for candidate in command_specs:
            if candidate.key != train_spec.key:
                test_spec = candidate
                break
    pareto_source = test_spec or train_spec
    return train_spec, test_spec, pareto_source


def _parse_experiment_bundle(exp_dir: Path) -> Optional[ExperimentBundle]:
    optuna_dir = exp_dir / "optuna"
    if not optuna_dir.is_dir():
        return None

    trial_files = pot._discover_trial_files(optuna_dir)
    log_metrics = pot._parse_log_metrics(optuna_dir)
    if not trial_files and not log_metrics:
        return None

    if not trial_files and log_metrics:
        for number in sorted(log_metrics):
            trial_dir = optuna_dir / f"trial_{number:04d}"
            trial_files.append(
                pot.TrialFiles(
                    number=number,
                    directory=trial_dir,
                    metrics_path=None,
                    context_path=None,
                    overrides_path=None,
                )
            )

    payload_cache: Dict[int, Dict] = {}
    metric_names, directions, fallback_config = _resolve_metric_names_and_directions(
        optuna_dir,
        trial_files,
        log_metrics,
        payload_cache,
    )

    if len(metric_names) < 2:
        print(f"[WARN] Skipping {exp_dir.name}: could not determine at least 2 metrics")
        return None

    command_specs = pot._load_commands_spec(trial_files, fallback_config) or pot._default_command_specs()
    train_spec, test_spec, _pareto_source = _pick_specs(command_specs)
    test_commands = _extract_test_commands(command_specs, trial_files, fallback_config)
    if not test_commands:
        print(f"[WARN] Skipping {exp_dir.name}: no test command signature detected")
        return None

    records: List[pot.TrialRecord] = []
    for files in trial_files:
        payload = pot._load_trial_payload(files, log_metrics, payload_cache)
        if payload is None:
            continue
        record = pot._build_trial_record(
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
        return None

    return ExperimentBundle(
        experiment_name=exp_dir.name,
        experiment_dir=exp_dir,
        optuna_dir=optuna_dir,
        metric_names=metric_names,
        directions=directions,
        test_commands=test_commands,
        records=records,
    )


def _record_to_row(env: TrialEnvelope, metric_names: Sequence[str], is_pareto: bool) -> Dict:
    pair = env.record.per_command.get(env.command_key, (None, None))
    return {
        "experiment": env.bundle.experiment_name,
        "trial": env.record.number,
        "classifier": env.record.classifier,
        "inner_classifier": env.record.inner_classifier,
        "scaler": env.record.scaler,
        "pca_n_components": env.record.pca_value,
        "test_command": env.command_signature_payload.get("name"),
        "test_mixer_type": env.command_signature_payload.get("mixer_type"),
        "test_datasets": env.command_signature_payload.get("datasets"),
        metric_names[0]: pair[0],
        metric_names[1]: pair[1],
        "pareto": is_pareto,
        "trial_dir": str(env.trial_dir),
    }


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_copytree(src: Path, dst: Path) -> Path:
    if not dst.exists():
        shutil.copytree(src, dst)
        return dst
    suffix = 2
    while True:
        candidate = dst.parent / f"{dst.name}_{suffix}"
        if not candidate.exists():
            shutil.copytree(src, candidate)
            return candidate
        suffix += 1


def _build_envelopes(bundle: ExperimentBundle, command_key: str, command_label: str, command_signature_payload: Dict) -> List[TrialEnvelope]:
    envelopes: List[TrialEnvelope] = []
    for record in bundle.records:
        trial_dir = record.metrics_path.parent if record.metrics_path else (bundle.optuna_dir / f"trial_{record.number:04d}")
        envelopes.append(
            TrialEnvelope(
                bundle=bundle,
                record=record,
                trial_dir=trial_dir,
                command_key=command_key,
                command_label=command_label,
                command_signature_payload=command_signature_payload,
            )
        )
    return envelopes


def _to_plot_records(envelopes: Sequence[TrialEnvelope], unified_key: str = "agg_test") -> List[pot.TrialRecord]:
    plot_records: List[pot.TrialRecord] = []
    for idx, env in enumerate(envelopes, start=1):
        pair = env.record.per_command.get(env.command_key, (None, None))
        plot_records.append(
            pot.TrialRecord(
                number=idx,
                classifier=env.record.classifier,
                inner_classifier=env.record.inner_classifier,
                scaler=env.record.scaler,
                pca_value=env.record.pca_value,
                train=(None, None),
                test=(None, None),
                metrics_path=env.record.metrics_path,
                per_command={unified_key: pair},
            )
        )
    return plot_records


def _parse_metric_filter_value(raw: str) -> Tuple[float, float]:
    try:
        parsed = ast.literal_eval(raw)
    except Exception as exc:
        raise ValueError(f"Invalid range tuple '{raw}'. Expected format like '(0.8, 1.0)'.") from exc

    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"Invalid range tuple '{raw}'. Expected exactly two values.")

    lo, hi = parsed
    try:
        lo_val = float(lo)
        hi_val = float(hi)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid range tuple '{raw}'. Both min/max must be numeric.") from exc

    if lo_val > hi_val:
        raise ValueError(f"Invalid range tuple '{raw}'. Min cannot be greater than max.")
    return lo_val, hi_val


def _parse_metric_filters(extra_args: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    filters: Dict[str, Tuple[float, float]] = {}
    idx = 0
    while idx < len(extra_args):
        flag = extra_args[idx]
        if not flag.startswith("--"):
            raise ValueError(f"Unexpected argument '{flag}'. Metric filters must be passed as --<metric> '(min,max)'.")
        metric_name = flag[2:].strip().lower()
        if not metric_name:
            raise ValueError(f"Invalid metric flag '{flag}'.")
        if idx + 1 >= len(extra_args):
            raise ValueError(f"Missing tuple value for filter '{flag}'.")
        raw_range = extra_args[idx + 1]
        if raw_range.startswith("--"):
            raise ValueError(f"Missing tuple value for filter '{flag}'.")
        filters[metric_name] = _parse_metric_filter_value(raw_range)
        idx += 2
    return filters


def _passes_metric_filters(
    env: TrialEnvelope,
    metric_names: Sequence[str],
    metric_filters: Dict[str, Tuple[float, float]],
) -> bool:
    if not metric_filters:
        return True
    pair = env.record.per_command.get(env.command_key)
    if not pair:
        return False

    for metric_idx, metric_name in enumerate(metric_names[:2]):
        bounds = metric_filters.get(metric_name)
        if bounds is None:
            continue
        value = pair[metric_idx]
        if value is None:
            return False
        lo, hi = bounds
        if value < lo or value > hi:
            return False
    return True


def aggregate_optuna_pareto(
    experiments_dir: Path,
    annotate: bool = True,
    metric_filters: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Path:
    experiments_dir = experiments_dir.expanduser().resolve()
    if not experiments_dir.is_dir():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    metric_filters = metric_filters or {}

    bundles: List[ExperimentBundle] = []
    for child in sorted(experiments_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("summar_runs_"):
            continue
        bundle = _parse_experiment_bundle(child)
        if bundle is None:
            continue
        bundles.append(bundle)

    if not bundles:
        raise RuntimeError("No Optuna experiment runs found with usable test command signatures")

    grouped: Dict[str, Dict] = {}
    for bundle in bundles:
        for command in bundle.test_commands:
            group_key_payload = {
                "test_command": command["signature_payload"],
                "metric_names": bundle.metric_names,
                "directions": bundle.directions,
            }
            group_key = json.dumps(group_key_payload, sort_keys=True, separators=(",", ":"))
            if group_key not in grouped:
                grouped[group_key] = {
                    "command_signature_payload": command["signature_payload"],
                    "metric_names": bundle.metric_names,
                    "directions": bundle.directions,
                    "members": [],
                }
            grouped[group_key]["members"].append(
                {
                    "bundle": bundle,
                    "command_key": command["key"],
                    "command_label": command["label"],
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_root = experiments_dir / f"summar_runs_{timestamp}"
    ensure_dir(summary_root)

    signatures_sorted = sorted(grouped.keys())
    aggregate_index: List[Dict] = []

    for idx, signature in enumerate(signatures_sorted, start=1):
        set_name = f"test_set_{idx}"
        set_dir = summary_root / set_name
        ensure_dir(set_dir)

        group = grouped[signature]
        members = group["members"]
        metric_names = group["metric_names"]
        directions = group["directions"]
        command_signature_payload = group["command_signature_payload"]
        pareto_label = command_signature_payload.get("name") or members[0]["command_label"]

        envelopes: List[TrialEnvelope] = []
        for member in members:
            envelopes.extend(
                _build_envelopes(
                    member["bundle"],
                    member["command_key"],
                    member["command_label"],
                    command_signature_payload,
                )
            )

        filtered_envelopes = [
            env for env in envelopes if _passes_metric_filters(env, metric_names, metric_filters)
        ]

        metrics: List[Tuple[TrialEnvelope, float, float]] = []
        for env in filtered_envelopes:
            pair = env.record.per_command.get(env.command_key)
            if not pair:
                continue
            if pair[0] is None or pair[1] is None:
                continue
            metrics.append((env, pair[0], pair[1]))

        front = pot._pareto_front(metrics, directions) if metrics else []
        pareto_envs = [item[0] for item in front]

        plot_records = _to_plot_records(pareto_envs)
        colors = pot._build_color_map(record.classifier for record in plot_records)
        markers = pot._build_marker_map(record.inner_classifier for record in plot_records)

        pareto_plot_path = set_dir / "visuals" / "pareto" / "aggregated_test_pareto.png"
        if pareto_envs:
            pot._scatter_points(
                plot_records,
                "agg_test",
                pareto_label,
                metric_names,
                colors,
                markers,
                pareto_plot_path,
                annotate=annotate,
                title=f"Pareto front (aggregated {pareto_label} metrics)",
            )
        else:
            print(f"[WARN] No Pareto-optimal records for {set_name}; skipping Pareto plot")

        pareto_env_ids = {id(env) for env in pareto_envs}
        all_rows = [_record_to_row(env, metric_names, is_pareto=(id(env) in pareto_env_ids)) for env in filtered_envelopes]
        pareto_rows = [row for row in all_rows if row.get("pareto")]

        _write_csv(set_dir / "summary" / "all_trials.csv", all_rows)
        _write_csv(set_dir / "summary" / "pareto_trials.csv", pareto_rows)

        with (set_dir / "summary" / "all_trials.json").open("w") as handle:
            json.dump(all_rows, handle, indent=2)
        with (set_dir / "summary" / "pareto_trials.json").open("w") as handle:
            json.dump(pareto_rows, handle, indent=2)

        signature_payload = command_signature_payload
        with (set_dir / "summary" / "test_signature.json").open("w") as handle:
            json.dump(signature_payload, handle, indent=2)

        copied_entries: List[Dict] = []
        model_root = set_dir / "pareto_models"
        ensure_dir(model_root)
        for env in pareto_envs:
            src = env.trial_dir
            if not src.is_dir():
                continue
            dst_name = f"{env.bundle.experiment_name}__trial_{env.record.number:04d}"
            dst = _safe_copytree(src, model_root / dst_name)
            copied_entries.append(
                {
                    "experiment": env.bundle.experiment_name,
                    "trial": env.record.number,
                    "source": str(src),
                    "copied_to": str(dst),
                }
            )

        with (set_dir / "summary" / "copied_pareto_models.json").open("w") as handle:
            json.dump(copied_entries, handle, indent=2)

        aggregate_index.append(
            {
                "test_set": set_name,
                "signature": signature_payload,
                "experiments": sorted({member["bundle"].experiment_name for member in members}),
                "metric_names": metric_names,
                "directions": directions,
                "pareto_metric_source": pareto_label,
                "total_trials": len(filtered_envelopes),
                "pareto_trials": len(pareto_envs),
                "metric_filters": {k: [v[0], v[1]] for k, v in metric_filters.items()},
                "plot": str(pareto_plot_path),
            }
        )

        print(
            f"[INFO] {set_name}: command={pareto_label} experiments={len(set(member['bundle'].experiment_name for member in members))} total_trials={len(filtered_envelopes)} pareto_trials={len(pareto_envs)}"
        )

    with (summary_root / "index.json").open("w") as handle:
        json.dump(aggregate_index, handle, indent=2)

    print(f"[INFO] Aggregated summary directory: {summary_root}")
    return summary_root


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Optuna experiments by test-set signature and plot Pareto fronts."
    )
    parser.add_argument(
        "experiments_dir",
        nargs="?",
        default="./experiments",
        help="Path to experiments root directory (default: ./experiments)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable point annotations in Pareto plots",
    )
    args, extra_args = parser.parse_known_args(argv)
    args.metric_filters = _parse_metric_filters(extra_args)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    aggregate_optuna_pareto(
        experiments_dir=Path(args.experiments_dir),
        annotate=not args.no_annotate,
        metric_filters=args.metric_filters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
