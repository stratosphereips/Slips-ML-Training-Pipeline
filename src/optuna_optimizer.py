
import itertools
import json
from copy import deepcopy
from pathlib import Path
import time
import traceback

import optuna

from plot_utils.plot_optuna_trials import generate_optuna_trial_plots
from optuna_validator import validate_optuna_search_space


class OptunaOptimizer:
    SPEC_TYPES = {"int", "float", "categorical", "constant", "group"}

    def __init__(self, pipeline_cls, config_reader_cls, base_config, exp_dir, metric_names, search_space, n_trials=20, directions=("maximize", "minimize"), n_jobs=1, optuna_run_dir=None, pruner_config=None, git_commit=None):
        self.pipeline_cls = pipeline_cls
        self.config_reader_cls = config_reader_cls
        self.base_config = deepcopy(base_config)
        self.exp_dir = Path(exp_dir)
        self.optuna_run_dir = Path(optuna_run_dir) if optuna_run_dir else (self.exp_dir / "optuna")
        self.optuna_run_dir.mkdir(parents=True, exist_ok=True)
        self.optuna_log_path = self.optuna_run_dir / "optuna_trials.log"
        self._log_optuna(f"[Optuna] Initialized optimizer at {self.optuna_run_dir}")
        self.metric_names = metric_names
        self.search_space = deepcopy(search_space or {})
        validate_optuna_search_space(self.search_space)
        self.n_trials = n_trials
        self.directions = directions
        self.n_jobs = n_jobs
        self.pruner = self._create_pruner(pruner_config or {})
        if self.pruner is not None:
            self._log_optuna(f"Using pruner {self.pruner.__class__.__name__}")
        self.trials_log = []
        self.git_commit = git_commit

    def _log_optuna(self, msg):
        from datetime import datetime
        with open(self.optuna_log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")

    def _sample_hyperparameters(self, trial, effective_config):
        overrides = {}
        if isinstance(self.search_space, dict):
            self._walk_search_space(trial, self.search_space, [], effective_config, overrides)
        self._normalize_sampled_config(effective_config)
        return overrides

    def _walk_search_space(self, trial, node, path_tokens, config, overrides):
        if self._is_param_spec(node):
            if not self._spec_conditions_met(node, config):
                return
            label = self._path_label(path_tokens)
            value = self._suggest_param_value(trial, label, node)
            self._assign_path(config, path_tokens, deepcopy(value))
            overrides[label] = value
            if label == "model.classifier_type":
                config.setdefault("model", {})["classifier_params"] = {}
            return
        if isinstance(node, dict):
            parent_label = self._path_label(path_tokens)
            items = list(node.items())
            if parent_label == "model":
                items.sort(key=lambda kv: 0 if kv[0] == "classifier_type" else 1)
            for key, child in items:
                if key == "when":
                    continue
                if not isinstance(child, (dict, list)):
                    continue
                if self._should_skip_branch(parent_label, key, config):
                    continue
                self._walk_search_space(trial, child, path_tokens + [key], config, overrides)
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                self._walk_search_space(trial, child, path_tokens + [idx], config, overrides)

    def _is_param_spec(self, node):
        if not isinstance(node, dict):
            return False
        spec_type = node.get("type")
        return isinstance(spec_type, str) and spec_type in self.SPEC_TYPES

    def _spec_conditions_met(self, spec, config):
        condition = spec.get("when")
        if not condition:
            return True
        for path, expected in condition.items():
            current = self._get_path_value(config, path)
            if isinstance(expected, (list, tuple, set)):
                if current not in expected:
                    return False
            else:
                if current != expected:
                    return False
        return True

    def _should_skip_branch(self, parent_label, key, config):
        if parent_label.endswith("classifier_params") and isinstance(key, str):
            cls = self._get_path_value(config, "model.classifier_type")
            if cls and key != cls:
                return True
        return False

    def _suggest_param_value(self, trial, path, spec):
        ptype = spec.get("type")
        name = spec.get("name") or path
        if ptype == "int":
            return trial.suggest_int(name, spec["low"], spec["high"])
        if ptype == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        if ptype == "categorical":
            choices = spec["choices"]
            if spec.get("multiple"):
                combos = self._build_combinations(choices, spec.get("min"), spec.get("max"))
                if not combos:
                    raise ValueError(f"No combinations available for parameter '{name}'")
                encoded_choices = ["|".join(combo) for combo in combos]
                picked = trial.suggest_categorical(name, encoded_choices)
                return picked.split("|") if picked else []
            return trial.suggest_categorical(name, choices)
        if ptype == "constant":
            return deepcopy(spec.get("value"))
        if ptype == "group":
            group_vals = {}
            for child_name, child_spec in (spec.get("params") or {}).items():
                child_path = f"{path}.{child_name}" if path else child_name
                group_vals[child_name] = self._suggest_param_value(trial, child_path, child_spec)
            return group_vals
        raise ValueError(f"Unsupported hyperparameter spec for '{name}': {spec}")

    def _build_combinations(self, choices, min_len=None, max_len=None):
        if not choices:
            return []
        lo = 1 if min_len is None else int(min_len)
        hi = len(choices) if max_len is None else int(max_len)
        hi = min(hi, len(choices))
        lo = max(1, lo)
        combos = []
        for size in range(lo, hi + 1):
            combos.extend(itertools.combinations(choices, size))
        return combos

    def objective(self, trial):
        self._log_optuna(f"Starting trial {trial.number}")
        t0 = time.time()
        config = deepcopy(self.base_config)
        sampled_overrides = self._sample_hyperparameters(trial, config)
        trial_dir = self.optuna_run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        overrides_snapshot = deepcopy(sampled_overrides)

        import yaml

        trial_cfg_path = trial_dir / "overrides.yaml"
        with open(trial_cfg_path, "w") as f:
            yaml.safe_dump(overrides_snapshot, f)

        # Save trial context (datasets, mixer, etc.)
        trial_context_path = trial_dir / "context.yaml"

        def extract_trial_context(cfg):
            context = deepcopy(cfg)
            context.pop("optuna", None)
            return context

        trial_context = extract_trial_context(config)
        with open(trial_context_path, "w") as f:
            yaml.safe_dump(trial_context, f)
        # Run pipeline (train+val), collect metrics
        try:
            pipeline = self.pipeline_cls(
                config,
                optuna_trial=trial,
                optuna_dir=str(trial_dir),
                git_commit=self.git_commit,
            )
            metrics = pipeline.run_optuna_trial(optuna_trial=trial, optuna_dir=str(trial_dir))
        except Exception as exc:
            t_fail = time.time()
            tb = traceback.format_exc()
            self._log_optuna(
                f"Trial {trial.number} FAILED in {t_fail - t0:.1f}s | overrides: {overrides_snapshot} | error: {exc}\n{tb}"
            )
            raise
        # Save results
        trial_result_path = trial_dir / "metrics.json"
        with open(trial_result_path, "w") as f:
            json.dump(metrics, f, indent=2)
        t1 = time.time()
        self._log_optuna(f"Finished trial {trial.number} in {t1-t0:.1f}s | overrides: {overrides_snapshot} | metrics: {metrics}")
        # Log for summary
        self.trials_log.append({
            "trial": trial.number,
            "classifier_type": config.get("model", {}).get("classifier_type"),
            "overrides": overrides_snapshot,
            "metrics": metrics,
        })
        return tuple(metrics[m] for m in self.metric_names)

    # -----------------
    # path utilities
    # -----------------
    def _path_label(self, tokens):
        if not tokens:
            return ""
        parts = []
        for token in tokens:
            if isinstance(token, str):
                if parts:
                    parts.append(".")
                parts.append(token)
            else:
                parts.append(f"[{token}]")
        return "".join(parts)

    def _tokenize_path(self, path):
        tokens = []
        buf = ""
        i = 0
        while i < len(path):
            ch = path[i]
            if ch == '.':
                if buf:
                    tokens.append(buf)
                    buf = ""
                i += 1
            elif ch == '[':
                if buf:
                    tokens.append(buf)
                    buf = ""
                j = path.find(']', i)
                if j == -1:
                    raise ValueError(f"Invalid path syntax: {path}")
                tokens.append(int(path[i + 1 : j]))
                i = j + 1
            else:
                buf += ch
                i += 1
        if buf:
            tokens.append(buf)
        return tokens

    def _assign_path(self, root, tokens, value):
        cur = root
        for i, token in enumerate(tokens):
            last = i == len(tokens) - 1
            next_token = tokens[i + 1] if not last else None
            if last:
                if isinstance(token, int):
                    if not isinstance(cur, list):
                        raise TypeError(f"Expected list while assigning index {token}, got {type(cur).__name__}")
                    while len(cur) <= token:
                        cur.append(None)
                    cur[token] = value
                else:
                    if not isinstance(cur, dict):
                        raise TypeError(f"Expected dict while assigning '{token}', got {type(cur).__name__}")
                    cur[token] = value
            else:
                if isinstance(token, int):
                    if not isinstance(cur, list):
                        raise TypeError(f"Expected list while traversing index {token}, got {type(cur).__name__}")
                    while len(cur) <= token:
                        cur.append(None)
                    if cur[token] is None:
                        cur[token] = {} if isinstance(next_token, str) else []
                    cur = cur[token]
                else:
                    if not isinstance(cur, dict):
                        raise TypeError(f"Expected dict while traversing '{token}', got {type(cur).__name__}")
                    if token not in cur or cur[token] is None:
                        cur[token] = {} if isinstance(next_token, str) else []
                    cur = cur[token]

    def _get_path_value(self, root, path):
        tokens = self._tokenize_path(path)
        cur = root
        for token in tokens:
            if isinstance(token, int):
                if not isinstance(cur, list) or token >= len(cur):
                    return None
                cur = cur[token]
            else:
                if not isinstance(cur, dict) or token not in cur:
                    return None
                cur = cur[token]
        return cur

    def _normalize_sampled_config(self, config):
        self._flatten_classifier_params(config)

    def _flatten_classifier_params(self, config):
        model = config.get("model")
        if not isinstance(model, dict):
            return
        cls_name = model.get("classifier_type")
        params = model.get("classifier_params")
        if isinstance(params, dict) and isinstance(cls_name, str):
            branch = None
            if cls_name in params and isinstance(params[cls_name], dict):
                branch = params[cls_name]
            else:
                short = cls_name.split(".")[-1]
                if short in params and isinstance(params[short], dict):
                    branch = params[short]
            if branch is not None:
                model["classifier_params"] = deepcopy(branch)
                params = model["classifier_params"]
        if isinstance(params, dict):
            self._flatten_nested_model(params)

    def _flatten_nested_model(self, params):
        nested_model = params.get("model")
        if not isinstance(nested_model, dict):
            return
        nested_type = nested_model.get("type")
        nested_params = nested_model.get("params")
        if not isinstance(nested_params, dict):
            return
        resolved_params = nested_params
        if isinstance(nested_type, str):
            candidate = nested_params.get(nested_type)
            if not isinstance(candidate, dict):
                short = nested_type.split(".")[-1]
                candidate = nested_params.get(short)
            if isinstance(candidate, dict):
                resolved_params = candidate
        nested_model["params"] = deepcopy(resolved_params)

    def optimize(self):
        self._log_optuna("Creating new study...")
        study = optuna.create_study(directions=list(self.directions), pruner=self.pruner)
        self._log_optuna(f"Study created. Study name: {getattr(study, 'study_name', 'N/A')}")

        self._log_optuna(f"Starting optimization with {self.n_trials} trials, {self.n_jobs} parallel jobs...")
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        self._log_optuna("Study optimization finished.")
        # Save all trials summary
        trials_csv = self.optuna_run_dir / "optuna_trials.csv"
        import pandas as pd
        pd.DataFrame(self.trials_log).to_csv(trials_csv, index=False)
        # Save best params/summary
        summary = {
            "best_trials": [
                {"number": t.number, "values": t.values, "params": t.params}
                for t in study.best_trials
            ],
            "metric_names": self.metric_names,
            "directions": self.directions,
            "n_trials": self.n_trials,
            "n_jobs": self.n_jobs
        }
        self._log_optuna(f"Optimization completed. Best trials summary: {json.dumps(summary, indent=2)}")
        with open(self.optuna_run_dir / "optuna_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        try:
            generated = generate_optuna_trial_plots(str(self.optuna_run_dir))
            if generated:
                rel = ", ".join(str(path.name) for path in generated)
                self._log_optuna(f"Optuna trial plots saved: {rel}")
            else:
                self._log_optuna("Optuna trial plot generation produced no files (insufficient data)")
        except Exception as exc:
            self._log_optuna(f"Failed to generate Optuna trial plots: {exc}")
        return study

    def _create_pruner(self, cfg):
        if not cfg:
            return None
        pruner_type = str(cfg.get("type", "median")).lower()
        kwargs = {k: v for k, v in cfg.items() if k != "type"}
        if pruner_type == "median":
            return optuna.pruners.MedianPruner(**kwargs)
        if pruner_type in {"nop", "none", "disabled"}:
            return optuna.pruners.NopPruner()
        if pruner_type in {"successive_halving", "sha"}:
            return optuna.pruners.SuccessiveHalvingPruner(**kwargs)
        if pruner_type == "hyperband":
            return optuna.pruners.HyperbandPruner(**kwargs)
        raise ValueError(f"Unknown pruner type '{pruner_type}'")
