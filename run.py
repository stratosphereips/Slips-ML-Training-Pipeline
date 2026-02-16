#!/usr/bin/env python3

import sys
from pathlib import Path
import traceback
import yaml
import json
import argparse
import subprocess
from datetime import datetime

# Insert src/ at front of sys.path so modules inside it can use relative imports
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

# Now import the pipeline module (the module lives in src/)

from pipeline import PipelineRunner  # imports src/pipeline.py as module 'pipeline'

from conf_reader import ConfigReader
from optuna_optimizer import OptunaOptimizer


def _log_optuna_message(log_dir: Path, message: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "optuna_trials.log"
    with open(log_path, "a") as log_file:
        log_file.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def _get_git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _has_tunable_params(search_space) -> bool:
    def _spec_has_params(spec):
        if not isinstance(spec, dict):
            return False
        if spec.get("choices"):
            return True
        if "type" in spec:
            return True
        return any(_spec_has_params(sub) for sub in spec.values() if isinstance(sub, dict))

    if not isinstance(search_space, dict):
        return False
    return any(_spec_has_params(spec) for spec in search_space.values())


def main(config_path: str = "./default_config.yaml", optuna_mode: bool = False):
    """
    Initialize and run the pipeline using the specified config file.
    Returns 0 on success, 1 on failure.
    """

    # Common setup: create unique experiment folder and update config
    cfg_reader = ConfigReader(config_path)
    base_config = cfg_reader.load()
    exp_base_dir = base_config.get("paths", {}).get("experiment_dir", "./experiments")
    exp_name = base_config["experiment_name"]
    exp_dir = Path(exp_base_dir) / exp_name
    # Find next available experiment folder (exp_name, exp_name_1, ...)
    if not exp_dir.exists():
        unique_exp_dir = exp_dir
    else:
        idx = 1
        while (Path(exp_base_dir) / f"{exp_name}_{idx}").exists():
            idx += 1
        unique_exp_dir = Path(exp_base_dir) / f"{exp_name}_{idx}"
    unique_exp_dir.mkdir(parents=True, exist_ok=True)
    # Update config to use the resolved experiment dir
    base_config["paths"]["experiment_dir_resolved"] = str(unique_exp_dir.resolve())

    # Unify command paths: always generate from resolved experiment directory
    # Overwrite experiment_name in config to unique experiment directory name
    base_config["experiment_name"] = unique_exp_dir.name
    base_config["paths"]["experiment_dir_resolved"] = str(unique_exp_dir.resolve())

    git_commit = _get_git_commit()

    with open(unique_exp_dir / "config_effective.yaml", "w") as f:
        yaml.safe_dump(base_config, f)
    with open(unique_exp_dir / "config_effective.json", "w") as f_json:
        json.dump(base_config, f_json, indent=2)

    if not optuna_mode:
        runner = PipelineRunner(
            str(unique_exp_dir / "config_effective.yaml"), optuna_mode=False, git_commit=git_commit
        )
        try:
            success = runner.run()
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
            traceback.print_exc()
            return 1
        return 0 if success else 1
    else:
        # Optuna mode: only create optuna/ subdirectory
        optuna_dir = unique_exp_dir / "optuna"
        optuna_dir.mkdir(exist_ok=True)
        cfg_reader = ConfigReader(config_path)
        optuna_cfg = cfg_reader.get_optuna_config()
        search_space = optuna_cfg["hyperparameters"]
        optuna_section_present = "optuna" in base_config
        search_space_populated = _has_tunable_params(search_space)
        if not optuna_section_present or not search_space_populated:
            if not optuna_section_present:
                reason = (
                    "[Optuna] Configuration does not define an 'optuna' section; cannot run hyperparameter search."
                )
            else:
                reason = (
                    "[Optuna] No tunable parameters were found in the optuna.hyperparameters section; nothing to optimize."
                )
            print(reason)
            _log_optuna_message(optuna_dir, reason)
            return 1
        metric_raw = optuna_cfg.get("metric", ["f1", "malware_fpr"])
        if isinstance(metric_raw, str):
            if metric_raw.startswith("(") and metric_raw.endswith(")"):
                metric_names = [m.strip() for m in metric_raw[1:-1].split(",")]
            else:
                metric_names = [m.strip() for m in metric_raw.split(",")]
        elif isinstance(metric_raw, (list, tuple)):
            metric_names = list(metric_raw)
        else:
            metric_names = ["f1", "malware_fpr"]
        directions = optuna_cfg.get("directions", ["maximize", "minimize"])
        n_trials = optuna_cfg.get("n_trials", 20)
        n_jobs = optuna_cfg.get("n_jobs", 1)
        try:
            optimizer = OptunaOptimizer(
                pipeline_cls=PipelineRunner,
                config_reader_cls=ConfigReader,
                base_config=base_config,
                exp_dir=str(unique_exp_dir),
                metric_names=metric_names,
                search_space=search_space,
                n_trials=n_trials,
                directions=directions,
                n_jobs=n_jobs,
                optuna_run_dir=str(optuna_dir),
                pruner_config=optuna_cfg.get("pruner"),
                git_commit=git_commit,
            )
            optimizer.optimize()
        except Exception as e:
            print(f"[ERROR] Optuna optimization failed: {e}")
            traceback.print_exc()
            return 2
        return 0

    # Save config in both yaml and json for both modes
    with open(unique_exp_dir / "config_effective.yaml", "w") as f:
        yaml.safe_dump(base_config, f)
    with open(unique_exp_dir / "config_effective.json", "w") as f_json:
        json.dump(base_config, f_json, indent=2)

    if optuna_mode:
        print("[INFO] Optuna mode enabled. This run may take much longer and will override config parameters for classifier, mixer, etc. to search for the best combination.")
        optuna_cfg = cfg_reader.get_optuna_config()
        # Always use a single optuna/ folder per experiment
        optuna_dir = unique_exp_dir / "optuna"
        optuna_dir.mkdir(exist_ok=True)
        # Prepare search space and metric names
        search_space = optuna_cfg["hyperparameters"]
        metric_raw = optuna_cfg.get("metric", ["f1", "malware_fpr"])
        if isinstance(metric_raw, str):
            if metric_raw.startswith("(") and metric_raw.endswith(")"):
                metric_names = [m.strip() for m in metric_raw[1:-1].split(",")]
            else:
                metric_names = [m.strip() for m in metric_raw.split(",")]
        elif isinstance(metric_raw, (list, tuple)):
            metric_names = list(metric_raw)
        else:
            metric_names = ["f1", "malware_fpr"]
        directions = optuna_cfg.get("directions", ["maximize", "minimize"])
        n_trials = optuna_cfg.get("n_trials", 20)
        n_jobs = optuna_cfg.get("n_jobs", 1)
        try:
            optimizer = OptunaOptimizer(
                pipeline_cls=PipelineRunner,
                config_reader_cls=ConfigReader,
                base_config=base_config,
                exp_dir=str(unique_exp_dir),
                metric_names=metric_names,
                search_space=search_space,
                n_trials=n_trials,
                directions=directions,
                n_jobs=n_jobs,
                optuna_run_dir=str(optuna_dir),
                pruner_config=optuna_cfg.get("pruner"),
                git_commit=git_commit,
            )
            optimizer.optimize()
        except Exception as e:
            print(f"[ERROR] Optuna optimization failed: {e}")
            traceback.print_exc()
            return 2
        return 0
    else:
        runner = PipelineRunner(
            str(unique_exp_dir / "config_effective.yaml"), optuna_mode=False, git_commit=git_commit
        )
        try:
            success = runner.run()
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
            traceback.print_exc()
            return 1
        return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML pipeline or Optuna tuner.")
    parser.add_argument("config", nargs="?", default="./default_config.yaml", help="Path to config file.")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter search mode.")
    args = parser.parse_args()
    sys.exit(main(args.config, optuna_mode=args.optuna))
