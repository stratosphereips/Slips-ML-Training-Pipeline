#!/usr/bin/env python3

import sys
from pathlib import Path
import traceback

# Insert src/ at front of sys.path so modules inside it can use relative imports
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

# Now import the pipeline module (the module lives in src/)

from pipeline import PipelineRunner  # imports src/pipeline.py as module 'pipeline'
from conf_reader import ConfigReader
from optuna_optimizer import OptunaOptimizer


def main(config_path: str = "./default_config.yaml", optuna_mode: bool = False):
    """
    Initialize and run the pipeline using the specified config file.
    Returns 0 on success, 1 on failure.
    """
    if optuna_mode:
        print("[INFO] Optuna mode enabled. This run may take much longer and will override config parameters for classifier, mixer, etc. to search for the best combination.")
        # Load config and optuna config
        cfg_reader = ConfigReader(config_path)
        base_config = cfg_reader.load()
        optuna_cfg = cfg_reader.get_optuna_config()
        # Get experiment dir for optuna output
        exp_dir = cfg_reader.get_paths()["experiment_dir_resolved"]
        # Prepare search space and metric names
        search_space = optuna_cfg["hyperparameters"]
        metric_raw = optuna_cfg.get("metric", ["f1", "malware_fpr"])
        # Parse metric field: support YAML list, tuple, or comma-separated string
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
        # Import pipeline class here to avoid circular import
        try:
            optimizer = OptunaOptimizer(
                pipeline_cls=PipelineRunner,
                config_reader_cls=ConfigReader,
                base_config=base_config,
                exp_dir=exp_dir,
                metric_names=metric_names,
                search_space=search_space,
                n_trials=n_trials,
                directions=directions,
                n_jobs=n_jobs,
            )
            optimizer.optimize()
        except Exception as e:
            print(f"[ERROR] Optuna optimization failed: {e}")
            traceback.print_exc()
            return 2
        return 0

    # Normal mode
    runner = PipelineRunner(config_path, optuna_mode=False)
    try:
        success = runner.run()
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        return 1
    return 0 if success else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the ML pipeline or Optuna tuner.")
    parser.add_argument("config", nargs="?", default="./default_config.yaml", help="Path to config file.")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter search mode.")
    args = parser.parse_args()
    sys.exit(main(args.config, optuna_mode=args.optuna))
