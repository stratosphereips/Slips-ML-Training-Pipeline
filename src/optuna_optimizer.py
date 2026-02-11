
import optuna
import json
from pathlib import Path
from copy import deepcopy
import time


class OptunaOptimizer:
    def __init__(self, pipeline_cls, config_reader_cls, base_config, exp_dir, metric_names, search_space, n_trials=20, directions=("maximize", "minimize"), n_jobs=1, optuna_run_dir=None):
        self.pipeline_cls = pipeline_cls
        self.config_reader_cls = config_reader_cls
        self.base_config = deepcopy(base_config)
        self.exp_dir = Path(exp_dir)
        self.optuna_run_dir = Path(optuna_run_dir) if optuna_run_dir else (self.exp_dir / "optuna")
        self.optuna_log_path = self.optuna_run_dir / "optuna_trials.log"
        self._log_optuna(f"[Optuna] Initialized optimizer at {self.optuna_run_dir}")
        self.metric_names = metric_names
        self.search_space = search_space
        self.n_trials = n_trials
        self.directions = directions
        self.n_jobs = n_jobs
        self.trials_log = []

    def _log_optuna(self, msg):
        from datetime import datetime
        with open(self.optuna_log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")

    def suggest_params(self, trial, classifier_type):
        params = {}
        # Only suggest params for the selected classifier_type
        if classifier_type in self.search_space:
            for pname, pdef in self.search_space[classifier_type].items():
                if pname == "classifier_type":
                    continue
                if pdef["type"] == "int":
                    params[pname] = trial.suggest_int(pname, pdef["low"], pdef["high"])
                elif pdef["type"] == "float":
                    params[pname] = trial.suggest_float(pname, pdef["low"], pdef["high"], log=pdef.get("log", False))
                elif pdef["type"] == "categorical":
                    params[pname] = trial.suggest_categorical(pname, pdef["choices"])
        return params

    def objective(self, trial):
        self._log_optuna(f"Starting trial {trial.number}")
        t0 = time.time()
        config = deepcopy(self.base_config)
        # Dynamically suggest classifier_type
        classifier_type_choices = self.search_space.get("classifier_type", {}).get("choices")
        if classifier_type_choices:
            classifier_type = trial.suggest_categorical("classifier_type", classifier_type_choices)
        else:
            classifier_type = config["model"]["classifier_type"]
        config["model"]["classifier_type"] = classifier_type
        params = self.suggest_params(trial, classifier_type)
        config["model"]["classifier_params"] = params
        # Save only changed params for this trial
        trial_cfg_path = self.optuna_run_dir / f"trial{trial.number}_conf.yaml"
        with open(trial_cfg_path, "w") as f:
            import yaml
            yaml.safe_dump({"classifier_type": classifier_type, **params}, f)
            # Save trial context (datasets, mixer, etc.)
            trial_context_path = self.optuna_run_dir / f"trial{trial.number}_context.yaml"
            def extract_trial_context(cfg):
                # Remove optuna-specific keys
                context = deepcopy(cfg)
                context.pop("optuna", None)
                # Optionally, only keep relevant keys
                # whitelist = ["dataset_loader", "features", "preprocessing", "model", "commands", "paths", "classes", "batch_size_train", "batch_size_test", "seed", "root"]
                # context = {k: context[k] for k in whitelist if k in context}
                return context
            trial_context = extract_trial_context(config)
            with open(trial_context_path, "w") as f:
                import yaml
                yaml.safe_dump(trial_context, f)
        # Run pipeline (train+val), collect metrics
        pipeline = self.pipeline_cls(config, optuna_trial=trial, optuna_dir=self.optuna_run_dir)
        metrics = pipeline.run_optuna_trial()  # Should return dict with metric_names
        # Save results
        trial_result_path = self.optuna_run_dir / f"trial{trial.number}_result.json"
        with open(trial_result_path, "w") as f:
            json.dump(metrics, f, indent=2)
        t1 = time.time()
        self._log_optuna(f"Finished trial {trial.number} in {t1-t0:.1f}s | params: {params} | metrics: {metrics}")
        # Log for summary
        self.trials_log.append({"trial": trial.number, "classifier_type": classifier_type, "params": params, "metrics": metrics})
        return tuple(metrics[m] for m in self.metric_names)

    def optimize(self):
        self._log_optuna("Creating new study...")
        study = optuna.create_study(directions=list(self.directions))
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
        return study
