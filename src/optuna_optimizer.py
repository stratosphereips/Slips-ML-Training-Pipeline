import optuna
import json
from pathlib import Path
from copy import deepcopy

class OptunaOptimizer:
    def __init__(self, pipeline_cls, config_reader_cls, base_config, exp_dir, metric_names, search_space, n_trials=20, directions=("maximize", "minimize"), n_jobs=1):
        self.pipeline_cls = pipeline_cls
        self.config_reader_cls = config_reader_cls
        self.base_config = deepcopy(base_config)
        self.exp_dir = Path(exp_dir)
        # Place optuna directory at the same level as output
        self.optuna_dir = self.exp_dir / "optuna"
        self.optuna_dir.mkdir(parents=True, exist_ok=True)
        self.metric_names = metric_names
        self.search_space = search_space
        self.n_trials = n_trials
        self.directions = directions
        self.n_jobs = n_jobs
        self.trials_log = []

    def suggest_params(self, trial, classifier_type):
        params = {}
        if classifier_type in self.search_space:
            for pname, pdef in self.search_space[classifier_type].items():
                if pdef["type"] == "int":
                    params[pname] = trial.suggest_int(pname, pdef["low"], pdef["high"])
                elif pdef["type"] == "float":
                    params[pname] = trial.suggest_float(pname, pdef["low"], pdef["high"], log=pdef.get("log", False))
                elif pdef["type"] == "categorical":
                    params[pname] = trial.suggest_categorical(pname, pdef["choices"])
        return params

    def objective(self, trial):
        config = deepcopy(self.base_config)
        classifier_type = config["model"]["classifier_type"]
        params = self.suggest_params(trial, classifier_type)
        config["model"]["classifier_params"].update(params)
        # Save only changed params for this trial
        trial_cfg_path = self.optuna_dir / f"trial{trial.number}_{self.exp_dir.name}_config.yaml"
        with open(trial_cfg_path, "w") as f:
            import yaml
            yaml.safe_dump(trial.params, f)
        # Run pipeline (train+val), collect metrics
        pipeline = self.pipeline_cls(config, optuna_trial=trial, optuna_dir=self.optuna_dir)
        metrics = pipeline.run_optuna_trial()  # Should return dict with metric_names
        # Save results
        trial_result_path = self.optuna_dir / f"trial{trial.number}_{self.exp_dir.name}_result.json"
        with open(trial_result_path, "w") as f:
            json.dump(metrics, f, indent=2)
        # Log for summary
        self.trials_log.append({"trial": trial.number, "params": params, "metrics": metrics})
        return tuple(metrics[m] for m in self.metric_names)

    def optimize(self):
        study = optuna.create_study(directions=list(self.directions))

        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        # Save all trials summary
        trials_csv = self.optuna_dir / "optuna_trials.csv"
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
        print(f"Optuna optimization completed. Best trials summary:\n{json.dumps(summary, indent=2)}")
        with open(self.optuna_dir / "optuna_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return study
