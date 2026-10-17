import copy
import json
from pathlib import Path

import optuna
import pytest
import yaml

from src.optuna_optimizer import OptunaOptimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
ADWIN = "river.ensemble.ADWINBoostingClassifier"


class DummyFailingPipeline:
    def __init__(self, *_args, **_kwargs):
        pass

    def run_optuna_trial(self, **_kwargs):
        raise RuntimeError("forced failure for testing")


class DummyTrial:
    def __init__(self, number=0):
        self.number = number
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def _optimizer(tmp_path, search_space=None, pipeline_cls=None, base_config=None, **kwargs):
    params = dict(
        pipeline_cls=pipeline_cls or DummyFailingPipeline,
        config_reader_cls=None,
        base_config=base_config or {"model": {"classifier_type": "Dummy"}},
        exp_dir=tmp_path,
        metric_names=["f1", "fpr"],
        search_space=search_space or {},
        directions=("maximize", "minimize"),
        n_trials=1,
        n_jobs=1,
    )
    params.update(kwargs)
    return OptunaOptimizer(**params)


def _sample(opt, config):
    trial = optuna.create_study(directions=["maximize", "minimize"]).ask()
    overrides = opt._sample_hyperparameters(trial, config)
    return overrides, trial


@pytest.fixture
def optimizer(tmp_path):
    return _optimizer(tmp_path)


def test_objective_handles_failure_and_continues(optimizer):
    trial = DummyTrial(number=1)
    values = optimizer.objective(trial)
    assert values[0] == pytest.approx(-1e9)
    assert values[1] == pytest.approx(1e9)
    assert optimizer.trials_log[-1]["status"] == "failed"
    trial_dir = Path(optimizer.optuna_run_dir) / "trial_0001"
    metrics_path = trial_dir / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert metrics["status"] == "failed"
    assert "error" in metrics
    assert trial.user_attrs["status"] == "failed"


def test_metric_and_direction_count_must_match(tmp_path):
    with pytest.raises(ValueError):
        _optimizer(tmp_path, metric_names=["f1"])


def test_seed_falls_back_to_random_seed(tmp_path):
    assert _optimizer(tmp_path, base_config={"random_seed": 5}).seed == 5


# ---------------- path utilities ----------------
def test_get_path_value_handles_dotted_keys(optimizer):
    cfg = {
        "model": {"classifier_params": {ADWIN: {"model": {"type": "HT"}}}},
        "commands": [{"mixer": {"type": "balanced"}}],
    }
    assert optimizer._get_path_value(cfg, f"model.classifier_params.{ADWIN}.model.type") == "HT"
    assert optimizer._get_path_value(cfg, "commands[0].mixer.type") == "balanced"
    assert optimizer._get_path_value(cfg, "model.classifier_params.missing.key") is None
    assert optimizer._get_path_value(cfg, "commands[3].mixer") is None


# ---------------- sampling ----------------
def test_nested_when_condition_samples_inner_params(tmp_path):
    space = {
        "model": {
            "classifier_type": {"type": "categorical", "choices": [ADWIN]},
            "classifier_params": {
                ADWIN: {
                    "model": {
                        "type": {"type": "categorical", "choices": ["HoeffdingTreeClassifier"]},
                        "params": {
                            "max_depth": {
                                "type": "int", "low": 5, "high": 30,
                                "when": {f"model.classifier_params.{ADWIN}.model.type": ["HoeffdingTreeClassifier"]},
                            },
                            "drift_window_threshold": {
                                "type": "int", "low": 100, "high": 200,
                                "when": {f"model.classifier_params.{ADWIN}.model.type": ["HoeffdingAdaptiveTreeClassifier"]},
                            },
                        },
                    }
                }
            },
        }
    }
    config = {"model": {"classifier_type": "x", "classifier_params": {}}}
    _sample(_optimizer(tmp_path, search_space=space), config)
    inner_params = config["model"]["classifier_params"]["model"]["params"]
    assert 5 <= inner_params["max_depth"] <= 30
    assert "drift_window_threshold" not in inner_params  # guard not met


def test_mixer_when_condition_uses_commands_path(tmp_path):
    space = {
        "commands": [
            {
                "mixer": {
                    "type": {"type": "categorical", "choices": ["balanced"]},
                    "balance": {
                        "type": "categorical",
                        "choices": [[0.25, 0.75]],
                        "when": {"commands[0].mixer.type": ["balanced"]},
                    },
                }
            }
        ]
    }
    config = {"model": {}, "commands": [{"name": "t", "command": "train", "mixer": {"type": "sequence"}}]}
    _sample(_optimizer(tmp_path, search_space=space), config)
    assert config["commands"][0]["mixer"] == {"type": "balanced", "balance": [0.25, 0.75]}


def test_preprocessing_step_params_are_addressed_by_index(tmp_path):
    space = {"preprocessing": {"steps": {1: {"params": {"n_components": {"type": "int", "low": 3, "high": 3}}}}}}
    config = {
        "model": {},
        "preprocessing": {
            "steps": [
                {"name": "scaler", "type": "StandardScaler", "params": {}},
                {"name": "pca", "type": "IncrementalPCA", "params": {"n_components": 11}},
            ]
        },
    }
    overrides, _ = _sample(_optimizer(tmp_path, search_space=space), config)
    assert overrides == {"preprocessing.steps[1].params.n_components": 3}
    assert config["preprocessing"]["steps"][1]["params"]["n_components"] == 3


def test_short_class_name_branch_and_fixed_params_are_kept(tmp_path):
    space = {
        "model": {"classifier_params": {"ARFClassifier": {"n_models": {"type": "int", "low": 4, "high": 4}}}}
    }
    config = {
        "model": {
            "classifier_type": "river.forest.ARFClassifier",
            "classifier_params": {"n_models": 10, "lambda_value": 7},
        }
    }
    _sample(_optimizer(tmp_path, search_space=space), config)
    assert config["model"]["classifier_params"] == {"n_models": 4, "lambda_value": 7}


def test_int_step_and_log_are_passed_to_optuna(tmp_path):
    opt = _optimizer(tmp_path)
    trial = optuna.create_study().ask()
    value = opt._suggest_param_value(trial, "p", {"type": "int", "low": 0, "high": 10, "step": 5})
    assert value in (0, 5, 10)
    dist = trial.distributions["p"]
    assert dist.step == 5
    opt._suggest_param_value(trial, "q", {"type": "int", "low": 1, "high": 1000, "log": True})
    assert trial.distributions["q"].log is True


def test_multiple_choice_combinations_built_once(tmp_path):
    opt = _optimizer(tmp_path)
    seen = []

    class FakeTrial:
        def suggest_categorical(self, name, choices):
            seen.append(choices)
            return choices[-1]

    spec = {"type": "categorical", "multiple": True, "choices": ["001", "002", "003"], "max": 2}
    first = opt._suggest_param_value(FakeTrial(), "datasets", spec)
    opt._suggest_param_value(FakeTrial(), "datasets", spec)
    assert seen[0] is seen[1]
    assert len(seen[0]) == 6  # 3 singles + 3 pairs
    assert first == ["002", "003"]


# ---------------- study outputs ----------------
class _InterruptingPipeline:
    calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def run_optuna_trial(self, **_kwargs):
        _InterruptingPipeline.calls += 1
        if _InterruptingPipeline.calls > 1:
            raise KeyboardInterrupt
        return {"f1": 0.5, "fpr": 0.1}


def test_summary_is_written_when_the_study_is_interrupted(tmp_path, monkeypatch):
    import src.optuna_optimizer as oo

    monkeypatch.setattr(oo, "generate_optuna_trial_plots", lambda *_: [])
    opt = _optimizer(tmp_path, pipeline_cls=_InterruptingPipeline, n_trials=5)
    with pytest.raises(KeyboardInterrupt):
        opt.optimize()
    summary = json.loads((opt.optuna_run_dir / "optuna_summary.json").read_text())
    assert [t["number"] for t in summary["best_trials"]] == [0]
    assert (opt.optuna_run_dir / "optuna_trials.csv").is_file()


@pytest.mark.parametrize("name", ["optuna_conf.yaml", "optuna_sklearn_conf.yaml", "demo_config.yaml"])
def test_shipped_search_spaces_sample_valid_configs(name, tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
    opt = _optimizer(tmp_path, search_space=cfg["optuna"]["hyperparameters"], base_config=cfg)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.RandomSampler(seed=0))
    sampled_any_balance = False
    for _ in range(40):
        config = copy.deepcopy(cfg)
        trial = study.ask()
        overrides = opt._sample_hyperparameters(trial, config)
        study.tell(trial, [0.0, 0.0])
        steps = config["preprocessing"]["steps"]
        assert all("type" in step for step in steps)  # never creates half-defined steps
        if "preprocessing.steps[1].params.n_components" in overrides:
            assert steps[1]["params"]["n_components"] == overrides["preprocessing.steps[1].params.n_components"]
        mixer = config["commands"][0]["mixer"]
        if mixer["type"] == "balanced":
            sampled_any_balance = True
            assert "balance" in mixer
        else:
            assert "balance" not in mixer
        datasets = mixer["datasets"]
        assert 1 <= len(datasets) <= 5
    assert sampled_any_balance
