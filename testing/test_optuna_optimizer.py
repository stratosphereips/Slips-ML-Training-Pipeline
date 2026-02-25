import json
from pathlib import Path

import pytest

from src.optuna_optimizer import OptunaOptimizer


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


@pytest.fixture
def optimizer(tmp_path):
    base_config = {"model": {"classifier_type": "Dummy"}}
    return OptunaOptimizer(
        pipeline_cls=DummyFailingPipeline,
        config_reader_cls=None,
        base_config=base_config,
        exp_dir=tmp_path,
        metric_names=["f1", "fpr"],
        search_space={},
        directions=("maximize", "minimize"),
        n_trials=1,
        n_jobs=1,
    )


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
