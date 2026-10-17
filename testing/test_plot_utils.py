"""Smoke and regression tests for the plotting / aggregation scripts in src/plot_utils."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.commons import BENIGN, MALICIOUS
from src.logger import Logger
from src.optuna_optimizer import OptunaOptimizer
from src.plot_utils import aggregate_optuna_pareto as agg
from src.plot_utils import plot_optuna_trials as pot

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "src" / "plot_utils"


def _write_train_log(path: Path, batches: int, with_validation: bool):
    rng = np.random.default_rng(0)
    logger = Logger(path, overwrite=True)
    for _ in range(batches):
        y = rng.choice([BENIGN, MALICIOUS], size=40)
        pred = np.where(rng.random(40) < 0.9, y, rng.choice([BENIGN, MALICIOUS], size=40))
        if with_validation:
            logger.save_training_results(pred[:30], y[:30], pred[30:], y[30:], 40)
        else:
            logger.save_training_results(pred, y, None, None, 40)


def _write_test_log(path: Path, batches: int):
    rng = np.random.default_rng(1)
    logger = Logger(path, overwrite=True)
    for _ in range(batches):
        y = rng.choice([BENIGN, MALICIOUS], size=50)
        logger.save_test_results(y, np.where(rng.random(50) < 0.9, y, BENIGN))


@pytest.mark.parametrize("with_validation", [True, False])
def test_plot_train_perf_runs_on_pipeline_logs(tmp_path, with_validation):
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_train_log(logs / "0_train_main_train.log", batches=25, with_validation=with_validation)
    out = tmp_path / "results"
    out.mkdir()
    # -f may point at the logs directory: the single *_train.log in it is used
    subprocess.run(
        [sys.executable, str(PLOT_DIR / "plot_train_perf.py"), "-f", str(logs), "-e", "0_train", "--save_folder", str(out)],
        check=True,
        capture_output=True,
    )
    base = out / "training" / "0_train"
    assert (base / "summary.txt").is_file()
    assert (base / "training" / "per_batch" / "class_counts_batch_training.png").is_file()
    if with_validation:
        assert (base / "comparison" / "last20" / "train_val_mcc_last20.png").is_file()


def test_plot_test_perf_runs_on_pipeline_log(tmp_path):
    log = tmp_path / "1_test_all_test.log"
    _write_test_log(log, batches=5)
    out = tmp_path / "results"
    out.mkdir()
    subprocess.run(
        [sys.executable, str(PLOT_DIR / "plot_test_perf.py"), "-f", str(log), "-e", "1_test", "--save_folder", str(out)],
        check=True,
        capture_output=True,
    )
    assert (out / "testing" / "1_test" / "summary.txt").is_file()


def test_train_only_class_counts_are_not_zero(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PLOT_DIR))
    import plot_train_perf
    from src.metrics_calculator import read_training_batches

    log = tmp_path / "t.log"
    _write_train_log(log, batches=3, with_validation=False)
    entries = read_training_batches(str(log))
    per_batch, _ = plot_train_perf.calculate_class_counts(entries, "training_per_class", [BENIGN, MALICIOUS])
    assert all(sum(counts.values()) == 40 for counts in per_batch)


# ---------------- Optuna plots and aggregation ----------------
class _ScorePipeline:
    """Stand-in for PipelineRunner: deterministic metrics per trial, no training."""

    def __init__(self, *_args, **_kwargs):
        pass

    def run_optuna_trial(self, optuna_trial=None, optuna_dir=None):
        n = optuna_trial.number
        f1, fpr = 0.5 + 0.04 * (n % 10), 0.02 * ((n * 7) % 10)
        Path(optuna_dir, "model.bin").write_bytes(b"model")
        return {
            "f1": f1, "fpr": fpr, "test_f1": f1, "test_fpr": fpr,
            "train_f1": f1 - 0.05, "train_fpr": fpr + 0.01,
            "objective_source": "test", "train_command": "train_main", "test_command": "test_all",
        }


def _run_study(exp_dir: Path, n_trials: int = 8) -> Path:
    base_config = {
        "experiment_name": exp_dir.name,
        "model": {"classifier_type": "river.forest.ARFClassifier", "classifier_params": {}},
        "preprocessing": {"steps": [{"name": "scaler", "type": "StandardScaler", "params": {}}]},
        "commands": [
            {"name": "train_main", "command": "train", "mixer": {"type": "sequence", "datasets": ["001"]}},
            {"name": "test_all", "command": "test", "mixer": {"type": "sequence", "datasets": ["002", "003"]}},
        ],
    }
    space = {"model": {"classifier_type": {"type": "categorical", "choices": ["river.forest.ARFClassifier", "river.tree.SGTClassifier"]}}}
    opt = OptunaOptimizer(
        pipeline_cls=_ScorePipeline, config_reader_cls=None, base_config=base_config, exp_dir=exp_dir,
        metric_names=["f1", "fpr"], search_space=space, n_trials=n_trials, directions=("maximize", "minimize"),
    )
    opt.optimize()
    return opt.optuna_run_dir


def test_optuna_study_outputs_and_plots(tmp_path):
    optuna_dir = _run_study(tmp_path / "experiments" / "study_a")
    assert (optuna_dir / "optuna_summary.json").is_file()
    assert (optuna_dir / "visuals" / "pareto" / "test_all_pareto.png").is_file()
    # the log fallback (used for interrupted studies) recovers every finished trial
    assert sorted(pot._parse_log_metrics(optuna_dir)) == list(range(8))


@pytest.fixture
def no_study_plots(monkeypatch):
    import src.optuna_optimizer as oo

    monkeypatch.setattr(oo, "generate_optuna_trial_plots", lambda *_args, **_kwargs: [])


def test_metric_inference_skips_trials_without_metrics(tmp_path, no_study_plots):
    optuna_dir = _run_study(tmp_path / "experiments" / "study_b", n_trials=3)
    (optuna_dir / "trial_0099").mkdir()  # killed trial: no metrics.json
    trial_files = pot._discover_trial_files(optuna_dir)
    names = pot._infer_metric_names_from_trials(trial_files, {}, {})
    assert "f1" in names and "fpr" in names


def test_aggregate_pareto_across_experiments(tmp_path, no_study_plots):
    experiments = tmp_path / "experiments"
    _run_study(experiments / "study_a")
    _run_study(experiments / "study_b", n_trials=4)
    summary_root = agg.aggregate_optuna_pareto(experiments, annotate=False, metric_filters={"f1": (0.0, 1.0)})
    set_dir = summary_root / "test_set_1"
    assert (set_dir / "summary" / "pareto_trials.csv").is_file()
    copied = list((set_dir / "pareto_models").iterdir())
    assert copied and all((path / "model.bin").is_file() for path in copied)


@pytest.mark.parametrize(
    "argv, expected_dir",
    [
        (["./exps", "--f1", "(0.8, 1.0)"], "./exps"),
        (["--f1", "(0.8, 1.0)", "./exps"], "./exps"),
        (["--f1", "(0.8, 1.0)", "--no-annotate"], "./experiments"),
        (["--f1=(0.8, 1.0)", "./exps"], "./exps"),
    ],
)
def test_aggregate_parse_args_accepts_filters_anywhere(argv, expected_dir):
    args = agg._parse_args(argv)
    assert args.experiments_dir == expected_dir
    assert args.metric_filters == {"f1": (0.8, 1.0)}


def test_aggregate_parse_args_missing_filter_value():
    with pytest.raises(ValueError):
        agg._parse_args(["./exps", "--f1"])
