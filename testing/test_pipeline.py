"""Pipeline-level tests: config -> loaders -> commands, Optuna trials, and run.py."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import src.pipeline as pipeline_mod
from src.conf_reader import ConfigReader

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = str(REPO_ROOT / "src" / "plot_utils" / "plot_train_perf.py")
TEST_SCRIPT = str(REPO_ROOT / "src" / "plot_utils" / "plot_test_perf.py")

_FIELDS = (
    "ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto service duration "
    "orig_bytes resp_bytes conn_state history orig_pkts resp_pkts label"
).split()
_TYPES = (
    "time string addr port addr port enum string interval count count string string count count string"
).split()


def _write_dataset(root: Path, name: str = "001_tiny", n: int = 240, seed: int = 0) -> Path:
    """Tiny labeled Zeek conn log: malicious flows are short and small, benign ones long and big."""
    rng = np.random.default_rng(seed)
    data_dir = root / name / "data"
    data_dir.mkdir(parents=True)
    lines = ["#separator \\x09", "#fields\t" + "\t".join(_FIELDS), "#types\t" + "\t".join(_TYPES)]
    for i in range(n):
        malicious = i % 3 == 0
        dur = rng.uniform(0.01, 0.2) if malicious else rng.uniform(5, 50)
        sbytes = int(rng.integers(40, 200)) if malicious else int(rng.integers(5000, 90000))
        lines.append(
            "\t".join(
                [
                    f"{1600000000 + i}.0", f"C{i}", "10.0.0.1", str(40000 + i), "10.0.0.2",
                    "443" if malicious else "80", "tcp", "-", f"{dur:.4f}", str(sbytes),
                    str(sbytes * 2), "S0" if malicious else "SF", "S" if malicious else "ShADadFf",
                    "1" if malicious else "20", "0" if malicious else "18",
                    "Malicious" if malicious else "Benign",
                ]
            )
        )
    (data_dir / "conn.log.labeled").write_text("\n".join(lines) + "\n")
    return root


def _tiny_config(tmp_path, **overrides):
    root = _write_dataset(tmp_path / "datasets")
    cfg = {
        "experiment_name": "tiny",
        "root": str(root),
        "seed": 1111,
        "validation_split": 0.2,
        "batch_size_train": 50,
        "batch_size_test": 100,
        "classes": ["Benign", "Malicious"],
        "plotting_pths": {"training": TRAIN_SCRIPT, "testing": TEST_SCRIPT},
        "paths": {"experiment_dir": str(tmp_path / "experiments")},
        "dataset_loader": {"cache_dir": str(tmp_path / "cache")},
        "features": {"protocols_to_discard": ["arp", "icmp", ""]},
        "preprocessing": {"steps": [{"name": "scaler", "type": "StandardScaler", "params": {}}]},
        "model": {"wrapper": "SklearnClassifierWrapper", "classifier_type": "SGDClassifier"},
        "commands": [
            {"name": "train_main", "command": "train", "mixer": {"type": "sequence", "datasets": ["001"]}},
            {"name": "test_all", "command": "test", "mixer": {"type": "sequence", "datasets": ["001"]}},
        ],
    }
    cfg.update(overrides)
    return cfg


def _stub_exp():
    return SimpleNamespace(rng=np.random.default_rng(0), log_event=lambda *_: None)


# ---------------- loaders / classifier construction ----------------
def test_build_loaders_forwards_dataset_loader_options(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(pipeline_mod, "find_and_load_datasets", lambda **kw: captured.update(kw) or {})
    cfg = _tiny_config(tmp_path)
    cfg["dataset_loader"].update(
        {"persist_cache_threshold": 7, "labeled_filenames": ["a.log"], "file_encoding": "latin-1", "file_errors": "strict"}
    )
    pipeline_mod.BuildManager(ConfigReader(cfg), _stub_exp()).build_loaders()
    assert captured["cache_dir"] == str((tmp_path / "cache").resolve())
    assert captured["persist_cache_threshold"] == 7
    assert captured["labeled_filenames"] == ["a.log"]
    assert captured["file_encoding"] == "latin-1"
    assert captured["file_errors"] == "strict"


def test_seed_reaches_nested_river_model(tmp_path):
    cfg = _tiny_config(tmp_path, seed=4242)
    cfg["model"] = {
        "wrapper": "RiverClassifierWrapper",
        "classifier_type": "river.ensemble.ADWINBoostingClassifier",
        "classifier_params": {"model": {"type": "HoeffdingAdaptiveTreeClassifier", "params": {}}},
    }
    wrapper = pipeline_mod.BuildManager(ConfigReader(cfg), _stub_exp()).build_classifier()
    assert wrapper.classifier.seed == 4242
    assert wrapper.classifier.model.seed == 4242


# ---------------- command execution ----------------
def _executor(cfg, frame, save_calls=None, features=None):
    reader = ConfigReader(cfg)
    save_calls = save_calls if save_calls is not None else []
    features = features or SimpleNamespace(process_batch=lambda b: (b[["x"]], b["ground_truth_label"]))
    prep = SimpleNamespace(transform=lambda X: X, partial_fit=lambda X: None, save=lambda **_: save_calls.append("prep"))
    clf = SimpleNamespace(
        predict=lambda X: np.array(["Benign"] * len(X)),
        partial_fit=lambda X, y: None,
        save_classifier=lambda **_: save_calls.append("model"),
    )
    return reader, pipeline_mod.CommandExecutor(reader, _stub_exp(), {"001": frame}, features, prep, clf)


def test_test_command_evaluates_every_flow(tmp_path):
    frame = pd.DataFrame({"x": range(100), "ground_truth_label": ["Benign"] * 100})
    reader, executor = _executor(_tiny_config(tmp_path), frame)
    cmd = reader.get_commands()[1]
    # even if a mixer spec asks for a validation split, a test command must not drop flows
    cmd["mixer"]["validation_split"] = 0.5
    cmd["effective_validation_split"] = 0.5
    cmd["paths"] = {"logs": str(tmp_path / "logs"), "model": str(tmp_path), "preprocessing": str(tmp_path)}
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    executor._run_test(1, cmd, test_metrics=counts)
    assert sum(counts.values()) == 100


def test_save_steps_false_skips_preprocessing_save(tmp_path):
    cfg = _tiny_config(tmp_path)
    cfg["preprocessing"]["save_steps"] = False
    frame = pd.DataFrame({"x": range(10), "ground_truth_label": ["Benign"] * 10})
    calls = []
    reader, executor = _executor(cfg, frame, save_calls=calls)
    cmd = reader.get_commands()[1]
    cmd["paths"] = {"logs": str(tmp_path / "logs"), "model": str(tmp_path), "preprocessing": str(tmp_path)}
    executor._run_test(1, cmd)
    assert calls == ["model"]


def test_feature_errors_propagate_out_of_training(tmp_path):
    frame = pd.DataFrame({"x": range(10), "ground_truth_label": ["Benign"] * 10})

    def broken(_batch):
        raise RuntimeError("bad batch")

    reader, executor = _executor(_tiny_config(tmp_path), frame, features=SimpleNamespace(process_batch=broken))
    cmd = reader.get_commands()[0]
    cmd["paths"] = {"logs": str(tmp_path / "logs"), "model": str(tmp_path), "preprocessing": str(tmp_path)}
    with pytest.raises(RuntimeError, match="bad batch"):
        executor._run_train(0, cmd)


# ---------------- Optuna trial ----------------
def test_run_optuna_trial_keeps_artifacts_per_trial(tmp_path):
    cfg = _tiny_config(tmp_path)
    exp_dir = tmp_path / "experiments" / "tiny"
    cfg["paths"]["experiment_dir_resolved"] = str(exp_dir)
    trial_dir = exp_dir / "optuna" / "trial_0000"

    runner = pipeline_mod.PipelineRunner(cfg, optuna_mode=True, optuna_dir=str(trial_dir))
    metrics = runner.run_optuna_trial(optuna_dir=str(trial_dir))

    assert 0.0 <= metrics["f1"] <= 1.0 and 0.0 <= metrics["fpr"] <= 1.0
    assert metrics["objective_source"] == "test"
    assert sum(metrics["test_confusion"].values()) == 240  # every test flow evaluated
    assert (trial_dir / "model.bin").is_file()
    assert (trial_dir / "scaler.bin").is_file()
    assert (trial_dir / "logs" / "0_train_main_train.log").is_file()
    assert (trial_dir / "logs" / "1_test_all_test.log").is_file()
    # nothing is written to the shared output dir, so parallel trials cannot collide
    assert not any((exp_dir / "output" / "models").iterdir())
    assert not any((exp_dir / "output" / "logs").iterdir())


# ---------------- run.py ----------------
def test_run_main_runs_pipeline_once(tmp_path, monkeypatch):
    import run

    calls = []

    class StubRunner:
        def __init__(self, *_args, **_kwargs):
            calls.append("init")

        def run(self):
            calls.append("run")
            return True

    monkeypatch.setattr(run, "PipelineRunner", StubRunner)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_tiny_config(tmp_path)))
    assert run.main(str(cfg_path)) == 0
    assert calls == ["init", "run"]


def test_run_main_end_to_end(tmp_path):
    import run

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_tiny_config(tmp_path)))
    assert run.main(str(cfg_path)) == 0

    exp_dir = tmp_path / "experiments" / "tiny"
    log = (exp_dir / "pipeline_run_log.txt").read_text()
    assert log.count("Starting standard pipeline execution") == 1
    assert (exp_dir / "output" / "models" / "classifier.bin").is_file()
    assert (exp_dir / "output" / "logs" / "0_train_main_train.log").is_file()
    assert (exp_dir / "output" / "results" / "training" / "0_train" / "summary.txt").is_file()
    assert (exp_dir / "output" / "results" / "testing" / "1_test" / "summary.txt").is_file()
