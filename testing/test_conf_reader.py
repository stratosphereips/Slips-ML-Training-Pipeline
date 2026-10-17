import copy
from pathlib import Path

import pytest
import yaml

from src.conf_reader import ConfigReader

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path, **overrides):
    cfg = {
        "experiment_name": "cfg_test",
        "root": str(tmp_path),
        "classes": ["Benign", "Malicious"],
        "batch_size_train": 100,
        "batch_size_test": 300,
        "validation_split": 0.2,
        "plotting_pths": {"training": "t.py", "testing": "t.py"},
        "paths": {"experiment_dir": str(tmp_path / "experiments")},
        "model": {"classifier_type": "SGDClassifier"},
        "commands": [
            {"name": "train_a", "command": "train", "mixer": {"type": "sequence", "datasets": ["x"]}},
            {"command": "train", "validation_split": 0.3, "mixer": {"type": "sequence", "datasets": ["x"]}},
            {"name": "train_c", "command": "train", "validation_split": 0.3,
             "mixer": {"type": "sequence", "datasets": ["x"], "validation_split": 0.4}},
            {"name": "test_all", "command": "test", "mixer": {"type": "sequence", "datasets": ["x"]}},
        ],
    }
    cfg.update(overrides)
    return cfg


def test_effective_batch_sizes_and_validation_splits(tmp_path):
    commands = ConfigReader(_config(tmp_path)).get_commands()
    assert [c["effective_batch_size"] for c in commands] == [100, 100, 100, 300]
    # mixer > command > global for train commands; test commands evaluate every flow
    assert [c["effective_validation_split"] for c in commands] == [0.2, 0.3, 0.4, 0.0]
    assert [c["mixer"]["validation_split"] for c in commands] == [0.2, 0.3, 0.4, 0.0]


def test_commands_without_name_get_a_default(tmp_path):
    commands = ConfigReader(_config(tmp_path)).get_commands()
    assert commands[1]["name"] == "cmd_1"


def test_input_dict_is_not_modified(tmp_path):
    cfg = _config(tmp_path)
    before = copy.deepcopy(cfg)
    ConfigReader(cfg).load()
    assert cfg == before


def test_null_sections_do_not_crash(tmp_path):
    reader = ConfigReader(_config(tmp_path, preprocessing={"steps": None}, optuna=None))
    assert reader.get_preprocessing_steps() == []
    assert reader.get_preprocessing_save_options()["save_steps"] is True
    optuna_cfg = reader.get_optuna_config()
    assert len(optuna_cfg["metric"]) == len(optuna_cfg["directions"])


def test_seed_and_random_seed(tmp_path):
    assert ConfigReader(_config(tmp_path, seed=7)).get_random_seed() == 7
    assert ConfigReader(_config(tmp_path, random_seed=9)).get_random_seed() == 9
    assert ConfigReader(_config(tmp_path)).get_random_seed() == 1111


def test_dataset_loader_defaults_and_resolved_cache_dir(tmp_path):
    ds = ConfigReader(_config(tmp_path)).get_dataset_loader_params()
    assert ds["batch_size"] == 100
    assert ds["data_subdir"] == "data"
    assert ds["persist_cache_threshold"] == 30000
    assert Path(ds["cache_dir_resolved"]).is_absolute()


def test_missing_required_key_raises(tmp_path):
    cfg = _config(tmp_path)
    del cfg["batch_size_test"]
    with pytest.raises(ValueError, match="batch_size_test"):
        ConfigReader(cfg).load()


@pytest.mark.parametrize("name", sorted(p.name for p in (REPO_ROOT / "configs").glob("*.yaml")))
def test_shipped_configs_load(name, tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
    cfg["paths"]["experiment_dir"] = str(tmp_path / "experiments")
    resolved = ConfigReader(cfg).load()
    assert resolved["commands"]
    for cmd in resolved["commands"]:
        assert cmd["name"]
        if cmd["command"] == "test":
            assert cmd["effective_validation_split"] == 0.0
