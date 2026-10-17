from pathlib import Path

import pytest
import yaml

from src import optuna_validator as validator

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validator_rejects_extra_keys():
    search_space = {
        "model": {
            "classifier_type": {
                "type": "categorical",
                "choices": ["river.forest.ARFClassifier"],
                "unexpected": True,
            }
        }
    }
    with pytest.raises(validator.OptunaConfigError):
        validator.validate_optuna_search_space(search_space)


def test_validator_accepts_minimal_space():
    search_space = {
        "model": {
            "classifier_type": {
                "type": "categorical",
                "choices": ["A", "B"],
            },
            "classifier_params": {
                "A": {
                    "alpha": {
                        "type": "float",
                        "low": 0.1,
                        "high": 1.0,
                    }
                }
            },
        }
    }
    # Should not raise
    validator.validate_optuna_search_space(search_space)


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"type": "categorical", "choices": []}, "non-empty 'choices'"),
        ({"type": "int", "low": 5, "high": 1}, "must not exceed"),
        ({"type": "float", "low": 0.1}, "requires 'high'"),
        ({"type": "int", "low": 1, "high": 10, "log": True, "step": 2}, "cannot be combined"),
        ({"type": "integer", "low": 1, "high": 2}, "unknown spec type"),
        ({"type": "categorical", "choices": ["a"], "when": ["x"]}, "non-empty mapping"),
        ({"type": "categorical", "multiple": True, "choices": [1, 2]}, "must be strings"),
        ({"type": "categorical", "multiple": True, "choices": ["a", "b"], "max": 0}, "min <= max"),
        ({"type": "group", "params": {"p": {"type": "int", "low": 1, "high": 2, "when": {"a": 1}}}}, "inside a group"),
        ({"type": "group", "params": {"p": 3}}, "group params must be specs"),
    ],
)
def test_validator_rejects_bad_specs(spec, message):
    with pytest.raises(validator.OptunaConfigError, match=message):
        validator.validate_optuna_search_space({"model": {"param": spec}})


def test_validator_accepts_runtime_mirroring_step_list():
    """A step written like the runtime config (name/type/params) is a container, not a spec."""
    search_space = {
        "preprocessing": {
            "steps": [
                {"name": "scaler", "type": "StandardScaler", "params": {}},
                {"name": "pca", "type": "IncrementalPCA", "params": {"n_components": {"type": "int", "low": 1, "high": 11}}},
            ]
        }
    }
    validator.validate_optuna_search_space(search_space)


@pytest.mark.parametrize("name", ["optuna_conf.yaml", "optuna_sklearn_conf.yaml", "demo_config.yaml"])
def test_shipped_search_spaces_validate(name):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
    validator.validate_optuna_search_space(cfg["optuna"]["hyperparameters"])
