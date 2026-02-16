import pytest

from src import optuna_validator as validator


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
