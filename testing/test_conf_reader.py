import pytest
import json
import tempfile
from pathlib import Path
from unittest import mock
import yaml

from src.conf_reader import (
    ConfigReader,
    _load_yaml_file,
    _deep_merge,
    YAML_NAMES,
    JSON_NAMES,
)


# ========== Helper Functions ==========
def write_yaml(path, data):
    """Write data to YAML file."""
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def write_json(path, data):
    """Write data to JSON file."""
    path.write_text(json.dumps(data), encoding="utf-8")


# ========== Deep Merge Tests ==========
class TestDeepMerge:
    """Test _deep_merge recursive merging logic."""
    
    def test_merge_simple_dicts_and_scalars(self):
        """Test merging dicts with no overlapping keys and scalar overrides."""
        # No conflict
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
        # Scalar override
        assert _deep_merge({"a": 1}, {"a": 10}) == {"a": 10}
        # None handling
        assert _deep_merge(None, {"a": 1}) == {"a": 1}
        assert _deep_merge({"a": 1}, None) == {"a": 1}
    
    def test_merge_nested_dicts_recursively(self):
        """Test recursive merging of nested dicts."""
        default = {"nested": {"x": 10, "y": 20}, "z": 30}
        override = {"nested": {"y": 200, "w": 40}}
        result = _deep_merge(default, override)
        assert result == {"nested": {"x": 10, "y": 200, "w": 40}, "z": 30}
    
    def test_merge_lists_replaced_not_merged(self):
        """Test that override lists replace default lists entirely."""
        default = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        assert _deep_merge(default, override) == {"items": [4, 5]}
    
    def test_merge_type_conversion(self):
        """Test that different types replace each other."""
        # Scalar to dict
        assert _deep_merge({"a": 5}, {"a": {"nested": 1}}) == {"a": {"nested": 1}}
        # Dict to scalar
        assert _deep_merge({"a": {"nested": 1}}, {"a": 5}) == {"a": 5}


# ========== File Loading Tests ==========
class TestLoadYamlFile:
    """Test _load_yaml_file file loading and parsing."""
    
    def test_load_yaml_and_yml_files(self, tmp_path):
        """Test loading YAML files (.yaml and .yml)."""
        # Test .yaml
        yaml_file = tmp_path / "config.yaml"
        write_yaml(yaml_file, {"key": "value", "number": 42})
        result = _load_yaml_file(yaml_file)
        assert result == {"key": "value", "number": 42}
        
        # Test .yml
        yml_file = tmp_path / "config.yml"
        write_yaml(yml_file, {"test": 123})
        result = _load_yaml_file(yml_file)
        assert result == {"test": 123}
    
    def test_load_json_file(self, tmp_path):
        """Test loading JSON file."""
        json_file = tmp_path / "config.json"
        write_json(json_file, {"key": "value"})
        result = _load_yaml_file(json_file)
        assert result == {"key": "value"}
    
    def test_load_empty_yaml_becomes_empty_dict(self, tmp_path):
        """Test that empty YAML file becomes empty dict."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        result = _load_yaml_file(yaml_file)
        assert result == {}
    
    def test_load_file_must_be_dict(self, tmp_path):
        """Test that top-level must be dict, not list."""
        yaml_file = tmp_path / "bad.yaml"
        write_yaml(yaml_file, [1, 2, 3])
        with pytest.raises(ValueError, match="must be a mapping"):
            _load_yaml_file(yaml_file)


# ========== ConfigReader File Discovery Tests ==========
class TestConfigReaderFileDiscovery:
    """Test file discovery and initialization."""
    
    def test_find_user_config_from_file_path(self, tmp_path):
        """Test finding config when base is a file."""
        config_file = tmp_path / "config.yaml"
        write_yaml(config_file, {"commands": []})
        
        cr = ConfigReader(str(config_file))
        found = cr._find_user_config()
        assert found == config_file
    
    def test_find_user_config_priority_in_directory(self, tmp_path):
        """Test config file priority: .yaml > .yml > .json."""
        yaml_file = tmp_path / "config.yaml"
        yml_file = tmp_path / "config.yml"
        json_file = tmp_path / "config.json"
        
        # Test with only json
        write_json(json_file, {"commands": []})
        cr = ConfigReader(str(tmp_path))
        assert cr._find_user_config() == json_file
        
        # Add yml, should now find that
        write_yaml(yml_file, {"commands": []})
        cr = ConfigReader(str(tmp_path))
        assert cr._find_user_config() == yml_file
        
        # Add yaml, should now find that (highest priority)
        write_yaml(yaml_file, {"commands": []})
        cr = ConfigReader(str(tmp_path))
        assert cr._find_user_config() == yaml_file
    
    def test_find_user_config_not_found(self, tmp_path):
        """Test that None is returned when no config found."""
        cr = ConfigReader(str(tmp_path))
        assert cr._find_user_config() is None


# ========== ConfigReader Validation Tests ==========
class TestConfigReaderValidation:
    """Test config validation rules."""
    
    def test_validate_commands_required_as_list(self, tmp_path):
        """Test that 'commands' must be present and be a list."""
        cr = ConfigReader(str(tmp_path))
        
        with pytest.raises(ValueError, match="must include 'commands'"):
            cr._validate({})
        
        with pytest.raises(ValueError, match="must include 'commands'"):
            cr._validate({"commands": "not_a_list"})
    
    def test_validate_train_test_require_mixer(self, tmp_path):
        """Test that train and test commands require mixer dict."""
        cr = ConfigReader(str(tmp_path))
        
        with pytest.raises(ValueError, match="'mixer' mapping is required"):
            cr._validate({"commands": [{"command": "train"}]})
        
        with pytest.raises(ValueError, match="'mixer' mapping is required"):
            cr._validate({"commands": [{"command": "test"}]})
    
    def test_validate_mixer_type_must_be_valid(self, tmp_path):
        """Test that mixer type must be sequence, random_batches, or balanced_by_label."""
        cr = ConfigReader(str(tmp_path))
        
        with pytest.raises(ValueError, match="unknown mixer.type"):
            cr._validate({
                "commands": [{
                    "command": "train",
                    "mixer": {"type": "invalid_mixer"}
                }]
            })
    
    def test_validate_mixer_types_require_datasets(self, tmp_path):
        """Test that all mixer types require 'datasets' list."""
        cr = ConfigReader(str(tmp_path))
        
        for mixer_type in ["sequence", "random_batches", "balanced_by_label"]:
            with pytest.raises(ValueError, match="requires 'datasets' list"):
                cr._validate({
                    "commands": [{
                        "command": "train",
                        "mixer": {"type": mixer_type}
                    }]
                })
    
    def test_validate_random_batches_weights_must_match_datasets(self, tmp_path):
        """Test that weights length must match datasets length."""
        cr = ConfigReader(str(tmp_path))
        
        with pytest.raises(ValueError, match="weights length mismatch"):
            cr._validate({
                "commands": [{
                    "command": "train",
                    "mixer": {
                        "type": "random_batches",
                        "datasets": ["ds1", "ds2"],
                        "weights": [1.0]  # Wrong length
                    }
                }]
            })
    
    def test_validate_load_and_new_mutually_exclusive(self, tmp_path):
        """Test that load and new cannot both be present."""
        cr = ConfigReader(str(tmp_path))
        
        with pytest.raises(ValueError, match="mutually exclusive"):
            cr._validate({
                "commands": [{
                    "command": "train",
                    "mixer": {"type": "sequence", "datasets": ["ds1"]},
                    "load": {"model": "path"},
                    "new": {"model": {}}
                }]
            })
    
    def test_validate_load_and_new_structure(self, tmp_path):
        """Test that load and new specs have correct structure."""
        cr = ConfigReader(str(tmp_path))
        
        # load must be dict
        with pytest.raises(ValueError, match="load must be a mapping"):
            cr._validate({
                "commands": [{
                    "command": "train",
                    "mixer": {"type": "sequence", "datasets": []},
                    "load": "not_dict"
                }]
            })
        
        # new must be dict
        with pytest.raises(ValueError, match="new must be a mapping"):
            cr._validate({
                "commands": [{
                    "command": "train",
                    "mixer": {"type": "sequence", "datasets": []},
                    "new": "not_dict"
                }]
            })
        
        # new can only have specific keys
        with pytest.raises(ValueError, match="unknown key"):
            cr._validate({
                "commands": [{
                    "command": "train",
                    "mixer": {"type": "sequence", "datasets": []},
                    "new": {"bad_key": "value"}
                }]
            })


# ========== ConfigReader Defaults and Fallback Tests ==========
class TestConfigReaderDefaults:
    """Test defaults loading and fallback behavior."""
    
    def test_defaults_fallback_sets_all_required_keys(self, tmp_path):
        """Test that _defaults_fallback ensures all required keys exist."""
        cr = ConfigReader(str(tmp_path))
        result = cr._defaults_fallback({})
        
        # Top-level keys
        assert "experiment_name" in result
        assert "root" in result
        assert "seed" in result
        assert "commands" in result
        
        # Nested configs
        assert "paths" in result
        assert "dataset_loader" in result
        assert "preprocessing" in result
        assert "features" in result
        assert "model" in result
    
    def test_defaults_fallback_preserves_user_values(self, tmp_path):
        """Test that user values are not overwritten by defaults."""
        cr = ConfigReader(str(tmp_path))
        user_config = {
            "experiment_name": "my_exp",
            "seed": 9999,
            "custom_key": "preserved"
        }
        result = cr._defaults_fallback(user_config)
        
        assert result["experiment_name"] == "my_exp"
        assert result["seed"] == 9999
        assert result["custom_key"] == "preserved"
    
    def test_defaults_fallback_nested_defaults(self, tmp_path):
        """Test that nested config defaults are set correctly."""
        cr = ConfigReader(str(tmp_path))
        result = cr._defaults_fallback({})
        
        # Dataset loader defaults
        ds = result["dataset_loader"]
        assert "batch_size" in ds
        assert "cache_dir" in ds
        assert "persist_cache_threshold" in ds
        
        # Model defaults
        model = result["model"]
        assert "wrapper" in model
        assert "classifier_type" in model
        assert "classifier_params" in model
        
        # Preprocessing defaults
        pc = result["preprocessing"]
        assert isinstance(pc.get("steps"), list)


# ========== Integration Tests ==========
class TestConfigReaderIntegration:
    """Integration tests for full load flow."""
    
    def test_user_overrides_defaults_and_computes_effective_values(self, tmp_path):
        """
        Comprehensive test: user config overrides defaults, lists are replaced,
        per-command effective values are computed, and paths are resolved.
        """
        # Create defaults
        defaults = {
            "experiment_name": "default_exp",
            "root": "/data/default",
            "batch_size_train": 500,
            "batch_size_test": 1000,
            "validation_split": 0.1,
            "paths": {"experiment_dir": "./experiments"},
            "dataset_loader": {"cache_dir": "/tmp/default_cache"},
            "preprocessing": {
                "steps": [{"name": "default_scaler", "type": "StandardScaler"}]
            },
            "model": {"classifier_params": {"alpha": 1.0}},
            "commands": [{
                "name": "default_cmd",
                "command": "train",
                "mixer": {"type": "sequence", "datasets": ["001"]},
            }],
        }
        defaults_path = tmp_path / "default_config.yaml"
        write_yaml(defaults_path, defaults)
        
        # Create user config with overrides
        user_cfg = {
            "experiment_name": "my_exp",
            "batch_size_train": 400,
            "paths": {"experiment_dir": "./my_exps"},
            "preprocessing": {
                "steps": [{"name": "my_scaler", "type": "StandardScaler"}]
            },
            "model": {"classifier_params": {"alpha": 0.5}},
            "commands": [{
                "name": "train_cmd",
                "command": "train",
                "mixer": {"type": "sequence", "datasets": ["008"]},
                "batch_size": 300,  # Override per-command
            }],
        }
        exp_dir = tmp_path / "exp_folder"
        exp_dir.mkdir()
        write_yaml(exp_dir / "config.yaml", user_cfg)
        
        # Load and verify
        cr = ConfigReader(str(exp_dir), defaults_path=str(defaults_path))
        resolved = cr.load()
        
        # Verify overrides
        assert resolved["experiment_name"] == "my_exp"
        assert resolved["batch_size_train"] == 400
        
        # Verify list replacement (not merge)
        assert len(resolved["preprocessing"]["steps"]) == 1
        assert resolved["preprocessing"]["steps"][0]["name"] == "my_scaler"
        
        # Verify nested override
        assert resolved["model"]["classifier_params"]["alpha"] == 0.5
        
        # Verify default preserved
        assert resolved["dataset_loader"]["cache_dir"] == "/tmp/default_cache"
        
        # Verify per-command effective values
        cmd = resolved["commands"][0]
        assert cmd["effective_batch_size"] == 300  # Command-level override
        assert "effective_validation_split" in cmd
        assert "paths" in cmd
        assert "command_dir" in cmd["paths"]
        
        # Verify paths are resolved
        assert "experiment_dir_resolved" in resolved["paths"]
        assert Path(resolved["paths"]["experiment_dir_resolved"]).is_absolute()
    
    def test_missing_explicit_defaults_raises_error(self, tmp_path):
        """Test that explicit missing defaults_path raises FileNotFoundError."""
        # Create user config only
        user_cfg = {
            "experiment_name": "exp",
            "commands": [{
                "command": "train",
                "mixer": {"type": "sequence", "datasets": ["ds1"]}
            }]
        }
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        write_yaml(exp_dir / "config.yaml", user_cfg)
        
        # Explicit bad defaults path
        cr = ConfigReader(str(exp_dir), defaults_path=str(tmp_path / "nonexistent.yaml"))
        with pytest.raises(FileNotFoundError, match="Defaults file"):
            cr.load()
    
    def test_missing_user_config_raises_error(self, tmp_path):
        """Test that missing user config raises FileNotFoundError."""
        cr = ConfigReader(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No config file found"):
            cr.load()
    
    def test_both_load_and_new_validation_fails(self, tmp_path):
        """Test that having both load and new in a command fails validation."""
        user_cfg = {
            "experiment_name": "bad",
            "commands": [{
                "command": "train",
                "mixer": {"type": "sequence", "datasets": ["ds1"]},
                "load": {"model": "path"},
                "new": {"model": {}}
            }]
        }
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        write_yaml(exp_dir / "config.yaml", user_cfg)
        
        cr = ConfigReader(str(exp_dir))
        with pytest.raises(ValueError, match="mutually exclusive"):
            cr.load()
    
    def test_load_caches_result(self, tmp_path):
        """Test that load caches result and doesn't reload."""
        user_cfg = {
            "experiment_name": "exp",
            "commands": [{
                "command": "train",
                "mixer": {"type": "sequence", "datasets": ["ds1"]}
            }]
        }
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        write_yaml(exp_dir / "config.yaml", user_cfg)
        
        cr = ConfigReader(str(exp_dir))
        result1 = cr.load()
        result2 = cr.load()
        
        assert result1 is result2  # Same object
    
    def test_save_effective_config(self, tmp_path):
        """Test saving effective config to JSON."""
        user_cfg = {
            "experiment_name": "exp",
            "commands": [{
                "command": "train",
                "mixer": {"type": "sequence", "datasets": ["ds1"]}
            }]
        }
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        write_yaml(exp_dir / "config.yaml", user_cfg)
        
        cr = ConfigReader(str(exp_dir))
        output_file = tmp_path / "effective.json"
        cr.save_effective_config(str(output_file))
        
        assert output_file.exists()
        loaded = json.loads(output_file.read_text(encoding="utf-8"))
        assert loaded["experiment_name"] == "exp"
        assert isinstance(loaded["commands"], list)
    
    def test_helper_accessors(self, tmp_path):
        """Test that helper accessor methods work correctly."""
        user_cfg = {
            "features": {
                "default_label": "CustomLabel",
                "protocols_to_discard": ["custom"]
            },
            "preprocessing": {"steps": ["step1"]},
            "model": {"wrapper": "CustomWrapper"},
            "commands": [{
                "command": "train",
                "mixer": {"type": "sequence", "datasets": ["ds1"]}
            }]
        }
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        write_yaml(exp_dir / "config.yaml", user_cfg)
        
        cr = ConfigReader(str(exp_dir))
        
        # Test accessors
        feat_params = cr.get_feature_extractor_params()
        assert feat_params["default_label"] == "CustomLabel"
        
        steps = cr.get_preprocessing_steps()
        assert steps == ["step1"]
        
        model = cr.get_model_spec()
        assert model["wrapper"] == "CustomWrapper"
        
        cmds = cr.get_commands()
        assert len(cmds) == 1