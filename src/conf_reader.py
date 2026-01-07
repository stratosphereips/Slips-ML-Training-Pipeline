# pipeline_ml_training/config_reader.py


import json
from pathlib import Path

try:
    import yaml
except Exception:
    raise ImportError("PyYAML required. Install with: pip install pyyaml")

YAML_NAMES = ("config.yaml", "config.yml")
JSON_NAMES = ("config.json",)


# ----------------------------- helpers -----------------------------

def _load_yaml_file(path):
    txt = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(txt) or {}
    except Exception:
        data = json.loads(txt)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config in {path} must be a mapping")
    return data


def _deep_merge(default, override):
    if default is None:
        return override
    if override is None:
        return default
    if isinstance(default, dict) and isinstance(override, dict):
        out = dict(default)
        for k, v in override.items():
            out[k] = _deep_merge(default.get(k), v)
        return out
    return override


# --------------------------- ConfigReader ---------------------------

class ConfigReader:
    def __init__(self, path_or_file, defaults_path=None):
        self.base = Path(path_or_file)
        self.defaults_path = Path(defaults_path) if defaults_path else None
        self.config_path = None
        self._raw_user = None
        self._raw_defaults = None
        self._resolved = None

    # ----------------------- discovery -----------------------

    def _find_user_config(self):
        if self.base.is_file():
            return self.base
        if self.base.is_dir():
            for name in YAML_NAMES + JSON_NAMES:
                p = self.base / name
                if p.exists():
                    return p
        return None

    def _find_default_config(self):
        if self.defaults_path:
            if not self.defaults_path.exists():
                raise FileNotFoundError(self.defaults_path)
            return self.defaults_path
        # Try to find default_config.yaml one level above this file
        p = Path(__file__).parent.parent / "default_config.yaml"
        return p if p.exists() else None

    # ----------------------- validation -----------------------

    def _validate(self, cfg):
        # classes must exist and be top-level only
        if "classes" not in cfg or not isinstance(cfg["classes"], list):
            raise ValueError("Top-level 'classes' (list) is required")

        if "model" in cfg and isinstance(cfg["model"], dict):
            if "classes" in cfg["model"]:
                raise ValueError("Do not set model.classes; use top-level classes")

        # forbid batch_size anywhere except top-level train/test
        if "dataset_loader" in cfg and "batch_size" in cfg["dataset_loader"]:
            raise ValueError(
                "Do not set dataset_loader.batch_size; use batch_size_train / batch_size_test"
            )

        # commands
        cmds = cfg.get("commands")
        if not isinstance(cmds, list):
            raise ValueError("'commands' must be a list")

        for i, cmd in enumerate(cmds):
            if not isinstance(cmd, dict):
                raise ValueError(f"commands[{i}] must be a mapping")
            if cmd.get("command") not in ("train", "test"):
                raise ValueError(f"commands[{i}].command must be 'train' or 'test'")
            mixer = cmd.get("mixer")
            if not isinstance(mixer, dict):
                raise ValueError(f"commands[{i}].mixer is required")
            if mixer.get("type") not in (
                "sequence",
                "random_batches",
                "balanced_by_label",
            ):
                raise ValueError(f"commands[{i}]: invalid mixer.type")

    # ----------------------- load -----------------------

    def load(self):
        if self._resolved is not None:
            return self._resolved

        # load defaults
        dpath = self._find_default_config()
        self._raw_defaults = _load_yaml_file(dpath) if dpath else {}

        # load user
        cfg_path = self._find_user_config()
        if cfg_path is None:
            raise FileNotFoundError(f"No config file found under {self.base}")
        self.config_path = cfg_path
        self._raw_user = _load_yaml_file(cfg_path)

        # merge defaults <- user
        merged = _deep_merge(self._raw_defaults, self._raw_user)

        # VALIDATE PURE DECLARATIVE CONFIG
        self._validate(merged)

        # ---------------- runtime fallbacks / derived ----------------

        merged.setdefault("seed", 1111)
        merged.setdefault("validation_split", 0.1)
        merged.setdefault("batch_size_train", 500)
        merged.setdefault("batch_size_test", 1000)

        # paths
        paths = merged.setdefault("paths", {})
        paths.setdefault("experiment_dir", "./experiments")
        paths.setdefault("preprocessing_dir_name", "preprocessing")

        exp = merged.get("experiment_name", "experiment")
        base = Path(paths["experiment_dir"])
        paths["experiment_dir_resolved"] = str((base / exp).resolve())
        paths["preprocessing_dir_resolved"] = str(
            (Path(paths["experiment_dir_resolved"]) / paths["preprocessing_dir_name"]).resolve()
        )

        # dataset loader (NO batch size here)
        ds = merged.setdefault("dataset_loader", {})
        ds.setdefault("data_subdir", "data")
        ds.setdefault("persist_cache_threshold", 30000)
        ds.setdefault("cache_dir", "./cache")
        ds["cache_dir_resolved"] = str(Path(ds["cache_dir"]).resolve())

        # preprocessing
        pc = merged.setdefault("preprocessing", {})
        pc.setdefault("steps", [])
        pc.setdefault("save_steps", True)
        pc.setdefault("step_filename_template", "{name}.bin")

        # model: IGNORE dummy_flows entirely
        model = merged.setdefault("model", {})
        model.pop("dummy_flows", None)

        # commands: effective values
        for idx, cmd in enumerate(merged.get("commands", [])):
            if "batch_size" in cmd:
                raise ValueError("Do not set command.batch_size")
            if cmd["command"] == "test":
                cmd["effective_batch_size"] = merged["batch_size_test"]
            else:
                cmd["effective_batch_size"] = merged["batch_size_train"]

            cmd_val = cmd.get("validation_split")
            mix_val = cmd.get("mixer", {}).get("validation_split")
            cmd["effective_validation_split"] = (
                mix_val if mix_val is not None else cmd_val if cmd_val is not None else merged["validation_split"]
            )

        self._resolved = merged
        return merged

    # ----------------------- accessors -----------------------

    def get_feature_extractor_params(self):
        cfg = self.load()
        feats = cfg.get("features", {})
        return {
            "default_label": feats.get("default_label"),
            "protocols_to_discard": feats.get("protocols_to_discard", []),
            "columns_to_discard": feats.get("columns_to_discard", []),
            "column_types": feats.get("column_types", {}),
        }

    def get_preprocessing_steps(self):
        cfg = self.load()
        return cfg.get("preprocessing", {}).get("steps", [])

    def get_preprocessing_save_options(self):
        cfg = self.load()
        return {
            "save_steps": cfg.get("preprocessing", {}).get("save_steps", True),
            "step_filename_template": cfg.get("preprocessing", {}).get(
                "step_filename_template", "{name}.bin"
            ),
            "preprocessing_dir": cfg.get("paths", {}).get(
                "preprocessing_dir_resolved"
            ),
        }

    def get_model_spec(self):
        """
        Model spec accessor.
        Enforces:
        - top-level classes only
        - dummy_flows ignored
        """
        cfg = self.load()
        model = dict(cfg.get("model", {}))
        model.pop("classes", None)
        model.pop("dummy_flows", None)
        return model

    def get_dataset_loader_params(self):
        """
        Dataset loader params.
        Injects batch_size at access time only.
        """
        cfg = self.load()
        ds = dict(cfg.get("dataset_loader", {}))
        ds["batch_size"] = int(cfg.get("batch_size_train"))
        return ds

    def get_paths(self):
        cfg = self.load()
        return cfg.get("paths", {})

    def get_commands(self):
        cfg = self.load()
        return cfg.get("commands", [])

    def get_command_load_spec(self, cmd_or_idx):
        cfg = self.load()
        cmds = cfg.get("commands", [])

        if isinstance(cmd_or_idx, int):
            if 0 <= cmd_or_idx < len(cmds):
                return cmds[cmd_or_idx].get("load")
            return None
        if isinstance(cmd_or_idx, dict):
            return cmd_or_idx.get("load")
        return None

    def get_command_new_spec(self, cmd_or_idx):
        cfg = self.load()
        cmds = cfg.get("commands", [])

        if isinstance(cmd_or_idx, int):
            if 0 <= cmd_or_idx < len(cmds):
                return cmds[cmd_or_idx].get("new")
            return None
        if isinstance(cmd_or_idx, dict):
            return cmd_or_idx.get("new")
        return None

    def get_plotting_paths(self):
        cfg = self.load()
        pths = cfg.get("plotting_pths", {}) or {}
        return pths.get("training"), pths.get("testing")

    def get_random_seed(self):
        """
        Original accessor, preserved.
        """
        cfg = self.load()
        return int(cfg.get("seed", cfg.get("random_seed", 1111)))

    def get_classes(self):
        """
        Single source of truth for label space.
        """
        cfg = self.load()
        classes = cfg.get("classes")
        if not classes:
            raise ValueError(
                "Top-level 'classes' must be defined and non-empty"
            )
        return classes
