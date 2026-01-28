
"""
Minimal ConfigReader
Usage:
    cr = ConfigReader(path_or_file)
    cfg = cr.load()
Access helpers:
    get_feature_extractor_params(), get_preprocessing_steps(), get_model_spec(),
    get_dataset_loader_params(), get_paths(), get_commands(),
    get_plotting_paths(), get_random_seed(), get_classes()
"""

from pathlib import Path
import json
from typing import Tuple, Optional

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML required. Install with: pip install pyyaml") from e

YAML_FILENAMES = ("config.yaml", "config.yml")
JSON_FILENAMES = ("config.json",)


def _load_file(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(txt) or {}
    elif suffix == ".json":
        data = json.loads(txt)
    else:
        # prefer yaml, fallback to json
        try:
            data = yaml.safe_load(txt) or {}
        except Exception:
            data = json.loads(txt)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level document in {path} must be a mapping (dict).")
    return data


class ConfigReader:
    """
    Minimal, strict config reader. Expects a single config file (yaml/json).
    No implicit defaults are injected except a few derived/resolved path fields.
    """

    # which fields must be present at top level
    REQUIRED_TOP_LEVEL = [
        "experiment_name",
        "root",
        "classes",
        "batch_size_train",
        "batch_size_test",
        "model",
        "commands",
        "plotting_pths",
    ]

    def __init__(self, path_or_dir: str):
        self.base = Path(path_or_dir)
        self.config_path: Optional[Path] = None
        self._raw_config: Optional[dict] = None
        self._resolved: Optional[dict] = None

    # ---------------- discovery ----------------
    def _find_config_file(self) -> Optional[Path]:
        p = self.base
        if p.is_file():
            return p
        if p.is_dir():
            for name in YAML_FILENAMES + JSON_FILENAMES:
                cand = p / name
                if cand.exists():
                    return cand
        return None

    # ---------------- validation ----------------
    def _assert_keys_present(self, d: dict, keys: list):
        missing = [k for k in keys if k not in d or d.get(k) is None]
        if missing:
            raise ValueError("Configuration missing required top-level keys: " + ", ".join(missing))

    def _validate_commands(self, cfg: dict):
        if not isinstance(cfg.get("commands"), list):
            raise ValueError("'commands' must be a list")
        for i, cmd in enumerate(cfg.get("commands", [])):
            if not isinstance(cmd, dict):
                raise ValueError(f"commands[{i}] must be a mapping")
            ctype = cmd.get("command")
            if ctype not in ("train", "test"):
                raise ValueError(f"commands[{i}].command must be 'train' or 'test'")
            mixer = cmd.get("mixer")
            if not isinstance(mixer, dict):
                raise ValueError(f"commands[{i}].mixer is required and must be a mapping")
            # Do not allow per-command batch_size; top-level sizes only
            if "batch_size" in cmd:
                raise ValueError(f"Do not set commands[{i}].batch_size — use top-level batch_size_train/test")

    def _validate_plotting_paths(self, cfg: dict):
        pths = cfg.get("plotting_pths")
        if not isinstance(pths, dict):
            raise ValueError("'plotting_pths' must be a mapping with 'training' and 'testing'")
        if "training" not in pths or "testing" not in pths:
            raise ValueError("'plotting_pths' must contain keys 'training' and 'testing'")

    # ---------------- load & resolve ----------------
    def load(self) -> dict:
        if self._resolved is not None:
            return self._resolved

        cfg_file = self._find_config_file()
        if cfg_file is None:
            raise FileNotFoundError(f"No config file found at or under: {self.base}")
        self.config_path = cfg_file
        self._raw_config = _load_file(cfg_file)

        # Validate required top-level keys (single source of truth)
        self._assert_keys_present(self._raw_config, self.REQUIRED_TOP_LEVEL)

        # Basic structural validations
        self._validate_plotting_paths(self._raw_config)
        self._validate_commands(self._raw_config)

        # create a copy we'll modify (clear intent)
        effective_config = dict(self._raw_config)

        # ------------------------------
        # propagate top-level batch sizes into dataset_loader mapping
        # ------------------------------
        ds = effective_config.get("dataset_loader", {}) or {}
        ds["batch_size"] = int(effective_config["batch_size_train"])
        ds["batch_size_test"] = int(effective_config["batch_size_test"])
        # fill minimal dataset_loader required keys if absent (not defaults — just required shapes)
        ds.setdefault("data_subdir", "data")
        ds.setdefault("persist_cache_threshold", 30000)
        ds.setdefault("cache_dir", "./cache")
        ds["cache_dir_resolved"] = str(Path(ds["cache_dir"]).resolve())
        effective_config["dataset_loader"] = ds

        # ------------------------------
        # resolve experiment paths (experiment_dir_resolved, preprocessing_dir_resolved)
        # ------------------------------
        paths = effective_config.get("paths", {}) or {}
        paths.setdefault("experiment_dir", "./experiments")
        exp_name = effective_config["experiment_name"]
        base_experiments = Path(paths["experiment_dir"])
        paths["experiment_dir_resolved"] = str((base_experiments / exp_name).resolve())
        effective_config["paths"] = paths

        # ------------------------------
        # compute per-command effective sizes/splits
        # ------------------------------
        commands = effective_config.get("commands", [])
        for idx, cmd in enumerate(commands):
            # effective batch size (test vs train)
            cmd["effective_batch_size"] = int(
                effective_config["batch_size_test"] if cmd.get("command") == "test" else effective_config["batch_size_train"]
            )
            # effective validation split: mixer -> command -> global
            mix_val = cmd.get("mixer", {}).get("validation_split")
            cmd_val = cmd.get("validation_split")
            if mix_val is not None:
                eff_val = float(mix_val)
            elif cmd_val is not None:
                eff_val = float(cmd_val)
            else:
                eff_val = float(effective_config.get("validation_split", 0.0))
            cmd["effective_validation_split"] = eff_val

            # per-command paths under experiment dir (helpful for pipeline layout)
            safe_name = cmd.get("name") or f"cmd_{idx}"
            cmd_base = Path(paths["experiment_dir_resolved"]) / "commands" / f"{idx}_{safe_name}"
            cmd["paths"] = {
                "command_dir": str(cmd_base.resolve()),
                "preprocessing": str((cmd_base / "preprocessing").resolve()),
                "model": str((cmd_base / "model").resolve()),
                "results": str((cmd_base / "results").resolve()),
                "logs": str((cmd_base / "logs").resolve()),
            }

            # ensure mixer sees the effective values
            # propagate effective values into mixer spec so mixers see them
            if isinstance(cmd.get("mixer"), dict):
                cmd["mixer"]["validation_split"] = cmd["effective_validation_split"]
                # give mixer a batch_size if not explicitly set
                cmd["mixer"].setdefault("batch_size", cmd["effective_batch_size"])

        # final store
        self._resolved = effective_config
        return self._resolved

    # ---------------- accessors ----------------
    def get_feature_extractor_params(self) -> dict:
        cfg = self.load()
        feats = cfg.get("features", {}) or {}
        return {
            "default_label": feats.get("default_label"),
            "protocols_to_discard": feats.get("protocols_to_discard"),
            "columns_to_discard": feats.get("columns_to_discard"),
            "column_types": feats.get("column_types"),
        }

    def get_preprocessing_steps(self) -> list:
        cfg = self.load()
        return cfg.get("preprocessing", {}).get("steps", [])

    def get_preprocessing_save_options(self) -> dict:
        cfg = self.load()
        return {
            "save_steps": cfg.get("preprocessing", {}).get("save_steps", True),
            "step_filename_template": cfg.get("preprocessing", {}).get("step_filename_template", "{name}.bin"),
            "preprocessing_dir": cfg.get("paths", {}).get("preprocessing"),
        }

    def get_model_spec(self) -> dict:
        cfg = self.load()
        # return shallow copy; top-level classes are authoritative
        m = dict(cfg.get("model", {}))
        m.pop("classes", None)
        m.pop("dummy_flows", None)
        return m

    def get_dataset_loader_params(self) -> dict:
        cfg = self.load()
        return dict(cfg.get("dataset_loader", {}))

    def get_paths(self) -> dict:
        cfg = self.load()
        return dict(cfg.get("paths", {}))

    def get_commands(self) -> list:
        cfg = self.load()
        return list(cfg.get("commands", []))

    def get_plotting_paths(self) -> Tuple[str, str]:
        cfg = self.load()
        p = cfg.get("plotting_pths", {}) or {}
        return p.get("training"), p.get("testing")

    def get_random_seed(self) -> int:
        cfg = self.load()
        return int(cfg.get("seed", cfg.get("random_seed", 1111)))

    def get_classes(self) -> list:
        cfg = self.load()
        classes = cfg.get("classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError("Top-level 'classes' must be a non-empty list")
        return classes
