# pipeline_ml_training/config_reader.py
"""
Minimal ConfigReader — single source of truth (no defaults).

Behavior:
 - Read exactly one config file (YAML or JSON).
 - Validate presence of required keys and structure.
 - Resolve experiment and preprocessing paths relative to config when needed.
 - Compute per-command effective_batch_size and effective_validation_split.
 - Resolve & validate plotting script paths (must exist).
 - Provide small accessor helpers used by the pipeline.

If anything required is missing or invalid, load() raises with a clear error.
"""

from pathlib import Path
import json

try:
    import yaml
except Exception:
    raise ImportError("PyYAML required. Install with: pip install pyyaml")

YAML_NAMES = ("config.yaml", "config.yml")
JSON_NAMES = ("config.json",)


def _load_file(path: Path):
    txt = path.read_text(encoding="utf-8")
    sfx = path.suffix.lower()
    if sfx in (".yaml", ".yml"):
        data = yaml.safe_load(txt) or {}
    elif sfx == ".json":
        data = json.loads(txt)
    else:
        # try yaml then json
        try:
            data = yaml.safe_load(txt) or {}
        except Exception:
            data = json.loads(txt)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config in {path} must be a mapping")
    return data


class ConfigReader:
    def __init__(self, path_or_dir: str):
        self.base = Path(path_or_dir)
        self.config_path: Path | None = None
        self._cfg: dict | None = None

    def _find_user_config(self) -> Path | None:
        p = self.base
        if p.is_file():
            return p
        if p.is_dir():
            for name in YAML_NAMES + JSON_NAMES:
                cand = p / name
                if cand.exists():
                    return cand
        return None

    def load(self) -> dict:
        if self._cfg is not None:
            return self._cfg

        cfg_path = self._find_user_config()
        if cfg_path is None:
            raise FileNotFoundError(f"No config file found under '{self.base}'")

        cfg = _load_file(cfg_path)
        self.config_path = cfg_path

        # --- Minimal strict validation of top-level keys ---
        required_top = ("experiment_name", "root", "classes", "commands", "paths", "model")
        for k in required_top:
            if k not in cfg:
                raise ValueError(f"Missing required top-level key: '{k}'")

        if not isinstance(cfg["classes"], list) or not cfg["classes"]:
            raise ValueError("Top-level 'classes' must be a non-empty list")

        if not isinstance(cfg["commands"], list):
            raise ValueError("'commands' must be a list")

        # paths: require experiment_dir
        if not isinstance(cfg["paths"], dict) or "experiment_dir" not in cfg["paths"]:
            raise ValueError("paths.experiment_dir is required")

        # plotting paths must be present and point to existing files (resolved relative to config file)
        plotting = cfg.get("plotting_pths")
        if not isinstance(plotting, dict):
            raise ValueError("plotting_pths mapping is required with keys 'training' and 'testing'")
        train_plot = plotting.get("training")
        test_plot = plotting.get("testing")
        if not train_plot or not test_plot:
            raise ValueError("plotting_pths.training and plotting_pths.testing must be provided")
        # resolve relative to config parent
        cfg_dir = cfg_path.parent
        train_path = (cfg_dir / train_plot).resolve() if not Path(train_plot).is_absolute() else Path(train_plot)
        test_path = (cfg_dir / test_plot).resolve() if not Path(test_plot).is_absolute() else Path(test_plot)
        if not train_path.exists():
            raise FileNotFoundError(f"Training plotting script not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Testing plotting script not found: {test_path}")
        # store resolved plotting paths back
        cfg["plotting_pths"] = {"training": str(train_path), "testing": str(test_path)}

        # dataset_loader: must be mapping and must include batch_size and data_subdir
        ds = cfg.get("dataset_loader")
        if not isinstance(ds, dict):
            raise ValueError("dataset_loader mapping required")
        if "batch_size" not in ds:
            raise ValueError("dataset_loader.batch_size is required")
        if "data_subdir" not in ds:
            raise ValueError("dataset_loader.data_subdir is required")
        if "cache_dir" in ds:
            ds_cache = Path(ds["cache_dir"])
            if not ds_cache.is_absolute():
                ds_cache = (cfg_dir / ds["cache_dir"]).resolve()
            ds["cache_dir_resolved"] = str(ds_cache)

        # model: must contain classifier_type (or type)
        model = cfg["model"]
        if not isinstance(model, dict):
            raise ValueError("model must be a mapping")
        if not (model.get("classifier_type") or model.get("type")):
            raise ValueError("model.classifier_type (or model.type) is required")

        # Validate commands and build per-command effective values & paths
        exp_name = cfg["experiment_name"]
        base_exp = Path(cfg["paths"]["experiment_dir"])
        exp_resolved = (base_exp / exp_name).resolve()
        cfg["paths"]["experiment_dir_resolved"] = str(exp_resolved)
        # preprocessing dir resolved
        pdir_name = cfg["paths"].get("preprocessing_dir_name", "preprocessing")
        cfg["paths"]["preprocessing_dir_resolved"] = str((exp_resolved / pdir_name).resolve())

        # ensure top-level batch_size_train/test exist if commands omit batch_size
        # but only raise if needed later
        for idx, cmd in enumerate(cfg["commands"]):
            if not isinstance(cmd, dict):
                raise ValueError(f"commands[{idx}] must be a mapping")
            if cmd.get("command") not in ("train", "test"):
                raise ValueError(f"commands[{idx}].command must be 'train' or 'test'")

            mixer = cmd.get("mixer")
            if not isinstance(mixer, dict):
                raise ValueError(f"commands[{idx}].mixer is required and must be a mapping")
            mtype = mixer.get("type")
            if mtype not in ("sequence", "random_batches", "balanced_by_label"):
                raise ValueError(f"commands[{idx}].mixer.type unknown: '{mtype}'")

            # compute effective_validation_split: command -> mixer -> top-level required 'validation_split'
            cmd_val = cmd.get("validation_split")
            mix_val = mixer.get("validation_split")
            if cmd_val is not None:
                eff_val = float(cmd_val)
            elif mix_val is not None:
                eff_val = float(mix_val)
            else:
                if "validation_split" not in cfg:
                    raise ValueError("Top-level 'validation_split' required when not provided per-command/mixer")
                eff_val = float(cfg["validation_split"])
            cmd["effective_validation_split"] = eff_val

            # effective batch size: command.batch_size if present else top-level batch_size_train/test required
            if "batch_size" in cmd:
                eff_bs = int(cmd["batch_size"])
            else:
                if cmd["command"] == "test":
                    if "batch_size_test" not in cfg:
                        raise ValueError("Top-level 'batch_size_test' required when command.batch_size not specified")
                    eff_bs = int(cfg["batch_size_test"])
                else:
                    if "batch_size_train" not in cfg:
                        raise ValueError("Top-level 'batch_size_train' required when command.batch_size not specified")
                    eff_bs = int(cfg["batch_size_train"])
            cmd["effective_batch_size"] = eff_bs

            # per-command file-system paths under experiment_dir
            safe_name = cmd.get("name") or f"cmd_{idx}"
            cmd_base = exp_resolved / "commands" / f"{idx}_{safe_name}"
            cmd["paths"] = {
                "command_dir": str(cmd_base.resolve()),
                "preprocessing": str((cmd_base / "preprocessing").resolve()),
                "model": str((cmd_base / "model").resolve()),
                "results": str((cmd_base / "results").resolve()),
                "logs": str((cmd_base / "logs").resolve()),
            }

            # write effective values into mixer mapping for downstream use
            mixer["validation_split"] = cmd["effective_validation_split"]
            if "batch_size" not in mixer:
                mixer["batch_size"] = cmd["effective_batch_size"]

        # final validation passed
        self._cfg = cfg
        return cfg

    # ----------------- small accessors -----------------
    def get_feature_extractor_params(self):
        cfg = self.load()
        feats = cfg.get("features", {})
        return {
            "default_label": feats.get("default_label"),
            "protocols_to_discard": feats.get("protocols_to_discard"),
            "columns_to_discard": feats.get("columns_to_discard"),
            "column_types": feats.get("column_types"),
        }

    def get_preprocessing_steps(self):
        return self.load().get("preprocessing", {}).get("steps", [])

    def get_preprocessing_save_options(self):
        cfg = self.load()
        return {
            "save_steps": cfg.get("preprocessing", {}).get("save_steps", True),
            "step_filename_template": cfg.get("preprocessing", {}).get("step_filename_template", "{name}.bin"),
            "preprocessing_dir": cfg["paths"]["preprocessing_dir_resolved"],
        }

    def get_model_spec(self):
        cfg = self.load()
        return cfg.get("model", {})

    def get_dataset_loader_params(self):
        cfg = self.load()
        return cfg.get("dataset_loader", {})

    def get_paths(self):
        return self.load().get("paths", {})

    def get_commands(self):
        return self.load().get("commands", [])

    def get_command_load_spec(self, cmd_or_idx):
        cfg = self.load()
        if isinstance(cmd_or_idx, int):
            cmds = cfg.get("commands", [])
            if 0 <= cmd_or_idx < len(cmds):
                return cmds[cmd_or_idx].get("load")
            return None
        if isinstance(cmd_or_idx, dict):
            return cmd_or_idx.get("load")
        return None

    def get_command_new_spec(self, cmd_or_idx):
        cfg = self.load()
        if isinstance(cmd_or_idx, int):
            cmds = cfg.get("commands", [])
            if 0 <= cmd_or_idx < len(cmds):
                return cmds[cmd_or_idx].get("new")
            return None
        if isinstance(cmd_or_idx, dict):
            return cmd_or_idx.get("new")
        return None

    def get_plotting_paths(self):
        cfg = self.load()
        p = cfg.get("plotting_pths", {})
        return p.get("training"), p.get("testing")

    def get_random_seed(self):
        cfg = self.load()
        if "seed" not in cfg:
            raise ValueError("Top-level 'seed' is required")
        return int(cfg["seed"])

    def get_classes(self):
        cfg = self.load()
        return cfg["classes"]

    def save_effective_config(self, target: str):
        cfg = self.load()
        p = Path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
