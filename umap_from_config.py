#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure local src/ is importable as a package when running from this folder.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.conf_reader import ConfigReader
from src.dataset_wrapper import find_and_load_datasets
from src.features import FeatureExtraction
from src.preprocessing_wrapper import PreprocessingWrapper
from src.class_factory import get_transformer_class, get_mixer_class

try:
    from umap import UMAP
except Exception as exc:
    print("ERROR: umap-learn is required. Install with: pip install umap-learn")
    raise SystemExit(1) from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except Exception as exc:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    raise SystemExit(1) from exc


def _normalize_sample_fraction(value: float) -> float:
    if value is None:
        return 1.0
    if value > 1.0:
        value = value / 100.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _as_numpy(X):
    try:
        return X.to_numpy()
    except Exception:
        return np.asarray(X)


def _add_samples(store, X, y, split, sample_frac, rng):
    X = _as_numpy(X)
    y = np.asarray(y)
    if X.size == 0:
        return
    if sample_frac < 1.0:
        mask = rng.random(X.shape[0]) < sample_frac
        if not mask.any():
            return
        X = X[mask]
        y = y[mask]
    store["X"].append(X)
    store["split"].append(np.array([split] * len(y), dtype="object"))
    store["label"].append(y)


def _build_preprocessor(cfg_reader: ConfigReader):
    cfg = cfg_reader.load()
    prep = PreprocessingWrapper(
        steps=[],
        experiment_name=cfg.get("experiment_name", "default"),
    )
    for step in cfg_reader.get_preprocessing_steps():
        cls = get_transformer_class(step["type"])
        prep.add_step(step.get("name"), cls(**(step.get("params") or {})))

    load_from = (cfg.get("preprocessing") or {}).get("load_from")
    if load_from:
        base = Path(load_from)
        if not base.is_absolute():
            if cfg_reader.config_path is None:
                raise RuntimeError("Config path not resolved for load_from.")
            base = cfg_reader.config_path.parent / base
        prep.load(base_path=str(base))
        return prep, True
    return prep, False


def _collect_from_train(
    cmd,
    loaders,
    feature_extractor,
    preprocessor,
    preprocessor_loaded,
    sample_frac,
    rng,
    store,
):
    mixer_spec = dict(cmd.get("mixer", {}) or {})
    mixer = get_mixer_class(mixer_spec["type"])(mixer_spec, loaders, rng)
    mixer.reset_epoch(int(cmd["effective_batch_size"]), epoch_idx=0)

    while True:
        train_batch, val_batch = mixer.next_batch()
        if train_batch is None and val_batch is None:
            break

        if train_batch is not None:
            X_train, y_train = feature_extractor.process_batch(train_batch)
            if X_train is not None and len(X_train) > 0:
                if not preprocessor_loaded and preprocessor.get_steps():
                    preprocessor.partial_fit(X_train)
                X_train_p = preprocessor.transform(X_train)
                _add_samples(store, X_train_p, y_train, "train", sample_frac, rng)

        if val_batch is not None:
            X_val, y_val = feature_extractor.process_batch(val_batch)
            if X_val is not None and len(X_val) > 0:
                X_val_p = preprocessor.transform(X_val)
                _add_samples(store, X_val_p, y_val, "eval", sample_frac, rng)


def _collect_from_test(
    cmd,
    loaders,
    feature_extractor,
    preprocessor,
    sample_frac,
    rng,
    store,
):
    mixer_spec = dict(cmd.get("mixer", {}) or {})
    mixer = get_mixer_class(mixer_spec["type"])(mixer_spec, loaders, rng)
    mixer.reset_epoch(int(cmd["effective_batch_size"]), epoch_idx=0)

    while True:
        batch, _ = mixer.next_batch()
        if batch is None:
            break
        X_test, y_test = feature_extractor.process_batch(batch)
        if X_test is None or len(X_test) == 0:
            continue
        X_test_p = preprocessor.transform(X_test)
        _add_samples(store, X_test_p, y_test, "test", sample_frac, rng)


def main():
    parser = argparse.ArgumentParser(
        description="UMAP visualization using pipeline config and datasets"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="./default_config.yaml",
        help="Path to config file (default: ./default_config.yaml)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction (0-1) or percent (0-100) to sample from each split.",
    )
    parser.add_argument(
        "--output",
        default="umap.png",
        help="Output image path (default: umap.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (default: config seed).",
    )
    args = parser.parse_args()

    sample_frac = _normalize_sample_fraction(args.sample)
    if sample_frac <= 0.0:
        raise SystemExit("Sample fraction must be > 0.")

    cfg_reader = ConfigReader(args.config)
    cfg = cfg_reader.load()
    seed = args.seed if args.seed is not None else cfg_reader.get_random_seed()
    rng = np.random.default_rng(int(seed))

    ds_params = cfg_reader.get_dataset_loader_params()
    loaders = find_and_load_datasets(
        root_dir=cfg.get("root"),
        batch_size=int(ds_params.get("batch_size", cfg.get("batch_size_train", 1000))),
        prefix_regex=ds_params.get("prefix_regex", r"^\d{3}"),
        data_subdir=ds_params.get("data_subdir", "data"),
        seed=int(seed),
        persist_cache_threshold=int(ds_params.get("persist_cache_threshold", 30000)),
        cache_dir=ds_params.get("cache_dir"),
        labeled_filenames=ds_params.get("labeled_filenames"),
        file_encoding=ds_params.get("file_encoding", "utf-8"),
        file_errors=ds_params.get("file_errors", "ignore"),
        shuffle_per_epoch=bool(ds_params.get("shuffle_per_epoch", False)),
    )

    feature_extractor = FeatureExtraction(**cfg_reader.get_feature_extractor_params())
    preprocessor, preprocessor_loaded = _build_preprocessor(cfg_reader)

    store = {"X": [], "split": [], "label": []}
    for cmd in cfg_reader.get_commands():
        if cmd.get("command") == "train":
            _collect_from_train(
                cmd,
                loaders,
                feature_extractor,
                preprocessor,
                preprocessor_loaded,
                sample_frac,
                rng,
                store,
            )

    if preprocessor.get_steps() and not preprocessor_loaded:
        if not any(preprocessor.is_fitted.values()):
            raise RuntimeError("Preprocessor has steps but was never fitted.")

    for cmd in cfg_reader.get_commands():
        if cmd.get("command") == "test":
            _collect_from_test(
                cmd,
                loaders,
                feature_extractor,
                preprocessor,
                sample_frac,
                rng,
                store,
            )

    if not store["X"]:
        raise SystemExit("No samples collected. Check datasets and sample fraction.")

    X_all = np.vstack(store["X"])
    split_all = np.concatenate(store["split"])
    label_all = np.concatenate(store["label"])

    umap = UMAP(n_components=2, random_state=int(seed))
    embedding = umap.fit_transform(X_all)

    label_colors = {"Benign": "#1f77b4", "Malicious": "#d62728"}
    split_markers = {"train": "o", "eval": "^", "test": "s"}
    split_order = ["train", "eval", "test"]
    label_order = ["Benign", "Malicious"]

    plt.figure(figsize=(10, 7))
    for split in split_order:
        for label in label_order:
            mask = (split_all == split) & (label_all == label)
            if not np.any(mask):
                continue
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=label_colors.get(label, "#888888"),
                marker=split_markers.get(split, "o"),
                s=14,
                alpha=0.7,
                linewidths=0,
            )

    label_handles = [
        Patch(facecolor=label_colors[k], edgecolor="none", label=k)
        for k in label_order
        if k in label_colors
    ]
    split_handles = [
        Line2D([0], [0], marker=split_markers[k], color="k", linestyle="None", label=k)
        for k in split_order
        if k in split_markers
    ]

    ax = plt.gca()
    legend_labels = ax.legend(handles=label_handles, title="Label", loc="upper right")
    ax.add_artist(legend_labels)
    ax.legend(handles=split_handles, title="Split", loc="lower right")

    plt.title(f"UMAP (sample={sample_frac:.3f}, n={X_all.shape[0]})")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved UMAP plot to {args.output}")


if __name__ == "__main__":
    main()
