# Slips ML Training Pipeline

Supplementary machine learning training pipeline for cybersecurity models used by the SLIPS project from the Stratosphere Laboratory.

This repository is intentionally separate from SLIPS itself, but it is designed to produce models and preprocessing artifacts that remain compatible with SLIPS ML modules.

## Overview

The pipeline trains and evaluates classifiers on labeled network-flow datasets, normalizes them to the SLIPS schema, extracts features, applies preprocessing, and stores reproducible experiment outputs. It also supports Optuna-based hyperparameter optimization for multi-objective model search.

Typical use cases:

- Train a baseline model on Zeek-style labeled flow datasets
- Evaluate trained models on held-out or unseen datasets
- Compare dataset mixing strategies such as sequential, random, balanced, and oversampled training
- Run Optuna studies to search classifier and preprocessing settings

## Key Capabilities

- End-to-end training and testing driven by YAML configuration
- Dataset normalization from Zeek connection logs to a SLIPS-compatible schema
- Modular feature extraction, preprocessing, and classifier wrappers
- Support for both `scikit-learn` and `river`-based models
- Experiment versioning through per-run output folders and effective config snapshots
- Built-in plotting utilities for training, testing, and Optuna studies

## Repository Structure

- `run.py`: entry point for standard runs and Optuna studies
- `configs/`: example pipeline configurations
- `src/pipeline.py`: runtime orchestration of the full pipeline
- `src/dataset_wrapper.py`: dataset discovery, loading, and caching
- `src/conn_normalizer.py`: Zeek-to-SLIPS normalization
- `src/features.py`: feature extraction
- `src/preprocessing_wrapper.py`: preprocessing pipeline assembly and persistence
- `src/classifier_wrapper.py`: model wrapper abstraction
- `src/data_selectors.py`: dataset mixing and batch selection strategies
- `src/plot_utils/`: plotting scripts and Optuna visualization helpers
- `docs/OPTUNA.md`: detailed Optuna usage and output guide
- `testing/`: automated test suite

## Requirements

- Python 3.10+ (CI uses 3.10; the Docker image uses 3.13)
- `pip` and a virtual environment tool recommended
- Labeled datasets in the format expected by the configured dataset loader

Install dependencies:

```bash
pip install -r requirements.txt
```

Main Python dependencies include `numpy`, `pandas`, `scikit-learn`, `river`, `PyYAML`, `optuna`, and `matplotlib`. `numpy`, `pandas`, `scikit-learn` and `river` are pinned to the versions SLIPS uses, so pickled models and preprocessing steps load in SLIPS.

## Datasets

By default, the pipeline expects datasets under `./datasets`.

Expected structure:

- `<root>/<NNN...>/data/` with one of `conn.log.labeled`, `labeled-conn.log`, `conn.log` (first match wins; configurable via `dataset_loader.labeled_filenames`)
- dataset folder names start with three digits (`dataset_loader.prefix_regex`, default `^\d{3}`); commands reference a dataset by a unique substring of its folder name, e.g. `"010"`
- logs are Zeek TSV logs with a `#fields` header and a `label` column; flows labeled Background are skipped, labels with a word starting with `MAL` (e.g. `Malicious`, `Malware`, `ATTACK_MALWARE`) map to Malicious and all others (including e.g. `Normal`) to Benign

You can:

- Use your own datasets if they match the expected structure
- Adapt feature extraction and loading logic for custom formats
- Use Stratosphere's labeled security datasets: https://github.com/stratosphereips/security-datasets-for-testing

To keep a consistent local path while storing the actual data elsewhere, use a symlink:

```bash
ln -s /absolute/path/to/your/datasets ./datasets
```

If you need the resolved absolute path later, for example when mounting the data into Docker, use:

```bash
readlink -f datasets
```

## Quick Start

### Native Run

Run the default configuration:

```bash
python run.py configs/default_config.yaml
```

Run an Optuna study:

```bash
python run.py configs/optuna_conf.yaml --optuna
```

`configs/optuna_conf.yaml` is the full search (512 trials, 16 parallel jobs); `configs/demo_config.yaml` runs a short 8-trial study that is useful as a smoke test:

```bash
python run.py configs/demo_config.yaml --optuna
```

Run commands from the repository root: dataset, cache, experiment and plotting-script paths in the configs are relative to the current working directory.

### Docker Run

Build the image:

```bash
docker build -t slips-pipeline:latest .
```

Run with Docker Compose:

```bash
docker compose run --rm pipeline python run.py configs/default_config.yaml
docker compose run --rm pipeline python run.py configs/optuna_conf.yaml --optuna
```

If `./datasets` is a symlink, mount the resolved target so the container sees the same data path:

```bash
DATA_ROOT=$(readlink -f datasets)
docker compose run --rm \
  -v "$DATA_ROOT":"$DATA_ROOT" \
  pipeline \
  python run.py configs/default_config.yaml
```

Current resource defaults in [`docker-compose.yml`](docker-compose.yml):

- `cpus: 4`
- `mem_reservation: 16g`
- `mem_limit: 20g`
- `memswap_limit: 32g`

## Configuration

Pipeline behavior is defined in YAML configuration files under `configs/`.

Required keys:

- `experiment_name`: base name for the run output directory
- `root`: dataset root, typically `./datasets`
- `classes`: label set used during training and evaluation (`["Benign", "Malicious"]`)
- `batch_size_train`, `batch_size_test`: batch sizes for train and test commands
- `plotting_pths`: `training` and `testing` plot scripts run after each command
- `model`: wrapper type, classifier type, and model parameters (or `load_path` to evaluate a saved model)
- `commands`: ordered runtime commands; each has a `name`, `command: train|test`, and a `mixer` (`type`: `sequence`, `random`, `balanced` or `oversampling`, plus `datasets`)

Optional keys:

- `seed` (default 1111), `validation_split` (default 0.0; applies to train commands)
- `paths.experiment_dir` (default `./experiments`)
- `dataset_loader`: file discovery, caching, and parsing settings (`data_subdir`, `prefix_regex`, `labeled_filenames`, `cache_dir`, `persist_cache_threshold`, `file_encoding`, `file_errors`)
- `features`: feature extraction behavior
- `preprocessing`: ordered sklearn-style preprocessing steps (`steps`, `save_steps`, `step_filename_template`, `load_steps`)
- `optuna`: search space and study settings, used with `--optuna` (see [`docs/OPTUNA.md`](docs/OPTUNA.md))

When a run starts, the pipeline creates a unique experiment directory (`experiments/<experiment_name>`, or `<experiment_name>_N` if it exists) and stores:

- `config_effective.yaml`, `config_effective.json`: the resolved config
- `pipeline_run_log.txt`: timestamped log of the run
- `output/logs/`: per-command logs (`{idx}_{name}_train.log`, `{idx}_{name}_test.log`)
- `output/models/`: the trained model (`classifier.bin` unless `model.save_name` is set)
- `output/preprocessing/`: the fitted preprocessing steps (`<step_name>.bin`)
- `output/results/`: plots and summaries from the plotting scripts

See [`configs/default_config.yaml`](configs/default_config.yaml) for a baseline example.

## Execution Model

At runtime, the pipeline is assembled from the configuration in these stages:

1. Configuration loading and path resolution
2. Dataset discovery and normalization into the SLIPS schema
3. Batch selection through the configured mixer
4. Feature extraction and preprocessing
5. Incremental training or testing through the classifier wrapper
6. Metric calculation, artifact persistence, and plotting

This separation keeps data handling, preprocessing, model execution, and experiment tracking modular and replaceable.

## Output

Experiment outputs are written under the configured experiment root, usually `./experiments/<experiment_name>/` (layout listed under [Configuration](#configuration)).

Optuna runs additionally create an `optuna/` subdirectory with trial-level metrics, trial configs, trained trial models, per-trial logs, summaries, and visualization outputs; in Optuna mode `output/` stays empty.

Errors are not swallowed: a failing batch stops a normal run with an error (exit code 1), and fails only that trial in Optuna mode (recorded in `optuna_trials.log` and the trial's `metrics.json`).

For full Optuna details, see [`docs/OPTUNA.md`](docs/OPTUNA.md).

## Testing

Run the full test suite with:

```bash
pytest
```

There is also a testing guide in [`testing/README.md`](testing/README.md).

## Profiling

To profile the pipeline with `cProfile`:

```bash
mkdir -p profiling
python -m cProfile -o profiling/profile.out run.py configs/default_config.yaml
```

To inspect the profile with `snakeviz`:

```bash
pip install snakeviz
snakeviz profiling/profile.out
```

## Development

For local quality checks:

```bash
pre-commit install
pre-commit run --all-files
pytest
```

If secret scanning is part of your workflow:

```bash
pip install detect-secrets
detect-secrets scan > .secrets.baseline
```

Contributions should keep tests passing and preserve compatibility with the existing configuration-driven pipeline structure.

## Extending the Pipeline

### Add a New Model

- Implement or extend a wrapper in `src/classifier_wrapper.py`
- Ensure the wrapper exposes the expected training, prediction, save, and load behavior
- Register the wrapper or classifier in `src/class_factory.py` if needed
- Point the `model` section of the config to the selected wrapper and classifier

### Add a New Preprocessing Step

- Use an sklearn-style transformer with `fit` or `partial_fit` and `transform`
- Add it through the `preprocessing.steps` config section
- Saved preprocessing artifacts are written under the experiment output directory

Example:

```python
preprocessor.add_step("scaler", StandardScaler())
```
