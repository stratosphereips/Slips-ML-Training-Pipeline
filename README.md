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

- Python 3
- `pip` and a virtual environment tool recommended
- Labeled datasets in the format expected by the configured dataset loader

Install dependencies:

```bash
pip install -r requirements.txt
```

Main Python dependencies include `numpy`, `pandas`, `scikit-learn`, `river`, `PyYAML`, `optuna`, and `matplotlib`.

## Datasets

By default, the pipeline expects datasets under `./datasets`.

You can:

- Use your own datasets if they match the expected structure
- Adapt feature extraction and loading logic for custom formats
- Use Stratosphere's labeled security datasets: https://github.com/stratosphereips/security-datasets-for-testing

To keep a consistent local path while storing the actual data elsewhere, use a symlink:

```bash
mkdir -p datasets
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

Current resource defaults in [`docker-compose.yml`](/Users/sebas/cloud9/Trabajo/Doctorado/WorkinCVUT/StratosphereProject/code/Slips-ML-Training-Pipeline/docker-compose.yml):

- `cpus: 4`
- `mem_reservation: 16g`
- `mem_limit: 20g`
- `memswap_limit: 32g`

## Configuration

Pipeline behavior is defined in YAML configuration files under `configs/`.

Core sections include:

- `experiment_name`: base name for the run output directory
- `root`: dataset root, typically `./datasets`
- `classes`: label set used during training and evaluation
- `dataset_loader`: file discovery, caching, and parsing settings
- `features`: feature extraction behavior
- `preprocessing`: ordered sklearn-style preprocessing steps
- `model`: wrapper type, classifier type, and model parameters
- `commands`: ordered runtime commands such as `train` and `test`

When a run starts, the pipeline creates a unique experiment directory and stores:

- `config_effective.yaml`
- `config_effective.json`
- model and preprocessing artifacts
- logs, metrics, and plots

See [`configs/default_config.yaml`](/Users/sebas/cloud9/Trabajo/Doctorado/WorkinCVUT/StratosphereProject/code/Slips-ML-Training-Pipeline/configs/default_config.yaml) for a baseline example.

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

Experiment outputs are written under the configured experiment root, usually `./experiments/<experiment_name>/`.

Common artifacts include:

- effective runtime configuration snapshots
- trained model files
- saved preprocessing steps
- training and testing logs
- metric summaries
- plots generated by the configured plotting scripts

Optuna runs additionally create an `optuna/` subdirectory with trial-level metrics, trial configs, summaries, and visualization outputs.

For full Optuna details, see [`docs/OPTUNA.md`](/Users/sebas/cloud9/Trabajo/Doctorado/WorkinCVUT/StratosphereProject/code/Slips-ML-Training-Pipeline/docs/OPTUNA.md).

## Testing

Run the full test suite with:

```bash
pytest
```

There is also a testing guide in [`testing/README.md`](/Users/sebas/cloud9/Trabajo/Doctorado/WorkinCVUT/StratosphereProject/code/Slips-ML-Training-Pipeline/testing/README.md).

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
