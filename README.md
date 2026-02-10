---
# Optuna Integration: Hyperparameter Search Architecture

## Optuna Architecture Overview

The pipeline now supports automated hyperparameter optimization using Optuna. This enables multi-objective search for the best classifier and mixer parameters, maximizing F1 and minimizing malware FPR.

### Key Features
- **Optuna mode**: Run with `--optuna` to enable hyperparameter search.
- **Multi-objective**: Simultaneously maximize F1 and minimize malware FPR.
- **Config-driven**: All search spaces and experiment settings are defined in the config file (see `optuna_conf.yaml`).
- **Logging**: All trial configs and results are stored in `optuna/` under each experiment folder.
- **Normal mode**: Running without `--optuna` executes the pipeline as before, with no changes to normal operation.

### Folder Structure
- `experiments/<experiment_name>/optuna/`
  - `optuna_trials.csv`: All trial parameters and results
  - `optuna_summary.json`: Best results and study info
  - `trial_{n}_config.yaml`: Config used for each trial
  - `trial_{n}_result.json`: Partial/epoch results, final metrics

### How to Run

**Normal mode:**
```bash
python run.py default_config.yaml
```

**Optuna mode:**
```bash
python run.py optuna_conf.yaml --optuna
```
This will run a multi-objective Optuna study, searching for the best combination of classifier, mixer, and other parameters as defined in the config.

### Example Optuna Config
See `optuna_conf.yaml` for a full example. Key section:
```yaml
optuna:
  enabled: true
  n_trials: 20
  metric: f1
  directions: ["maximize", "minimize"]
  hyperparameters:
    ARFClassifier:
      lambda_value:
        type: int
        low: 1
        high: 20
      n_models:
        type: int
        low: 5
        high: 50
    SGDClassifier:
      alpha:
        type: float
        low: 0.0001
        high: 0.1
        log: true
      loss:
        type: categorical
        choices: ["hinge", "log_loss"]
    Mixer:
      type:
        type: categorical
        choices: ["oversampling", "balanced", "random", "sequence"]
      round_robin_cycles:
        type: int
        low: 8
        high: 20
      stash_size_per_label:
        type: int
        low: 500
        high: 2000
      buffer_size_per_label:
        type: int
        low: 1000
        high: 3000
      micro_batch:
        type: int
        low: 16
        high: 64
      datasets:
        type: categorical
        choices: ["010", "015", "011", "010", "001", "008"]
```

### Notes
- Each Optuna trial runs a full pipeline training+validation, so expect long runtimes.
- All results are reproducible and logged for later inspection.
- Test/validation data is unified for all trials; only training datasets may change.

# How to run

Run the pipeline from the repository root (provide a config file or directory):

```bash
python run.py /path/to/config_or_config_dir
```

If you omit the argument the pipeline will look for a config in the current directory (`.`).

Results are written under `experiments/<experiment_name>`, but if a folder with the same name already exists, a numeric suffix is appended (e.g., `<experiment_name>_1`, `<experiment_name>_2`, etc.) to ensure previous results are not overwritten. The experiment folder name is generated centrally from the config and passed to all pipeline modules. Inner file names and subdirectory structures remain unchanged.

---

## Installation

Create an environment and install dependencies:

```bash
conda create -n slips-ml-pipeline python=3.10 pip -y
conda activate slips-ml-pipeline
pip install -r requirements.txt
```
Feel free to use simpler environments, like `venv` instead of conda
If you will fetch large Zeek datasets via Git, install Git LFS:

```bash
git lfs install
# optionally add the recommended dataset submodule
git submodule add https://github.com/stratosphereips/security-datasets-for-testing dataset-private
```

---
## Overview
This project provides a modular and scalable **machine learning pipeline** built in **Python** for data preprocessing, model training, evaluation for purpose of offline training models to be used in SLIPS ML modules. The models we are interested in support online learning and are able to be "extended" by partial fit, transfer learning and alike.

This repo is a compact, config-driven pipeline to:

* discover Zeek-style datasets in a dataset directory
* normalize flows to a canonical SLIPS format
* extract numeric features
* apply a configurable preprocessing pipeline
* train online-capable classifiers (sklearn / River wrapped)
* log metrics and invoke plotting scripts to generate figures

The configuration controls dataset roots, preprocessing steps, model spec, mixers and commands.

---

## Parts of the pipeline

Key modules:

* `src/dataset_wrapper.py` — Zeek dataset discovery + indexed loader
* `src/conn_normalizer.py` — Zeek → SLIPS normalization
* `src/features.py` — feature extraction (batch & single item)
* `src/preprocessing_wrapper.py` — sequence of transformers with save/load
* `src/classifier_wrapper.py` — unified interface for classifiers
* `src/data_selectors.py` — mixers / batch composition
* `src/conf_reader.py` — config loading & validation
* `src/plot_utils/` — plotting helpers used by the pipeline

---

## Usage (brief)

1. Prepare the `default_config.yaml` or modify it as you wish.
2. Ensure `root` points to your dataset root with subfolders (e.g. `root/001/data/conn.log.labeled`).
3. Run `python run.py /path/to/config`. If you don't provide config, a `default_config.yaml` is used
4. Inspect experiment outputs in `experiments/<experiment_name>`.

---

## Output

#### Training example output
```bash
=== VALIDATION Multi-class (Aggregated) ===
Accuracy:             0.9838
Malware F1:           0.9882
Malware FPR:          0.0345
Malware FNR:          0.0079
Macro F1:             0.9811
Precision:            0.9843
Recall:               0.9921
MCC:                  0.9622

=== TRAINING Multi-class (Aggregated) ===
Accuracy:             0.9758
Malware F1:           0.9828
Malware FPR:          0.0524
Malware FNR:          0.0121
Macro F1:             0.9710
Precision:            0.9776
Recall:               0.9879
MCC:                  0.9422

=== Per-class metrics (Aggregated) - VALIDATION ===
Class                 TP       TN       FP       FN      Acc     Prec      Rec       F1
Benign               336      754        6       12   0.9838   0.9825   0.9655   0.9739
Malicious            754      336       12        6   0.9838   0.9843   0.9921   0.9882

=== Per-class metrics (Aggregated) - TRAINING ===
Class                 TP       TN       FP       FN      Acc     Prec      Rec       F1
Benign              2842     6866       84      157   0.9758   0.9713   0.9476   0.9593
Malicious           6866     2842      157       84   0.9758   0.9776   0.9879   0.9828
```
#### Testing example output
```bash
[INFO] Plotting malware metrics (FPR, FNR, F1, Accuracy) over snapshots...
[INFO] Saving FPR/FNR-only plot...
[INFO] Plotting predicted vs seen counts (per-snapshot) for Malicious & Benign...
[INFO] Plotting final confusion matrix (final snapshot)...

=== Main final metrics (Aggregated so-far) ===
Accuracy:             0.9040
Malware F1:           0.9472
Malware FPR:          0.2532
Malware FNR:          0.0865
Macro F1:             0.7089
Precision:            0.9835
Recall:               0.9135

=== Per-class metrics (final snapshot) ===
Class                 TP       TN       FP       FN     Prec      Rec       F1
Malicious          59104     2929      993     5598   0.9835   0.9135   0.9472
Benign              2929    59104     5598      993   0.3435   0.7468   0.4706
```
---

##  Code Testing

Run unit tests:

```bash
pytest
```

---
# Developement
- If you want to add features, create an issue, or fork the repository and create a pull request with your new code.
- For the PR to be merged, we need all tests to be passing and pre-commit working without errors.
- Pre-commit runs linters and some checks based on the config. Here we use it to keep some code quality. If you want to contribute,
```bash
pre-commit install
pre-commit run --all-files
```

## Extending the Pipeline

### Add a new model or wrapper

* Implement a wrapper or classifier class in `src/classifier_wrapper.py`.

  * To reuse the pipeline’s training flow, extend `ClassifierWrapper` or implement the same `partial_fit`, `predict`, `save_classifier`, and `load_classifier` contract.
  * The factory `src.class_factory.get_wrapper_class` resolves wrapper names; add your wrapper class there (or reference it by dotted path in the config).
* In the `model` config, point to the classifier type (short name or dotted path) and optionally the wrapper name.

### Add a new preprocessing step

* Preprocessing steps are instantiated by the pipeline from the config. They must be sklearn-like objects with `fit`/`partial_fit` and `transform`.
* Programmatically you can add a step to `PreprocessingWrapper` with:

```python
preprocessor.add_step("scaler", StandardScaler())
```

* Saved preprocessing artifacts live under `output/preprocessing` and are re-loadable.

### Add a custom dataset loader

* `src.dataset_wrapper.find_and_load_datasets` returns a mapping of dataset_key -> loader.
* Your loader must support at least: `reset_epoch(batch_size)`, `next_batch()` and optionally `next_n(n)` to work with built-in mixers.
* Register/load your loader by replacing or extending `find_and_load_datasets` to return your loader instances keyed by dataset folder name.

### Add or modify mixers

* Mixers live in `src/data_selectors` and must follow the `DefaultMixer` contract (`reset_epoch`, `next_batch`).
* Update `src.class_factory.get_mixer_class` or reference a mixer by dotted path from the config to use a custom mixer.

---
## Loading saved models & preprocessing

The pipeline can reuse preprocessing steps and trained models from disk via configuration. This is useful for testing, fine-tuning, or continuing training from a previous run.

**Loading preprocessing steps**
* Use preprocessing.load_from to point to a directory containing saved preprocessing artifacts (*.bin).
* Absolute paths are used as-is.
* Relative paths are resolved relative to the experiment root (experiments/<experiment_name>).

```yaml
preprocessing:
  load_from: output/preprocessing
```

At startup, the pipeline calls PreprocessingWrapper.load(...) and expects one file per preprocessing step, named using the configured filename template (default: {name}.bin).
* Loading a trained model
* Use model.load_from to load a previously trained classifier from disk.
* The directory must contain a serialized classifier binary.
* The default filename is classifier.bin and can be overridden via model.load_name.

```yaml
model:
  load_from: output/models
  load_name: classifier.bin
```

During initialization, the pipeline:
* builds the classifier wrapper
* loads the classifier from the specified directory
* uses it for subsequent training or testing commands

----

## SLIPS

SLIPS (Stratosphere Linux IPS) is a behavioral machine-learning based intrusion detection and prevention system developed by the Stratosphere Laboratory. It detects malicious behaviors in network traffic and supports inputs such as PCAP files and flow logs from tools like Suricata and Zeek. For full details and installation instructions see the official project: https://github.com/stratosphereips/StratosphereLinuxIPS

## Author
**Jan Svoboda** **Stratosphere Lab**
GitHub: [@jsvobo](https://github.com/jsvobo)
