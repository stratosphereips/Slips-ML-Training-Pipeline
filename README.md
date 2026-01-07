
# How to run

Run the pipeline from the repository root (provide a config file or directory):

```bash
python run.py /path/to/config_or_config_dir
```

If you omit the argument the pipeline will look for a config in the current directory (`.`).

Results are written under `experiments/<experiment_name>` (see *Output*).

---

## Installation

Create an environment and install dependencies:

```bash
conda create -n slips-ml-pipeline python=3.10 pip -y
conda activate slips-ml-pipeline
pip install -r requirements.txt
```

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

1. Prepare the config directory or `config.yaml` (a default commented config is expected to exist).
2. Ensure `root` points to your dataset root with subfolders (e.g. `001/data/conn.log.labeled`).
3. Run `python run.py /path/to/config`.
4. Inspect experiment outputs in `experiments/<experiment_name>`.

---

## Output

#### Training example output


#### Testing example output

---

## Testing

Run unit tests:

```bash
pytest
```

---

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

## SLIPS

SLIPS (Stratosphere Linux IPS) is a behavioral machine-learning based intrusion detection and prevention system developed by the Stratosphere Laboratory. It detects malicious behaviors in network traffic and supports inputs such as PCAP files and flow logs from tools like Suricata and Zeek. For full details and installation instructions see the official project: https://github.com/stratosphereips/StratosphereLinuxIPS

## Author
**Jan Svoboda** **Stratosphere Lab**
GitHub: [@jsvobo](https://github.com/jsvobo)
