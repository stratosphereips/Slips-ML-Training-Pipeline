
## Architecture

The pipeline is assembled end-to-end at runtime from the YAML config:

1. **Config ingestion** – `ConfigReader` parses the experiment file, resolves effective paths, computes batch/validation sizes, and exposes typed accessors used everywhere else.
2. **Build phase** – `BuildManager` uses the config to instantiate the major subsystems in order: dataset loaders (via `find_and_load_datasets`), feature extractor, preprocessing stack, and classifier wrapper. Each subsystem only touches the configuration slice it needs.
3. **Dataset layer** – `ZeekDataset` eagerly loads every Zeek conn log, indexes metadata, and normalizes all flows to the canonical SLIPS schema through `ConnToSlipsConverter`. Loaders therefore expose a single `as_dataframe()` view that mixers can consume without re-running conversions.
4. **Mixing layer** – Mixers in `data_selectors.py` take the full DataFrames, optionally shuffle them per epoch, and emit train/validation splits following the selected strategy (sequence, random, balanced, oversampling). Sequence mixing now honors only the dataset order, so label-balancing knobs belong to the balanced mixers. Because mixers operate on cached tables, batching decisions are deterministic and memory-friendly.
5. **Feature + preprocessing stage** – For each emitted batch, `FeatureExtraction` cleans, engineers, and reorders numerical features while returning aligned labels. The batch then flows through the configured `PreprocessingWrapper`, which chains sklearn-compatible transformers (e.g., scalers, PCA) and persists their fitted state per experiment.
6. **Model stage** – `ClassifierWrapper` (Sklearn or River) abstracts the underlying estimator’s `partial_fit`, `predict`, and persistence routines, handles missing-class bootstrapping, and stores artifacts under the experiment directory.
- `optuna.hyperparameters` is now a list of declarative specs. Each entry describes a config `target` (dot/bracket path), a `type`, and optional `when` conditions. During a trial the optimizer samples each applicable spec, assigns the value directly to the target path (creating missing nodes automatically), and leaves untouched sections alone. Use `when` to scope parameters to certain classifiers or other choices, and `multiple: true` on categorical specs to sample combinations (encoded as strings internally).

This layered design keeps each concern isolated—configuration drives construction, loaders normalize once, mixers focus on selection, and downstream modules reuse the same batch contract—making it easy to swap models, features, or dataset roots without code edits.

---

## Parts of the pipeline

Key modules:

* `src/dataset_wrapper.py` — Zeek dataset discovery + indexed loader
* `src/conn_normalizer.py` — Zeek → SLIPS normalization
* `src/features.py` — feature extraction (whole df)
* `src/preprocessing_wrapper.py` — sequence of transforms with save/load
* `src/classifier_wrapper.py` — unified interface for classifiers
* `src/data_selectors.py` — mixers / batch composition
* `src/conf_reader.py` — config loading & validation
* `src/plot_utils/` — plotting helpers used by the pipeline, specifically the plotting scripts.
* `src/metrics_calculator.py` — unified metrics calculation for pipeline and plotting scripts

---
## Output

#### Training example output
```bash
=== VALIDATION Multi-class (Aggregated) ===
Accuracy:             0.9638
F1:                   0.9651
FPR:                  0.0629
FNR:                  0.0100
Macro F1:             0.9638
Precision:            0.9414
Recall:               0.9900
MCC:                  0.0000

=== TRAINING Multi-class (Aggregated) ===
Accuracy:             0.9680
F1:                   0.9694
FPR:                  0.0482
FNR:                  0.0170
Macro F1:             0.9679
Precision:            0.9561
Recall:               0.9830
MCC:                  0.0000

=== Per-class metrics (Aggregated) - VALIDATION ===
Class                 TP       TN       FP       FN      Acc     Prec      Rec       F1
Malicious            691      641       43        7   0.0000   0.9414   0.9900   0.9651

=== Per-class metrics (Aggregated) - TRAINING ===
Class                 TP       TN       FP       FN      Acc     Prec      Rec       F1
Malicious           6321     5732      290      109   0.0000   0.9561   0.9830   0.9694
```
#### Testing example output
```bash
[INFO] Output folder: /home/svobojan/pipeline_ml_training_for_SLIPS/experiments/lin_sequence_test_12/output/results/testing/1_test
[INFO] Reading testing logfile: /home/svobojan/pipeline_ml_training_for_SLIPS/experiments/lin_sequence_test_12/output/logs/1_test_all_test.log
[INFO] Plotting aggregated class counts (TP+FN per class so-far)...
[INFO] Plotting main metrics (FPR, FNR, F1, Accuracy) over snapshots...
[INFO] Saving FPR/FNR-only plot...
[INFO] Plotting predicted vs seen counts (per-snapshot) for Malicious & Benign...
[INFO] Plotting final confusion matrix (final snapshot)...

=== TESTING Multi-class (Aggregated) — 1_test ===
Accuracy:             0.8695
F1:                   0.9275
FPR:                  0.4375
FNR:                  0.1103
Macro F1:             0.6374
Precision:            0.9686
Recall:               0.8897
MCC:                  0.7389

=== Per-class metrics (Aggregated) - TESTING ===
Class                 TP       TN       FP       FN      Acc     Prec      Rec       F1
Malicious         240242     9998     7776    29793   0.8695   0.9686   0.8897   0.9275
Total test lines processed: 576
```

### Optuna study visuals

When the pipeline finishes an Optuna study it now drops a structured `optuna/visuals/` folder inside the experiment. Each configured command gets its own scatter plot under `visuals/commands/<command_slug>/<command_key>_metrics.png`, the primary train/test pair produces a single delta chart inside `visuals/deltas/`, and the Pareto-only views live under `visuals/pareto/`. All figures share the same color/marker legend so you can compare commands quickly, and the filenames make it obvious which command produced which metrics. See [docs/OPTUNA.md](docs/OPTUNA.md) for a detailed breakdown.

## Profiling the Pipeline

You can profile the pipeline to find performance bottlenecks using Python's built-in cProfile and visualize the results with snakeviz.

### Step-by-step:

1. **Create a directory for profiling outputs (recommended):**
  ```bash
  mkdir -p profiling
  ```

2. **Run the pipeline with cProfile:**
  ```bash
  python -m cProfile -o profiling/profile.out run.py /path/to/config.yaml
  ```
  - This will save the profiling results to `profiling/profile.out`.

3. **Visualize the results with snakeviz:**
  - First, install snakeviz if you haven't:
    ```bash
    pip install snakeviz
    ```
  - Then run:
    ```bash
    snakeviz profiling/profile.out
    ```
  - This will open an interactive browser view to explore the performance profile.

---

##  Code Testing
Run unit tests:

```bash
pytest
```

---
# Development
- If you want to add features, create an issue, or fork the repository and create a pull request with your new code.
- For the PR to be merged, we need all tests to be passing and pre-commit working without errors.
- Pre-commit runs linters and some checks based on the config. Here we use it to keep some code quality. If you want to contribute,
```bash
pre-commit install # if not installed yet
pre-commit run --all-files

pip install detect-secrets # if not installed yet
detect-secrets scan > .secrets.baseline
```

## Extending the Pipeline

### Add a new model or wrapper

* Implement a wrapper or classifier class in `src/classifier_wrapper.py`.

  * To reuse the pipeline’s training flow, extend `ClassifierWrapper` or implement the same `partial_fit`, `predict`, `save_classifier`, and `load_classifier` contract.
  * The factory `src.class_factory.get_wrapper_class` resolves wrapper names; add your wrapper class there (or reference it by dotted path in the config).
* The library and module, where the model is from, has to be added to a list of searched libraries at the top of `src.class_factory` file. Do not skip this step, factory would produce None instead of classifier.
* In the `model` config, point to the classifier type (short name or dotted path) and the wrapper name. **Note:** In Optuna mode, only the classifier type is varied dynamically; the wrapper is fixed per run (see notes below).

---

### Add a new preprocessing step

* Preprocessing steps are instantiated by the pipeline from the config. They must be sklearn-like objects with `fit`/`partial_fit` and `transform`.
* Programmatically you can add a step to `PreprocessingWrapper` with:

```python
preprocessor.add_step("scaler", StandardScaler())
```

* Saved preprocessing artifacts live under `output/preprocessing` and are re-loadable.


### Add a custom dataset loader

* `src.dataset_wrapper.find_and_load_datasets` returns a mapping of dataset_key -> loader.
* Your loader must expose a single `as_dataframe()` method that returns the entire dataset as a pandas DataFrame. The built-in mixers perform all batching and shuffling on top of these full DataFrames.
* Register/load your loader by replacing or extending `find_and_load_datasets` to return your loader instances keyed by dataset folder name.


### Add or modify mixers

* Mixers live in `src/data_selectors` and must follow the `DefaultMixer` contract (`reset_epoch`, `next_batch`). Each mixer pulls full DataFrames from loaders via `as_dataframe()` and is responsible for producing batches for downstream consumers.
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


## Parameter Dependencies & Notes

**Classifier/Wrapper Linkage:**
- The pipeline supports dynamic selection of `classifier_type` (e.g., `SGDClassifier`, `ARFClassifier`) as an Optuna hyperparameter.
- The `wrapper` (e.g., `SklearnWrapper`, `RiverWrapper`) is **fixed per run** and must be compatible with all classifiers being searched. This is by design: only one library's models are optimized per run for simplicity and reliability.
- If you wish to optimize across wrappers, run separate Optuna studies for each wrapper type.

**Config-driven Search Spaces:**
- All Optuna search spaces are defined in the config under `optuna.hyperparameters`.
- You can add new classifiers or parameters by extending this section.

**Parallelization:**
- The `optuna.n_jobs` parameter controls the number of parallel Optuna trials. Set this in your config to speed up search on multi-core systems.

**Trial Logging:**
- Every Optuna trial logs its full config and results in the `optuna/` subfolder of the experiment directory. This ensures full reproducibility and easy inspection.

**Experiment Directory Structure:**
- All experiment outputs (including Optuna logs) are written under a unique experiment directory, with numeric suffixes to avoid overwriting.

**Normal vs Optuna Mode:**
- Running without `--optuna` uses the config as-is for a single experiment. With `--optuna`, the pipeline performs a hyperparameter search as described below.

---


## Optuna Integration (Overview)

- Enable Optuna with `python run.py <config> --optuna`; the optimizer samples hyperparameters from the nested `optuna.hyperparameters` tree and spins up full train/validation trials.
- Each study writes logs, configs, and metrics to `experiments/<experiment_name>/optuna/`, so you can inspect trial configs and pick the best performer later.
- The provided `optuna_conf.yaml` includes a ready-to-run experiment and demonstrates how classifier/mixer comparisons are expressed; copy it as a starting point.
- The search-space definition is validated before any trials start: unknown keys, missing bounds, or mismatched `when` paths raise `OptunaConfigError` with a pointer to the broken section, keeping malformed configs out of the runtime.
- River classifiers can reference metrics by name (`"accuracy"`, `"f1"`, `"kappa"`, etc.)—the pipeline resolves these strings to real `river.metrics` instances and aborts immediately if a metric cannot be found.
- Classifier instantiation now fails fast with clear diagnostics; unsupported parameters are stripped (with a warning) and any constructor error stops the run instead of silently producing a `None` classifier.
- `optuna.pruner` is respected directly (`median`, `successive_halving`, `hyperband`, or `nop`), so you can fine-tune pruning strategy per study.
- If your config defines at least one `test` command, each Optuna trial now executes it right after training, uses those test metrics (`f1`, `fpr`) as the study objectives, and records the command name plus dataset list alongside the results.
- Full instructions—folder layout, normalized config example, validation details, and troubleshooting tips—live in [docs/OPTUNA.md](docs/OPTUNA.md).
- When the study is done, visualize every trial (training metrics, testing metrics, and signed train-test deltas) with `python src/plot_utils/plot_optuna_trials.py experiments/<experiment_name>/optuna`; the script automatically saves three scatter plots into the `optuna/` directory.

----

**Jan Svoboda** **Stratosphere Lab**
GitHub: [@jsvobo](https://github.com/jsvobo)
