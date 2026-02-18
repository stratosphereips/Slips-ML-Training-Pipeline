# Optuna Integration Guide

This guide expands on the short README summary and documents every detail required to run multi-objective Optuna studies with the SLIPS ML pipeline. It covers runtime modes, output layout, a copy/paste-ready configuration, and the exact rules the optimizer expects when you add new hyperparameters.

## How to Run
- **Normal mode** (no search, single experiment):
  ```bash
  python run.py default_config.yaml
  ```
- **Optuna mode** (multi-objective search):
  ```bash
  python run.py optuna_conf.yaml --optuna
  ```
  - `optuna_conf.yaml` is a complete example you can copy and adjust.
  - Use `--optuna` with any config that includes an `optuna` section; omit the flag to run that config verbatim.

## Study Outputs
Optuna writes every artifact into an `optuna/` subfolder inside the generated experiment directory, and once the study closes the pipeline automatically runs `generate_optuna_trial_plots` so the visual summaries are generated without any manual steps:
- `optuna_trials.csv` — flat table of trial parameters and metrics.
- `optuna_summary.json` — best trials, metric names, and study metadata.
- `trial_{n}_config.yaml` — the overrides sampled for a specific trial.
- `trial_{n}_context.yaml` — the full runtime config after overrides.
- `trial_{n}/metrics.json` — serialized train/test metrics for the trial, now including full confusion-matrix counts (`train_confusion` and `test_confusion`).
- `optuna_trials_training.png`, `optuna_trials_testing.png`, `optuna_trials_train_test_delta.png` — scatter plots produced by `src/plot_utils/plot_optuna_trials.py` (details below) that summarize every finished trial.

## Complete Config Example
The snippet below mirrors `optuna_conf.yaml` in the repo root so you can start from a known-good baseline:
```yaml
---
experiment_name: optuna_example
root: /opt/Datasets/security-datasets-for-testing/
seed: 1111
validation_split: 0.1
batch_size_train: 1000
batch_size_test: 2000

classes:
  - "Benign"
  - "Malicious"
plotting_pths:
  training: "./src/plot_utils/plot_train_perf.py"
  testing: "./src/plot_utils/plot_test_perf.py"
paths:
  experiment_dir: ./experiments

dataset_loader:
  data_subdir: data
  persist_cache_threshold: 20000
  cache_dir: ./cache
  labeled_filenames:
        # Mixer batch sizes are inherited from batch_size_train/test, so no per-command
        # chunk_size specification is required or supported here.
    - conn.log.labeled
    - labeled-conn.log
    - conn.log
  file_encoding: utf-8
  file_errors: ignore
  shuffle_per_epoch: false

features:
  default_label: Benign
  protocols_to_discard:
    - arp
    - ARP
    - icmp
    - igmp
    - ipv6-icmp
    - ""

preprocessing:
  steps:
    - name: "scaler"
      type: "StandardScaler"
      params: {}
  save_steps: true
  step_filename_template: "{name}.bin"

model:
  wrapper: RiverClassifierWrapper
  classifier_type: river.forest.ARFClassifier
  classifier_params:
    lambda_value: 10
    n_models: 10

commands:
  - name: "train_main"
    command: "train"
    mixer:
      type: "sequence"
      datasets:
        - "008"
        - "009"
        - "010"
        - "011"
        - "012"
        - "013"

  - name: "test_all"
    command: "test"
    mixer:
      type: "sequence"
      datasets:
        - "001"
        - "008"
        - "009"
        - "010"
        - "011"
        - "012"
        - "013"
        - "014"
        - "015"
        - "016"
        - "017"
        - "018"
        - "020"
        - "021"
        - "025"
        - "026"
        - "030"
        - "031"
        - "035"
        - "036"
        - "037"

optuna:
  n_jobs: 2
  enabled: true
  pruner:
    type: median
    n_warmup_steps: 5
  n_trials: 3
  metric:
    - "f1"
    - "fpr"
  directions:
    - "maximize"
    - "minimize"
  hyperparameters:
## Structural Validation & Fail-Fast Sampling
- The optimizer now validates the entire `optuna.hyperparameters` tree before starting a study. Unknown keys (for example adding `metric.max_depth` under the wrong parent) or missing `low/high` bounds cause an immediate `OptunaConfigError` that points to the offending path, so malformed YAML never survives to runtime.
- `when` clauses accept either a single value or a list, making it easy to reuse one spec across multiple nested models (`when: {model.classifier_params.river.ensemble.ADWINBoostingClassifier.model.type: ["HoeffdingAdaptiveTreeClassifier", "HoeffdingTreeClassifier"]}`).
- Every categorical spec must supply `choices`, numeric specs must define `low`/`high`, and `group` specs require a `params` mapping—these constraints are enforced by the validator, turning misconfigurations into clear error messages.

## River Metric Handling & Classifier Guards
- River classifiers can now reference metrics by simple names in either the base config or Optuna overrides; the pipeline converts strings such as `"f1"`, `"accuracy"`, or `"kappa"` into their corresponding `river.metrics` instances automatically.
- Passing an unknown metric name (or misspelling) raises a descriptive error before the classifier is created, avoiding the silent `Cls(**params)` failures we previously hit.
- After Optuna samples parameters, the pipeline cross-checks them against the classifier constructor signature. Unsupported knobs are removed with a warning, and true constructor errors propagate immediately so a trial never proceeds with a `None` classifier.

## Configurable Pruners
- The `optuna.pruner` section is wired directly into `optuna.create_study`. Supported `type` values are `median`, `successive_halving`, `hyperband`, and `nop` (disables pruning). Any extra keys are forwarded as constructor kwargs (`n_warmup_steps`, `min_resource`, etc.).
- Each study log records which pruner is active (`optuna_trials.log`) to aid reproducibility.

    model:
      classifier_type:
        type: categorical
        choices:
          - "river.forest.ARFClassifier"
          - "river.ensemble.ADWINBoostingClassifier"
      classifier_params:
        river.forest.ARFClassifier:
          lambda_value:
            type: int
            low: 1
            high: 20
          n_models:
            type: int
            low: 5
            high: 50
          grace_period:
            type: int
            low: 25
            high: 200
          max_depth:
            type: int
            low: 5
            high: 30
          split_criterion:
            type: categorical
            choices:
              - "gini"
              - "info_gain"
              - "hellinger"
          delta:
            type: float
            low: 1.0e-8
            high: 0.1
            log: true
          leaf_prediction:
            type: categorical
            choices:
              - "nba"
              - "nb"
              - "mc"
          metric:
            type: categorical
            choices:
              - "accuracy"
              - "kappa"
              - "f1"
              - "precision"
              - "recall"
        river.ensemble.ADWINBoostingClassifier:
          n_models:
            type: int
            low: 2
            high: 50
          model:
            type:
              type: categorical
              choices:
                - "HoeffdingAdaptiveTreeClassifier"
                - "HoeffdingTreeClassifier"
                - "SGTClassifier"
            params:
              max_depth:
                type: int
                low: 5
                high: 30
                when:
                  model.classifier_params.river.ensemble.ADWINBoostingClassifier.model.type:
                    - "HoeffdingAdaptiveTreeClassifier"
                    - "HoeffdingTreeClassifier"
              delta:
                type: float
                low: 1.0e-8
                high: 1.0e-5
                log: true
                when:
                  model.classifier_params.river.ensemble.ADWINBoostingClassifier.model.type:
                    - "HoeffdingAdaptiveTreeClassifier"
              split_criterion:
                type: categorical
                choices:
                  - "gini"
                  - "info_gain"
                  - "hellinger"
                when:
                  model.classifier_params.river.ensemble.ADWINBoostingClassifier.model.type:
                    - "HoeffdingAdaptiveTreeClassifier"
                    - "HoeffdingTreeClassifier"
              leaf_prediction:
                type: categorical
                choices:
                  - "nba"
                  - "nb"
                  - "mc"
                when:
                  model.classifier_params.river.ensemble.ADWINBoostingClassifier.model.type:
                    - "HoeffdingAdaptiveTreeClassifier"
                    - "HoeffdingTreeClassifier"
              drift_window_threshold:
                type: int
                low: 100
                high: 1000
                when:
                  model.classifier_params.river.ensemble.ADWINBoostingClassifier.model.type:
                    - "HoeffdingAdaptiveTreeClassifier"
              grace_period:
                type: int
                low: 50
                high: 500
                when:
                  model.classifier_params.river.ensemble.ADWINBoostingClassifier.model.type:
                    - "HoeffdingTreeClassifier"
              split_confidence:
                type: float
                low: 1.0e-7
                high: 1.0e-3
                log: true
                when:
                  model.classifier_params.river.ensemble.ADWINBoostingClassifier.model.type:
                    - "HoeffdingTreeClassifier"
    commands:
      - name: "train_main"
        command: "train"
        mixer:
          type:
            type: categorical
            choices:
              - "sequence"
              - "random"
          datasets:
            type: categorical
            multiple: true
            min: 3
            max: 6
            choices:
              - "001"
              - "008"
              - "009"
              - "010"
              - "011"
              - "012"
              - "013"
              - "014"
```

## Valid Optuna Config Rules
- **Mirror the runtime tree.** The nested structure under `optuna.hyperparameters` must be identical to the runtime config so the optimizer can build paths automatically.
- **Datasets stay lists.** When tuning which datasets a mixer uses, keep the list structure and just extend `choices` (set `multiple: true` when you want combinations).
- **Type selectors own their branches.** Every categorical `type` parameter must have either a matching key (`classifier_params.ARFClassifier`) or a `when` guard so only the active branch is sampled.
- **New parameters appear at the leaves.** Add a spec exactly where the runtime field lives (e.g., `model.classifier_params.ARFClassifier.grace_period`). No extra wiring is required.
- **Nested models need guards.** For inner classifiers, either wrap fields in a `group` spec or gate them with `when: {path: value}` so mutually exclusive branches stay isolated.
- **Conditions gate mutually exclusive knobs.** Use `when` with dot/list paths such as `commands[0].mixer.type` whenever a parameter should only apply in one branch. The guard accepts either a single value or a list, so you can reuse one spec across multiple base models.
- **Specs must stay self-contained.** A parameter spec can only use keys that belong to its declared `type` (e.g., `choices` for categorical, `low/high` for numeric). The validator rejects unknown keys early so malformed trees never reach the pipeline.

## Notes & Tips
- Every Optuna trial runs a full train + validation pass; reduce `n_trials` or leverage `n_jobs` to control runtime.
- The optimizer logs the sampled overrides (`trial*_config.yaml`) separately from the full resolved config (`trial*_context.yaml`) for reproducibility.
- Validation splits, dataset resolving, and command execution follow the same code paths as normal runs, so any config that works without Optuna will work with it as long as the `optuna` section is well-formed.
- Metric names entered as strings (e.g., `"f1"`, `"kappa"`) are turned into real `river.metrics` objects automatically before the classifier is created; typos will now fail fast.
- The `optuna.pruner` block is honored directly—set `type` to `median`, `nop`, `successive_halving`, or `hyperband` and pass any constructor kwargs alongside it.
- When a `test` command exists, every trial runs it immediately after training. The resulting `test_f1`/`test_fpr` become the Optuna objectives (`f1`, `fpr`), while the train metrics are reported separately. Trial logs also capture which command names ran and which dataset lists were applied.
- When in doubt, diff your current config against this document’s example and make sure each new spec mirrors the runtime structure.

## Visualizing Trials

After a study finishes the pipeline triggers `src/plot_utils/plot_optuna_trials.generate_optuna_trial_plots(...)` automatically, so the three PNGs listed above appear in the same `optuna/` folder as soon as the study completes. You can re-run or customize the plots manually with either the experiment root or the `optuna/` path itself:

```bash
python src/plot_utils/plot_optuna_trials.py experiments/<experiment_name>
python src/plot_utils/plot_optuna_trials.py experiments/<experiment_name>/optuna
```

Use `--annotate` to stamp the trial numbers next to every point when debugging specific overrides.

The script reads `optuna_summary.json` to determine the first/second objective (e.g., `f1` vs `fpr`). If the summary is missing—common when a study was interrupted—it falls back to scanning whatever `trial_XXXX/metrics.json` files exist, prints a warning that it is plotting an incomplete run, and infers the metric names automatically. Either way it saves the same three PNGs directly into the `optuna/` folder:

- `optuna_trials_training.png` — training metrics plotted with classifier-specific colors and inner-classifier specific markers.
- `optuna_trials_testing.png` — testing metrics plotted with the same legend so you can compare generalization directly.
- `optuna_trials_train_test_delta.png` — signed $(\text{train}-\text{test})$ differences for both objectives, with crosshairs at zero.

Colors distinguish the outer classifier (e.g., `ARFClassifier` vs `ADWINBoostingClassifier`) and marker shapes represent nested estimators (e.g., river trees inside an ensemble). Pass `--annotate` to label points with their trial numbers if you need to trace back to a specific configuration. Because the script exposes a callable `generate_optuna_trial_plots()` function, it can also be triggered programmatically from the pipeline after `optuna` runs.
