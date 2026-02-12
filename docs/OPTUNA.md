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
Optuna writes every artifact into an `optuna/` subfolder inside the generated experiment directory:
- `optuna_trials.csv` — flat table of trial parameters and metrics.
- `optuna_summary.json` — best trials, metric names, and study metadata.
- `trial_{n}_config.yaml` — the overrides sampled for a specific trial.
- `trial_{n}_context.yaml` — the full runtime config after overrides.
- `trial_{n}_result.json` — objective metrics reported by the pipeline.

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
        river.ensemble.ADWINBoostingClassifier:
          n_models:
            type: int
            low: 2
            high: 20
          model:
            type:
              type: categorical
              choices:
                - "HoeffdingAdaptiveTreeClassifier"
                - "HoeffdingTreeClassifier"
            params:
              HoeffdingAdaptiveTreeClassifier:
                max_depth:
                  type: int
                  low: 5
                  high: 30
                  when:
                    model.classifier_params.ADWINBoostingClassifier.model.type: HoeffdingAdaptiveTreeClassifier
                delta:
                  type: float
                  low: 1.0e-8
                  high: 1.0e-5
                  log: true
                  when:
                    model.classifier_params.ADWINBoostingClassifier.model.type: HoeffdingAdaptiveTreeClassifier
              HoeffdingTreeClassifier:
                grace_period:
                  type: int
                  low: 50
                  high: 500
                  when:
                    model.classifier_params.ADWINBoostingClassifier.model.type: HoeffdingTreeClassifier
                split_confidence:
                  type: float
                  low: 1.0e-7
                  high: 1.0e-3
                  log: true
                  when:
                    model.classifier_params.ADWINBoostingClassifier.model.type: HoeffdingTreeClassifier
    commands:
      - name: "train_main"
        command: "train"
        mixer:
          type:
            type: categorical
            choices:
              - "sequence"
              - "random"
          chunk_size:
            type: int
            low: 2000
            high: 10000
            when:
              commands[0].mixer.type: "random"
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
- **Conditions gate mutually exclusive knobs.** Use `when` with dot/list paths such as `commands[0].mixer.type` whenever a parameter should only apply in one branch.

## Notes & Tips
- Every Optuna trial runs a full train + validation pass; reduce `n_trials` or leverage `n_jobs` to control runtime.
- The optimizer logs the sampled overrides (`trial*_config.yaml`) separately from the full resolved config (`trial*_context.yaml`) for reproducibility.
- Validation splits, dataset resolving, and command execution follow the same code paths as normal runs, so any config that works without Optuna will work with it as long as the `optuna` section is well-formed.
- When in doubt, diff your current config against this document’s example and make sure each new spec mirrors the runtime structure.
