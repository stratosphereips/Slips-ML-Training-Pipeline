# Tuning the federated module's RLH model with the pipeline's Optuna

The SLIPS federated module's default model (`random_projection_mlp`, "RLH": frozen shared
random projection 18→256, fc1 256→16 + ReLU, head 16→2) runs inside this pipeline as an
ordinary classifier, so the pipeline's own Optuna search (`run.py --optuna`) tunes its
learning parameters. No warm start: every trial starts from the module's shared init.

## Run

```
# once: the fixed data split (writes ./datasets_fl_rlh, gitignored; ./datasets = the dataset root)
python3 -m src.fl_rlh.make_split --src ./datasets --out ./datasets_fl_rlh \
    --train 008 010 011 012 037 --outside 014 016 027

# the search (results in experiments/fl_rlh_optuna*/optuna/)
python3 run.py configs/fl_rlh_optuna.yaml --optuna
```

## What runs (`configs/fl_rlh_optuna.yaml`)

- **Data, the same in every trial.** Training = the 90% parts of the mixed captures 008,
  010, 011, 012, 037 (stratified by label, seeded, row order kept). Objective
  (`test_within`, the first test command) = their held-out 10%, never trained on.
  `test_outside` = captures of other networks (014 balanced, 016 mostly benign, 027 mostly
  malicious), recorded per trial in `command_metrics`, not optimized. No validation split,
  so nothing is taken out of the training rows; the mixer only changes how they are batched
  (`balanced` stops when one label runs out).
- **Features** (`features.feature_set: fl_rlh`): the module's 18 features from its own
  extraction, on the flow dict SLIPS stores for each Zeek record (`src/fl_rlh/data.py`).
  Checked identical to the raw-Zeek path for all flows of the eight captures and to the
  features of the flows SLIPS itself stored (dataset 011). `columns_to_discard` zeroes a
  feature (the input stays 18 wide). The config zeroes the two IP numbers: in the lab
  every host is in one /24, so after scaling they carry almost nothing for the module,
  while in these captures they alone would nearly separate the classes (benign = one
  local host, malicious = outside addresses).
- **Training** = the module's window: the StandardScaler step is `partial_fit` on the whole
  batch, the batch is scaled, then `FLRLHClassifier` (`src/fl_rlh/classifier.py`)
  undersamples it 50/50 and trains `epochs_per_batch` epochs of the module's manual Adam on
  it, one pass over the stream. The Adam state carries over between batches.
- **Search space**: lr (1e-5 to 1, log), epochs per batch (1 to 300), mini-batch size (full
  batch or 32/100/256), lr schedule (none, cosine or one-cycle per batch, cyclic across
  batches), Adam beta1 / beta2 / eps / weight decay, fc1 width, one optional extra layer,
  and the mixer (sequence, random, balanced with a ratio). Objective: f1 up, fpr down (the
  pipeline's default); confusion counts are in each trial's `metrics.json`.

## Same as the module, and not

Same: projection, shared init, the 18 features, the manual Adam, the scaler-then-
undersample order, unweighted cross-entropy (the module's `undersample` preset).
Different: Zeek conn logs have no ARP records; no federation and no fine-tune stage; a batch
of 1,000 rows stands in for a window; architecture and Adam settings other than the
defaults are not in the module.

`testing/test_fl_rlh.py` checks the model, init, features and Adam against
`src/fl_rlh/fl_rlh_reference.npz` (exported from the SLIPS image by
`src/fl_rlh/make_reference.py`), the split, the classifier and the config.
