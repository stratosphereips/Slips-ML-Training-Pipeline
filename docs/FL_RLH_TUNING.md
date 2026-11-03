# Offline learning-parameter study of the federated module's RLH model

`run_fl_rlh_tuning.py` trains the SLIPS federated module's default model
(`random_projection_mlp`: frozen shared random projection 18→256, fc1 256→16, head
16→2) offline on labelled Zeek conn logs, to see how Adam, the learning rate, a
learning-rate schedule and mini-batching behave, without SLIPS, peers or federation.

- **Same model and data path as the module.** The projection, shared init, 18
  features and manual Adam are copied from StratosphereLinuxIPS
  `fl_module_jan_rebased` @ dc15aad7d (`src/fl_rlh/model.py`).
  `testing/test_fl_rlh.py` checks them bit-for-bit against
  `src/fl_rlh/fl_rlh_reference.npz`, exported from the SLIPS image by
  `src/fl_rlh/make_reference.py`. Zeek records become the flow dicts SLIPS stores
  (checked on a SLIPS run: 3106 of 3106 conn flows identical).
- **Preprocessing as with `balancing: undersample` (SLIPS dc15aad7d):** a
  StandardScaler fit on all training rows (the module: the whole window) before
  50/50 undersampling, the undersampled rows scaled with it, unweighted
  cross-entropy. `scaler_fit_on: train_undersampled` gives the module's earlier
  behaviour (fit on the undersampled rows).
- **No warm start:** every run starts from the module's shared init.
- **Protocol:** a fixed, seeded, stratified 10% hold-out; stratified 5-fold CV on
  the other 90% over the grid in `configs/fl_rlh_tuning.yaml` (lr × schedule
  {none, cosine, cyclic triangular, one-cycle} × batch {full, 256, 100} × step
  budget). Then the best configs by mean CV MCC, plus the module's current
  settings, are retrained on all 90% and scored once on the hold-out.
- **Steps, not epochs:** budgets are optimizer steps. Full batch = one step per
  epoch, as in the module (15 steps per window).

## Run

```bash
cd <this repo>
mkdir -p experiments/fl_rlh_tuning
tmux new -s fltune 'python3 run_fl_rlh_tuning.py --config configs/fl_rlh_tuning.yaml \
  --workers 2 --threads 2 --out experiments/fl_rlh_tuning/full 2>&1 | tee -a experiments/fl_rlh_tuning/full.log'
```

`--out` with an existing directory resumes: finished config × fold results are
skipped. `--stage cv|final` runs one stage, `--smoke` runs a tiny grid (2 folds,
1500 flows per dataset) and `--root` overrides the dataset root.

## Output (`--out`)

| file | content |
|---|---|
| `data_summary.json` | flows and class balance per dataset, the split sizes, the environment |
| `cv_runs.jsonl` | one line per config × fold: final scores and the curve (step, epoch, lr, train/val loss, TP/FP/TN/FN, MCC, F1, P, R, FPR) |
| `cv_summary.csv` | per config: mean ± sd over folds, `mcc@<step>`, `steps_to_95pct_mcc`; best first |
| `test_results.csv`, `test_curves.jsonl` | the retrained configs on the hold-out |
| `plots/` | convergence by lr, by batch size, by schedule; the CV MCC heatmap; the hold-out curves |

## Caveats

- Zeek conn logs have no ARP records, so ARP (≈85% of the lab traffic) is absent here.
- The module's scaler is refit (`partial_fit`) on every window, accumulating
  over the run. Here it is fit once on the whole training split (each CV fold's
  training rows, or all 90% for the final runs), as if every window had been seen.
- `saddr_num` / `daddr_num` (IP % 1,000,000) are features, as in the module. In a
  capture of one attacked host they can help the model recognise addresses.
- Offline training sees all of the training data from the start. In the module,
  each window trains 15 steps on that window's flows only, so step counts transfer
  back, but final scores do not.
