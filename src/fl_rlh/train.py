# Author: Jan Svoboda
# functionality: Train one RLH configuration from the shared init and record its
# convergence (loss + confusion counts at checkpoint steps).
# behavior:
#   - Preprocessing as in the SLIPS federated module (since dc15aad7d): fit a
#     StandardScaler on all training rows (the module: the whole window, before
#     undersampling), undersample them to 50/50, scale train and eval rows with it.
#   - Optimizer = the module's manual Adam; the learning rate follows a schedule
#     (none / cosine / cyclic triangular / one-cycle) over the step budget.
#   - Full batch = one step per epoch (as in the module); with a batch size each
#     epoch is a class-stratified split into mini-batches, one step each.

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..metrics_calculator import compute_binary_metrics
from .data import undersample
from .model import adam_step, build_model, minibatch_indices, minibatch_rng, new_adam_state

SCHEDULERS = ("none", "cosine", "cyclic", "onecycle")


def lr_at(step: int, total: int, base_lr: float, kind: str, params: Optional[dict] = None) -> float:
    """Learning rate of 0-based step `step` out of `total` steps."""
    p = params or {}
    t = step / max(total, 1)
    if kind == "none":
        return base_lr
    if kind == "cosine":
        low = base_lr * float(p.get("min_factor", 0.01))
        return low + 0.5 * (base_lr - low) * (1 + math.cos(math.pi * t))
    if kind == "cyclic":
        # triangular: low -> base_lr -> low, `cycles` times over the budget
        cycles = max(int(p.get("cycles", 3)), 1)
        low = base_lr * float(p.get("low_factor", 0.1))
        pos = (t * cycles) % 1.0
        return low + (base_lr - low) * (1 - abs(2 * pos - 1))
    if kind == "onecycle":
        pct = float(p.get("pct_start", 0.3))
        start = base_lr / float(p.get("div_factor", 25))
        end = base_lr / float(p.get("final_div_factor", 1e4))
        if t < pct:
            return start + (base_lr - start) * (t / pct)
        u = (t - pct) / max(1 - pct, 1e-12)
        return end + 0.5 * (base_lr - end) * (1 + math.cos(math.pi * u))
    raise ValueError(f"unknown scheduler {kind!r} (expected one of {SCHEDULERS})")


def mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def scores(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    m = compute_binary_metrics({"TP": tp, "FP": fp, "TN": tn, "FN": fn})
    return {
        "mcc": mcc(tp, fp, tn, fn),
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "fpr": m["fpr"],
        "accuracy": m["accuracy"],
    }


def prepare(X_train, y_train, X_eval, *, undersample_train: bool, scaler_fit_on: str, rng):
    """Scale and undersample like the module; returns float32 arrays.

    scaler_fit_on="train_all" fits the scaler on all training rows before
    undersampling (the module since dc15aad7d); "train_undersampled" fits it
    on the undersampled rows (the module before)."""
    if scaler_fit_on not in ("train_all", "train_undersampled"):
        raise ValueError(f"unknown scaler_fit_on {scaler_fit_on!r}")
    scaler = StandardScaler().fit(X_train) if scaler_fit_on == "train_all" else None
    Xu, yu = undersample(X_train, y_train, rng) if undersample_train else (X_train, y_train)
    if scaler is None:
        scaler = StandardScaler().fit(Xu)
    return (
        scaler.transform(Xu).astype(np.float32),
        yu,
        scaler.transform(X_eval).astype(np.float32),
        scaler,
    )


def _criterion(y: np.ndarray, class_weighting: bool) -> nn.Module:
    if not class_weighting:
        return nn.CrossEntropyLoss()
    mal, ben = int((y == 1).sum()), int((y == 0).sum())
    total = mal + ben
    w = torch.tensor([total / (2.0 * max(ben, 1)), total / (2.0 * max(mal, 1))])
    return nn.CrossEntropyLoss(weight=w)


def train_eval(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xev: np.ndarray,
    yev: np.ndarray,
    *,
    lr: float,
    scheduler: str,
    batch_size: int,
    steps: int,
    checkpoints: List[int],
    seed: int,
    fold: int,
    scheduler_params: Optional[dict] = None,
    class_weighting: bool = False,
) -> List[dict]:
    """Train from the shared init for `steps` optimizer steps on the prepared
    (scaled) training rows; evaluate at step 0, every checkpoint <= steps and
    the last step. Returns one dict per evaluated step."""
    model = build_model(seed)
    state = new_adam_state(model)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    Xe, ye = torch.from_numpy(Xev), torch.from_numpy(yev.astype(np.int64))
    crit = _criterion(ytr, class_weighting)
    eval_crit = nn.CrossEntropyLoss()
    marks = sorted({c for c in checkpoints if 0 < c <= steps} | {steps})

    def evaluate(step: int, epoch: float, cur_lr: float) -> dict:
        with torch.no_grad():
            train_loss = float(crit(model(Xt), yt))
            logits = model(Xe)
            val_loss = float(eval_crit(logits, ye))
            pred = logits.argmax(dim=1)
            tp = int(((pred == 1) & (ye == 1)).sum())
            fp = int(((pred == 1) & (ye == 0)).sum())
            tn = int(((pred == 0) & (ye == 0)).sum())
            fn = int(((pred == 0) & (ye == 1)).sum())
        row = {"step": step, "epoch": round(epoch, 4), "lr": cur_lr,
               "train_loss": train_loss, "val_loss": val_loss,
               "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        row.update(scores(tp, fp, tn, fn))
        return row

    curve = [evaluate(0, 0.0, lr_at(0, steps, lr, scheduler, scheduler_params))]
    step, epoch = 0, 0
    while step < steps:
        if batch_size > 0:
            parts = minibatch_indices(ytr, batch_size, minibatch_rng(seed, fold, epoch, False))
        else:
            parts = [None]
        for i, part in enumerate(parts):
            if step >= steps:
                break
            if part is None:
                xb, yb = Xt, yt
            else:
                sel = torch.as_tensor(part)
                xb, yb = Xt[sel], yt[sel]
            cur_lr = lr_at(step, steps, lr, scheduler, scheduler_params)
            model.train()
            loss = crit(model(xb), yb)
            loss.backward()
            adam_step(model, state, cur_lr)
            step += 1
            if step in marks:
                curve.append(evaluate(step, epoch + (i + 1) / len(parts), cur_lr))
        epoch += 1
    return curve
