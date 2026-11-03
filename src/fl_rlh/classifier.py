# Author: Jan Svoboda
# functionality: The SLIPS federated module's RLH model as a pipeline classifier, so the
# pipeline's own Optuna search (run.py --optuna) tunes its learning parameters.
# behavior:
#   - One partial_fit = one training window of the module: the batch arrives scaled by the
#     pipeline's StandardScaler step (partial_fit on the whole batch, then transform: the
#     module's order since SLIPS dc15aad7d), is undersampled 50/50, then trained for
#     `epochs_per_batch` epochs of the module's manual Adam on an unweighted cross-entropy.
#     Full batch = one step per epoch (as in the module); with `minibatch_size` each epoch
#     is one step per class-stratified mini-batch.
#   - The Adam state carries over from batch to batch, like the module's local model.
#   - lr schedule: none (constant); cosine / onecycle over each batch's steps, restarting
#     every batch; cyclic = triangular over the whole stream, `cycle_batches` batches per
#     cycle (lr between lr * min_lr_factor and lr).
#   - Also tunable, beyond the module: Adam's beta1 / beta2 / eps / weight decay, the fc1
#     width (`hidden2`) and one extra ReLU layer (`extra_layer` nodes, 0 = none). The
#     defaults are the module's exact model and optimizer.
#   - No warm start: every instance starts from the shared init. No federation, no fine-tune.

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..commons import BENIGN, MALICIOUS
from .data import undersample
from .model import (
    BETA1,
    BETA2,
    EPS,
    HIDDEN2,
    SEED,
    WEIGHT_DECAY,
    adam_step,
    build_net,
    minibatch_indices,
    minibatch_rng,
    new_adam_state,
)

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


class FLRLHClassifier:
    """RLH (frozen shared projection 18->256, fc1 256->16, head 16->2) with the
    module's per-window training; sklearn-style partial_fit / predict."""

    def __init__(
        self,
        lr: float = 0.03,
        epochs_per_batch: int = 15,
        minibatch_size: int = 0,
        scheduler: str = "none",
        cycle_batches: int = 4,
        min_lr_factor: float = 0.01,
        undersample_batches: bool = True,
        beta1: float = BETA1,
        beta2: float = BETA2,
        eps: float = EPS,
        weight_decay: float = WEIGHT_DECAY,
        hidden2: int = HIDDEN2,
        extra_layer: int = 0,
        seed: int = SEED,
        torch_threads: int = 1,
    ):
        if scheduler not in SCHEDULERS:
            raise ValueError(f"unknown scheduler {scheduler!r} (expected one of {SCHEDULERS})")
        self.lr = float(lr)
        self.epochs_per_batch = int(epochs_per_batch)
        self.minibatch_size = int(minibatch_size)
        self.scheduler = scheduler
        self.cycle_batches = max(int(cycle_batches), 1)
        self.min_lr_factor = float(min_lr_factor)
        self.undersample_batches = bool(undersample_batches)
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.weight_decay = float(weight_decay)
        self.hidden2, self.extra_layer = int(hidden2), int(extra_layer)
        self.seed = int(seed)
        torch.set_num_threads(max(int(torch_threads), 1))
        self.model = build_net(self.seed, self.hidden2, self.extra_layer)
        self._adam = new_adam_state(self.model)
        self._criterion = nn.CrossEntropyLoss()
        self.batches_seen = 0
        self.steps_done = 0
        self.last_loss = None
        self.classes_ = np.array([BENIGN, MALICIOUS])

    def _steps_per_epoch(self, n: int) -> int:
        if self.minibatch_size <= 0 or n <= self.minibatch_size:
            return 1
        return -(-n // self.minibatch_size)

    def _lr(self, step_in_batch: int, steps_in_batch: int) -> float:
        if self.scheduler == "cyclic":
            period = self.cycle_batches * steps_in_batch
            return lr_at(
                self.steps_done % period,
                period,
                self.lr,
                "cyclic",
                {"cycles": 1, "low_factor": self.min_lr_factor},
            )
        return lr_at(
            step_in_batch, steps_in_batch, self.lr, self.scheduler, {"min_factor": self.min_lr_factor}
        )

    def partial_fit(self, X, y, classes=None):
        X = np.asarray(X, dtype=np.float32)
        y = (np.asarray(y) == MALICIOUS).astype(np.int64)
        if self.undersample_batches:
            # the module's _undersample_majority, seeded per window (seed + window)
            X, y = undersample(X, y, np.random.default_rng(self.seed + self.batches_seen))
        self.batches_seen += 1
        if len(y) == 0:
            return self
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        total = self.epochs_per_batch * self._steps_per_epoch(len(y))
        rng = minibatch_rng(self.seed, None, self.batches_seen, False)
        self.model.train()
        step, loss_sum = 0, 0.0
        for _ in range(self.epochs_per_batch):
            for part in minibatch_indices(y, self.minibatch_size, rng):
                sel = torch.as_tensor(part)
                loss = self._criterion(self.model(Xt[sel]), yt[sel])
                loss.backward()
                adam_step(
                    self.model,
                    self._adam,
                    self._lr(step, total),
                    wd=self.weight_decay,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                )
                loss_sum += loss.item() * len(part)
                step += 1
                self.steps_done += 1
        self.last_loss = loss_sum / (len(y) * self.epochs_per_batch)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.as_tensor(np.asarray(X, dtype=np.float32)))
        return np.where(out.argmax(dim=1).numpy() == 1, MALICIOUS, BENIGN)
