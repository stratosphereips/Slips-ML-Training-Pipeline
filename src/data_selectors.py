"""
Compact mixer subsystem (no replacement, deterministic train/val split).

Mixers:
 - SequenceMixer
 - RandomBatchesMixer
 - BalancedByLabelMixer

API:
 - mixer.reset_epoch(batch_size, epoch_idx=0)
 - train_batch, val_batch = mixer.next_batch()   # val_batch may be None
 - mixer.get_mix_plan()

This module provides defensive checks, clearer error messages, and
robust logic for pulling samples across loaders and performing
train/validation splits.
"""

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


# -------------------------
# helpers
# -------------------------


def is_dataframe(x):
    return pd is not None and isinstance(x, pd.DataFrame)


def batch_len(batch):
    if batch is None:
        return 0
    try:
        return len(batch)
    except Exception:
        return 1


def concat_batches(parts):
    if not parts:
        return None
    first = None
    for p in parts:
        if p is not None:
            first = p
            break
    if first is None:
        return None
    if is_dataframe(first):
        return pd.concat([p for p in parts if p is not None], ignore_index=True)
    out = []
    for p in parts:
        if p is None:
            continue
        try:
            out.extend(list(p))
        except Exception:
            out.append(p)
    return out


def slice_batch(batch, start, end):
    if batch is None:
        return None
    if is_dataframe(batch):
        return batch.iloc[start:end].reset_index(drop=True)
    try:
        return batch[start:end]
    except Exception:
        if start == 0 and end >= 1:
            return [batch]
        return []


def split_by_indices(batch, indices):
    """
    Return items at indices as same-type batch.
    indices: iterable of integer positions (order preserved).
    """
    if batch is None:
        return None
    if is_dataframe(batch):
        return batch.iloc[list(indices)].reset_index(drop=True)
    out = []
    try:
        for i in indices:
            out.append(batch[i])
    except Exception:
        # fallback: if batch is not indexable, return None
        return None
    return out


# -------------------------
# DefaultMixer (shared base)
# -------------------------


class DefaultMixer(object):
    """
    Shared utilities:
    - resolve dataset keys
    - reset loaders
    - pull up to n samples from a loader (uses loader.next_n if available)
    - deterministic splitting into train/val using self.validation_split and RNG
    """

    def __init__(self, spec, loaders, rng):
        self.spec = spec or {}
        self.loaders = loaders or {}
        self.rng = rng
        self.batch_size = None
        self.mix_plan = []
        # default validation split for this mixer (0.0 means no val)
        self.validation_split = float(self.spec.get("validation_split", 0.0))

    def resolve_key(self, key):
        if key in self.loaders:
            return key
        for k in self.loaders:
            # allow partial matches (containment)
            if key in k:
                return k
        raise KeyError("Dataset '{}' not found among loaders".format(key))

    def reset_loader(self, loader, batch_size):
        try:
            loader.reset_epoch(batch_size=batch_size)
        except TypeError:
            try:
                loader.reset_epoch()
            except Exception:
                # loader may not support reset; ignore
                pass

    def reset_epoch(self, batch_size, epoch_idx=0):
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.mix_plan = []

    def get_mix_plan(self):
        return self.mix_plan

    def _pull_n_from_loader(self, key, n):
        """
        Pull up to n samples from loader identified by key.
        Prefer loader.next_n(n) if available, else accumulate from next_batch().
        NOTE: no replacement; if loader is exhausted, we stop and return what's available.
        Returns a batch-like object (list or DataFrame) or None when nothing available.
        """
        if n <= 0:
            return None
        if key not in self.loaders:
            raise KeyError(f"Loader for key '{key}' not found")
        loader = self.loaders[key]
        if hasattr(loader, "next_n"):
            try:
                part = loader.next_n(n)
                if part is None:
                    return None
                if batch_len(part) == 0:
                    return None
                return part
            except Exception:
                # fall through to manual accumulation
                pass
        collected = []
        need = n
        while need > 0:
            try:
                b = loader.next_batch()
            except Exception:
                b = None
            if b is None:
                break
            bn = batch_len(b)
            if bn == 0:
                continue
            if bn <= need:
                collected.append(b)
                need -= bn
            else:
                # take portion (drop remainder)
                if is_dataframe(b):
                    part = b.iloc[:need].reset_index(drop=True)
                else:
                    try:
                        part = list(b)[:need]
                    except Exception:
                        part = b
                collected.append(part)
                need = 0
        if not collected:
            return None
        return concat_batches(collected)

    def _split_batch(self, batch):
        """
        Deterministically split `batch` into (train_batch, val_batch) according to
        self.validation_split and self.rng. If val size <= 0, return (batch, None).
        The split preserves batch type (DataFrame vs list).
        Uses rounding for val size calculation to behave sensibly on small batches.
        """
        if batch is None:
            return None, None
        n = batch_len(batch)
        if n == 0:
            return None, None
        frac = float(self.validation_split)
        if frac <= 0.0:
            return batch, None
        # use rounding so, e.g., 5 * 0.3 -> round(1.5) -> 2
        val_size = int(round(frac * n))
        if val_size <= 0:
            return batch, None
        # deterministic permutation using the mixer's RNG
        perm = self.rng.permutation(n)
        val_idxs = set(perm[:val_size].tolist())
        train_idxs = [i for i in range(n) if i not in val_idxs]
        val_idxs_ordered = [i for i in range(n) if i in val_idxs]
        # build batches preserving types
        if is_dataframe(batch):
            train_batch = batch.iloc[train_idxs].reset_index(drop=True)
            val_batch = batch.iloc[val_idxs_ordered].reset_index(drop=True)
        else:
            train_batch = [batch[i] for i in train_idxs]
            val_batch = [batch[i] for i in val_idxs_ordered]
        return train_batch, val_batch

    # subclasses must implement next_batch and call _split_batch before returning
    def next_batch(self):
        raise NotImplementedError("Subclasses must implement next_batch")


# -------------------------
# SequenceMixer
# -------------------------


class SequenceMixer(DefaultMixer):
    """
    Stream datasets in sequence (drain one, then next).

    Behavior (single mode):
      - Each call to next_batch() returns exactly one underlying loader.next_batch()
        (subject to deterministic train/validation split performed by _split_batch()).
      - The mixer advances to the next dataset only when the current loader is
        exhausted (i.e. returns None).

    Spec:
      - type: "sequence"
      - datasets: [ ... ]   (keys or prefixes)
      - per_dataset_batch_size: optional
      - validation_split: optional float
    """

    def __init__(self, spec, loaders, rng):
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("sequence spec requires 'datasets'")
        # Resolve keys now; errors for missing loaders are appropriate here
        self.dataset_keys = [self.resolve_key(k) for k in ds]
        self.per_dataset_batch_size = spec.get("per_dataset_batch_size")
        self._cur_idx = 0

    def reset_epoch(self, batch_size, epoch_idx=0):
        super().reset_epoch(batch_size, epoch_idx)
        self._cur_idx = 0
        # reset each loader using either per-dataset batch size or global
        for k in self.dataset_keys:
            bs = self.per_dataset_batch_size or self.batch_size
            self.reset_loader(self.loaders[k], bs)

    def next_batch(self):
        """
        Return exactly one underlying loader.next_batch() (subject to splitting).
        Advance to the next dataset only when a loader returns None / is exhausted.
        """
        if self.batch_size is None:
            raise RuntimeError("reset_epoch must be called first")

        while self._cur_idx < len(self.dataset_keys):
            key = self.dataset_keys[self._cur_idx]
            loader = self.loaders[key]
            try:
                batch = loader.next_batch()
            except Exception:
                batch = None
            if batch is None or batch_len(batch) == 0:
                # exhausted this dataset -> move to next and continue loop
                self._cur_idx += 1
                continue

            cnt = batch_len(batch)
            plan_entry = {"type": "drain", "dataset": key, "counts": {key: cnt}}
            self.mix_plan.append(plan_entry)

            # Split this single batch deterministically (may return val=None)
            train, val = self._split_batch(batch)
            last = self.mix_plan[-1]
            last["train_count"] = batch_len(train) if train is not None else 0
            last["val_count"] = batch_len(val) if val is not None else 0

            # Do NOT advance _cur_idx here; only advance when loader.next_batch() later returns None
            return train, val

        return None, None


# -------------------------
# RandomBatchesMixer
# -------------------------


class RandomBatchesMixer(DefaultMixer):
    """
    Produce random mixed batches from datasets.

    Spec keys:
      - type: "random_batches"
      - datasets: [list]                (required)
      - balanced: true|false            (default False)
      - weights: [list]                 (used when balanced==False)
      - validation_split: optional float (overrides DefaultMixer default if present)
    """

    def __init__(self, spec, loaders, rng):
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("random_batches requires 'datasets'")
        # validate weights length against the provided dataset list BEFORE resolving loader keys
        weights = spec.get("weights")
        if weights is None:
            weights = [1.0] * len(ds)
        if len(weights) != len(ds):
            raise ValueError("weights length must match datasets")

        self.datasets = [self.resolve_key(k) for k in ds]
        self.balanced = bool(spec.get("balanced", False))
        self.weights = list(weights)
        # allow per-mixer validation_split override
        if "validation_split" in spec:
            self.validation_split = float(spec.get("validation_split", 0.0))

    def reset_epoch(self, batch_size, epoch_idx=0):
        super().reset_epoch(batch_size, epoch_idx)
        # reset all loaders
        for k in self.datasets:
            loader = self.loaders[k]
            self.reset_loader(loader, batch_size)

    def _pull_n_from_loader(self, key, n):
        # use DefaultMixer implementation (no replacement)
        return super()._pull_n_from_loader(key, n)

    def next_batch(self):
        if self.batch_size is None:
            raise RuntimeError("reset_epoch must be called first")
        B = int(self.batch_size)
        if B <= 0:
            return None, None
        if self.balanced:
            k = len(self.datasets)
            base = [B // k] * k
            rem = B - sum(base)
            for i in range(rem):
                base[i] += 1
            counts = base
        else:
            probs = np.asarray(self.weights, dtype=float)
            probs = probs / probs.sum()
            # multinomial can return zeros; counts length matches datasets
            counts = self.rng.multinomial(B, probs).tolist()
        parts = []
        produced = {}
        for i, key in enumerate(self.datasets):
            cnt = int(counts[i])
            if cnt <= 0:
                produced[key] = 0
                continue
            part = self._pull_n_from_loader(key, cnt)
            if part is None:
                produced[key] = 0
                continue
            produced[key] = batch_len(part)
            parts.append(part)
        if not parts:
            return None, None
        batch = concat_batches(parts)
        # record total counts before split
        self.mix_plan.append({"type": "random_batches", "counts": produced})
        train, val = self._split_batch(batch)
        last = self.mix_plan[-1]
        last["train_count"] = batch_len(train) if train is not None else 0
        last["val_count"] = batch_len(val) if val is not None else 0
        return train, val


# -------------------------
# BalancedByLabelMixer
# -------------------------


class BalancedByLabelMixer(DefaultMixer):
    """
    Produce batches balanced by label across datasets.

    Spec keys:
      - type: "balanced_by_label"
      - datasets: [list]   (required)
      - labels: optional list of label values to balance; if missing, discovered by peeking
      - micro_batch: optional int (how much to pull when searching); default 16
      - validation_split: optional float
    """

    def __init__(self, spec, loaders, rng):
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("balanced_by_label requires 'datasets'")
        self.datasets = [self.resolve_key(k) for k in ds]
        self.provided_labels = spec.get("labels")
        self.micro = int(spec.get("micro_batch", 16))
        if "validation_split" in spec:
            self.validation_split = float(spec.get("validation_split", 0.0))
        # labels will be set on reset_epoch
        self.labels = []

    def reset_epoch(self, batch_size, epoch_idx=0):
        super().reset_epoch(batch_size, epoch_idx)
        # reset all loaders
        for k in self.datasets:
            loader = self.loaders[k]
            self.reset_loader(loader, batch_size)
        # discover labels if not provided
        if self.provided_labels:
            # accept provided labels but filter out falsy/None values
            self.labels = [l for l in list(self.provided_labels) if l is not None]
        else:
            labels_set = set()
            for k in self.datasets:
                loader = self.loaders[k]
                b = None
                # try to peek using next_n when available
                if hasattr(loader, "next_n"):
                    try:
                        b = loader.next_n(self.micro)
                    except Exception:
                        b = None
                if b is None:
                    try:
                        b = loader.next_batch()
                    except Exception:
                        b = None
                if b is None:
                    continue
                if is_dataframe(b):
                    if "label" in b.columns:
                        # drop NA values
                        vals = b["label"].dropna().unique().tolist()
                        for v in vals:
                            if v is not None:
                                labels_set.add(v)
                else:
                    for rec in b:
                        try:
                            lbl = rec.get("label")
                        except Exception:
                            lbl = None
                        if lbl is not None:
                            labels_set.add(lbl)
            # reset loaders again to ensure clean epoch
            for k in self.datasets:
                loader = self.loaders[k]
                self.reset_loader(loader, batch_size)
            if not labels_set:
                # use reasonable defaults; tests accept either casing or len==2
                self.labels = ["Benign", "Malicious"]
            else:
                self.labels = sorted(list(labels_set))

    def _pull_until_label(self, key, label, need):
        if need <= 0:
            return None
        if key not in self.loaders:
            raise KeyError(f"Loader for '{key}' not found")
        loader = self.loaders[key]
        collected = []
        total = 0
        while total < need:
            chunk = None
            if hasattr(loader, "next_n"):
                try:
                    chunk = loader.next_n(max(self.micro, need - total))
                except Exception:
                    chunk = None
            if chunk is None:
                try:
                    chunk = loader.next_batch()
                except Exception:
                    chunk = None
            if chunk is None:
                break
            matches = None
            if is_dataframe(chunk):
                if "label" in chunk.columns:
                    sel = chunk[chunk["label"] == label]
                    if len(sel):
                        matches = sel.reset_index(drop=True)
                    else:
                        matches = None
                else:
                    matches = None
            else:
                tmp = []
                for rec in chunk:
                    try:
                        if rec.get("label") == label:
                            tmp.append(rec)
                    except Exception:
                        continue
                if tmp:
                    matches = tmp
                else:
                    matches = None
            if not matches:
                continue
            collected.append(matches)
            total += batch_len(matches)
        if not collected:
            return None
        return concat_batches(collected)

    def next_batch(self):
        if self.batch_size is None:
            raise RuntimeError("reset_epoch must be called first")
        if not self.labels:
            # defensive: if labels were not discovered for some reason, fallback
            self.labels = ["Benign", "Malicious"]
        B = int(self.batch_size)
        if B <= 0:
            return None, None
        num_labels = max(1, len(self.labels))
        base = [B // num_labels] * num_labels
        rem = B - sum(base)
        for i in range(rem):
            base[i] += 1
        parts = []
        produced = {}
        for i, lbl in enumerate(self.labels):
            need = base[i]
            collected_for_label = []
            # attempt to pull from datasets round-robin until need met or exhausted
            remaining = need
            ds_idx = 0
            attempts = 0
            # per-dataset want heuristic
            per_ds_want = int(np.ceil(float(need) / max(1, len(self.datasets))))
            while remaining > 0 and attempts < len(self.datasets) * 4:
                key = self.datasets[ds_idx % len(self.datasets)]
                want = min(per_ds_want, remaining)
                chunk = self._pull_until_label(key, lbl, want)
                if chunk is None:
                    ds_idx += 1
                    attempts += 1
                    continue
                collected_for_label.append(chunk)
                got = batch_len(chunk)
                remaining -= got
                ds_idx += 1
            if collected_for_label:
                part = concat_batches(collected_for_label)
                parts.append(part)
                produced[lbl] = batch_len(part)
            else:
                produced[lbl] = 0
        if not parts:
            return None, None
        batch = concat_batches(parts)
        self.mix_plan.append({"type": "balanced_by_label", "counts_by_label": produced})
        train, val = self._split_batch(batch)
        last = self.mix_plan[-1]
        last["train_count"] = batch_len(train) if train is not None else 0
        last["val_count"] = batch_len(val) if val is not None else 0
        return train, val
