"""
Mixers: SequenceMixer, RandomBatchesMixer, BalancedByLabelMixer

Contracts:
 - mixer.reset_epoch(batch_size, epoch_idx=0)
 - mixer.next_batch() -> (train_batch, val_batch)
 - mixer.get_mix_plan() -> list of plan entries

Notes:
 - Uses loader API: reset_epoch(batch_size), next_n(n) if available, next_batch()
 - Attempts to provide full-sized batches until data are exhausted.
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
    """
    Keep semantics: if batch is DataFrame -> iloc slice, else attempt list slicing.
    """
    if batch is None:
        return None
    if is_dataframe(batch):
        return batch.iloc[start:end].reset_index(drop=True)
    try:
        return batch[start:end]
    except Exception:
        # fallback: can't slice, return original if asking for full
        if start == 0 and end >= 1:
            return [batch]
        return []

# -------------------------
# DefaultMixer (shared base)
# -------------------------
class DefaultMixer(object):
    def __init__(self, spec, loaders, rng):
        self.spec = spec or {}
        self.loaders = loaders or {}
        self.rng = rng
        self.batch_size = None
        self.mix_plan = []
        # validation split comes from spec; mixers/config_reader can set this
        self.validation_split = float(self.spec.get("validation_split", 0.0))

        # per-epoch bookkeeping to allocate validation counts fairly across batches
        self._epoch_seen = 0  # number of items seen so far this epoch

    def resolve_key(self, key):
        # exact match first, then substring fallback
        if key in self.loaders:
            return key
        for k in self.loaders:
            if key in k:
                return k
        raise KeyError(f"Dataset '{key}' not found among loaders")

    def reset_loader(self, loader, batch_size):
        # attempt to set loader internal batch size, best-effort
        try:
            loader.reset_epoch(batch_size=batch_size)
        except TypeError:
            try:
                loader.reset_epoch()
            except Exception:
                pass

    def reset_epoch(self, batch_size, epoch_idx=0):
        self.batch_size = batch_size
        self.mix_plan = []
        # reset epoch bookkeeping
        self._epoch_seen = 0

    def get_mix_plan(self):
        return self.mix_plan

    def _split_batch(self, batch):
        """
        Deterministic allocation of validation items across the epoch.

        Instead of computing val_size = int(self.validation_split * n) per-batch,
        we compute the desired cumulative number of validation items after this
        batch, and subtract the number already assigned. This ensures that the
        epoch-level number of validation samples is approximately
            round(validation_split * total_items_seen)
        without knowing the total epoch size up front.

        Returns (train_batch, val_batch).
        """
        if batch is None:
            return None, None

        n = batch_len(batch)
        if n == 0 or self.validation_split <= 0.0:
            # still advance epoch counter by n (n==0 -> no-op)
            self._epoch_seen += n
            return batch, None

        # desired cumulative validation count after including this batch
        total_before = self._epoch_seen
        total_after = self._epoch_seen + n

        desired_before = int(round(self.validation_split * total_before))
        desired_after = int(round(self.validation_split * total_after))
        val_size = desired_after - desired_before

        # advance epoch seen counter
        self._epoch_seen = total_after

        if val_size <= 0:
            return batch, None

        # choose val_size indices deterministically from RNG permutation
        perm = self.rng.permutation(n)
        val_idxs = set(perm[:val_size].tolist())
        train_idxs = [i for i in range(n) if i not in val_idxs]
        val_idxs_ordered = [i for i in range(n) if i in val_idxs]

        if is_dataframe(batch):
            train_batch = batch.iloc[train_idxs].reset_index(drop=True)
            val_batch = batch.iloc[val_idxs_ordered].reset_index(drop=True)
        else:
            train_batch = [batch[i] for i in train_idxs]
            val_batch = [batch[i] for i in val_idxs_ordered]
        return train_batch, val_batch

    def _pull_n_from_loader(self, key, n):
        """
        Pull up to n items from loader `key`. Uses loader.next_n when available,
        otherwise accumulates from next_batch(). Returns same-type batch or None.
        """
        if n <= 0:
            return None
        loader = self.loaders[key]
        if hasattr(loader, "next_n"):
            try:
                out = loader.next_n(n)
                if batch_len(out) == 0:
                    return None
                return out
            except Exception:
                pass
        # accumulate from next_batch
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
                # take only first `need` elements
                part = slice_batch(b, 0, need)
                collected.append(part)
                # If loader returned more than needed but we couldn't push remainder back into loader,
                # we attach the remainder to an internal per-loader buffer is out-of-scope here.
                need = 0
        if not collected:
            return None
        return concat_batches(collected)

# -------------------------
# SequenceMixer
# -------------------------
class SequenceMixer(DefaultMixer):
    """
    Stream datasets in sequence (drain one, then next).

    Spec:
      - type: "sequence"
      - datasets: [ ... ]   (keys or prefixes)
      - per_dataset_batch_size: optional int (requests this when resetting each loader)
      - validation_split: optional
    """
    def __init__(self, spec, loaders, rng):
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("sequence spec requires 'datasets'")
        self.dataset_keys = [self.resolve_key(k) for k in ds]
        self.per_dataset_batch_size = spec.get("per_dataset_batch_size")
        self._cur_idx = 0
        self._current_loader = None

    def reset_epoch(self, batch_size, epoch_idx=0):
        super().reset_epoch(batch_size, epoch_idx)
        self._cur_idx = 0
        self._current_loader = None
        if self.dataset_keys:
            self._prepare_current_loader()

    def _prepare_current_loader(self):
        if self._cur_idx >= len(self.dataset_keys):
            self._current_loader = None
            return
        key = self.dataset_keys[self._cur_idx]
        loader = self.loaders[key]
        # If per_dataset_batch_size is set, use it, otherwise use the overall batch_size
        bs = int(self.per_dataset_batch_size or self.batch_size or 1)
        # Reset loader so that its internal batch sizing aligns with requested size.
        self.reset_loader(loader, bs)
        self._current_loader = loader

    def next_batch(self):
        while self._cur_idx < len(self.dataset_keys):
            if self._current_loader is None:
                self._prepare_current_loader()
                if self._current_loader is None:
                    return None, None
            try:
                batch = self._current_loader.next_batch()
            except Exception:
                batch = None
            # If loader exhausted, move to next dataset
            if batch is None or batch_len(batch) == 0:
                self._cur_idx += 1
                self._current_loader = None
                continue
            key = self.dataset_keys[self._cur_idx]
            cnt = batch_len(batch)
            # record plan entry before split
            self.mix_plan.append({"type": "sequence", "dataset": key, "counts": {key: cnt}})
            train, val = self._split_batch(batch)
            last = self.mix_plan[-1]
            last["train_count"] = batch_len(train) if train is not None else 0
            last["val_count"] = batch_len(val) if val is not None else 0
            return train, val
        return None, None

# -------------------------
# RandomBatchesMixer
# -------------------------
class RandomBatchesMixer(DefaultMixer):
    """
    Produce random mixed batches from datasets, pulling without replacement until epoch exhausted.

    Spec keys:
      - type: "random_batches"
      - datasets: [list]         (required)
      - balanced: true|false     (default False)  # ignored here; use 'weights' instead
      - weights: [list]          (optional; must match datasets)
      - validation_split: optional float
    """
    def __init__(self, spec, loaders, rng):
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("random_batches mixer requires 'datasets'")
        self.datasets = [self.resolve_key(k) for k in ds]
        weights = spec.get("weights")
        if weights is None:
            weights = ([1.0/ len(self.datasets)] * len(self.datasets))
        if len(weights) != len(self.datasets):
            raise ValueError("random_batches weights length must match datasets")
        self.weights = np.asarray(weights, dtype=float)

    def reset_epoch(self, batch_size, epoch_idx=0):
        super().reset_epoch(batch_size, epoch_idx)
        # reset all loaders to have sensible internal batch sizing
        for k in self.datasets:
            loader = self.loaders[k]
            self.reset_loader(loader, batch_size)
        # normalize probs for sampling
        w = self.weights.copy().astype(float)
        w = np.where(w < 0, 0.0, w)
        if w.sum() <= 0:
            self.probs = np.ones_like(w) / float(len(w))
        else:
            self.probs = w / w.sum()

    def next_batch(self):
        if self.batch_size is None:
            raise RuntimeError("reset_epoch must be called first")
        B = int(self.batch_size)
        # initial draw: sample counts per dataset from multinomial
        counts = self.rng.multinomial(B, self.probs).tolist()
        # attempt to pull requested counts, redistribute shortfalls
        parts = []
        produced = {}
        remaining_to_fill = 0
        # first pass: ask each loader
        shortfall = 0
        for i, key in enumerate(self.datasets):
            want = int(counts[i])
            if want <= 0:
                produced[key] = 0
                continue
            part = self._pull_n_from_loader(key, want)
            got = batch_len(part)
            if got:
                parts.append(part)
                produced[key] = got
            else:
                produced[key] = 0
            if got < want:
                shortfall += want - got
        # if shortfall > 0, attempt to fill from other loaders in a second pass
        if shortfall > 0:
            # try to draw extra from datasets that still have data
            for i, key in enumerate(self.datasets):
                if shortfall <= 0:
                    break
                # request up to shortfall
                extra = self._pull_n_from_loader(key, shortfall)
                got = batch_len(extra)
                if got:
                    parts.append(extra)
                    produced[key] = produced.get(key, 0) + got
                    shortfall -= got
        if not parts:
            return None, None
        batch = concat_batches(parts)
        # record plan and split
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
      - balance: optional dict (label->fraction) OR list (fractions in order of 'labels' or discovered labels)
      - validation_split: optional float
    """
    def __init__(self, spec, loaders, rng):
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("balanced_by_label requires 'datasets'")
        self.datasets = [self.resolve_key(k) for k in ds]
        self.provided_labels = spec.get("labels")  # optional explicit labels list
        self.micro = int(spec.get("micro_batch", 16))
        self.balance_spec = spec.get("balance") or spec.get("balance_list") or spec.get("balance_dict")
        # per-label buffers to hold leftovers (list of batches)
        self.buffers = {}
        # internal label order (set on reset_epoch)
        self.labels = []
        if "validation_split" in spec:
            self.validation_split = float(spec.get("validation_split", 0.0))

    def reset_epoch(self, batch_size, epoch_idx=0):
        super().reset_epoch(batch_size, epoch_idx)
        # reset loaders
        for k in self.datasets:
            loader = self.loaders[k]
            self.reset_loader(loader, batch_size)
        # initialize buffers
        self.buffers = {}
        # discover labels if not provided
        if self.provided_labels:
            self.labels = [l for l in list(self.provided_labels) if l is not None]
        else:
            labels_set = set()
            for k in self.datasets:
                loader = self.loaders[k]
                b = None
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
            for k in self.datasets:
                loader = self.loaders[k]
                self.reset_loader(loader, batch_size)
            if not labels_set:
                # fallback to common binary labels
                self.labels = ["Benign", "Malicious"]
            else:
                self.labels = sorted(list(labels_set))
        # init buffers map
        for lbl in self.labels:
            self.buffers.setdefault(lbl, [])

        # parse balance_spec into fractions per label
        self.balance_map = self._parse_balance_spec(self.balance_spec, self.labels)

    def _parse_balance_spec(self, spec_value, labels):
        """
        Accepts:
          - None -> equal fractions
          - dict -> {label: fraction}
          - list -> [f1, f2, ...] mapped in order to 'labels'
        Returns normalized dict mapping label->fraction (sums to 1.0)
        """
        n = len(labels)
        if spec_value is None:
            return {lbl: 1.0 / max(1, n) for lbl in labels}

        if isinstance(spec_value, dict):
            # Only keep labels that exist; missing labels assumed 0
            out = {}
            for lbl in labels:
                out[lbl] = float(spec_value.get(lbl, 0.0))
            total = sum(out.values())
            if total <= 0:
                return {lbl: 1.0 / max(1, n) for lbl in labels}
            return {lbl: v / total for lbl, v in out.items()}

        if isinstance(spec_value, (list, tuple)):
            # If provided_labels existed and list length matches, map directly
            vals = list(spec_value)
            if len(vals) == len(labels):
                arr = np.asarray([float(x) for x in vals], dtype=float)
                s = arr.sum()
                if s <= 0:
                    return {lbl: 1.0 / max(1, n) for lbl in labels}
                arr = arr / s
                return {lbl: float(arr[i]) for i, lbl in enumerate(labels)}
            # if lengths differ, attempt to pad/truncate
            arr = np.asarray([float(x) for x in vals], dtype=float)
            if arr.size == 0:
                return {lbl: 1.0 / max(1, n) for lbl in labels}
            # normalize and map cyclically (best-effort)
            arr = arr / arr.sum()
            out = {}
            for i, lbl in enumerate(labels):
                out[lbl] = float(arr[i % arr.size])
            # re-normalize
            s = sum(out.values())
            if s <= 0:
                return {lbl: 1.0 / max(1, n) for lbl in labels}
            return {lbl: v / s for lbl, v in out.items()}

        # fallback
        return {lbl: 1.0 / max(1, n) for lbl in labels}

    def _consume_from_buffer(self, label, need):
        """
        Consume up to `need` items from the buffer for `label`.
        Returns a list of collected batches (may be single batch) or None.
        Leaves any remainder from partially-consumed buffered items back in buffer.
        """
        if need <= 0:
            return None
        buf = self.buffers.setdefault(label, [])
        if not buf:
            return None
        collected = []
        remaining = need
        new_buf = []
        for b in buf:
            if remaining <= 0:
                new_buf.append(b)
                continue
            blen = batch_len(b)
            if blen <= 0:
                continue
            if blen <= remaining:
                collected.append(b)
                remaining -= blen
            else:
                # take portion, keep remainder
                part = slice_batch(b, 0, remaining)
                rem = slice_batch(b, remaining, blen)
                if part is not None:
                    collected.append(part)
                if rem is not None and batch_len(rem) > 0:
                    new_buf.append(rem)
                remaining = 0
        # replace buffer with new_buf
        self.buffers[label] = new_buf
        if not collected:
            return None
        return concat_batches(collected)

    def _store_remainder_to_buffer(self, label, batch, consumed):
        """
        Given `batch` and that we consumed `consumed` items from its start,
        store the remainder (if any) into the buffer[label].
        `consumed` must be an int <= batch_len(batch).
        """
        if batch is None:
            return
        total = batch_len(batch)
        if consumed >= total:
            return
        rem = slice_batch(batch, consumed, total)
        if rem is None:
            return
        if batch_len(rem) == 0:
            return
        self.buffers.setdefault(label, []).append(rem)

    def _pull_for_label_from_loader(self, key, label, want):
        """
        Pull up to `want` items matching `label` from loader `key`.
        Uses loader.next_n/next_batch and filters by label.
        Returns a batch (same type) or None.
        """
        if want <= 0:
            return None
        loader = self.loaders[key]
        collected = []
        total = 0
        # We'll repeatedly ask loader for chunks and filter them for label matches
        while total < want:
            chunk = None
            if hasattr(loader, "next_n"):
                try:
                    chunk = loader.next_n(max(self.micro, want - total))
                except Exception:
                    chunk = None
            if chunk is None:
                try:
                    chunk = loader.next_batch()
                except Exception:
                    chunk = None
            if chunk is None:
                break
            # filter chunk by label
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
            # defensive fallback
            self.labels = ["Benign", "Malicious"]
            for lbl in self.labels:
                self.buffers.setdefault(lbl, [])
            self.balance_map = self._parse_balance_spec(self.balance_spec, self.labels)

        B = int(self.batch_size)
        if B <= 0:
            return None, None

        # compute desired counts per label (integers) with remainder handling
        fracs = [self.balance_map.get(lbl, 0.0) for lbl in self.labels]
        raw_counts = [float(f) * B for f in fracs]
        int_counts = [int(np.floor(c)) for c in raw_counts]
        rem = B - sum(int_counts)
        # distribute remainder to labels with largest fractional parts
        frac_parts = [(i, raw_counts[i] - int_counts[i]) for i in range(len(self.labels))]
        frac_parts.sort(key=lambda x: x[1], reverse=True)
        for i in range(rem):
            idx = frac_parts[i % len(frac_parts)][0]
            int_counts[idx] += 1

        parts = []
        produced = {}
        # For each label: first take from buffer, then pull from loaders until satisfied
        for i, lbl in enumerate(self.labels):
            need = int_counts[i]
            if need <= 0:
                produced[lbl] = 0
                continue
            collected_parts = []
            # consume buffer
            buf_part = self._consume_from_buffer(lbl, need)
            got = batch_len(buf_part)
            if got:
                collected_parts.append(buf_part)
                need -= got
            # round-robin pull across datasets until need satisfied or exhausted
            ds_idx = 0
            attempts = 0
            per_ds_want = int(np.ceil(float(max(1, need)) / max(1, len(self.datasets))))
            while need > 0 and attempts < len(self.datasets) * 4:
                key = self.datasets[ds_idx % len(self.datasets)]
                want = min(per_ds_want, need)
                chunk = self._pull_for_label_from_loader(key, lbl, want)
                if chunk is None:
                    ds_idx += 1
                    attempts += 1
                    continue
                # If chunk larger than needed, take consumed portion and buffer remainder
                clen = batch_len(chunk)
                if clen <= need:
                    collected_parts.append(chunk)
                    need -= clen
                else:
                    # take first `need` items, buffer remainder
                    taken = slice_batch(chunk, 0, need)
                    rem = slice_batch(chunk, need, clen)
                    if taken is not None and batch_len(taken) > 0:
                        collected_parts.append(taken)
                    if rem is not None and batch_len(rem) > 0:
                        self.buffers.setdefault(lbl, []).append(rem)
                    need = 0
                ds_idx += 1
            # finalize for this label
            if collected_parts:
                part = concat_batches(collected_parts)
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
