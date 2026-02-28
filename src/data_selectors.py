"""Mixers operating on full-dataframe loaders.

The goal of this module is to keep all batching and mixing logic in one place.
Loaders are expected to expose an :py:meth:`as_dataframe` method that returns
the entire dataset as a pandas ``DataFrame``. Mixers take those DataFrames,
apply optional shuffling, and emit train/validation batches according to the
requested strategy.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd



# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def is_dataframe(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.DataFrame)


def batch_len(batch: Any) -> int:
    if batch is None:
        return 0
    if is_dataframe(batch):
        return int(len(batch))
    try:
        return int(len(batch))
    except Exception:
        return 1


def concat_batches(parts: Sequence[Any]) -> Any:
    if not parts:
        return None
    cleaned = [p for p in parts if p is not None]
    if not cleaned:
        return None
    first = cleaned[0]
    if is_dataframe(first):
        dfs = [p.reset_index(drop=True) for p in cleaned if batch_len(p) > 0]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    out: List[Any] = []
    for part in cleaned:
        if part is None:
            continue
        if isinstance(part, list):
            out.extend(part)
        else:
            try:
                out.extend(list(part))
            except Exception:
                out.append(part)
    return out


def slice_batch(batch: Any, start: int, end: int) -> Any:
    if batch is None or start >= end:
        if is_dataframe(batch):
            return pd.DataFrame().iloc[:0]
        return []
    if is_dataframe(batch):
        return batch.iloc[start:end].reset_index(drop=True)
    try:
        return batch[start:end]
    except Exception:
        return []


def _allocate_from_fractions(fractions: Sequence[float], total: int) -> List[int]:
    if total <= 0 or not fractions:
        return [0 for _ in fractions]
    arr = np.asarray([max(0.0, float(f)) for f in fractions], dtype=float)
    if arr.sum() <= 0:
        arr = np.ones_like(arr)
    arr = (arr / arr.sum()) * float(total)
    ints = np.floor(arr).astype(int)
    remainder = total - int(ints.sum())
    order = np.argsort(-(arr - np.floor(arr)))
    for i in range(remainder):
        idx = int(order[i % len(order)])
        ints[idx] += 1
    return ints.tolist()


def _normalize_weights(weights: Optional[Sequence[float]], length: int) -> np.ndarray:
    if weights is None:
        return np.ones(length, dtype=float) / max(1, length)
    arr = np.asarray([max(0.0, float(w)) for w in weights], dtype=float)
    if arr.size != length or arr.sum() <= 0:
        return np.ones(length, dtype=float) / max(1, length)
    return arr / arr.sum()


# ---------------------------------------------------------------------------
# Base mixer
# ---------------------------------------------------------------------------


class BaseMixer:
    """Base functionality shared by all mixers."""

    def __init__(self, spec: Optional[Dict[str, Any]], loaders: Dict[str, Any], rng: np.random.RandomState):
        self.spec = spec or {}
        source_loaders = loaders or {}
        self.loaders = self._materialize_loaders(source_loaders)
        self.rng = rng or np.random.RandomState()
        self.validation_split = float(self.spec.get("validation_split", 0.0))
        self.shuffle_per_epoch = bool(self.spec.get("shuffle_per_epoch", False))

        self.batch_size: Optional[int] = None
        self.mix_plan: List[Dict[str, Any]] = []
        self.processed_dfs: Dict[str, Any] = {}
        self._orders: Dict[str, Optional[np.ndarray]] = {}
        self._epoch_seen = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset_epoch(self, batch_size: int, epoch_idx: int = 0) -> None:
        if batch_size is None:
            raise ValueError("batch_size must be provided")
        self.batch_size = int(batch_size)
        self.mix_plan = []
        self._epoch_seen = 0
        self.processed_dfs = {}
        self._orders = {}

        for key, data in self.loaders.items():
            if data is None:
                data = self._empty_frame()
            order = None
            if self.shuffle_per_epoch and is_dataframe(data) and batch_len(data) > 1:
                order = self.rng.permutation(len(data))
            self.processed_dfs[key] = data
            self._orders[key] = order

    def next_batch(self) -> Tuple[Any, Any]:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def get_mix_plan(self) -> List[Dict[str, Any]]:
        return list(self.mix_plan)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _empty_frame(self):
        if pd is None:
            return []
        return pd.DataFrame()

    def _random_seed(self) -> int:
        return int(self.rng.randint(0, 2**31 - 1))

    def _shuffle_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        perm = self.rng.permutation(len(df))
        return df.iloc[perm].reset_index(drop=True)

    def _ensure_dataframe(self, data: Any) -> Any:
        if data is None:
            return self._empty_frame()
        if is_dataframe(data):
            return data
        if pd is None:
            return data
        try:
            if isinstance(data, list):
                if not data:
                    return pd.DataFrame()
                return pd.DataFrame(data).reset_index(drop=True)
            if isinstance(data, dict):
                return pd.DataFrame([data]).reset_index(drop=True)
            iterable = list(data)
            if not iterable:
                return pd.DataFrame()
            return pd.DataFrame(iterable).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _extract_dataframe(self, loader: Any) -> Any:
        if loader is None:
            return self._empty_frame()
        if is_dataframe(loader):
            return loader.reset_index(drop=True)
        if hasattr(loader, "as_dataframe"):
            try:
                return loader.as_dataframe()
            except Exception:
                return self._empty_frame()
        return self._ensure_dataframe(loader)

    def _materialize_loaders(self, loaders: Dict[str, Any]) -> Dict[str, Any]:
        materialized: Dict[str, Any] = {}
        for key, loader in loaders.items():
            materialized[key] = self._ensure_dataframe(self._extract_dataframe(loader))
        return materialized

    def _split_batch(self, batch: Any) -> Tuple[Any, Any]:
        if batch is None:
            return None, None
        total = batch_len(batch)
        if total == 0 or self.validation_split <= 0.0:
            self._epoch_seen += total
            return batch, None

        before = self._epoch_seen
        after = before + total
        desired_before = int(round(self.validation_split * before))
        desired_after = int(round(self.validation_split * after))
        val_size = max(0, desired_after - desired_before)
        self._epoch_seen = after

        if val_size <= 0:
            return batch, None

        perm = self.rng.permutation(total)
        val_idx = set(perm[:val_size].tolist())
        train_idx = [i for i in range(total) if i not in val_idx]
        ordered_val_idx = [i for i in range(total) if i in val_idx]

        if is_dataframe(batch):
            train = batch.iloc[train_idx].reset_index(drop=True)
            val = batch.iloc[ordered_val_idx].reset_index(drop=True)
        else:
            train = [batch[i] for i in train_idx]
            val = [batch[i] for i in ordered_val_idx]
        return train, val

    def resolve_key(self, key: str) -> str:
        if key in self.loaders:
            return key
        matches = [name for name in self.loaders if key in name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"Dataset '{key}' not found among loaders")
        raise KeyError(f"Dataset '{key}' is ambiguous (matches: {matches})")

    # ------------------------------------------------------------------
    # shared dataset helpers
    # ------------------------------------------------------------------
    def _consume_dataset_rows(self, dataset_key: str, start_offset: int, count: int) -> Tuple[Any, int]:
        if count <= 0:
            return None, start_offset
        data = self.processed_dfs.get(dataset_key, self._empty_frame())
        total = batch_len(data)
        if start_offset >= total:
            return None, start_offset
        end = min(start_offset + count, total)
        order = self._orders.get(dataset_key)
        if is_dataframe(data) and order is not None:
            sel = order[start_offset:end]
            chunk = data.iloc[sel]
        else:
            chunk = slice_batch(data, start_offset, end)
        if chunk is None or batch_len(chunk) == 0:
            return None, end
        return chunk, end

    def _record_mix(self, entry: Dict[str, Any], train: Any, val: Any) -> None:
        record = dict(entry)
        record["train_count"] = batch_len(train) if train is not None else 0
        record["val_count"] = batch_len(val) if val is not None else 0
        self.mix_plan.append(record)


# ---------------------------------------------------------------------------
# Sequence mixer
# ---------------------------------------------------------------------------


class SequenceMixer(BaseMixer):
    """Drain datasets sequentially, preserving the per-dataset ordering."""

    def __init__(self, spec: Dict[str, Any], loaders: Dict[str, Any], rng: np.random.RandomState):
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("sequence mixer requires 'datasets'")
        self.dataset_keys = [self.resolve_key(k) for k in ds]
        self.per_dataset_batch_size = spec.get("per_dataset_batch_size")
        self._cursor = 0
        self._offsets: Dict[str, int] = {}

    def reset_epoch(self, batch_size: int, epoch_idx: int = 0) -> None:
        super().reset_epoch(batch_size, epoch_idx)
        self._cursor = 0
        self._offsets = {k: 0 for k in self.dataset_keys}

    def next_batch(self) -> Tuple[Any, Any]:
        if self.batch_size is None:
            raise RuntimeError("reset_epoch must be called first")
        per_ds_size = int(self.per_dataset_batch_size or self.batch_size)
        while self._cursor < len(self.dataset_keys):
            key = self.dataset_keys[self._cursor]
            start = self._offsets.get(key, 0)
            batch, new_offset = self._consume_dataset_rows(key, start, per_ds_size)
            self._offsets[key] = new_offset
            if batch is None or batch_len(batch) == 0:
                self._cursor += 1
                continue
            train, val = self._split_batch(batch)
            self._record_mix({"type": "sequence", "dataset": key, "dataset_batch_size": batch_len(batch)}, train, val)
            return train, val
        return None, None


# ---------------------------------------------------------------------------
# Random batches mixer
# ---------------------------------------------------------------------------


class RandomBatchesMixer(BaseMixer):
    """Mix samples randomly across datasets using multinomial allocation."""

    def __init__(self, spec: Dict[str, Any], loaders: Dict[str, Any], rng: np.random.RandomState):
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("random_batches mixer requires 'datasets'")
        self.datasets = [self.resolve_key(k) for k in ds]
        self.weights = _normalize_weights(spec.get("weights"), len(self.datasets))
        self.shuffle_within_batch = bool(spec.get("shuffle_within_batch", True))
        self._offsets: Dict[str, int] = {}

    def reset_epoch(self, batch_size: int, epoch_idx: int = 0) -> None:
        super().reset_epoch(batch_size, epoch_idx)
        self._offsets = {k: 0 for k in self.datasets}

    def next_batch(self) -> Tuple[Any, Any]:
        if self.batch_size is None:
            raise RuntimeError("reset_epoch must be called first")
        counts = self.rng.multinomial(int(self.batch_size), self.weights)
        chunks: List[Any] = []
        actual_counts: Dict[str, int] = {k: 0 for k in self.datasets}

        for dataset_key, need in zip(self.datasets, counts.tolist()):
            chunk = self._take_from_dataset(dataset_key, need)
            if chunk is None:
                continue
            actual_counts[dataset_key] += batch_len(chunk)
            chunks.append(chunk)

        missing = int(self.batch_size) - sum(actual_counts.values())
        if missing > 0:
            for dataset_key in self.datasets:
                if missing <= 0:
                    break
                chunk = self._take_from_dataset(dataset_key, missing)
                if chunk is None:
                    continue
                taken = batch_len(chunk)
                if taken == 0:
                    continue
                missing -= taken
                actual_counts[dataset_key] += taken
                chunks.append(chunk)

        if not chunks:
            return None, None

        batch = concat_batches(chunks)
        if is_dataframe(batch) and self.shuffle_within_batch and batch_len(batch) > 1:
            batch = self._shuffle_dataframe(batch)

        train, val = self._split_batch(batch)
        self._record_mix({"type": "random_batches", "counts_by_dataset": actual_counts}, train, val)
        return train, val

    def _take_from_dataset(self, key: str, requested: int) -> Any:
        if requested <= 0:
            return None
        start = self._offsets.get(key, 0)
        chunk, new_offset = self._consume_dataset_rows(key, start, requested)
        self._offsets[key] = new_offset
        return chunk


# ---------------------------------------------------------------------------
# Balanced mixers
# ---------------------------------------------------------------------------


class BalancedByLabelMixer(BaseMixer):
    """Return label-balanced batches until any label is exhausted."""

    def __init__(self, spec: Dict[str, Any], loaders: Dict[str, Any], rng: np.random.RandomState):
        if pd is None:
            raise RuntimeError("pandas is required for balanced mixers")
        super().__init__(spec, loaders, rng)
        ds = spec.get("datasets") or []
        if not ds:
            raise ValueError("balanced_by_label requires 'datasets'")
        self.datasets = [self.resolve_key(k) for k in ds]
        self.provided_labels = spec.get("labels")
        self.balance_spec = spec.get("balance") or spec.get("balance_list") or spec.get("balance_dict")
        self.shuffle_full_table = bool(spec.get("shuffle_labels", True))
        self.shuffle_labels = self.shuffle_full_table  # backwards compatibility
        self.shuffle_within_dataset = bool(spec.get("shuffle_within_dataset", True))

        self.labels: List[str] = []
        self.balance_map: Dict[str, float] = {}
        self._label_indices: Dict[str, List[tuple]] = {}
        self._label_offsets: Dict[str, int] = {}

    def reset_epoch(self, batch_size: int, epoch_idx: int = 0) -> None:
        super().reset_epoch(batch_size, epoch_idx)
        self._build_label_state()

    def _build_label_state(self) -> None:
        self.labels = []
        self.balance_map = {}
        self._label_indices = {}
        self._label_offsets = {}

        for key in self.datasets:
            frame = self.processed_dfs.get(key)
            if frame is None or not is_dataframe(frame) or frame.empty:
                continue
            if "label" not in frame.columns:
                continue
            order = self._orders.get(key)
            if order is None:
                idx_iter = list(range(len(frame)))
                if self.shuffle_within_dataset and len(frame) > 1:
                    idx_iter = self.rng.permutation(len(frame)).tolist()
            else:
                idx_iter = order.tolist()
            labels_series = frame["label"].reset_index(drop=True)
            for idx in idx_iter:
                try:
                    lbl = labels_series.iat[idx]
                except Exception:
                    continue
                if pd.isna(lbl):
                    continue
                bucket = self._label_indices.setdefault(lbl, [])
                bucket.append((key, idx))

        if not self._label_indices:
            return

        if self.provided_labels:
            labels = [lab for lab in list(self.provided_labels) if lab is not None]
        else:
            labels = sorted(self._label_indices.keys())
        self.labels = labels

        # optional shuffle across combined indices per label
        if self.shuffle_full_table:
            for lbl in self.labels:
                seq = self._label_indices.get(lbl, [])
                if len(seq) > 1:
                    self.rng.shuffle(seq)

        for label in self.labels:
            self._label_offsets[label] = 0

        self.balance_map = self._parse_balance_spec(self.balance_spec, self.labels)

    @staticmethod
    def _parse_balance_spec(spec_value: Any, labels: Sequence[str]) -> Dict[str, float]:
        n = len(labels)
        if n == 0:
            return {}
        if spec_value is None:
            return {lbl: 1.0 / n for lbl in labels}
        if isinstance(spec_value, dict):
            arr = np.asarray([float(spec_value.get(lbl, 0.0)) for lbl in labels], dtype=float)
        elif isinstance(spec_value, (list, tuple)):
            arr = np.asarray([float(x) for x in spec_value], dtype=float)
            if arr.size == n:
                pass
            elif arr.size > 0:
                arr = np.resize(arr, n)
            else:
                arr = np.ones(n, dtype=float)
        else:
            arr = np.ones(n, dtype=float)
        if arr.sum() <= 0:
            arr = np.ones(n, dtype=float)
        arr = arr / arr.sum()
        return {lbl: float(arr[i]) for i, lbl in enumerate(labels)}

    def _compute_target_counts(self) -> List[int]:
        fractions = [self.balance_map.get(lbl, 0.0) for lbl in self.labels]
        return _allocate_from_fractions(fractions, int(self.batch_size or 0))

    def _take_for_label(self, label: str, need: int, allow_resample: bool) -> Tuple[Optional[pd.DataFrame], int, int]:
        indices = self._label_indices.get(label)
        if not indices:
            return None, 0, 0
        start = self._label_offsets.get(label, 0)
        available = len(indices) - start
        take = min(need, max(0, available))

        selected = indices[start:start + take]
        self._label_offsets[label] = start + take

        missing = need - take
        if missing > 0:
            if not allow_resample or len(indices) == 0:
                return None, take, 0
            sampled_idx = self.rng.choice(len(indices), size=missing, replace=True)
            for idx in sampled_idx.tolist():
                selected.append(indices[idx])

        if not selected:
            return None, take, 0

        # gather rows grouped by dataset to minimize slicing overhead
        parts: List[pd.DataFrame] = []
        by_ds: Dict[str, List[int]] = {}
        for ds_key, row_idx in selected:
            by_ds.setdefault(ds_key, []).append(row_idx)
        for ds_key, idx_list in by_ds.items():
            frame = self.processed_dfs.get(ds_key)
            if frame is None or not is_dataframe(frame) or frame.empty:
                continue
            parts.append(frame.iloc[idx_list])

        if not parts:
            return None, take, 0

        resampled = max(0, missing)
        return pd.concat(parts, ignore_index=True), take, resampled

    def _shuffle_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        if batch_len(batch) <= 1:
            return batch
        return self._shuffle_dataframe(batch)

    def next_batch(self) -> Tuple[Any, Any]:
        if self.batch_size is None:
            raise RuntimeError("reset_epoch must be called first")
        if not self.labels:
            return None, None
        targets = self._compute_target_counts()
        if sum(targets) == 0:
            return None, None

        label_chunks: List[pd.DataFrame] = []
        counts: Dict[str, int] = {}
        for label, need in zip(self.labels, targets):
            if need <= 0:
                counts[label] = 0
                continue
            chunk, fresh, _ = self._take_for_label(label, need, allow_resample=False)
            if chunk is None or fresh < need:
                return None, None
            label_chunks.append(chunk)
            counts[label] = batch_len(chunk)

        if not label_chunks:
            return None, None
        batch = pd.concat(label_chunks, ignore_index=True)
        batch = self._shuffle_batch(batch)

        train, val = self._split_batch(batch)
        self._record_mix({"type": "balanced_by_label", "counts_by_label": counts}, train, val)
        return train, val


class BalancedByLabelWithOversamplingMixer(BalancedByLabelMixer):
    """Balanced mixer that oversamples (with replacement) when a label runs out."""

    def next_batch(self) -> Tuple[Any, Any]:
        if self.batch_size is None:
            raise RuntimeError("reset_epoch must be called first")
        if not self.labels:
            return None, None
        targets = self._compute_target_counts()
        if sum(targets) == 0:
            return None, None

        label_chunks: List[pd.DataFrame] = []
        counts: Dict[str, int] = {}
        resampled: Dict[str, int] = {}
        for label, need in zip(self.labels, targets):
            if need <= 0:
                counts[label] = 0
                resampled[label] = 0
                continue
            chunk, fresh, added = self._take_for_label(label, need, allow_resample=True)
            if chunk is None:
                return None, None
            counts[label] = batch_len(chunk)
            resampled[label] = added
            label_chunks.append(chunk)

        if not label_chunks:
            return None, None
        batch = pd.concat(label_chunks, ignore_index=True)
        batch = self._shuffle_batch(batch)

        train, val = self._split_batch(batch)
        self._record_mix({
            "type": "balanced_oversampling",
            "counts_by_label": counts,
            "resampled_by_label": resampled,
        }, train, val)
        return train, val
