# Author: Jan Svoboda
# functionality: Labelled Zeek conn logs -> the SLIPS federated module's 18 features,
# plus the fixed test split, CV folds and the module's 50/50 undersampling.
# behavior:
#   - Datasets are read with the pipeline's ZeekDataset (label mapping, Background skipped).
#   - Each Zeek record becomes the flow dict SLIPS's Zeek profiler would store
#     (slips_files/core/input_profilers/zeek.py, flows/zeek.py Conn): '-' -> "",
#     dur float, bytes/pkts/ports int, raw conn_state, dir_ "->". The pipeline's
#     ConnToSlipsConverter is NOT used: it turns state into 0/1 and leaves dir_ empty,
#     which would change two of the module's features.

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..commons import MALICIOUS
from ..dataset_wrapper import ZeekDataset
from .model import extract_flow_features

# Zeek conn field -> SLIPS flow key (slips_files/core/input_profilers/zeek_to_slips_maps.py)
CONN_TO_SLIPS = {
    "ts": "starttime",
    "uid": "uid",
    "id.orig_h": "saddr",
    "id.orig_p": "sport",
    "id.resp_h": "daddr",
    "id.resp_p": "dport",
    "proto": "proto",
    "service": "appproto",
    "duration": "dur",
    "orig_bytes": "sbytes",
    "resp_bytes": "dbytes",
    "conn_state": "state",
    "history": "history",
    "orig_pkts": "spkts",
    "resp_pkts": "dpkts",
}
_INT_FIELDS = ("sbytes", "dbytes", "spkts", "dpkts", "sport", "dport")


def zeek_to_slips_flow(record: Dict) -> Dict:
    """One Zeek conn record (ZeekDataset row or Zeek JSON line) -> SLIPS conn flow dict."""
    flow = {}
    for zeek_field, slips_field in CONN_TO_SLIPS.items():
        val = record.get(zeek_field, "")
        flow[slips_field] = "" if val is None or val == "-" else val
    flow["dur"] = float(flow.get("dur", 0) or 0)
    for f in _INT_FIELDS:
        flow[f] = int(float(flow.get(f, 0) or 0))
    if flow["proto"] == "unknown_transport":
        flow["proto"] = ""
    flow["type_"] = "conn"
    flow["dir_"] = "->"
    return flow


def resolve_dataset_dirs(root: str, ids: List[str], data_subdir: str = "data") -> Dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    out = {}
    for ds_id in ids:
        hits = sorted(p for p in root_path.iterdir() if p.name.startswith(str(ds_id)))
        if not hits or not (hits[0] / data_subdir).is_dir():
            raise FileNotFoundError(f"dataset {ds_id} not found under {root_path}")
        out[str(ds_id)] = hits[0] / data_subdir
    return out


def load_features(
    root: str,
    ids: List[str],
    cache_dir: Optional[str] = None,
    max_flows_per_dataset: Optional[int] = None,
    seed: int = 1111,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """X (n x 18, float64), y (1 = malicious), source dataset id per row, summary."""
    X, y, src, summary = [], [], [], {}
    for ds_id, path in resolve_dataset_dirs(root, ids).items():
        ds = ZeekDataset(path, batch_size=1, cache_dir=cache_dir)
        rows, labels = [], []
        for rec in ds._iter_lines():
            feats = extract_flow_features(zeek_to_slips_flow(rec))
            if feats is None:
                continue
            rows.append(feats)
            labels.append(1 if rec["label"] == MALICIOUS else 0)
        rows, labels = np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)
        if max_flows_per_dataset and len(labels) > max_flows_per_dataset:
            keep, _ = train_test_split(
                np.arange(len(labels)),
                train_size=max_flows_per_dataset,
                stratify=labels,
                random_state=seed,
            )
            keep = np.sort(keep)
            rows, labels = rows[keep], labels[keep]
        summary[ds_id] = {
            "path": str(path),
            "flows": int(len(labels)),
            "malicious": int(labels.sum()),
            "benign": int((labels == 0).sum()),
        }
        X.append(rows)
        y.append(labels)
        src.append(np.full(len(labels), ds_id))
    X, y, src = np.vstack(X), np.concatenate(y), np.concatenate(src)
    summary["total"] = {"flows": int(len(y)), "malicious": int(y.sum()), "benign": int((y == 0).sum())}
    return X, y, src, summary


def holdout_split(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed, seeded, stratified train/test indices."""
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed
    )
    return np.sort(train_idx), np.sort(test_idx)


def cv_folds(y_train: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold (positions within the training split)."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y_train)), y_train))


def undersample(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """The module's _undersample_majority: keep all of the smaller class and an
    equal random sample of the larger (original row order kept)."""
    idx_mal = np.where(y == 1)[0]
    idx_ben = np.where(y == 0)[0]
    n = min(len(idx_mal), len(idx_ben))
    if n == 0 or len(idx_mal) == len(idx_ben):
        return X, y
    keep_mal = rng.choice(idx_mal, n, replace=False) if len(idx_mal) > n else idx_mal
    keep_ben = rng.choice(idx_ben, n, replace=False) if len(idx_ben) > n else idx_ben
    keep = np.sort(np.concatenate([keep_mal, keep_ben]))
    return X[keep], y[keep]
