# Author: Jan Svoboda
# functionality: ZeekDataset loader specialized for conn.log.labeled files with caching. Handles casting, label mapping, batching, and efficient indexing for large files.
# behavior:
#   - Reads conn.log.labeled (or falls back to conn.log).
#   - Casts columns according to Zeek #types header.
#   - Skips flows with BACKGROUND label.
#   - Defaults unlabeled flows to BENIGN.
#   - Stores index in cache/ directory for large files (>50k valid flows) and reloads automatically.

from commons import BENIGN, MALICIOUS, BACKGROUND
from conn_normalizer import ConnToSlipsConverter
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import hashlib
import json
from functools import lru_cache


class ZeekDataset:
    def __init__(
        self,
        root,
        batch_size,
        seed=None,
        persist_cache_threshold=30000,
        cache_dir=None,
        labeled_filenames=None,
        file_encoding="utf-8",
        file_errors="ignore",
        shuffle_per_epoch=False,
    ):
        self.root = Path(root)
        self.seed = seed
        self.rng = random.Random(seed)
        self.persist_cache_threshold = persist_cache_threshold
        self.file_encoding = file_encoding
        self.file_errors = file_errors
        self.shuffle_per_epoch = shuffle_per_epoch
        self.batch_size = batch_size
        self.converter = ConnToSlipsConverter(default_label=str(BENIGN))

        # Always initialize _dataframe to None
        self._dataframe = None

        # cache dir
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        if labeled_filenames is None:
            self.labeled_filenames = [
                "conn.log.labeled",
                "labeled-conn.log",
                "conn.log",
            ]
        else:
            self.labeled_filenames = list(labeled_filenames)

        if not self.root.exists():
            raise FileNotFoundError(f"Root path {self.root} does not exist")

        self.find_labeled_logfile()
        self._index_file()
        # Always ensure DataFrame is loaded at init
        self.as_dataframe()

    def as_dataframe(self):
        """Load the entire dataset into a pandas DataFrame, and cache it."""
        if self._dataframe is not None:
            return self._dataframe
        records = list(self._iter_lines())
        normalized = self._normalize_records(records)
        self._dataframe = pd.DataFrame(normalized)
        return self._dataframe

    def _normalize_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
        try:
            return self.converter.normalize_batch(records)
        except Exception:
            return records

    def find_labeled_logfile(self) -> Path:
        for fname in self.labeled_filenames:
            candidate = self.root / fname
            if candidate.exists():
                self.current_file = candidate
                return candidate
        raise FileNotFoundError(
            f"No conn.log.labeled, labeled-conn.log or conn.log in {self.root}"
        )

    def __len__(self):
        return self.total_lines

    def _cache_path(self):
        base = self.cache_dir
        base.mkdir(exist_ok=True)
        file_hash = hashlib.sha1(str(self.current_file).encode()).hexdigest()[
            :16
        ]
        return base / f"{file_hash}.json"

    def clear_cache(self):
        """Remove the cache file for this dataset only."""
        cache_file = self._cache_path()
        if cache_file.exists():
            cache_file.unlink()

    def _index_file(self):
        cache_file = self._cache_path()
        file_stat = self.current_file.stat()

        # try loading from cache
        if cache_file.exists():
            with open(
                cache_file,
                "r",
                encoding=self.file_encoding,
                errors=self.file_errors,
            ) as f:
                data = json.load(f)
            if (
                data["__file_size"] == file_stat.st_size
                and data["__mtime"] == file_stat.st_mtime
            ):
                self.headers = data["headers"]
                self.types = data["types"]
                self.valid_indices = data["valid_indices"]
                self.labels = data["labels"]
                self.total_lines = len(self.valid_indices)
                return

        # else build fresh index
        headers, types = [], []
        valid_indices, labels = [], []

        with open(
            self.current_file,
            "r",
            encoding=self.file_encoding,
            errors=self.file_errors,
        ) as fh:
            idx = 0
            for line in fh:
                if line.startswith("#fields"):
                    headers = line.strip().split()[1:]
                elif line.startswith("#types"):
                    types = line.strip().split()[1:]
                elif line.startswith("#"):
                    continue
                else:
                    parts = line.strip().split("\t")
                    label_val = None
                    if "label" in headers:
                        try:
                            label_val = parts[headers.index("label")]
                        except Exception:
                            label_val = None
                    if not label_val:
                        label_val = str(BENIGN)

                    lab_up = label_val.upper()
                    if (
                        lab_up == str(BACKGROUND).upper()
                        or "BACKGROUND" in lab_up
                    ):
                        idx += 1
                        continue
                    if (
                        lab_up == str(MALICIOUS).upper()
                        or "MAL" in lab_up
                        or lab_up == "1"
                    ):
                        mapped = MALICIOUS
                    else:
                        mapped = BENIGN

                    valid_indices.append(idx)
                    labels.append(str(mapped))
                    idx += 1
            # Shuffle valid_indices and labels together with fixed seed (if provided)
            if self.seed is not None:
                combined = list(zip(valid_indices, labels))
                rng = random.Random(self.seed)
                rng.shuffle(combined)
                valid_indices, labels = zip(*combined)
                valid_indices = list(valid_indices)
                labels = list(labels)

        self.headers = headers
        self.types = types
        self.valid_indices = valid_indices
        self.labels = labels
        self.total_lines = len(valid_indices)

        # persist if large enough
        if len(valid_indices) > self.persist_cache_threshold:
            with open(
                cache_file,
                "w",
                encoding=self.file_encoding,
                errors=self.file_errors,
            ) as f:
                json.dump(
                    {
                        "__file_size": file_stat.st_size,
                        "__mtime": file_stat.st_mtime,
                        "headers": headers,
                        "types": types,
                        "valid_indices": valid_indices,
                        "labels": labels,
                    },
                    f,
                )

    def _iter_lines(self):
        headers, types = self.headers, self.types
        valid_set = set(self.valid_indices)
        labels = {
            index: label
            for index, label in zip(self.valid_indices, self.labels)
        }

        with open(
            self.current_file,
            "r",
            encoding=self.file_encoding,
            errors=self.file_errors,
        ) as fh:
            idx = 0
            for line in fh:
                if line.startswith("#"):
                    continue
                if idx not in valid_set:
                    idx += 1
                    continue
                parts = line.strip().split("\t")
                if len(parts) < len(headers):
                    parts.extend(["" for _ in range(len(headers) - len(parts))])
                record = {
                    h: self._cast(
                        parts[i], types[i] if i < len(types) else None
                    )
                    for i, h in enumerate(headers)
                }
                record["label"] = labels.get(idx, str(BENIGN))
                yield record
                idx += 1

    def _cast(self, value: str, typ: Optional[str]):
        if value in ("-", ""):
            return None
        if typ in ("int", "count", "port"):
            return int(value)
        if typ in ("double", "float"):
            return float(value)
        if typ == "bool":
            return value.lower() == "t"
        if typ == "time":
            return float(value)
        if typ == "interval":
            return float(value)
        if typ == "string" or typ == "str":
            try:
                return float(value)
            except ValueError:
                return value
        if typ == "":
            return float(value)
        return value

    # Batching and line access methods removed: batching is now handled by mixers only


# -------------------
# Helper functions
# -------------------


@lru_cache(maxsize=None)
def find_and_load_datasets(
    root_dir,
    batch_size=1000,
    prefix_regex=r"^\d{3}",
    data_subdir="data",
    seed=None,
    persist_cache_threshold=30000,
    cache_dir=None,
    labeled_filenames=None,
    file_encoding="utf-8",
    file_errors="ignore",
    shuffle_per_epoch=False,
) -> Dict[str, ZeekDataset]:

    import re

    # Normalize arguments before any filesystem access so the cache key is stable
    root_path = Path(root_dir).resolve()
    cache_path = Path(cache_dir).resolve() if cache_dir is not None else None
    labeled_files = (
        tuple(labeled_filenames)
        if labeled_filenames is not None
        else None
    )
    prefix_pattern = str(prefix_regex)
    data_subdir = str(data_subdir)
    file_encoding = str(file_encoding)
    file_errors = str(file_errors)
    shuffle_per_epoch = bool(shuffle_per_epoch)

    if not root_path.exists():
        raise FileNotFoundError(f"root_dir {root_path} does not exist")

    pattern = re.compile(prefix_pattern)
    loaders: Dict[str, ZeekDataset] = {}

    for entry in sorted(root_path.iterdir()):
        if not pattern.match(entry.name):
            continue
        data_path = entry / data_subdir
        if not data_path.is_dir():
            continue
        try:
            ds = ZeekDataset(
                data_path,
                batch_size=batch_size,
                seed=seed,
                persist_cache_threshold=persist_cache_threshold,
                cache_dir=cache_path,
                labeled_filenames=labeled_files,
                file_encoding=file_encoding,
                file_errors=file_errors,
                shuffle_per_epoch=shuffle_per_epoch,
            )
        except FileNotFoundError:
            # Skip datasets without conn.log/conn.log.labeled instead of failing the whole run
            continue
        loaders[entry.name] = ds

    return loaders
