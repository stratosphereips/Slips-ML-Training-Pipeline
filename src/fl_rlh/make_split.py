# Author: Jan Svoboda
# functionality: One-time fixed split of the data for configs/fl_rlh_optuna.yaml: each
# training capture's labelled Zeek conn log is cut into <id>-train-<name> (90%) and
# <id>-test-<name> (10%), stratified by label and seeded; the outside test captures are
# linked in unchanged.
# behavior:
#   - The log's header lines (#separator, #fields, #types, ...) and its #close line are
#     copied to both parts; rows keep their original (time) order within each part.
#   - Same seed -> the same files, so every Optuna trial trains on exactly the same rows
#     and is scored on rows it never saw, whatever mixer it samples.
#   - Writes split_manifest.json (rows per part and label) into the output root.
# usage:
#   python3 -m src.fl_rlh.make_split --src ./datasets --out ./datasets_fl_rlh \
#       --train 008 010 011 012 037 --outside 014 016 027 [--test-size 0.1] [--seed 1111]

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

LABELED_FILENAMES = ("conn.log.labeled", "labeled-conn.log", "conn.log")


def find_dataset(src: Path, ds_id: str) -> Path:
    hits = sorted(p for p in src.iterdir() if p.name.startswith(f"{ds_id}-"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one dataset '{ds_id}-*' under {src}, found {[p.name for p in hits]}")
    return hits[0]


def labeled_file(dataset_dir: Path, data_subdir: str = "data") -> Path:
    for name in LABELED_FILENAMES:
        path = dataset_dir / data_subdir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"no labelled conn log in {dataset_dir / data_subdir}")


def _label_class(fields: List[str], row: str, label_idx: int) -> str:
    cols = row.rstrip("\n").split("\t")
    value = cols[label_idx].lower() if label_idx < len(cols) else ""
    if "malicious" in value:
        return "malicious"
    if "benign" in value:
        return "benign"
    return "other"


def split_log(path: Path, test_size: float, seed: int) -> Dict[str, object]:
    """(header, train_rows, test_rows, closing, counts) of one labelled Zeek log."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    header = [ln for ln in lines if ln.startswith("#") and not ln.startswith("#close")]
    closing = [ln for ln in lines if ln.startswith("#close")]
    rows = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    fields_line = next((ln for ln in header if ln.startswith("#fields")), None)
    if fields_line is None:
        raise ValueError(f"{path} has no #fields header")
    fields = fields_line.rstrip("\n").split("\t")[1:]
    if "label" not in fields:
        raise ValueError(f"{path} has no 'label' field")
    label_idx = fields.index("label")
    classes = np.array([_label_class(fields, r, label_idx) for r in rows])
    rng = np.random.default_rng(seed)
    test_idx = []
    for cls in sorted(set(classes)):
        idx = np.where(classes == cls)[0]
        n_test = int(round(test_size * len(idx)))
        test_idx.extend(rng.permutation(idx)[:n_test].tolist())
    test_set = set(test_idx)
    train = [r for i, r in enumerate(rows) if i not in test_set]
    test = [r for i, r in enumerate(rows) if i in test_set]
    counts = {
        part: {cls: int(np.sum(classes[sel] == cls)) for cls in ("malicious", "benign", "other")}
        for part, sel in (
            ("train", np.array([i not in test_set for i in range(len(rows))], dtype=bool)),
            ("test", np.array([i in test_set for i in range(len(rows))], dtype=bool)),
        )
    }
    return {"header": header, "train": train, "test": test, "closing": closing, "counts": counts}


def write_part(out_dir: Path, filename: str, header, rows, closing) -> None:
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data" / filename, "w", encoding="utf-8") as fh:
        fh.writelines(header)
        fh.writelines(rows)
        fh.writelines(closing)


def make_split(src: Path, out: Path, train_ids, outside_ids, test_size: float, seed: int) -> dict:
    src, out = Path(src).expanduser().resolve(), Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(src), "test_size": test_size, "seed": seed, "train": {}, "outside": {}}
    for ds_id in train_ids:
        ds = find_dataset(src, ds_id)
        path = labeled_file(ds)
        parts = split_log(path, test_size, seed)
        rest = ds.name[len(ds_id) + 1:]
        for part in ("train", "test"):
            write_part(out / f"{ds_id}-{part}-{rest}", path.name, parts["header"], parts[part], parts["closing"])
        manifest["train"][ds_id] = {"source": str(path), **parts["counts"]}
    for ds_id in outside_ids:
        ds = find_dataset(src, ds_id)
        link = out / ds.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(ds.resolve(), target_is_directory=True)
        manifest["outside"][ds_id] = str(ds.resolve())
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="./datasets")
    ap.add_argument("--out", default="./datasets_fl_rlh")
    ap.add_argument("--train", nargs="+", default=["008", "010", "011", "012", "037"])
    ap.add_argument("--outside", nargs="*", default=["014", "016", "027"])
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1111)
    args = ap.parse_args(argv)
    manifest = make_split(Path(args.src), Path(args.out), args.train, args.outside, args.test_size, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
