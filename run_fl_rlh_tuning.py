#!/usr/bin/env python3
# Author: Jan Svoboda
# functionality: Offline learning-parameter study of the SLIPS federated module's
# RLH model on labelled captured traffic: fixed stratified 10% test split, k-fold
# CV over a grid (lr, lr schedule, mini-batch size, step budget) on the other 90%,
# then the best and the reference configurations retrained on all 90% and tested once.
# behavior:
#   - Writes into one output dir: config_used.yaml, data_summary.json,
#     cv_runs.jsonl (one line per config x fold, written as it finishes; a
#     re-run with the same --out skips what is done), cv_summary.csv,
#     test_results.csv, test_curves.jsonl and plots/*.png.
#   - Every run starts from the module's shared init (no warm start).
# usage:
#   python3 run_fl_rlh_tuning.py --config configs/fl_rlh_tuning.yaml [--out DIR]
#       [--workers N] [--threads T] [--stage cv|final|all] [--smoke]

import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
import os
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.fl_rlh.data import cv_folds, holdout_split, load_features  # noqa: E402
from src.fl_rlh.train import SCHEDULERS, prepare, train_eval  # noqa: E402

METRICS = ("mcc", "f1", "precision", "recall", "fpr", "accuracy", "val_loss", "train_loss")
_DATA = {}  # filled before the worker pool forks


# ---- config ---------------------------------------------------------------- #
def load_config(path: str, smoke: bool) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if smoke:
        cfg = copy.deepcopy(cfg)
        cfg["datasets"] = cfg["datasets"][:2]
        cfg["max_flows_per_dataset"] = 1500
        cfg["cv_folds"] = 2
        cfg["search"] = {"lr": [0.03], "scheduler": ["none", "cosine"], "batch_size": [0, 100], "steps": [15, 50]}
        cfg["final"]["top_k"] = 1
    for s in cfg["search"]["scheduler"]:
        if s not in SCHEDULERS:
            raise ValueError(f"unknown scheduler {s!r}; choose from {SCHEDULERS}")
    return cfg


def config_id(c: dict) -> str:
    b = "full" if not c["batch_size"] else str(c["batch_size"])
    return f"lr{c['lr']:g}-{c['scheduler']}-b{b}-s{c['steps']}"


def grid(search: dict) -> list:
    out = []
    for lr in search["lr"]:
        for sched in search["scheduler"]:
            for bs in search["batch_size"]:
                for steps in search["steps"]:
                    out.append({"lr": float(lr), "scheduler": sched, "batch_size": int(bs), "steps": int(steps)})
    return out


def tasks_for(configs: list) -> list:
    """Training jobs: a constant-lr run is the prefix of the longest one with the
    same lr and batch size, so those train once at the largest budget and are
    read off at each budget's checkpoint."""
    jobs = {}
    for c in configs:
        if c["scheduler"] == "none":
            key = (c["lr"], "none", c["batch_size"])
            job = jobs.setdefault(key, {**c, "budgets": []})
            job["steps"] = max(job["steps"], c["steps"])
            job["budgets"].append(c["steps"])
        else:
            jobs[(c["lr"], c["scheduler"], c["batch_size"], c["steps"])] = {**c, "budgets": [c["steps"]]}
    return list(jobs.values())


# ---- workers ----------------------------------------------------------------- #
def _worker_init(threads: int):
    import torch

    torch.set_num_threads(threads)


def _run(job_fold):
    job, fold = job_fold
    d = _DATA
    cfg = d["cfg"]
    if fold >= 0:
        tr_pos, ev_pos = d["folds"][fold]
        Xtr_raw, ytr_raw = d["Xtrain"][tr_pos], d["ytrain"][tr_pos]
        Xev_raw, yev = d["Xtrain"][ev_pos], d["ytrain"][ev_pos]
    else:  # final: all 90% -> test
        Xtr_raw, ytr_raw, Xev_raw, yev = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]
    pre = cfg["preprocessing"]
    rng = np.random.default_rng((int(cfg["seed"]), fold + 1))
    Xtr, ytr, Xev, _ = prepare(
        Xtr_raw, ytr_raw, Xev_raw,
        undersample_train=bool(pre.get("undersample", True)),
        scaler_fit_on=pre.get("scaler_fit_on", "train_all"),
        rng=rng,
    )
    t0 = time.time()
    curve = train_eval(
        Xtr, ytr, Xev, yev,
        lr=job["lr"], scheduler=job["scheduler"], batch_size=job["batch_size"], steps=job["steps"],
        checkpoints=sorted(set(cfg["checkpoints"]) | set(job["budgets"])),
        seed=int(cfg["seed"]), fold=fold,
        scheduler_params=cfg.get("scheduler_params", {}).get(job["scheduler"]),
        class_weighting=bool(pre.get("class_weighting", False)),
    )
    return job, fold, curve, time.time() - t0, len(ytr)


def run_jobs(job_folds: list, workers: int, threads: int):
    if workers <= 1:
        _worker_init(threads)
        for jf in job_folds:
            yield _run(jf)
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(threads,)) as pool:
        yield from pool.imap_unordered(_run, job_folds)


def rows_of(job: dict, fold: int, curve: list, seconds: float, n_train: int) -> list:
    """One result row per budget of the job (constant-lr jobs carry several)."""
    out = []
    for b in sorted(job["budgets"]):
        c = {k: job[k] for k in ("lr", "scheduler", "batch_size")}
        c["steps"] = b
        part = [p for p in curve if p["step"] <= b]
        out.append({"config_id": config_id(c), **c, "fold": fold, "n_train": n_train,
                    "seconds": round(seconds, 2), "final": part[-1], "curve": part})
    return out


def append_jsonl(path: Path, rows: list):
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line cut by a crash; its job is redone
    return out


# ---- summaries ------------------------------------------------------------- #
def summarize(rows: list, checkpoints: list) -> list:
    by = {}
    for r in rows:
        by.setdefault(r["config_id"], []).append(r)
    out = []
    for cid, rs in by.items():
        s = {k: rs[0][k] for k in ("config_id", "lr", "scheduler", "batch_size", "steps")}
        s["folds"] = len(rs)
        for m in METRICS:
            vals = [r["final"][m] for r in rs]
            s[f"{m}_mean"] = st.mean(vals)
            s[f"{m}_sd"] = st.stdev(vals) if len(vals) > 1 else 0.0
        s["epochs"] = rs[0]["final"]["epoch"]
        steps_seen = sorted({p["step"] for r in rs for p in r["curve"]})
        mean_mcc = {}
        for stp in steps_seen:
            v = [p["mcc"] for r in rs for p in r["curve"] if p["step"] == stp]
            if len(v) == len(rs):
                mean_mcc[stp] = st.mean(v)
        final = s["mcc_mean"]
        s["steps_to_95pct_mcc"] = next(
            (stp for stp in sorted(mean_mcc) if final > 0 and mean_mcc[stp] >= 0.95 * final), None
        )
        for stp in checkpoints:
            s[f"mcc@{stp}"] = mean_mcc.get(stp)
        s["seconds_mean"] = st.mean(r["seconds"] for r in rs)
        out.append(s)
    out.sort(key=lambda s: (-s["mcc_mean"], s["steps"]))
    return out


def write_csv(path: Path, rows: list):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


# ---- plots ----------------------------------------------------------------- #
def _mean_curve(rows: list, key: str):
    steps = sorted({p["step"] for r in rows for p in r["curve"]})
    xs, ys = [], []
    for stp in steps:
        v = [p[key] for r in rows for p in r["curve"] if p["step"] == stp]
        if len(v) == len(rows):
            xs.append(max(stp, 0.8))
            ys.append(st.mean(v))
    return xs, ys


def make_plots(out: Path, rows: list, summary: list, test_curves: list, cfg: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = out / "plots"
    pdir.mkdir(exist_ok=True)
    smax = max(cfg["search"]["steps"])
    by = {}
    for r in rows:
        by.setdefault(r["config_id"], []).append(r)

    def sel(lr=None, sched=None, bs=None, steps=smax):
        return [rs for rs in by.values()
                if (lr is None or rs[0]["lr"] == lr) and (sched is None or rs[0]["scheduler"] == sched)
                and (bs is None or rs[0]["batch_size"] == bs) and rs[0]["steps"] == steps]

    def panel(groups, label, title, fname, extra_lr=False):
        n = 3 if extra_lr else 2
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), layout="constrained")
        for rs in groups:
            for ax, key in zip(axes, ("mcc", "val_loss", "lr")[:n]):
                xs, ys = _mean_curve(rs, key)
                ax.plot(xs, ys, marker="o", markersize=3, label=label(rs[0]))
        for ax, name in zip(axes, ("validation MCC (CV mean)", "validation loss (CV mean)", "learning rate")[:n]):
            ax.set_xscale("log")
            ax.set_xlabel("optimizer step")
            ax.set_title(name, fontsize=10)
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(title, fontsize=11)
        fig.savefig(pdir / fname, dpi=120)
        plt.close(fig)

    if not summary:
        return
    panel(sel(sched="none", bs=0), lambda c: f"lr {c['lr']:g}",
          f"constant lr, full batch (1 step = 1 epoch), {smax} steps", "convergence_lr.png")
    best = summary[0]
    best_lr = best["lr"]
    panel(sel(lr=best_lr, sched="none"), lambda c: "full batch" if not c["batch_size"] else f"batch {c['batch_size']}",
          f"constant lr {best_lr:g}: mini-batch size, {smax} steps", "convergence_batch.png")
    panel(sel(lr=best_lr, bs=0), lambda c: c["scheduler"],
          f"lr {best_lr:g}, full batch: schedules over {smax} steps", "convergence_schedulers.png", extra_lr=True)

    # heatmap of mean CV MCC at the largest budget
    lrs = sorted({s["lr"] for s in summary})
    cols = [(sc, bs) for sc in cfg["search"]["scheduler"] for bs in cfg["search"]["batch_size"]]
    grid_v = np.full((len(lrs), len(cols)), np.nan)
    for s in summary:
        if s["steps"] == smax:
            grid_v[lrs.index(s["lr"]), cols.index((s["scheduler"], s["batch_size"]))] = s["mcc_mean"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cols), 1.2 + 0.6 * len(lrs)), layout="constrained")
    im = ax.imshow(grid_v, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)), [f"{sc}\n{'full' if not bs else bs}" for sc, bs in cols], fontsize=7)
    ax.set_yticks(range(len(lrs)), [f"lr {v:g}" for v in lrs])
    for i in range(len(lrs)):
        for j in range(len(cols)):
            if not math.isnan(grid_v[i, j]):
                ax.text(j, i, f"{grid_v[i, j]:.3f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax, label="mean CV MCC")
    ax.set_title(f"mean CV MCC after {smax} steps", fontsize=10)
    fig.savefig(pdir / "cv_mcc_heatmap.png", dpi=120)
    plt.close(fig)

    if test_curves:
        fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
        for tc in test_curves:
            xs = [max(p["step"], 0.8) for p in tc["curve"]]
            ax.plot(xs, [p["mcc"] for p in tc["curve"]], marker="o", markersize=3,
                    label=f"{tc['config_id']} ({tc['role']})")
        ax.set_xscale("log")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("test MCC")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title("retrained on all 90%, scored on the 10% hold-out", fontsize=10)
        fig.savefig(pdir / "test_curves.png", dpi=120)
        plt.close(fig)


# ---- main ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "fl_rlh_tuning.yaml"))
    ap.add_argument("--out", help="output dir; an existing one is resumed")
    ap.add_argument("--root", help="override the dataset root")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--threads", type=int, help="torch threads per worker")
    ap.add_argument("--stage", choices=("cv", "final", "all"), default="all")
    ap.add_argument("--smoke", action="store_true", help="tiny grid, 2 folds, 1500 flows per dataset")
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config, args.smoke)
    if args.root:
        cfg["root"] = args.root
    workers = args.workers or int(cfg["runtime"]["workers"])
    threads = args.threads or int(cfg["runtime"]["threads_per_worker"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "experiments" / "fl_rlh_tuning" / (stamp + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)  # noqa: E731
    log(f"output: {out}")

    seed = int(cfg["seed"])
    X, y, src, summary = load_features(cfg["root"], cfg["datasets"], cfg.get("cache_dir"),
                                       cfg.get("max_flows_per_dataset"), seed)
    tr, te = holdout_split(y, float(cfg["test_size"]), seed)
    folds = cv_folds(y[tr], int(cfg["cv_folds"]), seed)
    summary["split"] = {
        "train": {"flows": len(tr), "malicious": int(y[tr].sum())},
        "test": {"flows": len(te), "malicious": int(y[te].sum())},
        "cv_folds": len(folds),
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    import torch

    summary["environment"] = {"commit": commit, "torch": torch.__version__,
                              "workers": workers, "threads_per_worker": threads}
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"data: {summary['total']} | train {summary['split']['train']} | test {summary['split']['test']}")

    _DATA.update(cfg=cfg, Xtrain=X[tr], ytrain=y[tr], Xtest=X[te], ytest=y[te], folds=folds)
    configs = grid(cfg["search"])
    cv_path = out / "cv_runs.jsonl"

    if args.stage in ("cv", "all"):
        done = {(r["config_id"], r["fold"]) for r in read_jsonl(cv_path)}
        todo = []
        for job in tasks_for(configs):
            for f in range(len(folds)):
                ids = [config_id({**job, "steps": b}) for b in job["budgets"]]
                if any((i, f) not in done for i in ids):
                    todo.append((job, f))
        total = len(todo)
        log(f"CV: {len(configs)} configs x {len(folds)} folds -> {total} training jobs to run"
            f" ({len(done)} results already there); {workers} workers x {threads} threads")
        t0 = time.time()
        for n, (job, fold, curve, secs, n_train) in enumerate(run_jobs(todo, workers, threads), 1):
            rows = [r for r in rows_of(job, fold, curve, secs, n_train)
                    if (r["config_id"], r["fold"]) not in done]
            append_jsonl(cv_path, rows)
            eta = (time.time() - t0) / n * (total - n)
            best = curve[-1]
            log(f"[cv {n}/{total}] {config_id(job)} fold {fold}: MCC {best['mcc']:.4f} "
                f"val loss {best['val_loss']:.4f} ({secs:.1f}s, ETA {eta / 60:.1f} min)")

    rows = read_jsonl(cv_path)
    summ = summarize(rows, cfg["checkpoints"])
    write_csv(out / "cv_summary.csv", summ)
    if summ:
        log("best CV configs: " + "; ".join(f"{s['config_id']} MCC {s['mcc_mean']:.4f}±{s['mcc_sd']:.4f}"
                                             for s in summ[:5]))

    test_curves = read_jsonl(out / "test_curves.jsonl")
    if args.stage in ("final", "all") and summ:
        picks = [(s, "cv-best") for s in summ[: int(cfg["final"]["top_k"])]]
        by_id = {s["config_id"]: s for s in summ}
        for rc in cfg["final"].get("reference_configs", []):
            rc = {"lr": float(rc["lr"]), "scheduler": rc["scheduler"],
                  "batch_size": int(rc["batch_size"]), "steps": int(rc["steps"])}
            cid = config_id(rc)
            if cid not in [p[0]["config_id"] for p in picks]:
                picks.append((by_id.get(cid, {**rc, "config_id": cid}), "reference"))
        done = {t["config_id"] for t in test_curves}
        results = []
        for s, role in picks:
            c = {k: s[k] for k in ("lr", "scheduler", "batch_size", "steps")}
            if s["config_id"] not in done:
                job = {**c, "budgets": [c["steps"]]}
                _job, _f, curve, secs, n_train = next(run_jobs([(job, -1)], 1, threads))
                tc = {"config_id": s["config_id"], "role": role, **c, "n_train": n_train,
                      "seconds": round(secs, 2), "final": curve[-1], "curve": curve}
                append_jsonl(out / "test_curves.jsonl", [tc])
                test_curves.append(tc)
                log(f"[test] {s['config_id']} ({role}): MCC {curve[-1]['mcc']:.4f} F1 {curve[-1]['f1']:.4f} "
                    f"FPR {curve[-1]['fpr']:.4f}")
        for tc in test_curves:
            s = by_id.get(tc["config_id"], {})
            results.append({"config_id": tc["config_id"], "role": tc["role"],
                            "cv_mcc_mean": s.get("mcc_mean"), "cv_mcc_sd": s.get("mcc_sd"),
                            **{f"test_{m}": tc["final"][m] for m in METRICS},
                            "test_tp": tc["final"]["tp"], "test_fp": tc["final"]["fp"],
                            "test_tn": tc["final"]["tn"], "test_fn": tc["final"]["fn"]})
        write_csv(out / "test_results.csv", results)

    make_plots(out, rows, summ, test_curves, cfg)
    log(f"done: {out}")


if __name__ == "__main__":
    main()
