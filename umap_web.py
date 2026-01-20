#!/usr/bin/env python3

import argparse
import base64
import io
import json
import sys
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

# Ensure local src/ is importable as a package when running from this folder.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.conf_reader import ConfigReader
from src.dataset_wrapper import find_and_load_datasets
from src.features import FeatureExtraction
from src.preprocessing_wrapper import PreprocessingWrapper
from src.class_factory import get_transformer_class

try:
    from umap import UMAP
except Exception as exc:
    print("ERROR: umap-learn is required. Install with: pip install umap-learn")
    raise SystemExit(1) from exc

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except Exception as exc:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    raise SystemExit(1) from exc


HTML_PATH = ROOT / "umap_web.html"


def _normalize_sample_fraction(value):
    if value is None:
        return 1.0
    try:
        value = float(value)
    except Exception:
        return 1.0
    if value > 1.0:
        value = value / 100.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _as_numpy(X):
    try:
        return X.to_numpy()
    except Exception:
        return np.asarray(X)


def _adjust_color(rgb, factor):
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)


class JobStore:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.jobs_path = self.base_dir / "umap_jobs.json"
        self.lock = threading.Lock()
        self.jobs = {}
        self._load()

    def _load(self):
        if not self.jobs_path.exists():
            self.jobs = {}
            return
        try:
            data = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        except Exception:
            self.jobs = {}
            return
        if isinstance(data, list):
            self.jobs = {j.get("id"): j for j in data if isinstance(j, dict)}
        elif isinstance(data, dict):
            self.jobs = data
        else:
            self.jobs = {}

        for job in self.jobs.values():
            if job.get("status") in ("queued", "running"):
                job["status"] = "stale"
                job["error"] = "Server restarted while job was running."
        self._save()

    def _save(self):
        with self.lock:
            data = list(self.jobs.values())
            tmp = self.jobs_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.jobs_path)

    def create_job(self, params):
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "params": params,
        }
        self.jobs[job_id] = job
        self._save()
        return job_id

    def update_job(self, job_id, **fields):
        job = self.jobs.get(job_id)
        if not job:
            return
        job.update(fields)
        self.jobs[job_id] = job
        self._save()

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def list_jobs(self):
        jobs = list(self.jobs.values())
        jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
        return jobs


class UMAPService:
    def __init__(self, config_path, output_dir):
        self.cfg_reader = ConfigReader(config_path)
        self.cfg = self.cfg_reader.load()
        self.seed = int(self.cfg_reader.get_random_seed())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_png = None
        self.last_summary = None
        self.last_job_id = None
        self.jobs_dir = self.output_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.job_store = JobStore(self.output_dir)
        self.executor = ThreadPoolExecutor(max_workers=1)

        ds_params = self.cfg_reader.get_dataset_loader_params()
        root = self.cfg.get("root")
        if root is None:
            raise RuntimeError("Config missing 'root' path")
        root_path = Path(root)
        if not root_path.is_absolute():
            if self.cfg_reader.config_path:
                base = self.cfg_reader.config_path.resolve().parent
            else:
                base = ROOT
            root_path = (base / root_path).resolve()
        self.root_path = root_path
        self.loaders = find_and_load_datasets(
            root_dir=str(root_path),
            batch_size=int(ds_params.get("batch_size", 1000)),
            prefix_regex=ds_params.get("prefix_regex", r"^\d{3}"),
            data_subdir=ds_params.get("data_subdir", "data"),
            seed=self.seed,
            persist_cache_threshold=int(
                ds_params.get("persist_cache_threshold", 30000)
            ),
            cache_dir=ds_params.get("cache_dir"),
            labeled_filenames=ds_params.get("labeled_filenames"),
            file_encoding=ds_params.get("file_encoding", "utf-8"),
            file_errors=ds_params.get("file_errors", "ignore"),
            shuffle_per_epoch=bool(ds_params.get("shuffle_per_epoch", False)),
        )
        self.dataset_keys = sorted(self.loaders.keys())
        self.dataset_sizes = {
            name: int(len(loader)) for name, loader in self.loaders.items()
        }
        if not self.dataset_keys:
            # Fallback: list directories even if loaders failed (keeps UI usable).
            try:
                import re

                prefix_regex = ds_params.get("prefix_regex", r"^\d{3}")
                pattern = re.compile(prefix_regex)
                self.dataset_keys = sorted(
                    [
                        d.name
                        for d in root_path.iterdir()
                        if d.is_dir() and pattern.match(d.name)
                    ]
                )
                self.dataset_sizes = {name: 0 for name in self.dataset_keys}
            except Exception:
                self.dataset_keys = []
                self.dataset_sizes = {}
        self._init_color_map()

    def _init_color_map(self):
        self.dataset_index = {
            name: idx for idx, name in enumerate(self.dataset_keys)
        }
        self.cmap_benign = plt.get_cmap("Greens")
        self.cmap_malicious = plt.get_cmap("Reds")

        # Colors for dataset legend (use benign palette as the base)
        self.dataset_colors = {}
        total = max(1, len(self.dataset_keys) - 1)
        for name, idx in self.dataset_index.items():
            pos = idx / total if total else 0.0
            self.dataset_colors[name] = self.cmap_benign(0.35 + 0.55 * pos)[:3]

    def _label_color(self, label, dataset):
        total = max(1, len(self.dataset_keys) - 1)
        idx = self.dataset_index.get(dataset, 0)
        pos = idx / total if total else 0.0
        t = 0.35 + 0.55 * pos
        if label == "Malicious":
            return self.cmap_malicious(t)[:3]
        return self.cmap_benign(t)[:3]

    def _build_preprocessor(self):
        prep = PreprocessingWrapper(
            steps=[],
            experiment_name=self.cfg.get("experiment_name", "default"),
        )
        for step in self.cfg_reader.get_preprocessing_steps():
            cls = get_transformer_class(step["type"])
            prep.add_step(step.get("name"), cls(**(step.get("params") or {})))

        load_from = (self.cfg.get("preprocessing") or {}).get("load_from")
        if load_from:
            base = Path(load_from)
            if not base.is_absolute():
                if self.cfg_reader.config_path is None:
                    raise RuntimeError("Config path not resolved for load_from.")
                base = self.cfg_reader.config_path.parent / base
            prep.load(base_path=str(base))
            return prep, True
        return prep, False

    def _collect_samples(self, selected, sample_frac, rng, max_per_dataset=None):
        feature_extractor = FeatureExtraction(
            **self.cfg_reader.get_feature_extractor_params()
        )
        preprocessor, preprocessor_loaded = self._build_preprocessor()

        X_parts = []
        y_parts = []
        ds_parts = []

        for ds_name in selected:
            loader = self.loaders[ds_name]
            loader.reset_epoch(batch_size=loader.batch_size)
            collected = 0
            while True:
                if max_per_dataset is not None and collected >= max_per_dataset:
                    break
                batch = loader.next_batch()
                if batch is None:
                    break
                X, y = feature_extractor.process_batch(batch)
                if X is None or len(X) == 0:
                    continue
                X_np = _as_numpy(X)
                y_np = np.asarray(y)
                if sample_frac < 1.0:
                    mask = rng.random(X_np.shape[0]) < sample_frac
                    if not mask.any():
                        continue
                    X_np = X_np[mask]
                    y_np = y_np[mask]
                if max_per_dataset is not None:
                    remaining = max_per_dataset - collected
                    if remaining <= 0:
                        break
                    if X_np.shape[0] > remaining:
                        X_np = X_np[:remaining]
                        y_np = y_np[:remaining]
                X_parts.append(X_np)
                y_parts.append(y_np)
                ds_parts.append(
                    np.array([ds_name] * len(y_np), dtype="object")
                )
                collected += len(y_np)

        if not X_parts:
            raise RuntimeError("No samples collected for the selection.")

        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts)
        ds_all = np.concatenate(ds_parts)

        if preprocessor.get_steps():
            if not preprocessor_loaded:
                preprocessor.partial_fit(X_all)
            X_all = preprocessor.transform(X_all)

        return X_all, y_all, ds_all

    def _render_plot(self, embedding, labels, datasets, selected):
        label_markers = {"Benign": "o", "Malicious": "^"}
        label_order = ["Benign", "Malicious"]

        fig, ax = plt.subplots(figsize=(11, 9.5))
        # Reserve right margin so legends do not overlap plotted points.
        fig.subplots_adjust(right=0.74)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for ds_name in selected:
            for label in label_order:
                mask = (datasets == ds_name) & (labels == label)
                if not np.any(mask):
                    continue
                color = self._label_color(label, ds_name)
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[color],
                    marker=label_markers.get(label, "o"),
                    s=14,
                    alpha=0.75,
                    linewidths=0,
                )

        ax.set_title("UMAP by Dataset and Label")
        ax.set_xticks([])
        ax.set_yticks([])

        dataset_handles = [
            Patch(facecolor=self.dataset_colors[name], edgecolor="none", label=name)
            for name in selected
        ]
        label_handles = [
            Line2D(
                [0],
                [0],
                marker=label_markers[label],
                color="none",
                markerfacecolor=self._label_color(label, selected[0]),
                markeredgecolor="none",
                linestyle="None",
                label=label,
            )
            for label in label_order
        ]

        legend1 = ax.legend(
            handles=dataset_handles,
            title="Datasets",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            title_fontsize=9,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=label_handles,
            title="Labels",
            loc="upper left",
            bbox_to_anchor=(1.01, 0.62),
            borderaxespad=0.0,
            fontsize=8,
            title_fontsize=9,
        )

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _render_plotly(self, embedding, labels, datasets, selected):
        if not PLOTLY_AVAILABLE:
            return None

        label_symbols = {"Benign": "circle", "Malicious": "triangle-up"}
        label_order = ["Benign", "Malicious"]

        fig = go.Figure()
        for ds_name in selected:
            for label in label_order:
                mask = (datasets == ds_name) & (labels == label)
                if not np.any(mask):
                    continue
                color_rgb = self._label_color(label, ds_name)
                color = f"rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})"
                fig.add_trace(
                    go.Scattergl(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=color,
                            symbol=label_symbols.get(label, "circle"),
                            opacity=0.75,
                        ),
                        name=f"{ds_name} {label}",
                        customdata=np.column_stack(
                            (
                                np.full(np.sum(mask), ds_name),
                                np.full(np.sum(mask), label),
                            )
                        ),
                        hovertemplate="dataset: %{customdata[0]}<br>label: %{customdata[1]}<extra></extra>",
                    )
                )

        fig.update_layout(
            title="UMAP by Dataset and Label",
            showlegend=True,
            height=760,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(font=dict(size=9)),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        return pio.to_html(
            fig,
            include_plotlyjs="inline",
            full_html=True,
            config={"displaylogo": False, "scrollZoom": True},
        )

    def _parse_umap_params(self, params):
        defaults = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "spread": 1.0,
        }
        if not isinstance(params, dict):
            return defaults

        out = dict(defaults)
        if params.get("n_neighbors") is not None:
            try:
                nn = int(params.get("n_neighbors"))
                if nn >= 2:
                    out["n_neighbors"] = nn
            except Exception:
                pass
        if params.get("min_dist") is not None:
            try:
                md = float(params.get("min_dist"))
                if md >= 0.0:
                    out["min_dist"] = md
            except Exception:
                pass
        metric = params.get("metric")
        if isinstance(metric, str) and metric.strip():
            out["metric"] = metric.strip()
        if params.get("spread") is not None:
            try:
                sp = float(params.get("spread"))
                if sp > 0.0:
                    out["spread"] = sp
            except Exception:
                pass
        return out

    def compute_umap(self, selected, sample_frac, max_samples=None, umap_params=None):
        if not selected:
            selected = list(self.dataset_keys)
        selected = [s for s in selected if s in self.loaders]
        if not selected:
            raise RuntimeError("No valid datasets selected.")

        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        max_per_dataset = None
        if max_samples is not None:
            try:
                max_per_dataset = int(max_samples)
            except Exception:
                max_per_dataset = None
            if max_per_dataset is not None and max_per_dataset <= 0:
                max_per_dataset = None

        X_all, y_all, ds_all = self._collect_samples(
            selected, sample_frac, rng, max_per_dataset=max_per_dataset
        )

        params = self._parse_umap_params(umap_params or {})
        umap = UMAP(n_components=2, random_state=self.seed, **params)
        embedding = umap.fit_transform(X_all)

        png = self._render_plot(embedding, y_all, ds_all, selected)
        html = self._render_plotly(embedding, y_all, ds_all, selected)

        counts = {}
        for name in selected:
            counts[name] = {
                "Benign": int(np.sum((ds_all == name) & (y_all == "Benign"))),
                "Malicious": int(np.sum((ds_all == name) & (y_all == "Malicious"))),
            }

        elapsed = time.perf_counter() - start
        summary = {
            "total_points": int(len(y_all)),
            "duration_sec": float(elapsed),
            "sample_frac": float(sample_frac),
            "max_samples": max_per_dataset,
            "umap_params": params,
            "datasets": list(selected),
            "counts": counts,
        }
        self.last_png = png
        self.last_summary = summary

        return png, summary, html

    def start_job(self, params):
        job_id = self.job_store.create_job(params)
        self.executor.submit(self._run_job, job_id, params)
        return job_id

    def _run_job(self, job_id, params):
        try:
            self.job_store.update_job(
                job_id,
                status="running",
                started_at=time.time(),
            )
            selected = params.get("datasets") or []
            sample = params.get("sample")
            max_samples = params.get("max_samples")
            umap_params = params.get("umap_params") or {}

            png, summary, html = self.compute_umap(
                selected,
                sample,
                max_samples=max_samples,
                umap_params=umap_params,
            )

            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            png_path = job_dir / "umap.png"
            png_path.write_bytes(png)
            html_path = None
            if html:
                html_path = job_dir / "umap.html"
                html_path.write_text(html, encoding="utf-8")

            self.last_job_id = job_id
            self.job_store.update_job(
                job_id,
                status="done",
                finished_at=time.time(),
                summary=summary,
                result_png=str(png_path),
                result_html=str(html_path) if html_path else None,
            )
        except Exception as exc:
            self.job_store.update_job(
                job_id,
                status="error",
                finished_at=time.time(),
                error=str(exc),
            )

    def save_last(self, name=None):
        if not self.last_job_id:
            raise RuntimeError("No UMAP job has completed yet.")
        return self.save_job(self.last_job_id, name=name)

    def save_job(self, job_id, name=None):
        job = self.job_store.get_job(job_id)
        if not job or job.get("status") != "done":
            raise RuntimeError("Requested job is not complete.")
        png_path = job.get("result_png")
        if not png_path:
            raise RuntimeError("No image found for this job.")
        png_bytes = Path(png_path).read_bytes()
        if job.get("summary"):
            try:
                png_bytes = self._annotate_png(png_bytes, job.get("summary"))
            except Exception:
                # fall back to raw image on annotation failure
                png_bytes = png_bytes

        safe_name = None
        if name:
            cleaned = "".join(c for c in name if c.isalnum() or c in ("_", "-", "."))
            cleaned = cleaned.strip(".")
            if cleaned:
                safe_name = cleaned

        if safe_name is None:
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = f"umap_{ts}.png"

        if not safe_name.lower().endswith(".png"):
            safe_name += ".png"

        out_path = self.output_dir / safe_name
        out_path.write_bytes(png_bytes)
        html_out = None
        html_path = job.get("result_html")
        if html_path and Path(html_path).exists():
            html_out = out_path.with_suffix(".html")
            html_out.write_text(
                Path(html_path).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        return {"png": str(out_path), "html": str(html_out) if html_out else None}

    def _annotate_png(self, png_bytes, summary):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(io.BytesIO(png_bytes))
        if img is None:
            return png_bytes

        h, w = img.shape[:2]
        panel_w = 320
        dpi = 100
        fig_w = (w + panel_w) / dpi
        fig_h = h / dpi

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = fig.add_gridspec(
            1, 2, width_ratios=[w, panel_w], wspace=0.02
        )
        ax_img = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])

        ax_img.imshow(img)
        ax_img.axis("off")

        ax_txt.axis("off")
        ax_txt.set_facecolor("white")

        params = summary.get("umap_params", {})
        sample_frac = summary.get("sample_frac", 0.0)
        sample_pct = f"{sample_frac * 100:.2f}%"
        max_samples = summary.get("max_samples")
        max_label = "none" if max_samples is None else str(max_samples)
        datasets = summary.get("datasets") or []

        lines = [
            "UMAP settings",
            f"sample: {sample_pct}",
            f"max per dataset: {max_label}",
            f"n_neighbors: {params.get('n_neighbors', '-')}",
            f"min_dist: {params.get('min_dist', '-')}",
            f"metric: {params.get('metric', '-')}",
            f"spread: {params.get('spread', '-')}",
            "",
            "datasets:",
        ]
        if datasets:
            for name in datasets:
                lines.append(f"- {name}")
        else:
            lines.append("- none")

        ax_txt.text(
            0.02,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10,
            color="#1f1b16",
        )

        fig.patch.set_facecolor("white")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf.read()


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            if not HTML_PATH.exists():
                self._send_text(500, "Missing umap_web.html")
                return
            html = HTML_PATH.read_text(encoding="utf-8")
            bootstrap = {
                "datasets": [
                    {"name": name, "count": self.server.app.dataset_sizes.get(name, 0)}
                    for name in self.server.app.dataset_keys
                ],
                "output_dir": str(self.server.app.output_dir),
                "root": str(self.server.app.root_path),
            }
            marker = "window.__DATASETS_BOOTSTRAP__ = null;"
            if marker in html:
                html = html.replace(
                    marker,
                    f"window.__DATASETS_BOOTSTRAP__ = {json.dumps(bootstrap)};",
                )
            self._send_text(200, html, content_type="text/html")
            return
        if parsed.path == "/api/datasets":
            payload = {
                "datasets": [
                    {"name": name, "count": self.server.app.dataset_sizes.get(name, 0)}
                    for name in self.server.app.dataset_keys
                ],
                "output_dir": str(self.server.app.output_dir),
                "root": str(self.server.app.root_path),
            }
            self._send_json(200, payload)
            return
        if parsed.path == "/api/jobs":
            payload = {"jobs": self.server.app.job_store.list_jobs()}
            self._send_json(200, payload)
            return
        if parsed.path.startswith("/api/job/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 3:
                job_id = parts[2]
                if len(parts) == 3:
                    job = self.server.app.job_store.get_job(job_id)
                    if not job:
                        self._send_text(404, "Job not found")
                        return
                    self._send_json(200, job)
                    return
                if len(parts) == 4 and parts[3] == "result":
                    job = self.server.app.job_store.get_job(job_id)
                    if not job:
                        self._send_text(404, "Job not found")
                        return
                    if job.get("status") != "done":
                        self._send_text(400, "Job not finished")
                        return
                    png_path = job.get("result_png")
                    html_path = job.get("result_html")
                    data_url = None
                    if png_path and Path(png_path).exists():
                        png_bytes = Path(png_path).read_bytes()
                        b64 = base64.b64encode(png_bytes).decode("ascii")
                        data_url = f"data:image/png;base64,{b64}"
                    html = None
                    if html_path and Path(html_path).exists():
                        html = Path(html_path).read_text(encoding="utf-8")
                    self._send_json(
                        200,
                        {"image": data_url, "summary": job.get("summary"), "html": html},
                    )
                    return
        self._send_text(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_text(400, "Invalid JSON body")
                return
            name = payload.get("name")
            job_id = payload.get("job_id")
            try:
                if job_id:
                    path = self.server.app.save_job(job_id, name=name)
                else:
                    path = self.server.app.save_last(name=name)
            except Exception as exc:
                self._send_text(500, f"Save failed: {exc}")
                return
            if isinstance(path, dict):
                self._send_json(200, {"path": path.get("png"), "html": path.get("html")})
            else:
                self._send_json(200, {"path": path})
            return

        if parsed.path != "/api/umap":
            self._send_text(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_text(400, "Invalid JSON body")
            return

        selected = payload.get("datasets") or []
        sample = _normalize_sample_fraction(payload.get("sample"))
        if sample <= 0.0:
            self._send_text(400, "Sample fraction must be > 0")
            return

        max_samples = payload.get("max_samples")
        umap_params = payload.get("umap_params")
        job_params = {
            "datasets": selected,
            "sample": sample,
            "max_samples": max_samples,
            "umap_params": umap_params,
        }
        job_id = self.server.app.start_job(job_params)
        self._send_json(202, {"job_id": job_id})

    def log_message(self, fmt, *args):
        return

    def _send_text(self, code, text, content_type="text/plain"):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Local UMAP web viewer")
    parser.add_argument(
        "config",
        nargs="?",
        default="./default_config.yaml",
        help="Path to config file (default: ./default_config.yaml)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--output-dir",
        default="./umap_exports",
        help="Folder to save UMAP images",
    )
    args = parser.parse_args()

    app = UMAPService(args.config, args.output_dir)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    server.app = app
    print(f"UMAP web server running on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
