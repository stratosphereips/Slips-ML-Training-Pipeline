#!/usr/bin/env python3

import argparse
import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


class UMAPService:
    def __init__(self, config_path, output_dir):
        self.cfg_reader = ConfigReader(config_path)
        self.cfg = self.cfg_reader.load()
        self.seed = int(self.cfg_reader.get_random_seed())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_png = None
        self.last_summary = None

        ds_params = self.cfg_reader.get_dataset_loader_params()
        self.loaders = find_and_load_datasets(
            root_dir=self.cfg.get("root"),
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
        self._init_color_map()

    def _init_color_map(self):
        cmap = plt.get_cmap("tab20")
        self.dataset_colors = {}
        for i, name in enumerate(self.dataset_keys):
            self.dataset_colors[name] = cmap(i % cmap.N)[:3]

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

    def _collect_samples(self, selected, sample_frac, rng):
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
            while True:
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
                X_parts.append(X_np)
                y_parts.append(y_np)
                ds_parts.append(
                    np.array([ds_name] * len(y_np), dtype="object")
                )

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

        fig, ax = plt.subplots(figsize=(11, 7))
        for ds_name in selected:
            base = self.dataset_colors.get(ds_name, (0.4, 0.4, 0.4))
            for label in label_order:
                mask = (datasets == ds_name) & (labels == label)
                if not np.any(mask):
                    continue
                color = base if label == "Benign" else _adjust_color(base, 0.6)
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
            Line2D([0], [0], marker=label_markers[label], color="k", linestyle="None", label=label)
            for label in label_order
        ]

        legend1 = ax.legend(
            handles=dataset_handles,
            title="Datasets",
            loc="upper right",
            fontsize=8,
            title_fontsize=9,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=label_handles,
            title="Labels",
            loc="lower right",
            fontsize=8,
            title_fontsize=9,
        )

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def compute_umap(self, selected, sample_frac):
        if not selected:
            selected = list(self.dataset_keys)
        selected = [s for s in selected if s in self.loaders]
        if not selected:
            raise RuntimeError("No valid datasets selected.")

        rng = np.random.default_rng(self.seed)
        X_all, y_all, ds_all = self._collect_samples(selected, sample_frac, rng)

        umap = UMAP(n_components=2, random_state=self.seed)
        embedding = umap.fit_transform(X_all)

        png = self._render_plot(embedding, y_all, ds_all, selected)

        counts = {}
        for name in selected:
            counts[name] = {
                "Benign": int(np.sum((ds_all == name) & (y_all == "Benign"))),
                "Malicious": int(np.sum((ds_all == name) & (y_all == "Malicious"))),
            }

        summary = {
            "total_points": int(len(y_all)),
            "counts": counts,
        }
        self.last_png = png
        self.last_summary = summary

        return png, summary

    def save_last(self, name=None):
        if self.last_png is None:
            raise RuntimeError("No UMAP image has been generated yet.")

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
        out_path.write_bytes(self.last_png)
        return str(out_path)


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            if not HTML_PATH.exists():
                self._send_text(500, "Missing umap_web.html")
                return
            html = HTML_PATH.read_text(encoding="utf-8")
            self._send_text(200, html, content_type="text/html")
            return
        if self.path == "/api/datasets":
            payload = {
                "datasets": self.server.app.dataset_keys,
                "output_dir": str(self.server.app.output_dir),
            }
            self._send_json(200, payload)
            return
        self._send_text(404, "Not found")

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_text(400, "Invalid JSON body")
                return
            name = payload.get("name")
            try:
                path = self.server.app.save_last(name=name)
            except Exception as exc:
                self._send_text(500, f"Save failed: {exc}")
                return
            self._send_json(200, {"path": path})
            return

        if self.path != "/api/umap":
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

        try:
            png, summary = self.server.app.compute_umap(selected, sample)
        except Exception as exc:
            self._send_text(500, f"UMAP failed: {exc}")
            return

        b64 = base64.b64encode(png).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        self._send_json(
            200,
            {"image": data_url, "summary": summary},
        )

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
