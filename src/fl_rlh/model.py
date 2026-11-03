# Author: Jan Svoboda
# functionality: The SLIPS federated module's default model (random_projection_mlp,
# "RLH") and its 18-feature extraction, copied so it can be trained offline.
# behavior:
#   - Copied from StratosphereLinuxIPS, branch fl_module_jan_rebased @ dc15aad7d:
#     modules/federated_network_module/federated_network_module.py
#     (shared_random_projection, shared_init_seed, build_with_init_seed,
#     SimpleFederatedNet, minibatch_indices, minibatch_rng, _get_feature_order,
#     _to_num, _extract_flow_features, _encode_appproto) and
#     slips_files/common/abstracts/ml_module_base.py (_infer_state, _encode_proto).
#   - Same frozen projection (seed 1111; equal to the module's committed
#     artifacts/random_projection.bin), same shared init of fc1/head, same features.
#   - adam_step is the module's manual Adam (decoupled weight decay 1e-4).
#   testing/test_fl_rlh.py checks all of this against fl_rlh_reference.npz,
#   exported from the SLIPS image by src/fl_rlh/make_reference.py.

import ipaddress
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

SEED = 1111
FIXED_INPUT_DIM = 18
HIDDEN1 = 256
HIDDEN2 = 16
WEIGHT_DECAY = 1e-4
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

FEATURE_ORDER = [
    "dur",
    "sport",
    "dport",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "proto",
    "appproto",
    "state",
    "saddr_num",
    "daddr_num",
    "dir_num",
    "total_bytes",
    "total_pkts",
    "avg_pkt_size",
    "throughput",
    "history_len",
]


# ---- model ---------------------------------------------------------------- #
def shared_random_projection(input_dim: int, hidden1: int, seed: int) -> torch.Tensor:
    """The frozen random projection every peer shares (dedicated generator,
    never the global torch RNG)."""
    generator = torch.Generator().manual_seed(seed)
    weights = torch.empty(input_dim, hidden1)
    nn.init.kaiming_normal_(
        weights, mode="fan_in", nonlinearity="relu", generator=generator
    )
    return weights


def shared_init_seed(seed: int) -> int:
    """Shared seed of the trainable layers (every peer starts from it)."""
    return (int(seed) ^ 0x51ED5EED) % (2**31)


def build_with_init_seed(init_seed: int, build):
    """Construct a model with its trainable layers initialised from
    `init_seed`, without disturbing the caller's global torch RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        return build()


class SimpleFederatedNet(nn.Module):
    """input(18) -> RandomProjection(256, frozen) -> /sqrt(256) -> fc1(256->16)
    + ReLU -> head(16->2). Layers are created in the module's order, so the
    same init seed gives the same fc1/head weights."""

    def __init__(self, seed: int = SEED, hidden1: int = HIDDEN1, hidden2: int = HIDDEN2):
        super().__init__()
        random_weights = shared_random_projection(FIXED_INPUT_DIM, hidden1, seed)
        self.random_projection = nn.Linear(FIXED_INPUT_DIM, hidden1, bias=False)
        self.random_projection.weight.data = random_weights.T
        self.random_projection.weight.requires_grad = False
        self.projection_scale = hidden1**0.5
        self.fc1 = nn.Linear(hidden1, hidden2, bias=True)
        self.head = nn.Linear(hidden2, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.random_projection(x)
        x = x / self.projection_scale
        x = self.fc1(x)
        x = self.relu(x)
        return self.head(x)


def build_model(seed: int = SEED) -> SimpleFederatedNet:
    """A fresh RLH model at the shared init (no warm start)."""
    return build_with_init_seed(shared_init_seed(seed), lambda: SimpleFederatedNet(seed))


# ---- optimizer -------------------------------------------------------------- #
def new_adam_state(model: nn.Module) -> dict:
    return {
        name: {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "step": 0}
        for name, p in model.named_parameters()
    }


def adam_step(model: nn.Module, state: dict, lr: float, wd: float = WEIGHT_DECAY):
    """One update of the module's manual Adam: decoupled weight decay, then
    bias-corrected Adam. Frozen parameters (no grad) are skipped."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            st = state[name]
            st["step"] += 1
            g = param.grad.clone()
            param.mul_(1.0 - lr * wd)
            m, v = st["m"], st["v"]
            m.mul_(BETA1).add_(g, alpha=1.0 - BETA1)
            v.mul_(BETA2).addcmul_(g, g, value=1.0 - BETA2)
            m_hat = m / (1.0 - BETA1 ** st["step"])
            v_hat = v / (1.0 - BETA2 ** st["step"])
            param.addcdiv_(m_hat, v_hat.sqrt().add_(EPS), value=-lr)
            param.grad.zero_()


# ---- mini-batches (module's class-stratified split) ------------------------- #
def minibatch_rng(seed: int, run_seed: Optional[int], window: int, finetune: bool) -> np.random.Generator:
    run = 0 if run_seed is None else int(run_seed) + 1
    return np.random.default_rng((int(seed), run, int(window), int(finetune), 0x6D62))


def minibatch_indices(y, batch_size: int, rng: np.random.Generator) -> list:
    """ceil(n / batch_size) near-equal class-stratified parts of at most
    batch_size rows; batch_size <= 0 (or >= n) keeps the full batch."""
    y = np.asarray(y)
    n = len(y)
    if batch_size <= 0 or n <= batch_size:
        return [np.arange(n)]
    order, rank = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        order.append(idx)
        rank.append((np.arange(len(idx)) + 0.5) / len(idx))
    order = np.concatenate(order)[np.argsort(np.concatenate(rank), kind="stable")]
    return np.array_split(order, -(-n // batch_size))


# ---- features ------------------------------------------------------------- #
def _to_num(value) -> float:
    if value is None or isinstance(value, (list, dict)):
        return 0.0
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if num != num else num


def _encode_proto(proto: str) -> float:
    proto_map = {"tcp": 0.0, "udp": 1.0, "icmp-ipv6": 3.0, "icmp": 2.0, "arp": 4.0}
    return proto_map.get(str(proto).strip().lower(), 0.0)


def _encode_appproto(appproto) -> float:
    if not isinstance(appproto, str):
        return 0.0
    appproto = appproto.strip().lower()
    proto_map = {
        "http": 0.0,
        "dns": 1.0,
        "ssl": 2.0,
        "ssh": 3.0,
        "smtp": 4.0,
        "ftp": 5.0,
        "pop3": 6.0,
        "imap": 7.0,
        "telnet": 8.0,
        "https": 9.0,
    }
    return proto_map.get(appproto, 10.0)


def _infer_state(state: str, spkts: float, dpkts: float) -> float:
    pkts = int(float(spkts or 0) + float(dpkts or 0))
    pre = state.split("_")[0]
    st = state.lower()
    if "new" in st or st == "established":
        return 1.0
    if "closed" in st or st == "not established":
        return 0.0
    if state in ("S0", "REJ", "RSTOS0", "RSTRH", "SH", "SHR"):
        return 0.0
    if state in ("S1", "SF", "S2", "S3", "RSTO", "RSTP", "OTH"):
        return 1.0
    if "S" in pre and "A" in pre:
        return 1.0
    if "PA" in pre:
        return 1.0
    if any(x in pre for x in ("ECO", "ECR", "URH", "URP")):
        return 1.0
    if "EST" in pre:
        return 1.0
    if "RST" in pre or "FIN" in pre:
        return 0.0 if pkts <= 3 else 1.0
    return 0.0


def extract_flow_features(flow: dict) -> Optional[list]:
    """The module's 18 features of a SLIPS flow dict, in FEATURE_ORDER."""
    try:
        dur = _to_num(flow.get("dur"))
        spkts = _to_num(flow.get("spkts"))
        dpkts = _to_num(flow.get("dpkts"))
        sbytes = _to_num(flow.get("sbytes"))
        dbytes = _to_num(flow.get("dbytes"))
        appproto = flow.get("appproto")
        saddr = flow.get("saddr")
        daddr = flow.get("daddr")
        history = flow.get("history")
        total_bytes = sbytes + dbytes
        values = {
            "dur": dur,
            "sport": _to_num(flow.get("sport")),
            "dport": _to_num(flow.get("dport")),
            "spkts": spkts,
            "dpkts": dpkts,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "proto": _encode_proto(str(flow.get("proto", ""))),
            "appproto": (
                _encode_appproto(str(appproto))
                if appproto is not None and appproto == appproto
                else 10.0
            ),
            "state": _infer_state(str(flow.get("state", "")), spkts, dpkts),
            "saddr_num": (int(ipaddress.ip_address(str(saddr))) % 1000000 if saddr else 0.0),
            "daddr_num": (int(ipaddress.ip_address(str(daddr))) % 1000000 if daddr else 0.0),
            "dir_num": 1.0 if str(flow.get("dir_", "->")) == "->" else 0.0,
            "total_bytes": total_bytes,
            "total_pkts": spkts + dpkts,
            "avg_pkt_size": sbytes / max(spkts, 1.0),
            "throughput": total_bytes / max(dur, 0.001),
            "history_len": float(len(str(history))) if history else 0.0,
        }
        features = [float(values[feat]) for feat in FEATURE_ORDER]
        return features if len(features) == FIXED_INPUT_DIM else None
    except Exception:
        return None
