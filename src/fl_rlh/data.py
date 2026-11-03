# Author: Jan Svoboda
# functionality: Zeek conn records -> the flow dicts SLIPS stores (input to the federated
# module's 18-feature extraction), and the module's 50/50 undersampling.
# behavior:
#   - Each Zeek record becomes the flow dict SLIPS's Zeek profiler would store
#     (slips_files/core/input_profilers/zeek.py, flows/zeek.py Conn): '-' / unset -> "",
#     dur float, bytes/pkts/ports int, raw conn_state, dir_ "->". The pipeline's
#     ConnToSlipsConverter is NOT used for these features: it turns state into 0/1 and
#     leaves dir_ empty, which would change two of the module's features.
#   - The pipeline's ZeekDataset hands mixers records already normalized by
#     ConnToSlipsConverter (SLIPS key names, state as 0/1, dir_ empty): slips_flow()
#     accepts both forms. The converter's 0/1 state comes from the same rule the module
#     applies to the raw conn_state (SLIPS's _infer_state), so it is passed on as a state
#     string that rule maps to the same value; dir_ becomes "->" as SLIPS stores it.
#   - Used by FeatureExtraction(feature_set="fl_rlh") and FLRLHClassifier.

from typing import Dict, Tuple

import numpy as np

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
        unset = val is None or val == "-" or (isinstance(val, float) and val != val)
        flow[slips_field] = "" if unset else val
    flow["dur"] = float(flow.get("dur", 0) or 0)
    for f in _INT_FIELDS:
        flow[f] = int(float(flow.get(f, 0) or 0))
    if flow["proto"] == "unknown_transport":
        flow["proto"] = ""
    flow["type_"] = "conn"
    flow["dir_"] = "->"
    return flow


_SLIPS_KEYS = ("starttime", "uid", "saddr", "sport", "daddr", "dport", "proto",
               "appproto", "dur", "sbytes", "dbytes", "history", "spkts", "dpkts")


def _unset(val) -> bool:
    return val is None or val == "-" or (isinstance(val, float) and val != val)


def normalized_to_slips_flow(record: Dict) -> Dict:
    """One ConnToSlipsConverter record (what the pipeline's mixers yield) -> the
    SLIPS conn flow dict, as zeek_to_slips_flow would give for its raw Zeek line."""
    flow = {k: ("" if _unset(record.get(k, "")) else record.get(k, "")) for k in _SLIPS_KEYS}
    flow["dur"] = float(flow["dur"] or 0)
    for f in _INT_FIELDS:
        flow[f] = int(float(flow[f] or 0))
    if flow["proto"] == "unknown_transport":
        flow["proto"] = ""
    state = record.get("state", "")
    if isinstance(state, (int, float)) and not isinstance(state, bool) and state == state:
        # already the SLIPS rule's output: a state string the same rule maps back to it
        state = "established" if float(state) == 1.0 else "not established"
    flow["state"] = "" if _unset(state) else state
    flow["type_"] = "conn"
    flow["dir_"] = "->"
    return flow


def slips_flow(record: Dict) -> Dict:
    """A raw Zeek conn record or a ConnToSlipsConverter record -> SLIPS conn flow."""
    if "id.orig_h" in record or "conn_state" in record:
        return zeek_to_slips_flow(record)
    return normalized_to_slips_flow(record)


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
