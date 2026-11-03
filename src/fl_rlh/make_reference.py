# Author: Jan Svoboda
# functionality: Export the SLIPS federated module's own projection, shared init and
# features into fl_rlh_reference.npz, so testing/test_fl_rlh.py can check the copy in
# src/fl_rlh/model.py; optionally check the Zeek -> SLIPS flow mapping on a SLIPS run.
# usage (inside the SLIPS image, SLIPS repo at /src, this repo at /pipe):
#   docker run --rm --network none -u $(id -u):$(id -g) --group-add 136 -e HOME=/tmp \
#     -v ~/StratosphereLinuxIPS:/src:ro -v <this repo>:/pipe \
#     -v ~/pipeline_ml_training_for_SLIPS/datasets/011-zeek-mixed/data:/ds:ro \
#     [-v <slips output dir with zeek_files/ and databases/>:/lab:ro] \
#     -w /src --entrypoint python3 federation_network-slips:latest -I \
#     /pipe/src/fl_rlh/make_reference.py --conn /ds/conn.log.labeled [--lab /lab]

import argparse
import json
import sqlite3
import subprocess
import sys

sys.path.insert(0, "/src")
sys.path.insert(0, "/pipe")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from modules.federated_network_module import federated_network_module as fnm  # noqa: E402
from src.fl_rlh import model as copy_model  # noqa: E402
from src.fl_rlh.data import zeek_to_slips_flow  # noqa: E402

OUT = "/pipe/src/fl_rlh/fl_rlh_reference.npz"
RP_ARTIFACT = "/src/modules/federated_network_module/artifacts/random_projection.bin"


def module_instance():
    m = object.__new__(fnm.FederatedNetworkModule)
    m.print = lambda *a, **k: None
    return m


def read_tsv(path, limit):
    headers, out = None, []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("#fields"):
                headers = line.rstrip("\n").split("\t")[1:]
            elif line.startswith("#") or not line.strip():
                continue
            else:
                parts = line.rstrip("\n").split("\t")
                out.append({h: (None if v in ("-", "") else v) for h, v in zip(headers, parts)})
                if len(out) >= limit:
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", required=True, help="a labelled Zeek conn log (TSV)")
    ap.add_argument("--lab", help="a SLIPS output dir (zeek_files/conn.log + databases/flows.sqlite)")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    seed = fnm.SimpleFederatedNet(18).seed  # the module's default seed
    proj = fnm.shared_random_projection(18, 256, seed)
    artifact = torch.load(RP_ARTIFACT, weights_only=True)
    print(f"seed {seed}; generator projection == committed artifact: {torch.equal(proj, artifact)}")

    net = fnm.build_with_init_seed(fnm.shared_init_seed(seed), lambda: fnm.SimpleFederatedNet(18, seed=seed))
    mine = copy_model.build_model(seed)
    checks = {
        "projection": torch.equal(net.random_projection.weight, mine.random_projection.weight),
        "fc1": torch.equal(net.fc1.weight, mine.fc1.weight) and torch.equal(net.fc1.bias, mine.fc1.bias),
        "head": torch.equal(net.head.weight, mine.head.weight) and torch.equal(net.head.bias, mine.head.bias),
    }

    m = module_instance()
    assert m._get_feature_order() == copy_model.FEATURE_ORDER
    flows = [zeek_to_slips_flow(r) for r in read_tsv(args.conn, args.n)]
    feats_module = np.array([m._extract_flow_features(f) for f in flows], dtype=np.float64)
    feats_copy = np.array([copy_model.extract_flow_features(f) for f in flows], dtype=np.float64)
    checks["features"] = bool(np.array_equal(feats_module, feats_copy))
    print("copy == module:", checks)

    if args.lab:
        # the mapping: our flow dict from SLIPS's own Zeek conn.log line vs the flow SLIPS stored
        con = sqlite3.connect(f"{args.lab}/databases/flows.sqlite")
        stored = {uid: json.loads(raw) for uid, raw in con.execute("select uid, flow from flows")}
        same = diff = 0
        examples = []
        with open(f"{args.lab}/zeek_files/conn.log") as fh:
            for line in fh:
                if not line.startswith("{"):
                    continue
                rec = json.loads(line)
                f = stored.get(rec.get("uid"))
                if f is None:
                    continue
                a = m._extract_flow_features(f)
                b = m._extract_flow_features(zeek_to_slips_flow(rec))
                if a == b:
                    same += 1
                else:
                    diff += 1
                    if len(examples) < 3:
                        examples.append([n for n, x, y in zip(copy_model.FEATURE_ORDER, a, b) if x != y])
        print(f"mapping vs SLIPS-stored flows: {same} identical, {diff} different {examples}")
        checks["mapping_identical"] = same
        checks["mapping_different"] = diff

    try:
        commit = subprocess.run(["git", "-C", "/src", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    commit = commit or "a0b69abc6"
    np.savez_compressed(
        OUT,
        projection=artifact.numpy(),
        fc1_weight=net.fc1.weight.detach().numpy(),
        fc1_bias=net.fc1.bias.detach().numpy(),
        head_weight=net.head.weight.detach().numpy(),
        head_bias=net.head.bias.detach().numpy(),
        flows=np.array([json.dumps(f) for f in flows]),
        features=feats_module,
        slips_commit=np.array(commit),
        torch_version=np.array(torch.__version__),
    )
    print(f"wrote {OUT} (SLIPS {commit}, torch {torch.__version__})")
    if not all(v for k, v in checks.items() if not k.startswith("mapping")):
        sys.exit(1)


if __name__ == "__main__":
    main()
