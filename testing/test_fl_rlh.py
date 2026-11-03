# Tests for the offline RLH study (src/fl_rlh): the copied model and features match
# the SLIPS federated module (fl_rlh_reference.npz, made by src/fl_rlh/make_reference.py
# inside the SLIPS image), and the split / undersampling / scaler / schedules behave.

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.fl_rlh import model as M
from src.fl_rlh.data import cv_folds, holdout_split, undersample, zeek_to_slips_flow
from src.fl_rlh.train import lr_at, prepare, train_eval

REF = np.load(Path(M.__file__).with_name("fl_rlh_reference.npz"))
DATASET_011 = Path.home() / "pipeline_ml_training_for_SLIPS" / "datasets" / "011-zeek-mixed" / "data"


def test_projection_and_shared_init_match_the_slips_module():
    net = M.build_model(M.SEED)
    assert np.array_equal(net.random_projection.weight.detach().numpy().T, REF["projection"])
    assert np.array_equal(net.fc1.weight.detach().numpy(), REF["fc1_weight"])
    assert np.array_equal(net.fc1.bias.detach().numpy(), REF["fc1_bias"])
    assert np.array_equal(net.head.weight.detach().numpy(), REF["head_weight"])
    assert np.array_equal(net.head.bias.detach().numpy(), REF["head_bias"])
    assert not net.random_projection.weight.requires_grad


def test_every_run_starts_from_the_same_init():
    a, b = M.build_model(), M.build_model()
    torch.manual_seed(0)  # the global RNG does not matter
    c = M.build_model()
    for x, y in ((a, b), (a, c)):
        assert all(torch.equal(p, q) for p, q in zip(x.state_dict().values(), y.state_dict().values()))


def test_features_match_the_slips_module():
    flows = [json.loads(s) for s in REF["flows"]]
    feats = np.array([M.extract_flow_features(f) for f in flows], dtype=np.float64)
    assert np.array_equal(feats, REF["features"])


@pytest.mark.skipif(not DATASET_011.is_dir(), reason="dataset 011 not available")
def test_zeek_dataset_rows_give_the_reference_features():
    from src.dataset_wrapper import ZeekDataset

    want = {json.loads(s)["uid"]: f for s, f in zip(REF["flows"], REF["features"])}
    got = {}
    for rec in ZeekDataset(DATASET_011, batch_size=1)._iter_lines():
        if rec["uid"] in want:
            got[rec["uid"]] = M.extract_flow_features(zeek_to_slips_flow(rec))
    assert len(got) > 100
    for uid, feats in got.items():
        assert feats == list(want[uid]), uid


def test_zeek_unset_fields_become_what_slips_stores():
    flow = zeek_to_slips_flow({"id.orig_h": "10.0.0.1", "proto": "tcp", "service": None,
                               "duration": None, "orig_bytes": "-", "conn_state": "S0"})
    assert flow["appproto"] == "" and flow["dur"] == 0.0 and flow["sbytes"] == 0
    assert flow["state"] == "S0" and flow["dir_"] == "->"


def test_manual_adam_equals_torch_adamw():
    torch.manual_seed(0)
    x, y = torch.randn(64, 18), torch.randint(0, 2, (64,))
    a, b = M.build_model(), M.build_model()
    state = M.new_adam_state(a)
    opt = torch.optim.AdamW([p for p in b.parameters() if p.requires_grad], lr=0.03,
                            betas=(M.BETA1, M.BETA2), eps=M.EPS, weight_decay=M.WEIGHT_DECAY)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(5):
        crit(a(x), y).backward()
        M.adam_step(a, state, 0.03)
        opt.zero_grad()
        crit(b(x), y).backward()
        opt.step()
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_holdout_split_is_fixed_stratified_and_disjoint():
    y = np.array([1] * 700 + [0] * 300)
    tr, te = holdout_split(y, 0.1, 1111)
    tr2, te2 = holdout_split(y, 0.1, 1111)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)
    assert len(te) == 100 and abs(y[te].sum() - 70) <= 1
    assert not set(tr) & set(te) and len(tr) + len(te) == len(y)
    for fit, ev in cv_folds(y[tr], 5, 1111):
        assert not set(fit) & set(ev)


def test_undersample_keeps_the_smaller_class_and_balances():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([1] * 8 + [0] * 2)
    Xu, yu = undersample(X, y, np.random.default_rng(0))
    assert (yu == 0).sum() == (yu == 1).sum() == 2
    assert set(map(tuple, X[y == 0])) <= set(map(tuple, Xu))


def test_scaler_is_fit_on_the_training_rows_only():
    rng = np.random.default_rng(0)
    Xtr, ytr = rng.normal(size=(40, 18)), np.array([1] * 30 + [0] * 10)
    Xev = rng.normal(loc=50.0, size=(10, 18))
    _, yu, ev_scaled, scaler = prepare(Xtr, ytr, Xev, undersample_train=True,
                                       scaler_fit_on="train_undersampled", rng=np.random.default_rng(1))
    Xu, _ = undersample(Xtr, ytr, np.random.default_rng(1))
    assert np.allclose(scaler.mean_, Xu.mean(axis=0)) and len(yu) == 20
    _, _, _, scaler2 = prepare(Xtr, ytr, Xev * 3, undersample_train=True,
                               scaler_fit_on="train_undersampled", rng=np.random.default_rng(1))
    assert np.allclose(scaler.mean_, scaler2.mean_)  # eval rows never reach the scaler
    assert np.allclose(ev_scaled, scaler.transform(Xev).astype(np.float32))


def test_lr_schedules():
    assert lr_at(7, 15, 0.03, "none") == 0.03
    assert lr_at(0, 100, 0.1, "cosine") == pytest.approx(0.1)
    assert lr_at(99, 100, 0.1, "cosine", {"min_factor": 0.01}) < 0.002
    cyc = [lr_at(s, 90, 0.1, "cyclic", {"cycles": 3, "low_factor": 0.1}) for s in range(90)]
    assert cyc[0] == pytest.approx(0.01) and max(cyc) == pytest.approx(0.1)
    assert cyc[15] == pytest.approx(0.1) and cyc[30] == pytest.approx(0.01)  # up, down, up again
    one = [lr_at(s, 100, 0.1, "onecycle") for s in range(100)]
    assert one.index(max(one)) == 30 and one[-1] < one[0]


def test_a_constant_lr_run_is_the_prefix_of_a_longer_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 18)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    kw = dict(lr=0.03, scheduler="none", batch_size=64, seed=1111, fold=0, checkpoints=[5, 15, 40])
    short = train_eval(X, y, X, y, steps=15, **kw)
    long = train_eval(X, y, X, y, steps=40, **kw)
    assert short[-1] == next(p for p in long if p["step"] == 15)
