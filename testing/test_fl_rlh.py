# Tests for the RLH model in the pipeline (src/fl_rlh): the copied model and features
# match the SLIPS federated module (fl_rlh_reference.npz, made by
# src/fl_rlh/make_reference.py inside the SLIPS image), FLRLHClassifier trains one
# module window per partial_fit, and FeatureExtraction(feature_set="fl_rlh") gives the
# module's 18 features.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.commons import BENIGN, MALICIOUS
from src.features import FeatureExtraction
from src.fl_rlh import model as M
from src.fl_rlh.classifier import FLRLHClassifier, lr_at
from src.fl_rlh.data import undersample, zeek_to_slips_flow

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


def test_undersample_keeps_the_smaller_class_and_balances():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([1] * 8 + [0] * 2)
    Xu, yu = undersample(X, y, np.random.default_rng(0))
    assert (yu == 0).sum() == (yu == 1).sum() == 2
    assert set(map(tuple, X[y == 0])) <= set(map(tuple, Xu))


def test_lr_schedules():
    assert lr_at(7, 15, 0.03, "none") == 0.03
    assert lr_at(0, 100, 0.1, "cosine") == pytest.approx(0.1)
    assert lr_at(99, 100, 0.1, "cosine", {"min_factor": 0.01}) < 0.002
    cyc = [lr_at(s, 90, 0.1, "cyclic", {"cycles": 3, "low_factor": 0.1}) for s in range(90)]
    assert cyc[0] == pytest.approx(0.01) and max(cyc) == pytest.approx(0.1)
    assert cyc[15] == pytest.approx(0.1) and cyc[30] == pytest.approx(0.01)  # up, down, up again
    one = [lr_at(s, 100, 0.1, "onecycle") for s in range(100)]
    assert one.index(max(one)) == 30 and one[-1] < one[0]


def _batch(n_mal=8, n_ben=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_mal + n_ben, 18)).astype(np.float32)
    y = np.array([MALICIOUS] * n_mal + [BENIGN] * n_ben)
    return X, y


@pytest.mark.parametrize("minibatch, steps_per_epoch", [(0, 1), (3, 2), (100, 1)])
def test_one_partial_fit_is_one_undersampled_window(minibatch, steps_per_epoch):
    """8 mal / 2 ben -> 2/2 trained; 3 epochs = 3 steps on the full batch,
    ceil(4 / 3) = 2 steps per epoch with mini-batches of 3."""
    clf = FLRLHClassifier(epochs_per_batch=3, minibatch_size=minibatch)
    clf.partial_fit(*_batch())
    assert clf.batches_seen == 1 and clf.steps_done == 3 * steps_per_epoch
    assert clf._adam["fc1.weight"]["step"] == 3 * steps_per_epoch


def test_the_adam_state_carries_over_between_windows():
    clf = FLRLHClassifier(epochs_per_batch=2)
    clf.partial_fit(*_batch(seed=0))
    clf.partial_fit(*_batch(seed=1))
    assert clf.batches_seen == 2 and clf._adam["head.weight"]["step"] == 4


def test_every_classifier_starts_from_the_shared_init():
    a, b = FLRLHClassifier(), FLRLHClassifier(lr=0.1)
    ref = M.build_model()
    for clf in (a, b):
        assert all(
            torch.equal(p, q)
            for p, q in zip(clf.model.state_dict().values(), ref.state_dict().values())
        )


def test_predict_returns_the_pipeline_labels():
    clf = FLRLHClassifier(epochs_per_batch=5)
    X, y = _batch()
    clf.partial_fit(X, y)
    assert set(clf.predict(X)) <= {BENIGN, MALICIOUS}


def test_cyclic_lr_goes_up_and_down_across_windows():
    clf = FLRLHClassifier(lr=0.1, epochs_per_batch=4, scheduler="cyclic",
                          cycle_batches=2, min_lr_factor=0.1)
    seen = []
    clf._lr = (lambda real: lambda k, total: seen.append(real(k, total)) or seen[-1])(clf._lr)
    for i in range(4):
        clf.partial_fit(*_batch(seed=i))
    # 2 windows x 4 steps per cycle: 0.01 -> 0.1 -> 0.01, twice
    assert seen[0] == pytest.approx(0.01) and seen[4] == pytest.approx(0.1)
    assert seen[8] == pytest.approx(0.01) and max(seen) == pytest.approx(0.1)


def test_unknown_scheduler_is_rejected():
    with pytest.raises(ValueError):
        FLRLHClassifier(scheduler="warmup")


def test_fl_rlh_feature_set_gives_the_module_features():
    recs = [
        {"ts": 1.0, "uid": "C1", "id.orig_h": "10.0.0.1", "id.orig_p": 50000,
         "id.resp_h": "8.8.8.8", "id.resp_p": 53, "proto": "udp", "service": "dns",
         "duration": 0.02, "orig_bytes": 40, "resp_bytes": 120, "conn_state": "SF",
         "history": "Dd", "orig_pkts": 1, "resp_pkts": 1, "label": MALICIOUS},
        {"ts": 2.0, "uid": "C2", "id.orig_h": "10.0.0.1", "id.orig_p": 50001,
         "id.resp_h": "10.0.0.2", "id.resp_p": 0, "proto": "icmp", "service": "-",
         "duration": "-", "orig_bytes": "-", "resp_bytes": "-", "conn_state": "OTH",
         "history": "-", "orig_pkts": 1, "resp_pkts": 0, "label": BENIGN},
    ]
    fe = FeatureExtraction(feature_set="fl_rlh", protocols_to_discard=[])
    X, y = fe.process_batch(pd.DataFrame(recs))
    assert list(X.columns) == M.FEATURE_ORDER
    want = [M.extract_flow_features(zeek_to_slips_flow(r)) for r in recs]
    assert np.allclose(X.to_numpy(), np.array(want))
    assert list(y) == [MALICIOUS, BENIGN]
    # the pipeline's default extraction is unchanged
    assert FeatureExtraction().feature_set == "slips"


def test_unknown_feature_set_is_rejected():
    with pytest.raises(ValueError):
        FeatureExtraction(feature_set="other")



def test_the_optuna_config_samples_buildable_classifiers(tmp_path):
    """configs/fl_rlh_optuna.yaml through the pipeline's OptunaOptimizer: every
    sampled config flattens to FLRLHClassifier params that build, in range."""
    import copy

    import optuna
    import yaml

    from src.optuna_optimizer import OptunaOptimizer

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "fl_rlh_optuna.yaml").read_text())
    assert cfg["features"]["feature_set"] == "fl_rlh" and cfg["validation_split"] == 0.0
    assert cfg["features"]["columns_to_discard"] == ["saddr_num", "daddr_num"]
    train, objective, outside = cfg["commands"]
    assert train["command"] == "train" and objective["command"] == outside["command"] == "test"
    # the objective = the held-out rows of the training captures, never trained on
    assert all(d.endswith("-train") for d in train["mixer"]["datasets"])
    assert [d.replace("-test", "-train") for d in objective["mixer"]["datasets"]] == train["mixer"]["datasets"]
    assert not {d[:3] for d in train["mixer"]["datasets"]} & set(outside["mixer"]["datasets"])
    opt = OptunaOptimizer(
        pipeline_cls=None, config_reader_cls=None, base_config=cfg, exp_dir=tmp_path,
        metric_names=["f1", "fpr"], search_space=cfg["optuna"]["hyperparameters"],
        directions=("maximize", "minimize"), n_trials=1, n_jobs=1,
    )
    study = optuna.create_study(directions=["maximize", "minimize"],
                                sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(20):
        config = copy.deepcopy(cfg)
        trial = study.ask()
        overrides = opt._sample_hyperparameters(trial, config)
        study.tell(trial, [0.0, 0.0])
        params = config["model"]["classifier_params"]
        assert "FLRLHClassifier" not in params and params["seed"] == 1111
        assert 1e-5 <= params["lr"] <= 1.0 and 1 <= params["epochs_per_batch"] <= 300
        assert any(k.endswith("lr") for k in overrides)
        mixer = config["commands"][0]["mixer"]
        assert mixer["type"] in ("sequence", "random", "balanced")
        assert ("balance" in mixer) == (mixer["type"] == "balanced")
        assert mixer["datasets"] == cfg["commands"][0]["mixer"]["datasets"]  # same training set
        FLRLHClassifier(**params)


def test_default_net_is_the_module_model_and_shapes_are_tunable():
    ref, net = M.build_model(), M.build_net()
    assert all(torch.equal(p, q) for p, q in zip(ref.state_dict().values(), net.state_dict().values()))
    wide = M.build_net(hidden2=64, extra=32)
    assert wide.fc1.out_features == 64 and wide.extra.out_features == 32
    assert wide.head.in_features == 32 and wide(torch.zeros(3, 18)).shape == (3, 2)
    assert torch.equal(wide.random_projection.weight, ref.random_projection.weight)
    assert not wide.random_projection.weight.requires_grad


def test_adam_properties_match_torch_adamw():
    torch.manual_seed(0)
    x, y = torch.randn(64, 18), torch.randint(0, 2, (64,))
    a, b = M.build_model(), M.build_model()
    state = M.new_adam_state(a)
    opt = torch.optim.AdamW([p for p in b.parameters() if p.requires_grad], lr=0.01,
                            betas=(0.8, 0.95), eps=1e-6, weight_decay=0.01)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(5):
        crit(a(x), y).backward()
        M.adam_step(a, state, 0.01, wd=0.01, beta1=0.8, beta2=0.95, eps=1e-6)
        opt.zero_grad()
        crit(b(x), y).backward()
        opt.step()
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_classifier_with_extra_layer_and_custom_adam_trains():
    clf = FLRLHClassifier(epochs_per_batch=3, hidden2=32, extra_layer=8,
                          beta1=0.8, beta2=0.99, eps=1e-6, weight_decay=0.0)
    X, y = _batch()
    clf.partial_fit(X, y)
    assert clf.model.extra.out_features == 8 and clf.steps_done == 3
    assert set(clf.predict(X)) <= {BENIGN, MALICIOUS}


def _zeek_log(path, n_mal, n_ben, n_bg):
    header = ["#separator \\x09\n", "#set_separator\t,\n",
              "#fields\tts\tuid\tid.orig_h\tproto\tlabel\tdetailed-label\n",
              "#types\ttime\tstring\taddr\tenum\tstring\tstring\n"]
    labels = ["Malicious"] * n_mal + ["Benign"] * n_ben + ["Background"] * n_bg
    rows = [f"{i}.0\tC{i}\t10.0.0.1\ttcp\t{lab}\t-\n" for i, lab in enumerate(labels)]
    path.parent.mkdir(parents=True)
    path.write_text("".join(header + rows + ["#close\t2020-01-01\n"]))
    return header, rows


def test_make_split_is_fixed_stratified_disjoint_and_keeps_the_header(tmp_path):
    from src.fl_rlh.make_split import make_split

    src = tmp_path / "src"
    header, rows = _zeek_log(src / "008-zeek-mixed" / "data" / "conn.log.labeled", 70, 20, 10)
    (src / "014-zeek-malicious" / "data").mkdir(parents=True)
    out = tmp_path / "out"
    m = make_split(src, out, ["008"], ["014"], 0.1, 1111)
    tr = (out / "008-train-zeek-mixed" / "data" / "conn.log.labeled").read_text().splitlines(keepends=True)
    te = (out / "008-test-zeek-mixed" / "data" / "conn.log.labeled").read_text().splitlines(keepends=True)
    for part in (tr, te):
        assert part[: len(header)] == header and part[-1].startswith("#close")
    tr_rows, te_rows = tr[len(header):-1], te[len(header):-1]
    assert not set(tr_rows) & set(te_rows) and sorted(tr_rows + te_rows) == sorted(rows)
    assert tr_rows == [r for r in rows if r in set(tr_rows)]  # original order kept
    assert m["train"]["008"]["test"] == {"malicious": 7, "benign": 2, "other": 1}
    assert (out / "014-zeek-malicious").is_symlink()
    out2 = tmp_path / "out2"
    make_split(src, out2, ["008"], [], 0.1, 1111)
    assert (out2 / "008-test-zeek-mixed" / "data" / "conn.log.labeled").read_text() == "".join(te)


def test_fl_rlh_columns_to_discard_zero_a_feature_and_keep_18_inputs():
    rec = {"ts": 1.0, "uid": "C1", "id.orig_h": "147.32.81.167", "id.orig_p": 50000,
           "id.resp_h": "8.8.8.8", "id.resp_p": 53, "proto": "udp", "service": "dns",
           "duration": 0.02, "orig_bytes": 40, "resp_bytes": 120, "conn_state": "SF",
           "history": "Dd", "orig_pkts": 1, "resp_pkts": 1, "label": BENIGN}
    fe = FeatureExtraction(feature_set="fl_rlh", protocols_to_discard=[],
                           columns_to_discard=["saddr_num", "daddr_num"])
    X, _ = fe.process_batch(pd.DataFrame([rec]))
    assert list(X.columns) == M.FEATURE_ORDER
    assert X.loc[0, "saddr_num"] == 0.0 and X.loc[0, "daddr_num"] == 0.0
    assert X.loc[0, "dport"] == 53.0



def test_converter_records_give_the_same_features_as_raw_zeek_lines():
    """The pipeline's mixers yield ConnToSlipsConverter records (SLIPS names,
    state 0/1, dir_ empty); they must give the module's features exactly."""
    from src.conn_normalizer import ConnToSlipsConverter
    from src.fl_rlh.data import slips_flow

    raws = [
        {"ts": 1.0, "uid": "C1", "id.orig_h": "10.0.0.1", "id.orig_p": 50000, "id.resp_h": "8.8.8.8",
         "id.resp_p": 53, "proto": "udp", "service": "dns", "duration": 0.02, "orig_bytes": 40,
         "resp_bytes": 120, "conn_state": "SF", "history": "Dd", "orig_pkts": 1, "resp_pkts": 1},
        {"ts": 2.0, "uid": "C2", "id.orig_h": "167.94.138.99", "id.orig_p": 7789, "id.resp_h": "147.32.83.165",
         "id.resp_p": 445, "proto": "tcp", "service": "-", "duration": 2e-06, "orig_bytes": 0,
         "resp_bytes": 0, "conn_state": "S0", "history": "S", "orig_pkts": 2, "resp_pkts": 0},
        {"ts": 3.0, "uid": "C3", "id.orig_h": "10.0.0.1", "id.orig_p": 1, "id.resp_h": "10.0.0.2",
         "id.resp_p": 80, "proto": "tcp", "service": "http", "duration": "-", "orig_bytes": "-",
         "resp_bytes": "-", "conn_state": "RSTR", "history": "ShADadfR", "orig_pkts": 5, "resp_pkts": 4},
    ]
    conv = ConnToSlipsConverter(default_label=BENIGN)
    for raw in raws:
        normalized = conv.normalize_batch([raw])[0]
        assert "id.orig_h" not in normalized  # really the converter's form
        want = M.extract_flow_features(zeek_to_slips_flow(raw))
        assert M.extract_flow_features(slips_flow(normalized)) == want, raw["uid"]
        assert M.extract_flow_features(slips_flow(raw)) == want


@pytest.mark.skipif(not DATASET_011.is_dir(), reason="dataset 011 not available")
def test_the_pipeline_batch_path_gives_the_slips_stored_features():
    """Loader -> mixer -> FeatureExtraction(fl_rlh), as the pipeline runs it, against
    the features of the flows SLIPS itself stored for dataset 011."""
    from src.data_selectors import SequenceMixer
    from src.dataset_wrapper import find_and_load_datasets

    want = {json.loads(s)["uid"]: f for s, f in zip(REF["flows"], REF["features"])}
    loaders = find_and_load_datasets(str(DATASET_011.parents[1]), batch_size=1000,
                                     prefix_regex=r"^011-", seed=1111)
    mixer = SequenceMixer({"datasets": ["011"], "validation_split": 0.0}, loaders, np.random.RandomState(0))
    mixer.reset_epoch(1000)
    fe = FeatureExtraction(feature_set="fl_rlh", protocols_to_discard=[])
    seen = 0
    while True:
        batch, _ = mixer.next_batch()
        if batch is None:
            break
        X, _ = fe.process_batch(batch)
        for uid, row in zip(batch["uid"], X.to_numpy()):
            if uid in want:
                assert np.array_equal(row, want[uid]), uid
                seen += 1
    assert seen > 100
