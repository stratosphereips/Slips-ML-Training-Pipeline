import numpy as np
import pytest
from sklearn.metrics import accuracy_score, matthews_corrcoef

from src.commons import BENIGN, MALICIOUS
from src.logger import Logger
from src import metrics_calculator as mc


def _one_vs_rest_counts(y_true, y_pred, classes):
    return {
        cls: {
            "TP": int(np.sum((y_pred == cls) & (y_true == cls))),
            "FP": int(np.sum((y_pred == cls) & (y_true != cls))),
            "FN": int(np.sum((y_pred != cls) & (y_true == cls))),
            "TN": int(np.sum((y_pred != cls) & (y_true != cls))),
        }
        for cls in classes
    }


@pytest.mark.parametrize("classes", [["Benign", "Malicious"], ["a", "b", "c"]])
def test_multi_metrics_match_sklearn(classes):
    rng = np.random.default_rng(0)
    y_true = rng.choice(classes, size=600)
    noise = rng.choice(classes, size=600)
    y_pred = np.where(rng.random(600) < 0.8, y_true, noise)
    metrics = mc.compute_multi_metrics(_one_vs_rest_counts(y_true, y_pred, classes))
    assert metrics["MCC"] == pytest.approx(matthews_corrcoef(y_true, y_pred))
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))


def test_multi_metrics_degenerate_inputs():
    assert mc.compute_multi_metrics({})["MCC"] == 0.0
    all_benign = {
        "Benign": {"TP": 10, "FP": 0, "FN": 0, "TN": 0},
        "Malicious": {"TP": 0, "FP": 0, "FN": 0, "TN": 10},
    }
    metrics = mc.compute_multi_metrics(all_benign)
    assert metrics["MCC"] == 0.0  # undefined -> 0, no ZeroDivisionError
    assert metrics["accuracy"] == 1.0


def test_binary_metrics_values_and_zero_division():
    m = mc.compute_binary_metrics({"TP": 8, "FP": 2, "FN": 4, "TN": 86})
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(8 / 12)
    assert m["fpr"] == pytest.approx(2 / 88)
    assert m["fnr"] == pytest.approx(4 / 12)
    assert m["accuracy"] == pytest.approx(0.94)
    zero = mc.compute_binary_metrics({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    assert zero == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0, "fnr": 0.0, "accuracy": 0.0}


def test_malware_metrics_fp_over_predicted_is_fdr():
    counts = {
        "Malicious": {"TP": 6, "FP": 2, "FN": 1, "TN": 11},
        "Benign": {"TP": 11, "FP": 1, "FN": 2, "TN": 6},
    }
    m = mc.compute_malware_metrics(counts)
    assert m["fp_over_predicted"] == pytest.approx(2 / 8)
    assert m["fpr"] == pytest.approx(2 / 13)


def test_training_log_round_trip(tmp_path):
    logger = Logger(tmp_path / "0_train_train.log", overwrite=True)
    y_gt_train = np.array([MALICIOUS, MALICIOUS, BENIGN, BENIGN, BENIGN])
    y_pred_train = np.array([MALICIOUS, BENIGN, BENIGN, MALICIOUS, BENIGN])
    y_gt_val = np.array([MALICIOUS, BENIGN])
    y_pred_val = np.array([MALICIOUS, MALICIOUS])
    written = logger.save_training_results(y_pred_train, y_gt_train, y_pred_val, y_gt_val, 7)

    entries = mc.read_training_batches(str(tmp_path / "0_train_train.log"))
    assert len(entries) == 1
    entry = entries[0]
    assert entry["per_class"]["Malicious"] == written["val"]
    assert entry["training_per_class"]["Malicious"] == written["train"]
    # Benign is the mirror image of Malicious
    mal = written["train"]
    assert entry["training_per_class"]["Benign"] == {"TP": mal["TN"], "FP": mal["FN"], "TN": mal["TP"], "FN": mal["FP"]}
    assert mc.detect_validation_split(entries) is True


def test_training_log_round_trip_without_validation(tmp_path):
    logger = Logger(tmp_path / "t.log", overwrite=True)
    logger.save_training_results(np.array([BENIGN, MALICIOUS]), np.array([BENIGN, BENIGN]), None, None, 2)
    entries = mc.read_training_batches(str(tmp_path / "t.log"))
    assert "per_class" not in entries[0]
    assert entries[0]["training_per_class"]["Malicious"]["FP"] == 1
    assert mc.detect_validation_split(entries) is False


def test_testing_log_round_trip_is_cumulative(tmp_path):
    logger = Logger(tmp_path / "1_test_test.log", overwrite=True)
    logger.save_test_results([MALICIOUS, BENIGN], [MALICIOUS, MALICIOUS])
    last = logger.save_test_results([MALICIOUS, BENIGN, "Background"], [BENIGN, BENIGN, BENIGN])
    assert last == {"TP": 1, "FP": 1, "TN": 1, "FN": 1}

    entries = mc.read_testing_snapshots(str(tmp_path / "1_test_test.log"))
    assert len(entries) == 2
    assert entries[-1]["binary_summary"] == last
    *_, totals = mc.accumulate_testing_metrics(entries)
    assert totals == [2, 4]


def test_per_class_table_lists_every_class():
    table = mc._format_per_class_table(
        "Per-class metrics",
        {
            "Benign": {"TP": 5, "TN": 3, "FP": 1, "FN": 0},
            "Malicious": {"TP": 3, "TN": 5, "FP": 0, "FN": 1},
        },
    )
    rows = [line.split()[0] for line in table[2:]]
    assert rows == ["Malicious", "Benign"]
