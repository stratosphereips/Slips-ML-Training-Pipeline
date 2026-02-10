import json
from .commons import BENIGN, MALICIOUS, BACKGROUND
import numpy as np
from pathlib import Path

# -------------------------
# Optuna Logger for trial/config/result logging
# -------------------------
class OptunaLogger:
    def __init__(self, optuna_dir):
        self.optuna_dir = Path(optuna_dir)
        self.optuna_dir.mkdir(parents=True, exist_ok=True)

    def log_trial_config(self, trial_number, config):
        cfg_path = self.optuna_dir / f"trial_{trial_number}_config.yaml"
        try:
            import yaml
            with open(cfg_path, "w") as f:
                yaml.safe_dump(config, f)
        except Exception:
            with open(cfg_path, "w") as f:
                f.write(str(config))

    def log_trial_result(self, trial_number, result):
        result_path = self.optuna_dir / f"trial_{trial_number}_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    def log_summary(self, summary):
        summary_path = self.optuna_dir / "optuna_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)




class Logger:
    def __init__(
        self,
        logfile_path,
        overwrite: bool = False,
    ):
        """
        Logger that writes strictly to a single logfile path.

        Parameters
        ----------
        logfile_path : str or Path
            Full path to the logfile (including filename).
        overwrite : bool
            If False and file exists, raises FileExistsError.
        """
        self.logfile_path = Path(logfile_path)

        # ensure parent directory exists
        self.logfile_path.parent.mkdir(parents=True, exist_ok=True)

        if self.logfile_path.exists() and not overwrite:
            raise FileExistsError(
                f"Logfile '{self.logfile_path}' already exists."
            )

        # create / truncate logfile
        self.logfile_path.write_text("")

        # Only consider MALICIOUS and BENIGN labels for metrics
        self.relevant_labels = [MALICIOUS, BENIGN]

    # -------------------------
    # basic logging
    # -------------------------
    def write_to_log(self, message: str):
        with open(self.logfile_path, "a") as f:
            f.write(message + "\n")

    def log(self, message: str):
        print(message)

    # -------------------------
    # metrics helpers
    # -------------------------
    def compute_metrics(self, y_true, y_pred, relevant_labels=None):
        if relevant_labels is None:
            relevant_labels = self.relevant_labels

        metrics = {
            "TP": int(np.sum((y_pred == MALICIOUS) & (y_true == MALICIOUS))),
            "FP": int(np.sum((y_pred == MALICIOUS) & (y_true == BENIGN))),
            "FN": int(np.sum((y_pred == BENIGN) & (y_true == MALICIOUS))),
            "TN": int(np.sum((y_pred == BENIGN) & (y_true == BENIGN))),
        }

        seen_labels = {
            label: int(np.sum(y_true == label)) for label in relevant_labels
        }
        predicted_labels = {
            label: int(np.sum(y_pred == label)) for label in relevant_labels
        }

        return metrics, seen_labels, predicted_labels

    def _filter_labels(self, y_true, y_pred, relevant_labels=None):
        if relevant_labels is None:
            relevant_labels = self.relevant_labels

        mask = np.isin(y_true, relevant_labels)
        return y_true[mask], y_pred[mask]

    # -------------------------
    # training logging
    # -------------------------
    def save_training_results(
        self,
        y_pred_train,
        y_gt_train,
        y_pred_val,
        y_gt_val,
        sum_labeled_flows,
    ):
        relevant_labels = self.relevant_labels

        # train + validation
        if y_pred_val is not None and y_gt_val is not None and len(y_gt_val) > 0:
            y_gt_val_filt, y_pred_val_filt = self._filter_labels(
                y_gt_val, y_pred_val, relevant_labels
            )
            y_gt_train_filt, y_pred_train_filt = self._filter_labels(
                y_gt_train, y_pred_train, relevant_labels
            )

            metrics_val, seen_val, pred_val = self.compute_metrics(
                y_gt_val_filt, y_pred_val_filt, relevant_labels
            )
            metrics_train, seen_train, pred_train = self.compute_metrics(
                y_gt_train_filt, y_pred_train_filt, relevant_labels
            )

            self.write_to_log(
                f"Total labels: {sum_labeled_flows}, "
                f"Validation size: {len(y_pred_val_filt)}, "
                f"Validation seen labels: {seen_val}, "
                f"Validation predicted labels: {pred_val}, "
                f"Validation metrics: {metrics_val}, "
                f"Training size: {len(y_gt_train_filt)}, "
                f"Training seen labels: {seen_train}, "
                f"Training predicted labels: {pred_train}, "
                f"Training metrics: {metrics_train}"
            )

        # train only
        else:
            y_gt_train_filt, y_pred_train_filt = self._filter_labels(
                y_gt_train, y_pred_train, relevant_labels
            )
            metrics, seen, pred = self.compute_metrics(
                y_gt_train_filt, y_pred_train_filt, relevant_labels
            )

            self.write_to_log(
                f"Total labels: {sum_labeled_flows}, "
                f"Training size: {len(y_pred_train_filt)}, "
                f"Training seen labels: {seen}, "
                f"Training predicted labels: {pred}, "
                f"Training metrics: {metrics}"
            )

    # -------------------------
    # test logging
    # -------------------------
    def save_test_results(self, original_labels, predicted_labels):
        original_labels = np.asarray(original_labels)
        predicted_labels = np.asarray(predicted_labels)

        mask = (original_labels != BACKGROUND) & (
            predicted_labels != BACKGROUND
        )
        filtered_orig = original_labels[mask]
        filtered_pred = predicted_labels[mask]

        if not hasattr(self, "malware_metrics"):
            self.malware_metrics = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
            self.seen_labels = {MALICIOUS: 0, BENIGN: 0}
            self.predicted_labels = {MALICIOUS: 0, BENIGN: 0}

        for label in [MALICIOUS, BENIGN]:
            self.seen_labels[label] += int(np.sum(filtered_orig == label))
            self.predicted_labels[label] += int(
                np.sum(filtered_pred == label)
            )

        self.malware_metrics["TP"] += int(
            np.sum(
                (filtered_orig == MALICIOUS)
                & (filtered_pred == MALICIOUS)
            )
        )
        self.malware_metrics["FP"] += int(
            np.sum(
                (filtered_orig == BENIGN)
                & (filtered_pred == MALICIOUS)
            )
        )
        self.malware_metrics["FN"] += int(
            np.sum(
                (filtered_orig == MALICIOUS)
                & (filtered_pred == BENIGN)
            )
        )
        self.malware_metrics["TN"] += int(
            np.sum(
                (filtered_orig == BENIGN)
                & (filtered_pred == BENIGN)
            )
        )

        total_flows = sum(self.seen_labels.values())
        self.write_to_log(
            f"Total flows: {total_flows}; "
            f"Seen labels: {self.seen_labels}; "
            f"Predicted labels: {self.predicted_labels}; "
            f"Malware metrics (TP/FP/TN/FN): {self.malware_metrics}"
        )
