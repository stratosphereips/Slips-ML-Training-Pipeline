# pipeline_ml_training/runner.py
"""
 - saves effective config into experiment folder BEFORE running
 - supports loading preprocessing steps and classifier from config
   (preprocessing.load_from, model.load_from + model.load_name)
 - saves model & preprocessing after every command (train or test)
 - writes per-command logs under command folders and a separate experiment-level
   "checkpoints" log for test checkpoints (different from results logs)
 - includes an if __name__ == "__main__" entrypoint for simple CLI usage
"""

import os
import json
import shutil
from pathlib import Path
import sys

import numpy as np

from .conf_reader import ConfigReader
from .class_factory import (
    get_transformer_class,
    get_classifier_class,
    prepare_river_nested_model_params,
    get_wrapper_class,
    get_mixer_class,
)
from .dataset_wrapper import find_and_load_datasets
from .features import FeatureExtraction
from .preprocessing_wrapper import PreprocessingWrapper
from .logger import Logger


class PipelineRunner:
    def __init__(self, config_path_or_dir):
        # load config
        self.cfg_reader = ConfigReader(config_path_or_dir)
        self.cfg = self.cfg_reader.load()

        # deterministic rng for mixers / sampling
        self.rng = np.random.default_rng(self.cfg.get("seed", 1111))

        # components to be built
        self.loaders = None
        self.feature_extractor = None
        self.preprocessor = None
        self.classifier_wrapper = None

        # experiment directory paths resolved by config_reader
        self.paths = self.cfg_reader.get_paths()
        self.expdir = Path(self.paths["experiment_dir_resolved"])
        self.checkpoints_dir = self.expdir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # filesystem / layout helpers
    # ---------------------------
    def _prepare_experiment_dirs(self):
        """
        Create the top-level experiment layout and save effective config before run.
        """
        # base experiment dir and standard children
        self.expdir.mkdir(parents=True, exist_ok=True)
        (self.expdir / "models").mkdir(exist_ok=True)
        (self.expdir / "results").mkdir(exist_ok=True)
        (self.expdir / "logs").mkdir(exist_ok=True)
        (self.expdir / self.cfg["paths"]["preprocessing_dir_name"]).mkdir(
            exist_ok=True
        )

        # Save effective config early (so it's always present)
        cfg_path = self.expdir / "config_effective.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

        readme = self.expdir / "readme.txt"
        if not readme.exists():
            readme.write_text(
                f"Experiment: {self.cfg.get('experiment_name')}\n"
                "Created by PipelineRunner.\n"
                "Effective config saved in config_effective.json\n",
                encoding="utf-8",
            )

    # ---------------------------
    # construction helpers
    # ---------------------------
    def _build_loaders(self):
        ds_params = self.cfg_reader.get_dataset_loader_params()
        root = self.cfg.get("root")
        if root is None:
            raise ValueError("Config must specify 'root' dataset path")

        # call the dataset helper (your existing loader)
        self.loaders = find_and_load_datasets(
            root_dir=root,
            batch_size=int(ds_params.get("batch_size", 1000)),
            prefix_regex=ds_params.get("prefix_regex", r"^\d{3}"),
            data_subdir=ds_params.get("data_subdir", "data"),
            seed=self.cfg.get("seed", None),
        )

        # apply extra loader-level options (cache_dir, labeled_filenames, shuffle_per_epoch)
        for key, loader in self.loaders.items():
            try:
                if "persist_cache_threshold" in ds_params and hasattr(loader, "persist_cache_threshold"):
                    loader.persist_cache_threshold = ds_params.get("persist_cache_threshold")
            except Exception:
                pass
            try:
                if "cache_dir" in ds_params and hasattr(loader, "cache_dir"):
                    loader.cache_dir = ds_params.get("cache_dir")
            except Exception:
                pass
            try:
                if "labeled_filenames" in ds_params and hasattr(loader, "labeled_filenames"):
                    loader.labeled_filenames = ds_params.get("labeled_filenames")
            except Exception:
                pass
            try:
                if "file_encoding" in ds_params and hasattr(loader, "file_encoding"):
                    loader.file_encoding = ds_params.get("file_encoding")
                if "file_errors" in ds_params and hasattr(loader, "file_errors"):
                    loader.file_errors = ds_params.get("file_errors")
            except Exception:
                pass
            try:
                if "shuffle_per_epoch" in ds_params and hasattr(loader, "shuffle_per_epoch"):
                    loader.shuffle_per_epoch = ds_params.get("shuffle_per_epoch")
            except Exception:
                pass

        return self.loaders

    def _build_feature_extractor(self):
        params = self.cfg_reader.get_feature_extractor_params()
        self.feature_extractor = FeatureExtraction(**params)
        return self.feature_extractor

    def _build_preprocessor(self):
        """
        Build PreprocessingWrapper with steps from config.
        If `preprocessing.load_from` is present in config, try to load existing steps.
        """
        exp_name = self.cfg.get("experiment_name", "experiment")
        save_opts = self.cfg_reader.get_preprocessing_save_options()
        base_preproc_dir = save_opts.get("preprocessing_dir") or str((self.expdir / "preprocessing").resolve())
        step_template = save_opts.get("step_filename_template", "{name}.bin")

        prep = PreprocessingWrapper(
            steps=[],
            experiment_name=exp_name,
            base_models_dir=base_preproc_dir,
            step_filename_template=step_template,
        )

        # instantiate steps defined in config
        steps_spec = self.cfg_reader.get_preprocessing_steps()
        for s in steps_spec:
            try:
                Cls = get_transformer_class(s.get("type"))
            except Exception as e:
                raise RuntimeError(f"Failed to resolve transformer '{s.get('type')}': {e}")
            params = s.get("params", {}) or {}
            try:
                inst = Cls(**params)
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate transformer '{s.get('type')}': {e}")
            prep.add_step(s.get("name"), inst)

        # support loading existing preprocessing steps if configured
        p_conf = self.cfg.get("preprocessing", {}) or {}
        load_from = p_conf.get("load_from")
        if load_from:
            # interpret relative paths relative to experiment dir
            ppath = Path(load_from)
            if not ppath.is_absolute():
                ppath = self.expdir / load_from
            try:
                prep.load(base_path=str(ppath))
            except Exception as e:
                # fail early — better to know before training starts
                raise RuntimeError(f"Failed to load preprocessing from {ppath}: {e}")

        self.preprocessor = prep
        return prep

    def _build_classifier_wrapper(self):
        """
        Instantiate the classifier and the wrapper.
        Support 'model.load_from' in config to load an existing classifier.
        """
        model_spec = self.cfg_reader.get_model_spec() or {}
        classifier_type = model_spec.get("classifier_type") or model_spec.get("type")
        classifier_params = dict(model_spec.get("classifier_params", {}) or {})

        # handle nested river inner 'model' spec (instantiate inner model if needed)
        if isinstance(classifier_params, dict) and "model" in classifier_params and isinstance(classifier_params["model"], dict):
            classifier_params = prepare_river_nested_model_params(classifier_params)

        # If user requested to load existing classifier, try to load instead of instantiate
        load_conf = model_spec.get("load_from")
        load_name = model_spec.get("load_name", model_spec.get("save_name", "classifier.bin"))

        wrapper_name = model_spec.get("wrapper", "SKLearnClassifierWrapper")
        WrapperCls = get_wrapper_class(wrapper_name)

        if load_conf:
            # interpret relative path relative to experiment dir
            load_path = Path(load_conf)
            if not load_path.is_absolute():
                load_path = self.expdir / load_conf
            # Attempt to load: We will create a dummy wrapper first (no classifier) then call load_classifier
            try:
                # instantiate a minimal classifier object (some wrappers expect a classifier object on init)
                # try to build a classifier object if possible; if it fails, pass None and rely on wrapper.load
                classifier_obj = None
                try:
                    Cls = get_classifier_class(classifier_type)
                    classifier_obj = Cls(**classifier_params)
                except Exception:
                    classifier_obj = None

                # try wrapper construction with available parameters
                try:
                    wrapper_obj = WrapperCls(classifier_obj, preprocessing_handler=self.preprocessor, classes=model_spec.get("classes", []))
                except TypeError:
                    try:
                        wrapper_obj = WrapperCls(classifier_obj, preprocessing_handler=self.preprocessor)
                    except TypeError:
                        wrapper_obj = WrapperCls(classifier_obj)
                # now call the wrapper's load method; wrappers offer load methods in your code:
                # - PreprocessingWrapper.load exists; ClassifierWrapper has load_classifier(path,name)
                try:
                    wrapper_obj.load_classifier(str(load_path), name=load_name)
                except Exception as e:
                    # maybe wrapper provides different signature - fallback to wrapper.classifier.load or direct pickle load
                    raise RuntimeError(f"Wrapper failed to load classifier from {load_path}: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to load classifier from {load_conf}: {e}")
            self.classifier_wrapper = wrapper_obj
            return wrapper_obj

        # otherwise instantiate classifier from scratch
        if not classifier_type:
            raise ValueError("model.classifier_type must be present when not loading an existing model")
        Cls = get_classifier_class(classifier_type)
        try:
            classifier_obj = Cls(**classifier_params)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate classifier '{classifier_type}': {e}")

        # instantiate wrapper
        try:
            wrapper_obj = WrapperCls(classifier_obj, preprocessing_handler=self.preprocessor, classes=model_spec.get("classes", []))
        except TypeError:
            try:
                wrapper_obj = WrapperCls(classifier_obj, preprocessing_handler=self.preprocessor)
            except TypeError:
                wrapper_obj = WrapperCls(classifier_obj)

        # allow dummy_flows override
        if hasattr(wrapper_obj, "dummy_flows") and model_spec.get("dummy_flows"):
            try:
                wrapper_obj.dummy_flows.update(model_spec.get("dummy_flows") or {})
            except Exception:
                pass

        self.classifier_wrapper = wrapper_obj
        return wrapper_obj

    # ---------------------------
    #  structure helpers
    # ---------------------------
    def _ensure_command_dirs(self, cmd_idx, cmd):
        cmd_paths = cmd.get("paths", {})
        for v in cmd_paths.values():
            try:
                Path(v).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        # return full log path
        log_path = Path(cmd_paths.get("logs", ".")) / "pipeline.log"
        return str(log_path)

    def _save_command_config(self, cmd_idx, cmd):
        cmd_dir = Path(cmd["paths"]["command_dir"])
        target = cmd_dir / "config_used.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(cmd, f, indent=2)

    def _append_checkpoint(self, cmd_idx, cmd, text):
        """
        Save a line (text) into experiment-level checkpoints log for this command.
        This is separate from the per-command logs/results.
        """
        safe_name = cmd.get("name") or f"cmd_{cmd_idx}"
        fname = self.checkpoints_dir / f"{cmd_idx}_{safe_name}_checkpoints.log"
        with open(fname, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    # ---------------------------
    # command implementations
    # ---------------------------
    def _run_train_command(self, cmd_idx, cmd):
        mixer_spec = cmd.get("mixer", {})
        MixerCls = get_mixer_class(mixer_spec.get("type"))
        mixer = MixerCls(mixer_spec, self.loaders, self.rng)

        batch_size = int(cmd.get("effective_batch_size"))
        mixer.reset_epoch(batch_size, epoch_idx=0)

        log_file = self._ensure_command_dirs(cmd_idx, cmd)
        logger = Logger(experiment_name=self.cfg.get("experiment_name"), path_to_logging_dir=os.path.dirname(log_file), path_to_logfile=os.path.basename(log_file), overwrite=True)

        batch_counter = 0
        while True:
            train_batch, val_batch = mixer.next_batch()
            if train_batch is None and val_batch is None:
                break
            batch_counter += 1

            # feature extraction
            try:
                X_train_df, y_train = self.feature_extractor.process_batch(train_batch)
            except Exception as e:
                logger.write_to_log(f"[ERROR] feature extraction failed for train batch {batch_counter}: {e}")
                continue

            if getattr(X_train_df, "shape", (0, 0))[0] == 0:
                continue

            # preprocessing
            try:
                self.preprocessor.partial_fit(X_train_df)
                X_train_processed = self.preprocessor.transform(X_train_df)
            except Exception as e:
                logger.write_to_log(f"[ERROR] preprocessing failed on train batch {batch_counter}: {e}")
                continue

            # classifier update
            try:
                y_train_arr = np.asarray(y_train)
                self.classifier_wrapper.partial_fit(X_train_processed, y_train_arr)
                y_pred_train = self.classifier_wrapper.predict(X_train_processed)
            except Exception as e:
                logger.write_to_log(f"[ERROR] classifier partial_fit failed on train batch {batch_counter}: {e}")
                continue

            # validation handling (mixer already returns split)
            y_pred_val = None
            y_val_arr = None
            if val_batch is not None:
                try:
                    X_val_df, y_val = self.feature_extractor.process_batch(val_batch)
                    if getattr(X_val_df, "shape", (0, 0))[0] > 0:
                        X_val_processed = self.preprocessor.transform(X_val_df)
                        y_pred_val = self.classifier_wrapper.predict(X_val_processed)
                        y_val_arr = np.asarray(y_val)
                except Exception as e:
                    logger.write_to_log(f"[WARN] validation processing failed for batch {batch_counter}: {e}")
                    y_pred_val = None
                    y_val_arr = None

            # logging metrics
            try:
                logger.save_training_results(y_pred_train, np.asarray(y_train), y_pred_val, y_val_arr, sum_labeled_flows=len(y_train))
            except Exception as e:
                logger.write_to_log(f"[WARN] saving training metrics failed for batch {batch_counter}: {e}")

        # Save preprocessing and model after the command finishes (required)
        # Preprocessor
        try:
            prep_path = Path(cmd["paths"]["preprocessing"])
            prep_path.mkdir(parents=True, exist_ok=True)
            self.preprocessor.save(base_path=str(prep_path))
        except Exception as e:
            logger.write_to_log(f"[ERROR] saving preprocessing failed: {e}")

        # classifier
        try:
            model_path = Path(cmd["paths"]["model"])
            model_path.mkdir(parents=True, exist_ok=True)
            model_spec = self.cfg_reader.get_model_spec()
            save_name = model_spec.get("save_name", "classifier.bin")
            self.classifier_wrapper.save_classifier(path=str(model_path), name=save_name)
        except Exception as e:
            logger.write_to_log(f"[ERROR] saving classifier failed: {e}")

    def _run_test_command(self, cmd_idx, cmd):
        mixer_spec = cmd.get("mixer", {})
        MixerCls = get_mixer_class(mixer_spec.get("type"))
        mixer = MixerCls(mixer_spec, self.loaders, self.rng)

        batch_size = int(cmd.get("effective_batch_size"))
        mixer.reset_epoch(batch_size, epoch_idx=0)

        log_file = self._ensure_command_dirs(cmd_idx, cmd)
        logger = Logger(experiment_name=self.cfg.get("experiment_name"), path_to_logging_dir=os.path.dirname(log_file), path_to_logfile=os.path.basename(log_file), overwrite=True)

        batch_counter = 0
        while True:
            train_batch, val_batch = mixer.next_batch()
            # for tests mixers typically return (batch, None)
            batch = train_batch
            if batch is None:
                break
            batch_counter += 1
            try:
                X_df, y = self.feature_extractor.process_batch(batch)
                if getattr(X_df, "shape", (0, 0))[0] == 0:
                    continue
                X_proc = self.preprocessor.transform(X_df)
                y_pred = self.classifier_wrapper.predict(X_proc)
                logger.save_test_results(np.asarray(y), np.asarray(y_pred))
                # Also append a simple checkpoint summary line to the experiment checkpoints log
                summary = f"cmd={cmd_idx} name={cmd.get('name')} batch={batch_counter} test_lines={len(y)}"
                self._append_checkpoint(cmd_idx, cmd, summary)
            except Exception as e:
                logger.write_to_log(f"[ERROR] testing batch {batch_counter} failed: {e}")
                continue

        # Save preprocessing and model after test command as well (required)
        try:
            prep_path = Path(cmd["paths"]["preprocessing"])
            prep_path.mkdir(parents=True, exist_ok=True)
            self.preprocessor.save(base_path=str(prep_path))
        except Exception as e:
            logger.write_to_log(f"[ERROR] saving preprocessing after test failed: {e}")

        try:
            model_path = Path(cmd["paths"]["model"])
            model_path.mkdir(parents=True, exist_ok=True)
            model_spec = self.cfg_reader.get_model_spec()
            save_name = model_spec.get("save_name", "classifier.bin")
            self.classifier_wrapper.save_classifier(path=str(model_path), name=save_name)
        except Exception as e:
            logger.write_to_log(f"[ERROR] saving classifier after test failed: {e}")

    # ---------------------------
    # orchestration
    # ---------------------------
    def run(self):
        # prepare experiment directory and save effective config early
        self._prepare_experiment_dirs()

        # build components (loaders, feature extraction, preprocessor, classifier)
        self._build_loaders()
        self._build_feature_extractor()
        self._build_preprocessor()
        self._build_classifier_wrapper()

        # execute commands
        commands = self.cfg_reader.get_commands()
        for idx, cmd in enumerate(commands):
            # create cmd dirs and store used command config
            Path(cmd["paths"]["command_dir"]).mkdir(parents=True, exist_ok=True)
            self._save_command_config(idx, cmd)

            what = cmd.get("command")
            if what == "train":
                self._run_train_command(idx, cmd)
            elif what == "test":
                self._run_test_command(idx, cmd)
            else:
                log_file = self._ensure_command_dirs(idx, cmd)
                logger = Logger(experiment_name=self.cfg.get("experiment_name"), path_to_logging_dir=os.path.dirname(log_file), path_to_logfile=os.path.basename(log_file), overwrite=True)
                logger.write_to_log(f"[WARN] Unknown command '{what}' - skipping")
                continue

        return True



def _cli_main(argv):
    if len(argv) < 2:
        print("Usage: python -m pipeline_ml_training.runner ")
        return 2
    path = "."
    runner = PipelineRunner(path)
    try:
        runner.run()
    except Exception as e:
        print("Pipeline run failed:", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main(sys.argv))
