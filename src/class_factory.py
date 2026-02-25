"""
Exports:
 - get_transformer_class(type_name)
 - get_classifier_class(classifier_type)
 - prepare_river_nested_model_params(params)
 - prepare_classifier_params(classifier_cls, params)
 - get_wrapper_class(wrapper_name)
 - get_mixer_class(mixer_type)

Dotted-path import supported. Module search lists are intentionally small.

Notes:
 - Functions probe a small set of well-known modules (sklearn, river, xgboost)
     and will raise informative ValueError/RuntimeError when packages/classes are
     not found.
"""


from typing import Any
import importlib
from copy import deepcopy

# Central list of sklearn modules for classifier resolution
sklearn_modules = [
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.svm",
    "sklearn.naive_bayes",
    "sklearn.tree",
    # Add more as needed for future support
]

# Central list of river modules for classifier resolution (primary and nested)
river_modules = [
    "river.tree", #sgt tree, hoeffding tree, etc.
    "river.ensemble", #adwin boosting
    "river.linear_model",  # e.g., ALMAClassifier and other online linear models
    "river.forest", #ARF
    "river.neighbors", #KNN
    #Rest is for compatibility with nested models etc.
    "river.naive_bayes",
    "river.neural_net",
    "river.preprocessing",
    "river.compose",
    "river.base",
    # Add more as needed for future support
]


# -------------------------
# helpers
# -------------------------
def _import_from_path(path: str):
    """
    Import an attribute given a dotted path like "package.module.Name".
    Raises ImportError with helpful message when module or attribute missing.
    """
    parts = path.split(".")
    if len(parts) < 2:
        raise ImportError(f"dotted path expected, got: {path!r}")
    module = ".".join(parts[:-1])
    name = parts[-1]
    try:
        mod = importlib.import_module(module)
    except Exception as e:
        raise ImportError(f"Failed to import module '{module}' for path '{path}': {e}")
    if not hasattr(mod, name):
        raise ImportError(f"Module '{module}' does not define attribute '{name}'")
    return getattr(mod, name)


def _find_in_modules(name: str, modules):
    """
    Search for a symbol name in a list of module names.
    Tries exact name and capitalized variant (e.g. 'logisticregression' -> 'Logisticregression').
    Returns the found attribute or None.
    """
    candidates = [name]
    if name:
        candidates.append(name[0].upper() + name[1:])
    for m in modules:
        try:
            mod = importlib.import_module(m)
        except Exception:
            # library not installed or import error — skip safely
            continue
        for c in candidates:
            if hasattr(mod, c):
                return getattr(mod, c)
    return None


# -------------------------
# transformer mapper
# -------------------------
def get_transformer_class(type_name: str):
    """
    Resolve a transformer class by short name (e.g. 'StandardScaler') or dotted path.
    Tries a small set of sklearn modules.
    """
    if not type_name:
        raise ValueError("type_name required")

    # dotted path
    if "." in type_name:
        return _import_from_path(type_name)

    modules = [
        "sklearn.preprocessing",
        "sklearn.decomposition",
        "sklearn.impute",
    ]
    Cls = _find_in_modules(type_name, modules)
    if Cls is not None:
        return Cls
    raise ValueError(f"Transformer '{type_name}' not found (tried sklearn modules)")


# -------------------------
# classifier mapper
# -------------------------
def get_classifier_class(classifier_type: str):
    """
    Resolve a classifier class by short name or dotted path.
    Tries sklearn first, then river, and finally xgboost aliases.
    """
    if not classifier_type:
        raise ValueError("classifier_type required")

    # dotted path
    if "." in classifier_type:
        return _import_from_path(classifier_type)

    Cls = _find_in_modules(classifier_type, sklearn_modules)
    if Cls is not None:
        return Cls

    # try a central set of river modules (if installed)
    Cls = _find_in_modules(classifier_type, river_modules)
    if Cls is not None:
        return Cls

    # try xgboost common alias (only import if alias requested)
    try:
        low = classifier_type.lower()
        if low in ("xgbclassifier", "xgboost", "xgboostclassifier", "xgb"):
            from xgboost import XGBClassifier  # type: ignore

            return XGBClassifier

        # fall back to looking inside xgboost module for named classes
        Cls = _find_in_modules(classifier_type, ["xgboost"])
        if Cls is not None:
            return Cls
    except Exception:
        # xgboost not installed or import error — ignore and proceed to final error
        pass

    raise ValueError(f"Classifier '{classifier_type}' not found")


def prepare_river_nested_model_params(params: Any) -> Any:
    """
    If params is a dict containing {'model': {'type': ..., 'params': {...}}},
    instantiate the nested river model and return a NEW dict with 'model'
    replaced by the instantiated model.

    Returns the original params unchanged when no nested model spec is present.
    """
    if not isinstance(params, dict):
        return params
    nested = params.get("model")
    if not isinstance(nested, dict):
        return params
    inner_type = nested.get("type")
    inner_params = nested.get("params", {}) or {}
    if not inner_type:
        raise ValueError("Nested model spec missing 'type'")

    # try to resolve the inner class (dotted or via river search)
    try:
        if "." in inner_type:
            InnerCls = _import_from_path(inner_type)
        else:
            InnerCls = _find_in_modules(inner_type, river_modules)
            if InnerCls is None:
                # try dotted path fallback — may raise ImportError
                InnerCls = _import_from_path(inner_type)

        # instantiate and return a shallow copy with the instantiated model
        out = dict(params)
        out["model"] = InnerCls(**inner_params)
        return out
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate nested river model '{inner_type}': {e}"
        )


def prepare_classifier_params(classifier_cls, params: Any):
    """Return a parameter mapping ready for instantiating ``classifier_cls``."""
    if params is None:
        return params
    if isinstance(params, dict):
        prepared = dict(params)
    else:
        prepared = deepcopy(params)
    if classifier_cls is None or not isinstance(prepared, dict):
        return prepared
    module_name = getattr(classifier_cls, "__module__", "")
    if module_name.startswith("river.") and "metric" in prepared:
        prepared["metric"] = _resolve_river_metric(prepared["metric"])
    classifier_name = getattr(classifier_cls, "__name__", "").lower()
    targets_knn = module_name.startswith("river.neighbors") and (
        "knn_classifier" in module_name or classifier_name.endswith("knnclassifier")
    )
    if targets_knn:
        _ensure_knn_stable_optimizer(prepared)
    return prepared


def _resolve_river_metric(metric_spec):
    if metric_spec is None:
        return None
    if not isinstance(metric_spec, (str, dict)):
        return metric_spec
    try:
        from river import metrics as river_metrics
    except Exception as e:
        raise RuntimeError(
            "River metrics requested in classifier parameters but the 'river' package is unavailable"
        ) from e

    if isinstance(metric_spec, dict):
        metric_name = metric_spec.get("name") or metric_spec.get("type")
        metric_params = metric_spec.get("params", {}) or {}
    else:
        metric_name = metric_spec
        metric_params = {}

    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("Metric specification must include a non-empty string name")

    import inspect

    for candidate in _candidate_metric_class_names(metric_name):
        if not hasattr(river_metrics, candidate):
            continue
        MetricCls = getattr(river_metrics, candidate)
        if inspect.ismodule(MetricCls):
            # Try same-name attribute inside the module (common for grouped metrics)
            inner = getattr(MetricCls, candidate, None)
            if inner is None:
                continue
            MetricCls = inner
        if not inspect.isclass(MetricCls):
            continue
        return MetricCls(**metric_params)
    raise ValueError(f"Unknown river metric '{metric_name}'")


def _candidate_metric_class_names(raw_name: str):
    normalized = raw_name.replace("-", "_").replace(" ", "_").lower()
    pieces = [segment.capitalize() for segment in normalized.split("_") if segment]
    camel_case = "".join(pieces)
    candidates = [raw_name, camel_case, normalized.capitalize()]
    alias_map = {
        "f1": ["F1"],
        "precision": ["Precision"],
        "recall": ["Recall"],
        "accuracy": ["Accuracy"],
        "kappa": ["Kappa", "CohenKappa"],
        #add here more aliases if you want to use these metrics
    }
    candidates.extend(alias_map.get(normalized, []))
    # Preserve order but drop duplicates
    seen = set()
    ordered = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        ordered.append(cand)
    return ordered


def _ensure_knn_stable_optimizer(params: dict):
    """Fallback to a brute-force optimizer to avoid SWINN graph corruption."""
    if params.get("optimizer") is not None:
        return
    try:
        from river.neighbors import BruteForce  # Lazy import for optional dep
    except Exception:
        return
    params["optimizer"] = BruteForce()


# -------------------------
# wrapper mapper
# -------------------------
def get_wrapper_class(wrapper_name: str):
    """
    Resolve a local wrapper class by short name or dotted path.

    Expected local module: src.classifier_wrapper
    """
    if not wrapper_name:
        raise ValueError("wrapper_name required")
    # dotted path
    if "." in wrapper_name:
        return _import_from_path(wrapper_name)

    try:
        mod = importlib.import_module("src.classifier_wrapper")
    except Exception as e:
        raise RuntimeError(
            f"Failed to import local classifier_wrapper module: {e}"
        )

    wn = wrapper_name.lower()
    if "sklearn" in wn:
        if hasattr(mod, "SKLearnClassifierWrapper"):
            return getattr(mod, "SKLearnClassifierWrapper")
    if "river" in wn:
        if hasattr(mod, "RiverClassifierWrapper"):
            return getattr(mod, "RiverClassifierWrapper")

    # try direct attribute name
    if hasattr(mod, wrapper_name):
        return getattr(mod, wrapper_name)

    raise ValueError(f"Wrapper '{wrapper_name}' not found in local wrapper module")


# -------------------------
# mixer mapper
# -------------------------
def get_mixer_class(mixer_type: str):
    """
    Resolve a mixer class by short name or dotted path. Returns class object.
    """
    if not mixer_type:
        raise ValueError("mixer_type required")
    # dotted path
    if "." in mixer_type:
        return _import_from_path(mixer_type)
    try:
        mod = importlib.import_module("src.data_selectors")
    except Exception as e:
        raise RuntimeError(f"Failed to import built-in mixers module: {e}")

    mapping = {
        "sequence": "SequenceMixer",
        "random": "RandomBatchesMixer",
        "balanced": "BalancedByLabelMixer",
        "oversampling": "BalancedByLabelWithOversamplingMixer",
    }
    if mixer_type in mapping:
        name = mapping[mixer_type]
        if not hasattr(mod, name):
            raise RuntimeError(f"Mixers module missing expected class '{name}'")
        return getattr(mod, name)
    raise ValueError(f"Unknown mixer type '{mixer_type}'")
