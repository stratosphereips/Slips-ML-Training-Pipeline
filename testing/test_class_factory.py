import types
import sys
import pytest

from src import class_factory as mm

# -------------------------
# helpers: simulate modules
# -------------------------
@pytest.fixture
def fake_module(tmp_path):
    mod_name = "fake_module"
    cls_name = "FakeClass"
    mod = types.ModuleType(mod_name)

    # class that can accept any kwargs
    class FakeClass:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    setattr(mod, cls_name, FakeClass)
    sys.modules[mod_name] = mod
    yield mod_name, cls_name, mod
    del sys.modules[mod_name]



# -------------------------
# _import_from_path
# -------------------------
def test_import_from_path_success(fake_module):
    mod_name, cls_name, _ = fake_module
    path = f"{mod_name}.{cls_name}"
    cls = mm._import_from_path(path)
    assert cls.__name__ == cls_name


def test_import_from_path_invalid_module():
    with pytest.raises(ImportError):
        mm._import_from_path("nonexistent.module.Class")


def test_import_from_path_invalid_attr(fake_module):
    mod_name, _, _ = fake_module
    with pytest.raises(ImportError):
        mm._import_from_path(f"{mod_name}.NoSuchClass")


# -------------------------
# _find_in_modules
# -------------------------
def test_find_in_modules_finds_class(fake_module):
    mod_name, cls_name, _ = fake_module
    cls = mm._find_in_modules(cls_name, [mod_name])
    assert cls.__name__ == cls_name


def test_find_in_modules_returns_none():
    cls = mm._find_in_modules("NoClass", ["math", "sys"])
    assert cls is None


# -------------------------
# get_transformer_class
# -------------------------
def test_get_transformer_class_dotted(fake_module):
    mod_name, cls_name, _ = fake_module
    cls = mm.get_transformer_class(f"{mod_name}.{cls_name}")
    assert cls.__name__ == cls_name


def test_get_transformer_class_invalid():
    with pytest.raises(ValueError):
        mm.get_transformer_class("NonExistentTransformer")


# -------------------------
# get_classifier_class
# -------------------------
def test_get_classifier_class_dotted(fake_module):
    mod_name, cls_name, _ = fake_module
    cls = mm.get_classifier_class(f"{mod_name}.{cls_name}")
    assert cls.__name__ == cls_name


def test_get_classifier_class_xgb(monkeypatch):
    import types
    import sys
    fake_xgb = types.ModuleType("xgboost")

    class XGBClassifier:
        pass

    setattr(fake_xgb, "XGBClassifier", XGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgb)

    cls = mm.get_classifier_class("xgbclassifier")
    assert cls is XGBClassifier



def test_get_classifier_class_not_found():
    with pytest.raises(ValueError):
        mm.get_classifier_class("NoSuchClassifier")


# -------------------------
# prepare_river_nested_model_params
# -------------------------
def test_prepare_river_nested_model_params_noop():
    assert mm.prepare_river_nested_model_params(None) is None
    assert mm.prepare_river_nested_model_params({}) == {}


def test_prepare_river_nested_model_params_missing_type():
    with pytest.raises(ValueError):
        mm.prepare_river_nested_model_params({"model": {"params": {}}})


def test_prepare_river_nested_model_params_instantiate(fake_module):
    mod_name, cls_name, _ = fake_module
    params = {"model": {"type": f"{mod_name}.{cls_name}", "params": {"a": 1}}}
    out = mm.prepare_river_nested_model_params(params)

    # check it's an instance
    cls_obj = getattr(sys.modules[mod_name], cls_name)
    assert isinstance(out["model"], cls_obj)
    assert out["model"].kwargs["a"] == 1
    assert out is not params  # ensure new dict returned



# -------------------------
# get_wrapper_class
# -------------------------
def test_get_wrapper_class_invalid_name():
    with pytest.raises(ValueError):
        mm.get_wrapper_class("NoWrapper")


def test_get_mixer_class_unknown():
    with pytest.raises(ValueError):
        mm.get_mixer_class("nope")


def test_get_mixer_class_dotted(fake_module):
    mod_name, cls_name, _ = fake_module
    cls = mm.get_mixer_class(f"{mod_name}.{cls_name}")
    assert cls.__name__ == cls_name


def test_prepare_classifier_params_noop_for_non_river():
    class Dummy:
        __module__ = "sklearn.linear_model"

    params = {"metric": "accuracy"}
    out = mm.prepare_classifier_params(Dummy, params)
    assert out == params
    assert out is not params


def test_prepare_classifier_params_resolves_metric(monkeypatch):
    import types

    metrics_mod = types.ModuleType("river.metrics")

    class Accuracy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    setattr(metrics_mod, "Accuracy", Accuracy)

    river_mod = types.ModuleType("river")
    river_mod.metrics = metrics_mod

    monkeypatch.setitem(sys.modules, "river", river_mod)
    monkeypatch.setitem(sys.modules, "river.metrics", metrics_mod)

    class Dummy:
        __module__ = "river.testing"

    params = {"metric": "accuracy"}
    out = mm.prepare_classifier_params(Dummy, params)
    assert isinstance(out["metric"], Accuracy)


def test_prepare_classifier_params_knn_uses_default_engine():
    from river.neighbors import KNNClassifier

    params = {"n_neighbors": 3}
    out = mm.prepare_classifier_params(KNNClassifier, params)
    assert out == {"n_neighbors": 3}
    assert out is not params
    model = KNNClassifier(**out)
    assert model.engine is None or type(model.engine).__name__ == "SWINN"


# -------------------------
# positive resolution paths (real libraries)
# -------------------------
@pytest.mark.parametrize(
    "name, expected",
    [
        ("SklearnClassifierWrapper", "SKLearnClassifierWrapper"),
        ("SKLearnClassifierWrapper", "SKLearnClassifierWrapper"),
        ("RiverClassifierWrapper", "RiverClassifierWrapper"),
    ],
)
def test_get_wrapper_class_resolves_builtin(name, expected):
    assert mm.get_wrapper_class(name).__name__ == expected


@pytest.mark.parametrize(
    "mixer_type, expected",
    [
        ("sequence", "SequenceMixer"),
        ("random", "RandomBatchesMixer"),
        ("balanced", "BalancedByLabelMixer"),
        ("oversampling", "BalancedByLabelWithOversamplingMixer"),
    ],
)
def test_get_mixer_class_resolves_builtin(mixer_type, expected):
    assert mm.get_mixer_class(mixer_type).__name__ == expected


@pytest.mark.parametrize(
    "name, module_prefix",
    [
        ("SGDClassifier", "sklearn.linear_model"),
        ("ARFClassifier", "river.forest"),
        ("HoeffdingTreeClassifier", "river.tree"),
        ("river.ensemble.ADWINBoostingClassifier", "river.ensemble"),
    ],
)
def test_get_classifier_class_short_and_dotted_names(name, module_prefix):
    cls = mm.get_classifier_class(name)
    assert cls.__module__.startswith(module_prefix)


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("f1", "F1"),
        ("kappa", "CohenKappa"),
        ("accuracy", "Accuracy"),
        ({"name": "precision"}, "Precision"),
    ],
)
def test_prepare_classifier_params_resolves_real_river_metric(spec, expected):
    from river.forest import ARFClassifier

    out = mm.prepare_classifier_params(ARFClassifier, {"metric": spec})
    assert type(out["metric"]).__name__ == expected


def test_prepare_classifier_params_unknown_river_metric():
    from river.forest import ARFClassifier

    with pytest.raises(ValueError):
        mm.prepare_classifier_params(ARFClassifier, {"metric": "no_such_metric"})


def test_prepare_river_nested_model_params_short_river_name():
    out = mm.prepare_river_nested_model_params(
        {"model": {"type": "HoeffdingTreeClassifier", "params": {"max_depth": 7}}}
    )
    assert type(out["model"]).__name__ == "HoeffdingTreeClassifier"
    assert out["model"].max_depth == 7


def test_prepare_river_nested_model_params_unknown_short_name():
    with pytest.raises(RuntimeError, match="not found in river modules"):
        mm.prepare_river_nested_model_params({"model": {"type": "NoSuchRiverModel"}})
