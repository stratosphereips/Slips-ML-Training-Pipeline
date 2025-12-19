import types
import sys
import pytest
from unittest.mock import MagicMock

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
    import types, sys
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
    from types import ModuleType

    # check it's an instance
    cls_obj = getattr(sys.modules[mod_name], cls_name)
    assert isinstance(out["model"], cls_obj)
    assert out["model"].kwargs["a"] == 1
    assert out is not params  # ensure new dict returned



# -------------------------
# get_wrapper_class
# -------------------------
def test_get_wrapper_class_invalid_name(monkeypatch):
    fake_mod = types.ModuleType("pipeline_ml_training.classifier_wrapper")
    monkeypatch.setitem(sys.modules, "pipeline_ml_training.classifier_wrapper", fake_mod)
    with pytest.raises(ValueError):
        mm.get_wrapper_class("NoWrapper")



def test_get_mixer_class_unknown(monkeypatch):
    fake_mixers = types.ModuleType("pipeline_ml_training.mixers")
    monkeypatch.setitem(sys.modules, "pipeline_ml_training.mixers", fake_mixers)
    with pytest.raises(ValueError):
        mm.get_mixer_class("nope")


def test_get_mixer_class_dotted(fake_module):
    mod_name, cls_name, _ = fake_module
    cls = mm.get_mixer_class(f"{mod_name}.{cls_name}")
    assert cls.__name__ == cls_name
