"""Validation helpers for Optuna hyperparameter search spaces."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


class OptunaConfigError(ValueError):
    """Raised when the Optuna hyperparameter section is malformed."""


_COMMON_SPEC_KEYS = {"type", "name", "when"}
_SPEC_ALLOWED_KEYS: Dict[str, Iterable[str]] = {
    "int": _COMMON_SPEC_KEYS | {"low", "high", "step", "log"},
    "float": _COMMON_SPEC_KEYS | {"low", "high", "step", "log"},
    "categorical": _COMMON_SPEC_KEYS | {"choices", "multiple", "min", "max"},
    "constant": _COMMON_SPEC_KEYS | {"value"},
    "group": _COMMON_SPEC_KEYS | {"params"},
}


def validate_optuna_search_space(search_space: Any) -> None:
    """Raise OptunaConfigError when the search space contains structural issues."""

    errors: List[str] = []

    def check_spec(node: Dict[str, Any], path: List[str], in_group: bool = False):
        label = _path_label(path)
        spec_type = node.get("type")
        allowed_keys = set(_SPEC_ALLOWED_KEYS[spec_type])
        extra_keys = [key for key in node.keys() if key not in allowed_keys]
        if extra_keys:
            errors.append(f"{label}: unexpected keys {sorted(extra_keys)} for spec type '{spec_type}'")
        if "when" in node:
            if in_group:
                errors.append(f"{label}: 'when' is not supported inside a group (guard the group instead)")
            elif not isinstance(node["when"], dict) or not node["when"]:
                errors.append(f"{label}: 'when' must be a non-empty mapping of path -> value(s)")
        if spec_type == "categorical":
            choices = node.get("choices")
            if not isinstance(choices, list) or not choices:
                errors.append(f"{label}: categorical spec requires a non-empty 'choices' list")
            elif node.get("multiple"):
                if not all(isinstance(c, str) and "|" not in c for c in choices):
                    errors.append(f"{label}: 'multiple' choices must be strings without '|'")
                lo, hi = node.get("min", 1), node.get("max", len(choices))
                if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi):
                    errors.append(f"{label}: 'multiple' needs integer 1 <= min <= max (got min={lo}, max={hi})")
        if spec_type in {"int", "float"}:
            for required in ("low", "high"):
                if required not in node:
                    errors.append(f"{label}: {spec_type} spec requires '{required}'")
            low, high = node.get("low"), node.get("high")
            if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low > high:
                errors.append(f"{label}: low ({low}) must not exceed high ({high})")
            if node.get("log") and node.get("step") not in (None, 1):
                errors.append(f"{label}: 'log' and 'step' cannot be combined")
        if spec_type == "constant" and "value" not in node:
            errors.append(f"{label}: constant spec requires 'value'")
        if spec_type == "group":
            params = node.get("params")
            if not isinstance(params, dict):
                errors.append(f"{label}: group spec requires 'params' mapping")
            else:
                for child_name, child_spec in params.items():
                    if not _is_param_spec(child_spec):
                        errors.append(f"{_path_label(path + [child_name])}: group params must be specs")
                    else:
                        check_spec(child_spec, path + [child_name], in_group=True)

    def visit(node: Any, path: List[str]):
        if _is_param_spec(node):
            check_spec(node, path)
            return

        if isinstance(node, dict):
            spec_type = node.get("type")
            has_children = any(isinstance(v, (dict, list)) for v in node.values())
            if isinstance(spec_type, str) and not has_children:
                # a leaf with an unknown spec type is almost certainly a typo (e.g. "integer")
                errors.append(
                    f"{_path_label(path)}: unknown spec type '{spec_type}' "
                    f"(expected one of {sorted(_SPEC_ALLOWED_KEYS)})"
                )
                return
            for key, child in node.items():
                if key == "when":
                    continue
                visit(child, path + [str(key)])
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                visit(child, path + [f"[{idx}]"])

    visit(search_space, [])

    if errors:
        raise OptunaConfigError(
            "Optuna hyperparameter definition has structural issues:\n" + "\n".join(f" - {err}" for err in errors)
        )


def _is_param_spec(node: Any) -> bool:
    """Same rule as OptunaOptimizer._is_param_spec: a dict whose 'type' is a known spec type."""
    if not isinstance(node, dict):
        return False
    spec_type = node.get("type")
    return isinstance(spec_type, str) and spec_type in _SPEC_ALLOWED_KEYS


def _path_label(tokens: Sequence[str]) -> str:
    if not tokens:
        return "root"
    label = []
    for token in tokens:
        if token.startswith("["):
            label.append(token)
        elif label:
            label.append(f".{token}")
        else:
            label.append(token)
    return "".join(label)
