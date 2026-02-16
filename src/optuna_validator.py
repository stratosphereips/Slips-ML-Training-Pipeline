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

    def visit(node: Any, path: List[str]):
        if _is_param_spec(node):
            spec_type = node.get("type")
            allowed_keys = set(_SPEC_ALLOWED_KEYS.get(spec_type, _COMMON_SPEC_KEYS))
            extra_keys = [key for key in node.keys() if key not in allowed_keys]
            if extra_keys:
                errors.append(
                    f"{_path_label(path)}: unexpected keys {sorted(extra_keys)} for spec type '{spec_type}'"
                )
            if spec_type == "categorical" and "choices" not in node:
                errors.append(f"{_path_label(path)}: categorical spec requires 'choices'")
            if spec_type in {"int", "float"}:
                for required in ("low", "high"):
                    if required not in node:
                        errors.append(f"{_path_label(path)}: {spec_type} spec requires '{required}'")
            if spec_type == "constant" and "value" not in node:
                errors.append(f"{_path_label(path)}: constant spec requires 'value'")
            if spec_type == "group":
                params = node.get("params")
                if not isinstance(params, dict):
                    errors.append(f"{_path_label(path)}: group spec requires 'params' mapping")
                else:
                    for child_name, child_spec in params.items():
                        visit(child_spec, path + [child_name])
            return

        if isinstance(node, dict):
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
    if not isinstance(node, dict):
        return False
    spec_type = node.get("type")
    return isinstance(spec_type, str)


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
