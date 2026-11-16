"""English and Russian catalogs for the dashboards.

Data values (priority codes, feature names, failure-mode keys) stay in English.
Only display strings live here. ``bundle`` raises if the two languages diverge.
"""

from __future__ import annotations

import re
from typing import Literal

Lang = Literal["en", "ru"]
DEFAULT_LANG: Lang = "en"
_FIELD = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)")
_BUNDLES: dict[str, dict] | None = None


def _shape(value: object) -> object:
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    if isinstance(value, tuple):
        return [_shape(item) for item in value]
    return "str" if isinstance(value, str) else type(value).__name__


def _fields(value: object, path: str, found: dict[str, set[str]]) -> None:
    if isinstance(value, str):
        found[path] = set(_FIELD.findall(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            _fields(item, f"{path}.{key}" if path else key, found)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _fields(item, f"{path}[{index}]", found)


def _check(left: dict, right: dict) -> None:
    if _shape(left) != _shape(right):
        raise RuntimeError("English and Russian catalogs have different structures")
    en_fields: dict[str, set[str]] = {}
    ru_fields: dict[str, set[str]] = {}
    _fields(left, "", en_fields)
    _fields(right, "", ru_fields)
    mismatch = [
        path
        for path, fields in en_fields.items()
        if fields != ru_fields.get(path, set())
    ]
    if mismatch:
        raise RuntimeError(
            "Format placeholders differ between languages: " + ", ".join(mismatch[:12])
        )


def _load() -> dict[str, dict]:
    global _BUNDLES
    if _BUNDLES is None:
        from dashboard.locale import en, ru

        _check(en.CATALOG, ru.CATALOG)
        _BUNDLES = {"en": en.CATALOG, "ru": ru.CATALOG}
    return _BUNDLES


def bundle(lang: str) -> dict:
    """Catalog for ``lang`` (``en`` or ``ru``)."""
    try:
        return _load()[lang]
    except KeyError as exc:
        raise ValueError(f"Unsupported language: {lang}") from exc


def tr(lang: str, key: str, **kwargs: object) -> str:
    """Look up a dotted string key and format it.

    Parameters
    ----------
    lang : str
        ``en`` or ``ru``.
    key : str
        Dotted path, for example ``ui.brand``.
    **kwargs
        Values for ``str.format`` placeholders.

    Returns
    -------
    str
        Localized string.
    """
    node: object = bundle(lang)
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    if not isinstance(node, str):
        raise KeyError(key)
    return node.format(**kwargs) if kwargs else node
