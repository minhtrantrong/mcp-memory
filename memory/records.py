"""Normalisation for JSON columns that models sometimes fill with plain strings.

A model occasionally answers `entities` or `topics` with one comma-separated
string instead of a list. Storing that verbatim makes search and the dashboard
iterate over characters, so list-shaped columns are normalised on write and on
read.
"""

from __future__ import annotations

import json
from typing import Any


def as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return []
    return [text for text in (str(part).strip() for part in parts) if text]


def parse_text_list(raw: str) -> list[str]:
    try:
        return as_text_list(json.loads(raw))
    except ValueError:
        return []


def parse_json_list(raw: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except ValueError:
        return []
    if isinstance(value, str):
        return as_text_list(value)
    return value if isinstance(value, list) else []


def as_dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def as_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = value.replace("[", "").replace("]", "").split(",")
    if not isinstance(value, (list, tuple)):
        return []
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids
