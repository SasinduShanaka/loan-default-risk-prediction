"""Metadata-driven validation for raw model inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


class ValidationError(ValueError):
    """A user-correctable input error with an API-safe detail payload."""

    def __init__(self, detail: dict[str, Any]):
        self.detail = detail
        super().__init__(str(detail.get("message", "Invalid input")))


def _raise(message: str, **detail: Any) -> None:
    raise ValidationError({"message": message, **detail})


def validate_payload(
    payload: Any, schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate raw values without applying learned preprocessing."""

    if not isinstance(payload, Mapping):
        _raise("Request body must be a JSON object")
    if not payload:
        _raise("Request body must not be empty")

    expected_features = list(schema["expected_features"])
    expected_set = set(expected_features)
    supplied_set = set(payload)

    unexpected = sorted(supplied_set - expected_set)
    if unexpected:
        _raise("Unexpected fields", fields=unexpected)

    missing = [name for name in expected_features if name not in supplied_set]
    if missing:
        _raise("Missing required fields", fields=missing)

    cleaned: dict[str, Any] = {}
    fields = schema["fields"]
    for name in expected_features:
        value = payload[name]
        field = fields[name]

        if value is None:
            if not field.get("nullable", False):
                _raise(f"{name} may not be null", field=name)
            cleaned[name] = None
            continue

        if field["type"] == "number":
            if isinstance(value, bool) or not isinstance(value, Real):
                _raise(f"Invalid numeric value for {name}", field=name)
            if not math.isfinite(float(value)):
                _raise(f"Invalid numeric value for {name}", field=name)
        elif field["type"] == "category":
            allowed_values = field.get("allowed_values", [])
            if value not in allowed_values:
                _raise(
                    f"Invalid value for {name}",
                    field=name,
                    allowed_values=allowed_values,
                )
        else:
            raise RuntimeError(f"Unsupported schema type for {name}: {field['type']}")

        cleaned[name] = value

    for name in ("loan_amount", "term"):
        value = cleaned.get(name)
        if value is not None and value <= 0:
            _raise(f"{name} must be greater than zero", field=name)

    for name in ("income", "property_value", "Credit_Score", "LTV", "dtir1"):
        value = cleaned.get(name)
        if value is not None and value < 0:
            _raise(f"{name} must not be negative", field=name)

    return cleaned
