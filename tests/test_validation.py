from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from backend.validation import ValidationError, validate_payload


SCHEMA = json.loads(
    (Path(__file__).parents[1] / "models" / "input_schema.json").read_text(
        encoding="utf-8"
    )
)


def valid_payload() -> dict[str, object]:
    return {name: field["default"] for name, field in SCHEMA["fields"].items()}


def assert_validation_error(
    payload: object, message: str, *, field: str | None = None
) -> dict[str, object]:
    with pytest.raises(ValidationError) as caught:
        validate_payload(payload, SCHEMA)
    assert caught.value.detail["message"] == message
    if field is not None:
        assert caught.value.detail["field"] == field
    return caught.value.detail


def test_valid_payload_is_returned_in_expected_feature_order() -> None:
    payload = valid_payload()

    cleaned = validate_payload(payload, SCHEMA)

    assert list(cleaned) == SCHEMA["expected_features"]
    assert cleaned == payload


@pytest.mark.parametrize("payload", [None, [], "not an object", 42])
def test_non_object_payload_is_rejected(payload: object) -> None:
    assert_validation_error(payload, "Request body must be a JSON object")


def test_empty_payload_is_rejected() -> None:
    assert_validation_error({}, "Request body must not be empty")


def test_missing_field_is_rejected_even_when_field_is_nullable() -> None:
    payload = valid_payload()
    payload.pop("loan_amount")

    detail = assert_validation_error(payload, "Missing required fields")

    assert detail["fields"] == ["loan_amount"]


def test_unexpected_field_is_rejected() -> None:
    payload = valid_payload()
    payload["unexpected"] = "value"

    detail = assert_validation_error(payload, "Unexpected fields")

    assert detail["fields"] == ["unexpected"]


@pytest.mark.parametrize("field", ["term", "loan_limit"])
def test_nullable_field_accepts_null(field: str) -> None:
    payload = valid_payload()
    payload[field] = None

    cleaned = validate_payload(payload, SCHEMA)

    assert cleaned[field] is None


@pytest.mark.parametrize("field", ["loan_amount", "Gender"])
def test_required_field_rejects_null(field: str) -> None:
    payload = valid_payload()
    payload[field] = None

    assert_validation_error(payload, f"{field} may not be null", field=field)


@pytest.mark.parametrize("value", [True, False, "296500", math.nan, math.inf, -math.inf])
def test_invalid_numeric_type_or_non_finite_value_is_rejected(value: object) -> None:
    payload = valid_payload()
    payload["loan_amount"] = value

    assert_validation_error(
        payload, "Invalid numeric value for loan_amount", field="loan_amount"
    )


def test_invalid_category_lists_actual_fitted_values() -> None:
    payload = valid_payload()
    payload["loan_type"] = "invented-type"

    detail = assert_validation_error(
        payload, "Invalid value for loan_type", field="loan_type"
    )

    assert detail["allowed_values"] == ["type1", "type2", "type3"]


def test_fitted_sex_not_available_category_is_preserved() -> None:
    payload = valid_payload()
    payload["Gender"] = "Sex Not Available"

    cleaned = validate_payload(payload, SCHEMA)

    assert cleaned["Gender"] == "Sex Not Available"


@pytest.mark.parametrize("field", ["loan_amount", "term"])
@pytest.mark.parametrize("value", [0, -1])
def test_positive_fields_reject_zero_and_negative_values(
    field: str, value: int
) -> None:
    payload = valid_payload()
    payload[field] = value

    assert_validation_error(
        payload, f"{field} must be greater than zero", field=field
    )


@pytest.mark.parametrize(
    "field", ["income", "property_value", "Credit_Score", "LTV", "dtir1"]
)
def test_non_negative_fields_reject_negative_values(field: str) -> None:
    payload = valid_payload()
    payload[field] = -0.01

    assert_validation_error(
        payload, f"{field} must not be negative", field=field
    )


def test_extreme_but_finite_positive_value_is_accepted() -> None:
    payload = valid_payload()
    payload["income"] = 1_000_000_000

    cleaned = validate_payload(payload, SCHEMA)

    assert cleaned["income"] == 1_000_000_000
