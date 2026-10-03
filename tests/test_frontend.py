from __future__ import annotations

from typing import Any

import pytest
import requests

from frontend.app import (
    FrontendServiceError,
    fetch_metadata,
    friendly_label,
    group_for_field,
    submit_prediction,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def complete_metadata() -> dict[str, Any]:
    return {
        "selected_model": "XGBoost",
        "target": "Status",
        "expected_features": ["loan_amount", "loan_type"],
        "numeric_features": ["loan_amount"],
        "categorical_features": ["loan_type"],
        "fields": {
            "loan_amount": {
                "type": "number",
                "nullable": True,
                "default": 296500.0,
            },
            "loan_type": {
                "type": "category",
                "nullable": True,
                "allowed_values": ["type1", "type2", "type3"],
                "default": "type1",
            },
        },
    }


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("loan_amount", "Loan Amount"),
        ("property_value", "Property Value"),
        ("Credit_Score", "Credit Score"),
        ("dtir1", "Debt-to-Income Ratio (dtir1)"),
        ("Gender", "Gender"),
    ],
)
def test_friendly_label_uses_readable_presentation_only_names(
    name: str, label: str
) -> None:
    assert friendly_label(name) == label


def test_unknown_field_is_grouped_conservatively() -> None:
    assert group_for_field("unfamiliar_field") == "Additional Information"
    assert group_for_field("loan_amount") == "Loan Information"
    assert group_for_field("Credit_Score") == "Credit Information"


def test_fetch_metadata_uses_normalized_url_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_get(url: str, *, timeout: float) -> FakeResponse:
        observed.update(url=url, timeout=timeout)
        return FakeResponse(200, complete_metadata())

    monkeypatch.setattr(requests, "get", fake_get)

    metadata = fetch_metadata("http://127.0.0.1:8000/", timeout=3.5)

    assert metadata == complete_metadata()
    assert observed == {"url": "http://127.0.0.1:8000/metadata", "timeout": 3.5}


def test_backend_connection_error_becomes_user_facing_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get(url: str, *, timeout: float) -> FakeResponse:
        raise requests.ConnectionError("socket details")

    monkeypatch.setattr(requests, "get", fail_get)

    with pytest.raises(
        FrontendServiceError,
        match="Could not connect to the prediction service",
    ):
        fetch_metadata("http://127.0.0.1:8000")


def test_structured_validation_error_is_preserved_for_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(
        url: str, *, json: dict[str, Any], timeout: float
    ) -> FakeResponse:
        return FakeResponse(
            422,
            {
                "detail": {
                    "message": "Invalid value for loan_type",
                    "allowed_values": ["type1", "type2", "type3"],
                }
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(FrontendServiceError) as caught:
        submit_prediction("http://127.0.0.1:8000", {"loan_type": "bad"})

    assert str(caught.value) == (
        "Invalid value for loan_type. Allowed values: type1, type2, type3"
    )


def test_submit_prediction_preserves_null_and_uses_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_post(
        url: str, *, json: dict[str, Any], timeout: float
    ) -> FakeResponse:
        observed.update(url=url, json=json, timeout=timeout)
        return FakeResponse(
            200,
            {
                "predicted_class": 1,
                "probability_class_0": 0.2,
                "probability_class_1": 0.8,
                "model": "XGBoost",
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)
    payload = {"loan_amount": None, "loan_type": "type1"}

    result = submit_prediction(
        "http://127.0.0.1:8000/", payload, timeout=7.5
    )

    assert result["predicted_class"] == 1
    assert observed == {
        "url": "http://127.0.0.1:8000/predict",
        "json": payload,
        "timeout": 7.5,
    }


def test_malformed_success_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(
        url: str, *, json: dict[str, Any], timeout: float
    ) -> FakeResponse:
        return FakeResponse(200, {"model": "XGBoost"})

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(
        FrontendServiceError, match="malformed response"
    ):
        submit_prediction("http://127.0.0.1:8000", {"loan_amount": 1})
