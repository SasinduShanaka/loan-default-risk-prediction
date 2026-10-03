from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app, service


def test_valid_prediction_matches_direct_pipeline_probabilities(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    result = response.json()
    direct_frame = pd.DataFrame([valid_payload], columns=service.expected_features)
    direct_prediction = service.pipeline.predict(direct_frame)[0].item()
    direct_probabilities = service.pipeline.predict_proba(direct_frame)[0]

    assert result["predicted_class"] == direct_prediction
    assert result["predicted_class"] in service.classes
    assert result["model"] == "XGBoost"
    for class_value, probability in zip(service.classes, direct_probabilities):
        key = f"probability_class_{class_value}"
        assert 0 <= result[key] <= 1
        assert result[key] == pytest.approx(float(probability))


def test_repeated_prediction_is_consistent(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    first = client.post("/predict", json=valid_payload)
    second = client.post("/predict", json=valid_payload)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_nullable_numeric_field_reaches_pipeline(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    valid_payload["dtir1"] = None

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["predicted_class"] in service.classes


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.clear(), "Request body must not be empty"),
        (lambda payload: payload.pop("loan_amount"), "Missing required fields"),
        (lambda payload: payload.update({"extra": 1}), "Unexpected fields"),
        (
            lambda payload: payload.update({"loan_amount": "100000"}),
            "Invalid numeric value for loan_amount",
        ),
        (
            lambda payload: payload.update({"loan_type": "invalid"}),
            "Invalid value for loan_type",
        ),
        (
            lambda payload: payload.update({"loan_amount": -1}),
            "loan_amount must be greater than zero",
        ),
    ],
)
def test_invalid_request_returns_structured_422(
    client: TestClient,
    valid_payload: dict[str, Any],
    mutate: Any,
    message: str,
) -> None:
    mutate(valid_payload)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert response.json()["detail"]["message"] == message


def test_unexpected_inference_failure_returns_sanitized_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_prediction(payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("C:\\secret\\model-path: internal details")

    monkeypatch.setattr(service, "predict", fail_prediction)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/predict",
            json={
                name: field["default"]
                for name, field in service.metadata["fields"].items()
            },
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed unexpectedly"}
    assert "secret" not in response.text
