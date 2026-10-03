from fastapi.testclient import TestClient

from backend.main import service


def test_metadata_returns_frontend_input_contract(client: TestClient) -> None:
    response = client.get("/metadata")

    assert response.status_code == 200
    metadata = response.json()
    assert set(metadata) == {
        "selected_model",
        "target",
        "expected_features",
        "numeric_features",
        "categorical_features",
        "fields",
    }
    assert metadata["selected_model"] == "XGBoost"
    assert metadata["target"] == "Status"
    assert metadata["expected_features"] == service.expected_features
    assert metadata["fields"]["Gender"]["allowed_values"] == [
        "Female",
        "Joint",
        "Male",
        "Sex Not Available",
    ]
