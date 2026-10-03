from fastapi.testclient import TestClient


def test_root_identifies_running_service(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "Loan Status Prediction API",
        "status": "running",
        "model": "XGBoost",
    }


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model": "XGBoost",
    }


def test_local_streamlit_origin_is_allowed(client: TestClient) -> None:
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
