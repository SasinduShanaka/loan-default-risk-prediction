from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.main import app, service


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        name: field["default"] for name, field in service.metadata["fields"].items()
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
