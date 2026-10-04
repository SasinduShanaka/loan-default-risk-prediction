from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from backend.config import MODEL_PATH, SCHEMA_PATH
from backend.model_service import (
    LoanPredictionService,
    derive_input_schema,
    inspect_pipeline,
    load_pipeline,
)


EXPECTED_FEATURES = [
    "loan_limit",
    "Gender",
    "approv_in_adv",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "open_credit",
    "business_or_commercial",
    "loan_amount",
    "term",
    "Neg_ammortization",
    "interest_only",
    "lump_sum_payment",
    "property_value",
    "construction_type",
    "occupancy_type",
    "Secured_by",
    "total_units",
    "income",
    "credit_type",
    "Credit_Score",
    "co-applicant_credit_type",
    "age",
    "submission_of_application",
    "LTV",
    "Region",
    "Security_Type",
    "dtir1",
]

NUMERIC_FEATURES = [
    "loan_amount",
    "term",
    "property_value",
    "income",
    "Credit_Score",
    "LTV",
    "dtir1",
]

CATEGORICAL_FEATURES = [
    "loan_limit",
    "Gender",
    "approv_in_adv",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "open_credit",
    "business_or_commercial",
    "Neg_ammortization",
    "interest_only",
    "lump_sum_payment",
    "construction_type",
    "occupancy_type",
    "Secured_by",
    "total_units",
    "credit_type",
    "co-applicant_credit_type",
    "age",
    "submission_of_application",
    "Region",
    "Security_Type",
]

NULLABLE_FEATURES = {
    "loan_limit",
    "approv_in_adv",
    "loan_purpose",
    "term",
    "Neg_ammortization",
    "property_value",
    "income",
    "age",
    "submission_of_application",
    "LTV",
    "dtir1",
}


def test_model_artifact_loads_as_expected_fitted_pipeline() -> None:
    assert MODEL_PATH.exists()
    assert MODEL_PATH.stat().st_size == 746_857

    pipeline = load_pipeline(MODEL_PATH)
    inspection = inspect_pipeline(pipeline)

    assert isinstance(pipeline, Pipeline)
    assert inspection.preprocessor_step == "preprocessor"
    assert inspection.model_step == "model"
    assert isinstance(pipeline.named_steps[inspection.model_step], XGBClassifier)
    assert inspection.classes == [0, 1]
    assert inspection.expected_features == EXPECTED_FEATURES
    assert inspection.numeric_features == NUMERIC_FEATURES
    assert inspection.categorical_features == CATEGORICAL_FEATURES
    assert inspection.categories["Gender"] == [
        "Female",
        "Joint",
        "Male",
        "Sex Not Available",
    ]
    assert inspection.categories["Security_Type"] == ["Indriect", "direct"]


def test_derived_schema_is_json_native_and_matches_pipeline() -> None:
    pipeline = load_pipeline(MODEL_PATH)

    schema = derive_input_schema(pipeline)

    assert schema["selected_model"] == "XGBoost"
    assert schema["target"] == "Status"
    assert schema["expected_features"] == EXPECTED_FEATURES
    assert schema["numeric_features"] == NUMERIC_FEATURES
    assert schema["categorical_features"] == CATEGORICAL_FEATURES
    assert list(schema["fields"]) == EXPECTED_FEATURES
    assert schema["fields"]["loan_amount"] == {
        "type": "number",
        "nullable": False,
        "default": 296500.0,
    }
    assert {
        name for name, field in schema["fields"].items() if field["nullable"]
    } == NULLABLE_FEATURES
    assert schema["fields"]["term"]["nullable"] is True
    assert schema["fields"]["Gender"]["nullable"] is False
    assert schema["fields"]["Gender"]["allowed_values"][-1] == "Sex Not Available"
    assert schema["fields"]["Gender"]["default"] == "Male"
    assert all(
        not isinstance(value, np.generic)
        for field in schema["fields"].values()
        for value in field.get("allowed_values", [])
    )
    json.dumps(schema)


def test_service_rejects_schema_with_wrong_feature_order(tmp_path: Path) -> None:
    schema = derive_input_schema(load_pipeline(MODEL_PATH))
    schema["expected_features"] = list(reversed(schema["expected_features"]))
    invalid_schema_path = tmp_path / "input_schema.json"
    invalid_schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(RuntimeError, match="feature order"):
        LoanPredictionService(MODEL_PATH, invalid_schema_path)


def test_service_rejects_schema_with_wrong_field_type(tmp_path: Path) -> None:
    schema = derive_input_schema(load_pipeline(MODEL_PATH))
    schema["fields"]["loan_type"]["type"] = "number"
    invalid_schema_path = tmp_path / "input_schema.json"
    invalid_schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(RuntimeError, match="type for loan_type"):
        LoanPredictionService(MODEL_PATH, invalid_schema_path)


def test_service_rejects_schema_with_wrong_nullability(tmp_path: Path) -> None:
    schema = derive_input_schema(load_pipeline(MODEL_PATH))
    schema["fields"]["Gender"]["nullable"] = True
    invalid_schema_path = tmp_path / "input_schema.json"
    invalid_schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(RuntimeError, match="nullable flag for Gender"):
        LoanPredictionService(MODEL_PATH, invalid_schema_path)


def test_service_normalizes_categorical_null_to_pipeline_missing_marker() -> None:
    service = LoanPredictionService(MODEL_PATH, SCHEMA_PATH)
    payload = {
        name: field["default"] for name, field in service.metadata["fields"].items()
    }
    payload["loan_limit"] = None
    direct_payload = dict(payload)
    direct_payload["loan_limit"] = np.nan

    result = service.predict(payload)
    direct_frame = pd.DataFrame([direct_payload], columns=service.expected_features)
    direct_probabilities = service.pipeline.predict_proba(direct_frame)[0]

    assert result["probability_class_0"] == pytest.approx(direct_probabilities[0])
    assert result["probability_class_1"] == pytest.approx(direct_probabilities[1])


def test_service_predicts_and_maps_probabilities_by_actual_classes() -> None:
    service = LoanPredictionService(MODEL_PATH, SCHEMA_PATH)
    payload = {
        name: (
            field["default"]
            if field["type"] == "number"
            else field["allowed_values"][0]
        )
        for name, field in service.metadata["fields"].items()
    }

    result = service.predict(payload)

    assert result["predicted_class"] in service.classes
    assert result["model"] == "XGBoost"
    assert set(result) == {
        "predicted_class",
        "probability_class_0",
        "probability_class_1",
        "model",
    }
    assert result["probability_class_0"] == pytest.approx(0.10565197467803955)
    assert result["probability_class_1"] == pytest.approx(0.8943480253219604)
    assert all(type(value) in {int, float, str} for value in result.values())
