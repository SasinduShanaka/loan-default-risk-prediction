"""Streamlit client for the Loan Status Prediction API."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


class FrontendServiceError(RuntimeError):
    """An API or transport error that is safe to display to the user."""


def friendly_label(name: str) -> str:
    special_labels = {
        "Credit_Score": "Credit Score",
        "Credit_Worthiness": "Credit Worthiness",
        "LTV": "Loan-to-Value Ratio (LTV)",
        "dtir1": "Debt-to-Income Ratio (dtir1)",
        "Neg_ammortization": "Negative Amortization",
        "Secured_by": "Secured By",
        "Security_Type": "Security Type",
    }
    return special_labels.get(name, name.replace("_", " ").title())


def group_for_field(name: str) -> str:
    groups = {
        "Applicant Information": {
            "Gender",
            "age",
            "income",
            "co-applicant_credit_type",
        },
        "Loan Information": {
            "loan_limit",
            "approv_in_adv",
            "loan_type",
            "loan_purpose",
            "loan_amount",
            "term",
            "open_credit",
            "business_or_commercial",
            "Neg_ammortization",
            "interest_only",
            "lump_sum_payment",
            "submission_of_application",
        },
        "Credit Information": {
            "Credit_Worthiness",
            "credit_type",
            "Credit_Score",
            "dtir1",
        },
        "Property Information": {
            "property_value",
            "construction_type",
            "occupancy_type",
            "Secured_by",
            "total_units",
            "LTV",
            "Region",
            "Security_Type",
        },
    }
    return next(
        (group for group, names in groups.items() if name in names),
        "Additional Information",
    )


def _json_response(response: Any) -> Any:
    try:
        return response.json()
    except (ValueError, TypeError) as exc:
        raise FrontendServiceError(
            "The prediction service returned a malformed response."
        ) from exc


def _error_message(payload: Any, fallback: str) -> str:
    if not isinstance(payload, dict):
        return fallback
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if not isinstance(detail, dict):
        return fallback
    message = str(detail.get("message", fallback))
    allowed = detail.get("allowed_values")
    if isinstance(allowed, list) and allowed:
        message += ". Allowed values: " + ", ".join(map(str, allowed))
    fields = detail.get("fields")
    if isinstance(fields, list) and fields:
        message += ". Fields: " + ", ".join(map(str, fields))
    return message


def fetch_metadata(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/metadata", timeout=timeout)
    except requests.RequestException as exc:
        raise FrontendServiceError(
            "Could not connect to the prediction service. "
            "Please make sure the backend is running."
        ) from exc

    payload = _json_response(response)
    if response.status_code != 200:
        raise FrontendServiceError(
            _error_message(payload, "Could not load model metadata.")
        )
    required = {
        "selected_model",
        "expected_features",
        "numeric_features",
        "categorical_features",
        "fields",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise FrontendServiceError(
            "The prediction service returned malformed metadata."
        )
    return payload


def submit_prediction(
    base_url: str, payload: dict[str, Any], timeout: float = 10.0
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/predict", json=payload, timeout=timeout
        )
    except requests.RequestException as exc:
        raise FrontendServiceError(
            "Could not connect to the prediction service. "
            "Please make sure the backend is running."
        ) from exc

    response_payload = _json_response(response)
    if response.status_code != 200:
        raise FrontendServiceError(
            _error_message(response_payload, "Prediction request failed.")
        )
    if not isinstance(response_payload, dict) or "predicted_class" not in response_payload:
        raise FrontendServiceError(
            "The prediction service returned a malformed response."
        )
    return response_payload


def _render_field(name: str, field: dict[str, Any]) -> Any:
    label = friendly_label(name)
    use_null = False
    if field.get("nullable", False):
        use_null = st.checkbox(
            f"Leave {label} blank",
            key=f"null_{name}",
            help="The trained pipeline will handle this missing value.",
        )

    if field["type"] == "number":
        value = st.number_input(
            label,
            value=float(field.get("default", 0.0)),
            disabled=use_null,
            key=f"value_{name}",
        )
    else:
        options = list(field["allowed_values"])
        default = field.get("default")
        default_index = options.index(default) if default in options else 0
        value = st.selectbox(
            label,
            options=options,
            index=default_index,
            disabled=use_null,
            key=f"value_{name}",
        )
    return None if use_null else value


def _render_result(result: dict[str, Any]) -> None:
    st.subheader("Prediction Result")
    st.success("Prediction completed successfully.")
    class_one_probability = result.get("probability_class_1")
    class_zero_probability = result.get("probability_class_0")
    columns = st.columns(3)
    columns[0].metric("Predicted Class", str(result["predicted_class"]))
    if class_one_probability is not None:
        columns[1].metric(
            "Probability of Class 1", f"{float(class_one_probability):.2%}"
        )
        st.progress(min(max(float(class_one_probability), 0.0), 1.0))
    if class_zero_probability is not None:
        columns[2].metric(
            "Probability of Class 0", f"{float(class_zero_probability):.2%}"
        )


def run() -> None:
    st.set_page_config(
        page_title="Loan Status Prediction System",
        page_icon="🏦",
        layout="wide",
    )
    st.title("Loan Status Prediction System")
    st.caption("Machine-learning decision support for loan assessment.")
    st.write(
        "Enter applicant and loan details below. The backend sends the raw values "
        "through the existing trained preprocessing and XGBoost pipeline."
    )

    base_url = os.getenv("LOAN_API_URL", DEFAULT_BACKEND_URL)
    try:
        with st.spinner("Connecting to the prediction service..."):
            metadata = fetch_metadata(base_url)
    except FrontendServiceError as exc:
        st.error(str(exc))
        st.info(f"Expected backend address: {base_url}")
        st.stop()

    with st.sidebar:
        st.header("System Information")
        st.write("**System:** Loan Status Prediction")
        st.write(f"**Model:** {metadata['selected_model']}")
        st.write("**Task:** Binary Classification")
        st.caption(f"API: {base_url}")

    grouped_fields: dict[str, list[str]] = {}
    for name in metadata["expected_features"]:
        grouped_fields.setdefault(group_for_field(name), []).append(name)

    payload: dict[str, Any] = {}
    group_order = [
        "Applicant Information",
        "Loan Information",
        "Credit Information",
        "Property Information",
        "Additional Information",
    ]
    with st.form("loan_prediction_form"):
        for group in group_order:
            names = grouped_fields.get(group, [])
            if not names:
                continue
            st.subheader(group)
            columns = st.columns(2)
            for index, name in enumerate(names):
                with columns[index % 2]:
                    payload[name] = _render_field(name, metadata["fields"][name])
        submitted = st.form_submit_button(
            "Predict Loan Status", type="primary", use_container_width=True
        )

    if submitted:
        try:
            with st.spinner("Generating prediction..."):
                result = submit_prediction(base_url, payload)
            _render_result(result)
        except FrontendServiceError as exc:
            st.error(str(exc))

    st.info(
        "This prediction is intended to support decision-making and should not "
        "be used as the sole basis for a financial decision."
    )


if __name__ == "__main__":
    run()
