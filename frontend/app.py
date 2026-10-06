"""Streamlit client for the Loan Status Prediction API."""

from __future__ import annotations

from html import escape
import math
import numbers
import os
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

FIELD_GROUPS: dict[str, frozenset[str]] = {
    "Applicant Information": frozenset(
        {
            "Gender",
            "age",
            "income",
            "co-applicant_credit_type",
        }
    ),
    "Loan Information": frozenset(
        {
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
        }
    ),
    "Credit Information": frozenset(
        {
            "Credit_Worthiness",
            "credit_type",
            "Credit_Score",
            "dtir1",
        }
    ),
    "Property Information": frozenset(
        {
            "property_value",
            "construction_type",
            "occupancy_type",
            "Secured_by",
            "total_units",
            "LTV",
            "Region",
            "Security_Type",
        }
    ),
}

FALLBACK_GROUP = "Additional Information"
GROUP_ORDER: tuple[str, ...] = (*FIELD_GROUPS, FALLBACK_GROUP)
REVIEW_STEP = "Review & Predict"

GROUP_INTROS: dict[str, str] = {
    "Applicant Information": "Who is applying, and what they earn.",
    "Loan Information": "The loan being requested and how it is structured.",
    "Credit Information": "Credit standing as reported by the bureau.",
    "Property Information": "The property the loan is secured against.",
    FALLBACK_GROUP: "Remaining details required by the model.",
}

# Display order within a step, so related inputs sit next to each other
# instead of following the order the backend happens to list them in.
FIELD_ORDER: tuple[str, ...] = (
    "Gender",
    "age",
    "income",
    "co-applicant_credit_type",
    "loan_amount",
    "term",
    "loan_type",
    "loan_purpose",
    "loan_limit",
    "approv_in_adv",
    "business_or_commercial",
    "open_credit",
    "interest_only",
    "Neg_ammortization",
    "lump_sum_payment",
    "submission_of_application",
    "Credit_Score",
    "credit_type",
    "Credit_Worthiness",
    "dtir1",
    "property_value",
    "LTV",
    "occupancy_type",
    "construction_type",
    "Secured_by",
    "total_units",
    "Region",
    "Security_Type",
)

NOT_PROVIDED = "__not_provided__"
NOT_PROVIDED_LABEL = "Not provided"

# Four inputs per row keeps each step short, which leaves every field about a
# quarter of the page wide; these labels are shortened so they still fit on one
# or two lines there. The full name is kept for the review step and warnings.
COLUMNS_PER_ROW = 4
COMPACT_LABELS: dict[str, str] = {
    "LTV": "LTV",
    "dtir1": "Debt-to-Income",
    "approv_in_adv": "Pre-approved",
    "business_or_commercial": "Business/Commercial",
    "co-applicant_credit_type": "Co-applicant Bureau",
    "credit_type": "Credit Bureau",
    "Neg_ammortization": "Neg. Amortization",
    "construction_type": "Construction",
    "occupancy_type": "Occupancy",
}

# Units belong in the label; the raw feature names carry none.
LABEL_SUFFIXES: dict[str, str] = {
    "term": "(months)",
    "income": "(per month)",
    "LTV": "(%)",
    "dtir1": "(%)",
}

# Readable names for the dataset's abbreviated codes. Only codes whose meaning
# is established are translated -- loan_type, loan_purpose and
# Credit_Worthiness stay as-is rather than inventing a meaning for them.
VALUE_LABELS: dict[str, dict[str, str]] = {
    "loan_limit": {"cf": "Conforming", "ncf": "Non-conforming"},
    "Gender": {"Sex Not Available": "Other"},
    "approv_in_adv": {"pre": "Pre-approved", "nopre": "Not pre-approved"},
    "open_credit": {"opc": "Open credit", "nopc": "No open credit"},
    "business_or_commercial": {
        "b/c": "Business/commercial",
        "nob/c": "Not business/commercial",
    },
    "Neg_ammortization": {
        "neg_amm": "Negative amortization",
        "not_neg": "No negative amortization",
    },
    "interest_only": {
        "int_only": "Interest only",
        "not_int": "Principal and interest",
    },
    "lump_sum_payment": {
        "lpsm": "Lump-sum payment",
        "not_lpsm": "No lump-sum payment",
    },
    "construction_type": {"sb": "Site-built", "mh": "Manufactured home"},
    "occupancy_type": {
        "pr": "Principal residence",
        "sr": "Secondary residence",
        "ir": "Investment property",
    },
    "Secured_by": {"home": "Home", "land": "Land"},
    "total_units": {
        "1U": "1 unit",
        "2U": "2 units",
        "3U": "3 units",
        "4U": "4 units",
    },
    "credit_type": {
        "EXP": "Experian",
        "EQUI": "Equifax",
        "CRIF": "CRIF",
        "CIB": "Credit Information Bureau",
    },
    "co-applicant_credit_type": {
        "EXP": "Experian",
        "CIB": "Credit Information Bureau",
    },
    "submission_of_application": {
        "to_inst": "Submitted to institution",
        "not_inst": "Not submitted to institution",
    },
    "Security_Type": {"direct": "Direct", "Indriect": "Indirect"},
    "Region": {"central": "Central", "south": "South"},
    "age": {"<25": "Under 25", ">74": "75 and over"},
}

FIELD_HELP: dict[str, str] = {
    "loan_amount": "Principal being requested.",
    "term": "Loan term in months. 360 months is a 30-year loan.",
    "property_value": "Appraised value of the property.",
    "income": "Applicant income per month, as recorded in the training data.",
    "Credit_Score": "Score reported by the credit bureau.",
    "LTV": "Loan amount as a percentage of the property value.",
    "dtir1": "Monthly debt repayments as a percentage of income.",
    "loan_type": "Loan category from the training data; the codes are opaque.",
    "loan_purpose": "Purpose code from the training data; the codes are opaque.",
    "Credit_Worthiness": (
        "Bureau grade from the training data; the codes are opaque."
    ),
    "credit_type": "Bureau that supplied the applicant's credit score.",
    "co-applicant_credit_type": "Bureau that reported on the co-applicant.",
    "Region": "Region the property sits in.",
    "Security_Type": "Whether the security is held directly or indirectly.",
}

# Keep the custom chrome aligned with .streamlit/config.toml.
STYLES = """
<style>
:root {
    --loan-navy: #16232A;
    --loan-teal: #075056;
    --loan-orange: #FF5B04;
    --loan-ink: #16232A;
    --loan-muted: #52636A;
    --loan-line: #C9D7DA;
}
[data-testid="stAppViewContainer"] { background: #E4EEF0; }
[data-testid="stAppHeader"] { background: #E4EEF0; }
[data-testid="stMainMenu"] { display: none; }
[data-testid="stMainBlockContainer"] {
    max-width: 1320px;
    padding-top: 3.5rem;
    padding-bottom: 2.5rem;
}
.loan-brand {
    display: flex; align-items: center; gap: 0.7rem;
    color: var(--loan-navy); font-size: 0.85rem; font-weight: 700;
    letter-spacing: 0.08em; margin-bottom: 0.4rem;
}
.loan-brand-mark {
    display: grid; place-items: center; width: 34px; height: 34px;
    background: var(--loan-orange); color: var(--loan-navy); border-radius: 10px;
}
.loan-hero {
    background: linear-gradient(115deg, #16232A 0%, #12393E 62%, #075056 100%);
    padding: 2.4rem 2.6rem; border-radius: 18px;
    margin: 0.2rem 0 0.6rem; color: white;
    box-shadow: 0 8px 24px #16232A18;
}
.loan-eyebrow {
    color: #FF5B04; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
}
.loan-hero h1 {
    color: #FFFFFF; font-size: clamp(1.8rem, 3vw, 2.7rem);
    font-weight: 650; letter-spacing: -0.035em;
    line-height: 1.2; padding: 0.65rem 0 0.8rem;
}
.loan-hero p { color: #E4EEF0; max-width: 660px; margin: 0; line-height: 1.7; }
.loan-hero-meta {
    display: flex; gap: 1.6rem; flex-wrap: wrap; margin-top: 1.6rem;
    padding-top: 1.1rem; border-top: 1px solid #FFFFFF26;
    font-size: 0.78rem; color: #E4EEF0;
}
.loan-hero-meta span::before { content: "·"; color: #FF5B04; margin-right: 0.5rem; }
[data-testid="stCaptionContainer"] { color: var(--loan-muted); }
[data-testid="stHeading"] h3 {
    color: var(--loan-navy); font-size: 1.3rem;
    font-weight: 650; letter-spacing: -0.025em;
}
.st-key-assessment {
    background: #FFFFFF; border: 1px solid var(--loan-line);
    border-radius: 16px; padding: 1.8rem 2rem;
    box-shadow: 0 4px 20px #16232A0D;
}
[data-testid="stTextInput"] [data-testid="stWidgetLabel"],
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {
    min-height: 2.6rem; align-items: flex-start;
}
[data-testid="stWidgetLabel"] p { font-size: 0.82rem; font-weight: 600; color: var(--loan-ink); }
[data-testid="stTextInput"], [data-testid="stSelectbox"] { margin-bottom: 0.6rem; }
[data-testid="stTextInputRootElement"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #F8FBFC; border-color: #C9D7DA; border-radius: 8px;
    min-height: 44px;
}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color: var(--loan-orange); box-shadow: 0 0 0 3px #FF5B0426;
}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
    min-height: 44px; border-radius: 8px; transition: background 150ms ease;
}
[data-testid="stBaseButton-secondary"] {
    background: white; color: var(--loan-ink); border-color: var(--loan-line);
}
[data-testid="stBaseButton-secondary"]:hover {
    color: #075056; background: #EDF5F6; border-color: #7CA4A7;
}
[data-testid="stBaseButton-primary"]:not(:disabled):hover {
    background: #E64F00; border-color: #E64F00; color: #16232A;
}
[data-testid="stBaseButton-primary"] { color: #16232A; }
.st-key-nav_next [data-testid="stBaseButton-primary"],
.st-key-nav_next [data-testid="stBaseButton-primary"] p {
    color: #FFFFFF;
    font-weight: 700;
}
.st-key-nav_next button:disabled {
    background: #E4EAEC;
    border-color: #C9D7DA;
    opacity: 1;
    cursor: not-allowed;
}
.st-key-nav_next button:disabled p {
    color: #607078;
    font-weight: 700;
}
[data-testid="stButton"] button:focus-visible {
    outline: 3px solid #FF5B04; outline-offset: 3px;
}
/* The current step is disabled for navigation, but remains clearly selected. */
[class*="st-key-step_"] [data-testid="stBaseButton-primary"]:disabled {
    background: var(--loan-teal); color: #FFFFFF;
    border-color: var(--loan-teal); opacity: 1;
}
[class*="st-key-step_"] [data-testid="stBaseButton-secondary"] {
    background: #FFFFFF; color: var(--loan-teal);
    border-color: #7CA4A7;
}
[class*="st-key-step_"] [data-testid="stBaseButton-secondary"]:hover {
    background: #EDF5F6; color: var(--loan-teal);
    border-color: var(--loan-teal);
}
[class*="st-key-step_"] button { min-height: 58px; }
[class*="st-key-step_"] button p { font-size: 0.8rem; font-weight: 600; }
/* A continuous connector sits behind the native, keyboard-accessible steps. */
.st-key-stepper [data-testid="stHorizontalBlock"] { position: relative; }
.st-key-stepper [data-testid="stHorizontalBlock"]::before {
    content: ""; position: absolute; left: 1rem; right: 1rem; top: 29px;
    height: 2px; background: #9CB4B8; pointer-events: none;
}
.st-key-stepper [data-testid="stColumn"] { position: relative; z-index: 1; }
/* Predict is the one commitment on the page, so it is the one control that
   moves: it lifts under the cursor, presses in on click, and a single sheen
   crosses it per hover. Nothing loops -- idle motion on a submit button reads
   as a distraction rather than an affordance. */
.st-key-predict button {
    min-height: 46px; box-shadow: 0 4px 12px #FF5B0433;
    position: relative; overflow: hidden;
    transition: transform 160ms ease, box-shadow 160ms ease,
                background 150ms ease;
}
.st-key-predict button:not(:disabled):hover {
    transform: translateY(-2px); box-shadow: 0 9px 22px #FF5B044D;
}
/* The press needs to beat the hover lift, hence the shorter duration. */
.st-key-predict button:not(:disabled):active {
    transform: translateY(0); box-shadow: 0 2px 8px #FF5B0440;
    transition-duration: 70ms;
}
.st-key-predict button::after {
    content: ""; position: absolute; top: 0; bottom: 0; left: -60%; width: 45%;
    background: linear-gradient(90deg, transparent, #FFFFFF59, transparent);
    transform: skewX(-20deg); pointer-events: none;
}
.st-key-predict button:not(:disabled):hover::after {
    animation: loan-predict-sheen 720ms ease-out;
}
@keyframes loan-predict-sheen {
    from { left: -60%; }
    to { left: 125%; }
}
/* Matches the specificity of the primary-button hover rule above, which
   would otherwise repaint this label dark ink mid-hover. */
.st-key-predict [data-testid="stBaseButton-primary"]:not(:disabled),
.st-key-predict [data-testid="stBaseButton-primary"]:not(:disabled) p {
    color: #FFFFFF;
}
.st-key-predict button p { font-size: 0.9rem; font-weight: 700; }
.st-key-predict button:disabled { box-shadow: none; }
.loan-review-summary {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px; background: var(--loan-line); border: 1px solid var(--loan-line);
    border-radius: 10px; overflow: hidden; margin: 0.2rem 0 0.7rem;
}
.loan-review-stat { background: #F1F7F8; padding: 1rem 1.2rem; }
.loan-review-stat span {
    display: block; color: var(--loan-muted); font-size: 0.76rem; margin-bottom: 0.4rem;
}
.loan-review-stat strong {
    color: var(--loan-navy); font-size: 1.2rem; font-weight: 650;
    overflow-wrap: anywhere;
}
.loan-review-details { margin: 0; }
.loan-review-details > div {
    display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
    gap: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #DFEAEC;
    font-size: 0.8rem; line-height: 1.45;
}
.loan-review-details dt { color: var(--loan-muted); }
.loan-review-details dd {
    margin: 0; text-align: right; font-weight: 600; color: var(--loan-ink);
    overflow-wrap: anywhere;
}
.loan-result-card {
    padding: 1.4rem 1.5rem;
    border: 1px solid var(--loan-line);
    border-left-width: 6px;
    border-radius: 12px;
    margin: 0.35rem 0 1rem;
}
.loan-result--normal {
    background: #E5F1F2;
    border-left-color: var(--loan-teal);
}
.loan-result--failure {
    background: #FFF0E8;
    border-left-color: var(--loan-orange);
}
.loan-result-label,
.loan-probability-card span {
    color: var(--loan-muted);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.loan-result-card h3 {
    color: var(--loan-ink);
    font-size: clamp(1.35rem, 2.4vw, 2rem);
    line-height: 1.25;
    margin: 0.45rem 0 0;
    overflow-wrap: anywhere;
}
.loan-probability-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.25rem;
}
.loan-probability-card {
    background: #FFFFFF;
    border: 1px solid var(--loan-line);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
}
.loan-probability-card strong {
    display: block;
    color: var(--loan-ink);
    font-size: 1.8rem;
    margin-top: 0.35rem;
}
@media (max-width: 740px) {
    .st-key-stepper [data-testid="stHorizontalBlock"]::before { top: 22px; }
    .loan-review-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .loan-probability-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
    .st-key-stepper [data-testid="stHorizontalBlock"]::before {
        left: 50%; right: auto; top: 1rem; bottom: 1rem; width: 2px; height: auto;
    }
}
[data-testid="stProgress"] { margin-top: 0.2rem; }
[data-testid="stExpander"] {
    background: #FFFFFF; border-color: var(--loan-line); border-radius: 10px;
}
[data-testid="stMetric"] {
    background: #F1F7F8; border: 1px solid var(--loan-line);
    border-top: 3px solid var(--loan-teal); border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
[data-testid="stMetricLabel"] { color: var(--loan-muted); }
[data-testid="stMetricValue"] { color: var(--loan-navy); font-weight: 650; }
[data-testid="stAlert"] { border-radius: 10px; }
@media (max-width: 740px) {
    [data-testid="stMainBlockContainer"] { padding: 2.8rem 1rem 1.5rem; }
    .loan-hero { padding: 1.6rem; border-radius: 12px; }
    .loan-hero-meta { gap: 0.6rem 1rem; }
    .st-key-assessment { padding: 1.1rem; }
    [class*="st-key-step_"] button { min-height: 44px; }
}
@media (prefers-reduced-motion: reduce) {
    [data-testid="stButton"] button { transition: none; }
    .st-key-predict button:not(:disabled):hover,
    .st-key-predict button:not(:disabled):active { transform: none; }
    .st-key-predict button:not(:disabled):hover::after { animation: none; }
}
</style>
"""


class FrontendServiceError(RuntimeError):
    """An API or transport error that is safe to display to the user."""


def friendly_label(name: str) -> str:
    special_labels = {
        "approv_in_adv": "Approved in Advance",
        "business_or_commercial": "Business or Commercial",
        "co-applicant_credit_type": "Co-applicant Credit Type",
        "submission_of_application": "Application Submission",
        "Credit_Score": "Credit Score",
        "Credit_Worthiness": "Credit Worthiness",
        "LTV": "Loan-to-Value Ratio (LTV)",
        "dtir1": "Debt-to-Income Ratio (dtir1)",
        "Neg_ammortization": "Negative Amortization",
        "Secured_by": "Secured By",
        "Security_Type": "Security Type",
    }
    return special_labels.get(name, name.replace("_", " ").title())


def input_label(name: str) -> str:
    """The label shown above an input: compact, with its unit."""
    label = COMPACT_LABELS.get(name, friendly_label(name))
    suffix = LABEL_SUFFIXES.get(name)
    return f"{label} {suffix}" if suffix else label


def group_for_field(name: str) -> str:
    return next(
        (group for group, names in FIELD_GROUPS.items() if name in names),
        FALLBACK_GROUP,
    )


def sort_fields(names: list[str]) -> list[str]:
    """Order a step's fields for display; unlisted fields sort last."""
    positions = {name: index for index, name in enumerate(FIELD_ORDER)}
    return sorted(names, key=lambda name: (positions.get(name, len(positions)), name))


def columnize(names: list[str], columns: int = COLUMNS_PER_ROW) -> list[list[str]]:
    """Split a step's fields into balanced, column-major chunks.

    Column-major keeps the tab order running straight down each column rather
    than hopping back up the page.
    """
    base, extra = divmod(len(names), columns)
    chunks: list[list[str]] = []
    start = 0
    for index in range(columns):
        size = base + (1 if index < extra else 0)
        chunks.append(names[start : start + size])
        start += size
    return chunks


def build_steps(feature_names: list[str]) -> list[tuple[str, list[str]]]:
    """Group features into the ordered data-entry steps of the stepper."""
    grouped: dict[str, list[str]] = {}
    for name in feature_names:
        grouped.setdefault(group_for_field(name), []).append(name)
    return [
        (group, sort_fields(grouped[group]))
        for group in GROUP_ORDER
        if group in grouped
    ]


def step_titles(steps: list[tuple[str, list[str]]]) -> list[str]:
    """Every step label shown in the indicator, including the final review."""
    return [title for title, _ in steps] + [REVIEW_STEP]


def initial_answers(metadata: dict[str, Any]) -> dict[str, Any]:
    return {name: None for name in metadata["expected_features"]}


def missing_required(
    names: list[str], fields: dict[str, Any], answers: dict[str, Any]
) -> list[str]:
    """Fields the form user has left empty, regardless of model nullability."""
    return [name for name in names if answers.get(name) is None]


def parse_number(raw: str) -> tuple[float | None, bool]:
    """Read a typed number, tolerating thousands separators and a % sign.

    Returns the value and whether the text was acceptable. Empty text is
    acceptable and means "not provided".
    """
    text = raw.strip().replace(",", "").replace("_", "").removesuffix("%").strip()
    if not text:
        return None, True
    try:
        value = float(text)
    except ValueError:
        return None, False
    if not math.isfinite(value) or value < 0:
        return None, False
    return value, True


def calculate_ltv(loan_amount: Any, property_value: Any) -> float | None:
    """Calculate loan-to-value percentage when both source values are usable."""
    for value in (loan_amount, property_value):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
        ):
            return None
    if float(loan_amount) <= 0 or float(property_value) <= 0:
        return None
    return float(loan_amount) / float(property_value) * 100


def format_number_input(value: Any) -> str:
    """Seed a numeric text box without losing precision on a round trip."""
    if value is None:
        return ""
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.10f}".rstrip("0").rstrip(".")


def format_answer(value: Any) -> str:
    """Render a captured answer for the review step."""
    if value is None:
        return NOT_PROVIDED_LABEL
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            return str(value)
        return f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}"
    return str(value)


def display_value(name: str, value: Any) -> str:
    """The readable form of a stored code, used in dropdowns and the review."""
    if value is None or value == NOT_PROVIDED:
        return NOT_PROVIDED_LABEL
    return VALUE_LABELS.get(name, {}).get(value, format_answer(value))


def field_help(name: str, field: dict[str, Any]) -> str | None:
    """Tooltip text, listing the raw codes a dropdown will send."""
    described = FIELD_HELP.get(name)
    if field["type"] == "number":
        return described
    codes = ", ".join(str(code) for code in field.get("allowed_values", []))
    sends = f"Sends one of: {codes}." if codes else ""
    return " ".join(part for part in (described, sends) if part) or None


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


def _validate_metadata(payload: Any) -> dict[str, Any]:
    malformed = FrontendServiceError(
        "The prediction service returned malformed metadata."
    )
    required = {
        "selected_model",
        "expected_features",
        "numeric_features",
        "categorical_features",
        "fields",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise malformed

    expected = payload["expected_features"]
    numeric = payload["numeric_features"]
    categorical = payload["categorical_features"]
    fields = payload["fields"]
    if (
        not isinstance(payload["selected_model"], str)
        or not isinstance(expected, list)
        or not expected
        or not all(isinstance(name, str) for name in expected)
        or len(expected) != len(set(expected))
        or not isinstance(numeric, list)
        or not isinstance(categorical, list)
        or not all(isinstance(name, str) for name in numeric)
        or not all(isinstance(name, str) for name in categorical)
        or not set(numeric).isdisjoint(categorical)
        or set(numeric) | set(categorical) != set(expected)
        or not isinstance(fields, dict)
        or set(fields) != set(expected)
    ):
        raise malformed

    numeric_names = set(numeric)
    for name in expected:
        field = fields[name]
        expected_type = "number" if name in numeric_names else "category"
        if (
            not isinstance(field, dict)
            or field.get("type") != expected_type
            or not isinstance(field.get("nullable"), bool)
            or "default" not in field
        ):
            raise malformed
        default = field["default"]
        if expected_type == "number":
            if (
                isinstance(default, bool)
                or not isinstance(default, numbers.Real)
                or not math.isfinite(float(default))
            ):
                raise malformed
        else:
            allowed = field.get("allowed_values")
            if not isinstance(allowed, list) or not allowed or default not in allowed:
                raise malformed
    return payload


def _validate_prediction(payload: Any) -> dict[str, Any]:
    malformed = FrontendServiceError(
        "The prediction service returned a malformed response."
    )
    if not isinstance(payload, dict) or "predicted_class" not in payload:
        raise malformed
    predicted_class = payload["predicted_class"]
    if isinstance(predicted_class, (dict, list, bool)) or predicted_class is None:
        raise malformed
    if "model" in payload and not isinstance(payload["model"], str):
        raise malformed
    for key, value in payload.items():
        if not key.startswith("probability_class_"):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise malformed
    return payload


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
    return _validate_metadata(payload)


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
    return _validate_prediction(response_payload)


def _render_number_field(
    name: str, field: dict[str, Any], stored: Any
) -> tuple[Any, bool]:
    """A typed number box.

    st.number_input is deliberately avoided: its +/- steppers imply an
    increment that means nothing for an amount, a score or a ratio. A text
    input also lets the validation message explain when a required value was
    cleared or entered in an unreadable format.
    """
    key = f"value_{name}"
    if key not in st.session_state:
        st.session_state[key] = format_number_input(stored)
    raw = st.text_input(
        input_label(name),
        key=key,
        placeholder="Required",
        help=field_help(name, field),
    )
    value, accepted = parse_number(raw)
    if not accepted:
        st.caption(":red[Enter a positive number, for example 296500.]")
        return stored, False
    return value, True


def _render_category_field(
    name: str, field: dict[str, Any], stored: Any
) -> tuple[Any, bool]:
    key = f"value_{name}"
    options = list(field["allowed_values"])
    if key not in st.session_state and stored in options:
        st.session_state[key] = stored
    chosen = st.selectbox(
        input_label(name),
        options=options,
        index=None,
        key=key,
        placeholder="Select an option",
        format_func=lambda value: display_value(name, value),
        help=field_help(name, field),
    )
    return chosen, True


def _render_field(
    name: str, field: dict[str, Any], stored: Any
) -> tuple[Any, bool]:
    """Render one input, seeding it from the answer captured on an earlier visit."""
    if field["type"] == "number":
        return _render_number_field(name, field, stored)
    return _render_category_field(name, field, stored)


def _goto_step(index: int) -> None:
    st.session_state["step"] = index
    st.rerun()


def _render_step_indicator(
    titles: list[str], current: int, incomplete: set[int]
) -> None:
    """Draw a visual step header without allowing sections to be skipped."""
    with st.container(key="stepper"):
        for column, (index, title) in zip(st.columns(len(titles)), enumerate(titles)):
            if index >= current:
                marker = ""
            elif index in incomplete:
                marker = "! "
            else:
                marker = "✓ "
            column.button(
                f"{marker}{index + 1}. {title}",
                key=f"step_{index}",
                disabled=True,
                type="primary" if index == current else "secondary",
                use_container_width=True,
            )
    st.caption(f"Step {current + 1} of {len(titles)} — {titles[current]}")


def _render_navigation(current: int, blocked: bool) -> None:
    back_column, _, next_column = st.columns([1, 2, 1])
    if current > 0 and back_column.button(
        "← Back", key="nav_back", use_container_width=True
    ):
        _goto_step(current - 1)
    if next_column.button(
        "Next →",
        key="nav_next",
        type="primary",
        disabled=blocked,
        use_container_width=True,
    ):
        _goto_step(current + 1)


def _render_step_fields(
    names: list[str], metadata: dict[str, Any], answers: dict[str, Any]
) -> list[str]:
    """Render a step as aligned columns; returns any unreadable fields."""
    rejected: list[str] = []
    chunks = columnize(names)
    for column, chunk in zip(st.columns(len(chunks), gap="medium"), chunks):
        with column:
            for name in chunk:
                if name == "LTV":
                    value = calculate_ltv(
                        answers.get("loan_amount"), answers.get("property_value")
                    )
                    st.text_input(
                        "Loan-to-Value Ratio (%)",
                        value="" if value is None else f"{value:.2f}",
                        disabled=True,
                        help="Calculated automatically from loan amount and property value.",
                    )
                    answers[name] = value
                    if value is None:
                        rejected.append(name)
                    continue
                value, accepted = _render_field(
                    name, metadata["fields"][name], answers.get(name)
                )
                answers[name] = value
                if not accepted:
                    rejected.append(name)
    return rejected


def _render_review(
    steps: list[tuple[str, list[str]]], answers: dict[str, Any]
) -> None:
    summary_fields = [
        ("loan_amount", "Loan amount"),
        ("term", "Term (months)"),
        ("income", "Monthly income"),
        ("Credit_Score", "Credit score"),
    ]
    stats = "".join(
        '<div class="loan-review-stat">'
        f'<span>{escape(label)}</span>'
        f'<strong>{escape(display_value(name, answers[name]))}</strong></div>'
        for name, label in summary_fields
        if name in answers
    )
    if stats:
        st.markdown(f'<div class="loan-review-summary">{stats}</div>', unsafe_allow_html=True)
    st.caption("Application details · Expand a section to check its values or make changes.")
    for start in range(0, len(steps), 2):
        for offset, column in enumerate(st.columns(2)):
            index = start + offset
            if index >= len(steps):
                break
            title, names = steps[index]
            provided = sum(answers.get(name) is not None for name in names)
            with column.expander(f"{title} · {provided}/{len(names)} provided", expanded=False):
                rows = "".join(
                    f'<div><dt>{escape(friendly_label(name))}</dt>'
                    f'<dd>{escape(display_value(name, answers.get(name)))}</dd></div>'
                    for name in names
                )
                st.markdown(f'<dl class="loan-review-details">{rows}</dl>', unsafe_allow_html=True)
                if st.button(f"Edit {title}", key=f"review_edit_{index}", use_container_width=True):
                    _goto_step(index)


def _render_result(result: dict[str, Any]) -> None:
    st.subheader("Prediction Result")
    outcome = {
        0: "Repayments were handled normally",
        1: "The borrower failed to meet the repayment obligation",
    }[result["predicted_class"]]
    outcome_class = "normal" if result["predicted_class"] == 0 else "failure"
    class_one_probability = result.get("probability_class_1")
    class_zero_probability = result.get("probability_class_0")
    probability_cards = ""
    if class_one_probability is not None:
        probability_cards += (
            '<article class="loan-probability-card">'
            '<span>Probability of repayment failure</span>'
            f'<strong>{float(class_one_probability):.2%}</strong></article>'
        )
    if class_zero_probability is not None:
        probability_cards += (
            '<article class="loan-probability-card">'
            '<span>Probability of normal repayment</span>'
            f'<strong>{float(class_zero_probability):.2%}</strong></article>'
        )
    st.markdown(
        f'<section class="loan-result-card loan-result--{outcome_class}" '
        'aria-live="polite"><div class="loan-result-label">Prediction</div>'
        f'<h3>{escape(outcome)}</h3></section>'
        f'<div class="loan-probability-grid">{probability_cards}</div>',
        unsafe_allow_html=True,
    )


def _reset_stepper(metadata: dict[str, Any]) -> None:
    for name in metadata["expected_features"]:
        st.session_state.pop(f"value_{name}", None)
    st.session_state["answers"] = initial_answers(metadata)
    st.session_state["step"] = 0
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)


def run() -> None:
    st.set_page_config(
        page_title="Loan Status Prediction System",
        page_icon="🏦",
        layout="wide",
    )
    st.markdown(STYLES, unsafe_allow_html=True)
    st.markdown(
        '<div class="loan-brand"><span class="loan-brand-mark" aria-hidden="true">'
        '◈</span> LOAN ASSESSMENT</div>'
        '<section class="loan-hero" aria-labelledby="loan-title">'
        '<div class="loan-eyebrow">Intelligent lending insights</div>'
        '<h1 id="loan-title">Loan Status Prediction</h1>'
        '<p>A clearer view of every application. Enter the applicant’s details, '
        'review the loan profile, and generate a model-based assessment.</p>'
        '<div class="loan-hero-meta"><span>Guided application</span>'
        '<span>Credit &amp; property profile</span><span>Prediction insights</span></div>'
        '</section>',
        unsafe_allow_html=True,
    )

    base_url = os.getenv("LOAN_API_URL", DEFAULT_BACKEND_URL)
    try:
        with st.spinner("Connecting to the prediction service..."):
            metadata = fetch_metadata(base_url)
    except FrontendServiceError as exc:
        st.error(str(exc))
        st.info(f"Expected backend address: {base_url}")
        st.stop()

    st.caption(
        f"Assessment model: {metadata['selected_model']} · Complete each step, then review your details."
    )

    steps = build_steps(metadata["expected_features"])
    titles = step_titles(steps)
    fields = metadata["fields"]

    # Restart cleanly whenever the backend starts serving a different schema.
    if st.session_state.get("field_signature") != metadata["expected_features"]:
        _reset_stepper(metadata)
        st.session_state["field_signature"] = list(metadata["expected_features"])

    answers = st.session_state["answers"]
    current = min(st.session_state.get("step", 0), len(titles) - 1)
    st.session_state["step"] = current

    incomplete = {
        index
        for index, (_, names) in enumerate(steps)
        if missing_required(names, fields, answers)
    }
    _render_step_indicator(titles, current, incomplete)

    with st.container(key="assessment"):
        if current < len(steps):
            title, names = steps[current]
            st.subheader(title)
            st.caption(GROUP_INTROS.get(title, ""))
            rejected = _render_step_fields(names, metadata, answers)
            outstanding = missing_required(names, fields, answers)
            _render_navigation(current, blocked=bool(outstanding or rejected))
        else:
            outstanding = missing_required(
                list(metadata["expected_features"]), fields, answers
            )
            if outstanding:
                st.warning(
                    "Required details are still missing: "
                    + ", ".join(friendly_label(name) for name in outstanding)
                    + ". Use the steps above to complete them."
                )
            st.subheader(REVIEW_STEP)
            st.caption(
                "Complete the missing details to generate your prediction."
                if outstanding else "Your application is ready for prediction."
            )
            _render_review(steps, answers)
            with st.container(key="review_actions"):
                back_column, _, predict_column = st.columns(
                    [1, 2, 1], vertical_alignment="center"
                )
                if back_column.button(
                    "← Back", key="review_back", use_container_width=True
                ):
                    _goto_step(current - 1)
                if predict_column.button(
                    "Predict Loan Status",
                    key="predict",
                    type="primary",
                    disabled=bool(outstanding),
                    use_container_width=True,
                ):
                    try:
                        with st.spinner("Generating prediction..."):
                            st.session_state["result"] = submit_prediction(
                                base_url, dict(answers)
                            )
                        st.session_state.pop("error", None)
                    except FrontendServiceError as exc:
                        st.session_state.pop("result", None)
                        st.session_state["error"] = str(exc)

            if st.session_state.get("error"):
                st.error(st.session_state["error"])
            if st.session_state.get("result"):
                _render_result(st.session_state["result"])
                if st.button("Start Over", key="start_over", use_container_width=True):
                    _reset_stepper(metadata)
                    st.rerun()

    st.info(
        "This prediction is intended to support decision-making and should not "
        "be used as the sole basis for a financial decision."
    )


if __name__ == "__main__":
    run()
