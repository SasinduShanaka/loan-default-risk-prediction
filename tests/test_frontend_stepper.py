"""Behavioural tests for the multi-step data-entry flow in the Streamlit UI."""

from __future__ import annotations

from typing import Any

import pytest
import requests
from streamlit.testing.v1 import AppTest

from frontend.app import (
    COLUMNS_PER_ROW,
    NOT_PROVIDED,
    NOT_PROVIDED_LABEL,
    REVIEW_STEP,
    build_steps,
    columnize,
    display_value,
    field_help,
    format_answer,
    format_number_input,
    friendly_label,
    initial_answers,
    input_label,
    missing_required,
    parse_number,
    sort_fields,
    step_titles,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def stepper_metadata() -> dict[str, Any]:
    """Four features spanning two steps, both field types and both nullabilities."""
    return {
        "selected_model": "XGBoost",
        "expected_features": ["Gender", "age", "loan_amount", "term"],
        "numeric_features": ["loan_amount", "term"],
        "categorical_features": ["Gender", "age"],
        "fields": {
            "Gender": {
                "type": "category",
                "nullable": False,
                "allowed_values": ["Male", "Female"],
                "default": "Male",
            },
            "age": {
                "type": "category",
                "nullable": True,
                "allowed_values": ["<25", "25-34"],
                "default": "25-34",
            },
            "loan_amount": {
                "type": "number",
                "nullable": False,
                "default": 296500.0,
            },
            "term": {
                "type": "number",
                "nullable": True,
                "default": 360.0,
            },
        },
    }


def prediction_payload() -> dict[str, Any]:
    return {
        "predicted_class": 1,
        "probability_class_0": 0.25,
        "probability_class_1": 0.75,
        "model": "XGBoost",
    }


def _entrypoint() -> None:
    from frontend.app import run

    run()


def start_app(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
) -> AppTest:
    metadata = metadata if metadata is not None else stepper_metadata()
    prediction = prediction if prediction is not None else prediction_payload()
    monkeypatch.setattr(
        requests, "get", lambda url, *, timeout: FakeResponse(200, metadata)
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, *, json, timeout: FakeResponse(200, prediction),
    )
    app = AppTest.from_function(_entrypoint)
    app.run()
    return app


def recording_post(observed: dict[str, Any]) -> Any:
    def fake_post(url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
        observed.update(json)
        return FakeResponse(200, prediction_payload())

    return fake_post


# --- grouping and ordering ------------------------------------------------


def test_build_steps_keeps_canonical_group_order_and_skips_empty_groups() -> None:
    steps = build_steps(["loan_amount", "Credit_Score", "Gender"])

    assert steps == [
        ("Applicant Information", ["Gender"]),
        ("Loan Information", ["loan_amount"]),
        ("Credit Information", ["Credit_Score"]),
    ]


def test_unknown_fields_become_a_trailing_step() -> None:
    steps = build_steps(["mystery_field", "Gender"])

    assert steps[-1] == ("Additional Information", ["mystery_field"])


def test_fields_follow_display_order_not_backend_order() -> None:
    assert sort_fields(["loan_purpose", "term", "loan_amount"]) == [
        "loan_amount",
        "term",
        "loan_purpose",
    ]


def test_unordered_fields_sort_last_deterministically() -> None:
    assert sort_fields(["zz_extra", "aa_extra", "Gender"]) == [
        "Gender",
        "aa_extra",
        "zz_extra",
    ]


def test_step_titles_end_with_the_review_step() -> None:
    titles = step_titles(build_steps(["Gender", "loan_amount"]))

    assert titles == ["Applicant Information", "Loan Information", REVIEW_STEP]


def test_initial_answers_start_from_metadata_defaults() -> None:
    assert initial_answers(stepper_metadata()) == {
        "Gender": "Male",
        "age": "25-34",
        "loan_amount": 296500.0,
        "term": 360.0,
    }


# --- labelling ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("term", "Term (months)"),
        ("income", "Income (per month)"),
        ("LTV", "LTV (%)"),
        ("dtir1", "Debt-to-Income (%)"),
        ("loan_amount", "Loan Amount"),
    ],
)
def test_input_label_is_compact_and_carries_its_unit(name: str, label: str) -> None:
    assert input_label(name) == label


@pytest.mark.parametrize(
    ("name", "narrow", "full"),
    [
        ("LTV", "LTV (%)", "Loan-to-Value Ratio (LTV)"),
        ("dtir1", "Debt-to-Income (%)", "Debt-to-Income Ratio (dtir1)"),
        ("occupancy_type", "Occupancy", "Occupancy Type"),
        ("Neg_ammortization", "Neg. Amortization", "Negative Amortization"),
    ],
)
def test_review_keeps_the_full_name_where_the_input_is_shortened(
    name: str, narrow: str, full: str
) -> None:
    """Narrow inputs get a short label; the review has room for the real name."""
    assert input_label(name) == narrow
    assert friendly_label(name) == full


# --- grid layout ----------------------------------------------------------


def test_columnize_balances_fields_across_the_row() -> None:
    assert columnize(["a", "b", "c", "d", "e", "f"], columns=4) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
        ["f"],
    ]


def test_columnize_fills_down_each_column_to_keep_tab_order_sane() -> None:
    """Column-major: tabbing runs down a column, not back up the page."""
    assert columnize(["a", "b", "c", "d"], columns=2) == [["a", "b"], ["c", "d"]]


def test_columnize_pads_with_empty_columns_so_widths_stay_consistent() -> None:
    assert columnize(["a", "b"], columns=4) == [["a"], ["b"], [], []]


def test_columnize_keeps_every_field_exactly_once() -> None:
    names = [f"f{index}" for index in range(13)]

    chunks = columnize(names, columns=COLUMNS_PER_ROW)

    assert [name for chunk in chunks for name in chunk] == names


def test_four_columns_halve_the_rows_of_the_largest_step() -> None:
    names = [f"f{index}" for index in range(12)]

    rows = max(len(chunk) for chunk in columnize(names, columns=COLUMNS_PER_ROW))

    assert COLUMNS_PER_ROW == 4
    assert rows == 3


@pytest.mark.parametrize(
    ("name", "value", "rendered"),
    [
        ("construction_type", "sb", "Site built"),
        ("occupancy_type", "ir", "Investment property"),
        ("total_units", "2U", "2 units"),
        ("age", "<25", "Under 25"),
        ("loan_type", "type1", "type1"),
        ("loan_purpose", "p3", "p3"),
        ("Credit_Worthiness", "l1", "l1"),
        ("Gender", None, NOT_PROVIDED_LABEL),
        ("Gender", NOT_PROVIDED, NOT_PROVIDED_LABEL),
    ],
)
def test_display_value_translates_only_codes_with_known_meanings(
    name: str, value: Any, rendered: str
) -> None:
    assert display_value(name, value) == rendered


def test_field_help_lists_the_codes_a_dropdown_will_send() -> None:
    metadata = stepper_metadata()

    help_text = field_help("age", metadata["fields"]["age"])

    assert help_text is not None
    assert "Sends one of: <25, 25-34." in help_text


def test_field_help_for_a_number_has_no_code_list() -> None:
    metadata = stepper_metadata()

    assert field_help("term", metadata["fields"]["term"]) == (
        "Loan term in months. 360 months is a 30-year loan."
    )


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (None, NOT_PROVIDED_LABEL),
        (296500.0, "296,500"),
        (43.5, "43.50"),
        ("type1", "type1"),
    ],
)
def test_format_answer_presents_review_values_readably(
    value: Any, rendered: str
) -> None:
    assert format_answer(value) == rendered


# --- required-field tracking ---------------------------------------------


def test_missing_required_ignores_nullable_fields() -> None:
    metadata = stepper_metadata()
    answers = {"Gender": "Male", "age": None, "loan_amount": None, "term": None}

    assert missing_required(
        list(metadata["expected_features"]), metadata["fields"], answers
    ) == ["loan_amount"]


# --- rendering and navigation -------------------------------------------


def test_only_the_current_step_is_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    app = start_app(monkeypatch)

    assert [box.label for box in app.selectbox] == ["Gender", "Age"]
    assert len(app.text_input) == 0


def test_no_field_uses_a_separate_blank_checkbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nullability is expressed inside each control, which keeps rows aligned."""
    app = start_app(monkeypatch)
    assert len(app.checkbox) == 0

    app.button(key="nav_next").click().run()

    assert len(app.checkbox) == 0


def test_nullable_dropdown_offers_a_not_provided_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    required, nullable = app.selectbox[0], app.selectbox[1]

    assert NOT_PROVIDED_LABEL not in required.options
    assert NOT_PROVIDED_LABEL in nullable.options


def test_next_advances_and_back_restores_the_earlier_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    app.selectbox[0].set_value("Female").run()
    app.button(key="nav_next").click().run()

    assert [field.label for field in app.text_input] == [
        "Loan Amount",
        "Term (months)",
    ]
    assert len(app.selectbox) == 0

    app.button(key="nav_back").click().run()

    assert app.selectbox[0].value == "Female"


def test_answers_from_every_step_reach_the_prediction_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    app = start_app(monkeypatch)
    monkeypatch.setattr(requests, "post", recording_post(observed))

    app.selectbox[0].set_value("Female").run()
    app.button(key="nav_next").click().run()
    app.text_input[0].set_value("123456").run()
    app.button(key="nav_next").click().run()
    app.button(key="predict").click().run()

    assert observed == {
        "Gender": "Female",
        "age": "25-34",
        "loan_amount": 123456.0,
        "term": 360.0,
    }
    assert app.metric[0].value == "1"


def test_not_provided_dropdown_choice_sends_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    app = start_app(monkeypatch)
    monkeypatch.setattr(requests, "post", recording_post(observed))

    app.selectbox[1].set_value(NOT_PROVIDED).run()
    app.button(key="step_2").click().run()
    app.button(key="predict").click().run()

    assert observed["age"] is None


def test_cleared_nullable_number_sends_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    app = start_app(monkeypatch)
    monkeypatch.setattr(requests, "post", recording_post(observed))

    app.button(key="nav_next").click().run()
    app.text_input[1].set_value("").run()
    app.button(key="nav_next").click().run()
    app.button(key="predict").click().run()

    assert observed["term"] is None


def test_clearing_a_required_number_blocks_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    app.button(key="nav_next").click().run()
    app.text_input[0].set_value("").run()

    assert any("Loan Amount" in block.value for block in app.warning)
    assert app.button(key="nav_next").disabled


def test_missing_required_value_blocks_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    app.button(key="nav_next").click().run()
    app.text_input[0].set_value("").run()
    app.button(key="step_2").click().run()

    assert any("still missing" in block.value for block in app.warning)
    assert app.button(key="predict").disabled


def test_step_indicator_flags_an_incomplete_earlier_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    app.button(key="nav_next").click().run()
    app.text_input[0].set_value("").run()
    app.button(key="step_2").click().run()

    assert app.button(key="step_1").label.startswith("! ")
    assert app.button(key="step_0").label.startswith("✓ ")


def test_step_indicator_jumps_straight_to_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    app.button(key="step_2").click().run()

    assert any(REVIEW_STEP in block.value for block in app.subheader)
    assert not app.button(key="predict").disabled


def test_failed_prediction_shows_the_service_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, *, json, timeout: FakeResponse(
            422, {"detail": {"message": "Invalid value for Gender"}}
        ),
    )

    app.button(key="step_2").click().run()
    app.button(key="predict").click().run()

    assert any("Invalid value for Gender" in block.value for block in app.error)


def test_start_over_returns_to_the_first_step_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    app.selectbox[0].set_value("Female").run()
    app.button(key="step_2").click().run()
    app.button(key="predict").click().run()
    app.button(key="start_over").click().run()

    assert app.selectbox[0].value == "Male"


# --- numeric text entry ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("296500", 296500.0),
        ("296,500", 296500.0),
        (" 296500 ", 296500.0),
        ("75.12", 75.12),
        ("39%", 39.0),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_number_accepts_the_ways_people_type_amounts(
    raw: str, value: float | None
) -> None:
    assert parse_number(raw) == (value, True)


@pytest.mark.parametrize("raw", ["abc", "12abc", "-5", "nan", "inf", "1/2"])
def test_parse_number_rejects_unusable_text(raw: str) -> None:
    assert parse_number(raw) == (None, False)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, ""),
        (296500.0, "296,500"),
        (360.0, "360"),
        (75.12254902, "75.12254902"),
    ],
)
def test_format_number_input_round_trips_without_losing_precision(
    value: Any, text: str
) -> None:
    assert format_number_input(value) == text
    assert parse_number(text)[0] == value


def test_typed_amount_keeps_its_thousands_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    app = start_app(monkeypatch)
    monkeypatch.setattr(requests, "post", recording_post(observed))

    app.button(key="nav_next").click().run()
    app.text_input[0].set_value("1,250,000").run()
    app.button(key="nav_next").click().run()
    app.button(key="predict").click().run()

    assert observed["loan_amount"] == 1250000.0


def test_unreadable_amount_is_flagged_and_blocks_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    app.button(key="nav_next").click().run()
    app.text_input[0].set_value("about 300k").run()

    assert any("Enter a positive number" in block.value for block in app.caption)
    assert app.button(key="nav_next").disabled


def test_unreadable_amount_does_not_overwrite_the_last_good_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    app = start_app(monkeypatch)
    monkeypatch.setattr(requests, "post", recording_post(observed))

    app.button(key="nav_next").click().run()
    app.text_input[0].set_value("about 300k").run()
    app.text_input[0].set_value("250000").run()
    app.button(key="nav_next").click().run()
    app.button(key="predict").click().run()

    assert observed["loan_amount"] == 250000.0


def test_numeric_fields_are_seeded_from_the_schema_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    app.button(key="nav_next").click().run()

    assert [field.value for field in app.text_input] == ["296,500", "360"]


# --- chrome ---------------------------------------------------------------


def test_no_system_information_sidebar_is_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    assert len(app.sidebar) == 0


def test_model_name_survives_the_removed_sidebar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = start_app(monkeypatch)

    assert any("XGBoost" in block.value for block in app.caption)
