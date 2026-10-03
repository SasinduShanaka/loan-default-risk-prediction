# Stage 9 Backend Design

## Purpose

Build the Stage 9 backend for the IT3051 Loan Default Risk Prediction mini project. The service will expose the already-trained XGBoost classification pipeline through a metadata-driven FastAPI API. It will accept raw feature values, validate them, and delegate all imputation, encoding, scaling, and prediction to the saved sklearn pipeline.

Stage 10 frontend work is outside this design.

## Source-of-truth artifacts

Runtime behavior depends on these files:

- `models/final_loan_status_pipeline.joblib` — the only runtime model artifact
- `models/deployment_input_metadata.json` — the feature and validation contract
- `models/final_loan_status_pipeline.pkl` — retained only as an alternative submission artifact

The repository currently does not contain these files. A `.pkl` and the original notebook have been identified in `C:\Users\ASUS\Downloads`, but the required `.joblib` and metadata JSON have not yet been supplied. Implementation and end-to-end verification must not invent metadata or silently substitute the `.pkl` for the required runtime file.

The notebook will be preserved without modifying Stages 1–8.

## Architecture

The backend consists of four focused modules:

- `backend/config.py` resolves project-relative artifact paths with `pathlib.Path`.
- `backend/validation.py` validates and cleans raw JSON using deployment metadata.
- `backend/model_service.py` loads and verifies artifacts once, constructs ordered one-row DataFrames, and performs inference.
- `backend/main.py` configures FastAPI, CORS, endpoint handling, and safe HTTP errors.

The request flow is:

```text
User / future Streamlit frontend
        |
        | HTTP JSON containing raw feature values
        v
FastAPI endpoint
        |
        v
Metadata-driven validation
        |
        v
One-row DataFrame in expected_features order
        |
        v
Saved sklearn Pipeline
  - trained numeric preprocessing
  - trained categorical preprocessing
  - final XGBoost classifier
        |
        v
Class prediction and class-aware probabilities
        |
        v
JSON response
```

## Startup and model readiness

The application will load a single `LoanPredictionService` instance. Startup fails clearly if:

- either required runtime artifact is absent or unreadable;
- metadata is invalid or `expected_features` is empty;
- the loaded object is not an sklearn `Pipeline`;
- `named_steps` does not contain `preprocessor` and `model`;
- the final estimator is not the saved XGBoost classifier; or
- the pipeline cannot accept an appropriate raw sample during verification.

Health must never report success when initialization failed. The service will not retrain, replace, or modify the estimator.

## Metadata contract

`deployment_input_metadata.json` is authoritative for:

- selected model name;
- target name;
- exact ordered feature names;
- numeric and categorical feature groups;
- per-field metadata;
- allowed categorical values; and
- nullable fields.

The API and documentation will use the exact metadata spelling, including values such as `Sex Not Available`. Removed training columns—including `Interest_rate_spread`, `rate_of_interest`, `Upfront_charges`, `ID`, and `year`—will not be reintroduced unless they unexpectedly appear in the supplied final metadata, in which case the artifact inconsistency must be reported rather than guessed around.

## Input validation

`POST /predict` accepts a JSON object containing the raw expected features. Validation will:

1. reject missing non-nullable fields;
2. reject unexpected fields;
3. accept `null` only where metadata permits it;
4. reject booleans and non-numeric values for numeric fields;
5. reject categorical values not listed in `allowed_values`;
6. preserve accepted category strings exactly;
7. reject negative values for logically non-negative financial fields when those fields exist; and
8. require `term` to be greater than zero when present and non-null.

Training minimum and maximum values are reference information, not universal hard bounds. Extreme but logically valid numeric values remain acceptable. Accepted null numeric values are represented in a form the trained pipeline imputer can consume.

Validation does not encode, scale, or impute values.

## Prediction behavior

The model service builds input as:

```python
pd.DataFrame([cleaned_payload], columns=expected_features)
```

It calls `pipeline.predict()` and, when supported, `pipeline.predict_proba()`. Probability values are paired with `pipeline.named_steps["model"].classes_`; no fixed probability-column position is assumed.

The response uses semantically neutral class labels because the business meaning of target values 0 and 1 is not verified. A binary response will resemble:

```json
{
  "predicted_class": 1,
  "probability_class_0": 0.2,
  "probability_class_1": 0.8,
  "model": "XGBoost"
}
```

If probabilities are unavailable, the response contains only `predicted_class` and `model`. NumPy scalar values will be converted to JSON-safe Python values.

## API contract

### `GET /`

Returns the service name, running status, and metadata-derived model name.

### `GET /health`

Returns HTTP 200 with `status: ok`, `model_loaded: true`, and the model name only when initialization succeeded.

### `GET /metadata`

Returns the frontend-safe deployment contract, including `selected_model`, `target`, `expected_features`, `numeric_features`, `categorical_features`, and `fields`.

### `POST /predict`

Validates a complete raw input object and returns the neutral prediction response. User-input failures return HTTP 422 with readable structured details. Unexpected inference failures return HTTP 500 without a traceback or internal path disclosure.

## CORS

Local development origins for common Streamlit and FastAPI hosts will be allowed explicitly. Documentation will note that deployed environments must restrict origins to the real frontend host.

## Testing strategy

Implementation will follow test-driven development. Tests will be written and observed failing before the corresponding production behavior is added.

Automated tests will cover:

- healthy startup and health response;
- metadata response and required keys;
- a successful raw-value prediction;
- valid predicted classes;
- probability bounds and mapping against the classifier's actual `classes_` ordering;
- empty objects, missing fields, and extra fields;
- wrong numeric types, including boolean values;
- invalid categories with helpful allowed-value feedback;
- negative loan amounts where that field exists;
- `term <= 0` where that field exists;
- permitted nullable fields; and
- repeatable predictions for identical input.

A valid test payload will be built only from supplied metadata defaults or a sanitized compatible source row. Categorical values will not be invented.

Live verification will start Uvicorn, call `/health`, `/metadata`, and `/predict`, and then run the full `pytest -q` suite. Verification must also confirm that the runtime model is non-empty, preprocessing is not duplicated, raw values reach the pipeline, and probability mapping uses `classes_`.

## Documentation and repository structure

The implementation will create the requested `backend/`, `tests/`, `docs/`, `models/`, `notebooks/`, and empty `frontend/` structure where absent. It will preserve existing work and copy supplied artifacts only after their identity and role are verified.

Documentation will include:

- API endpoints with examples generated from actual metadata;
- system architecture and the reasons for bundling preprocessing with the estimator;
- at least 15 manual test cases in the requested table format; and
- an expanded academic project README with team members Shanaka, Praneesha, Dilki, and Dinuja.

`requirements.txt` will contain only the requested runtime, testing, and future Streamlit dependencies, preserving any existing compatible pins. `.gitignore` will cover standard Python development artifacts without excluding model files. Model size will be checked against normal GitHub limits and documented if necessary.

## Constraints and exclusions

- Do not retrain or replace XGBoost.
- Do not modify notebook Stages 1–8.
- Do not add SMOTE.
- Do not reproduce preprocessing outside the pipeline.
- Do not load both serialized model variants at runtime.
- Do not infer target semantics such as approved, rejected, default, safe, or risk.
- Do not build the Stage 10 frontend.
- Do not invent feature names, categories, defaults, or nullable rules while metadata is unavailable.

## Completion criteria

Stage 9 is complete only when the required artifacts are present and verified, the API starts from the repository root, all four endpoints behave as specified, automated tests pass, live endpoint checks pass, and the final handoff includes the repository tree, commands, real metadata summary, real request/response examples, test evidence, and any unresolved issues.
