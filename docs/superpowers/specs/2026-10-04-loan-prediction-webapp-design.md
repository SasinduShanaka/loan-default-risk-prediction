# Loan Prediction Web Application Design

## Purpose

Build Stages 9 and 10 of the IT3051 Loan Default Risk Prediction mini project around the supplied, already-trained classification pipeline. A FastAPI backend will validate raw loan inputs and run the fitted pipeline. A Streamlit frontend will obtain the input contract from FastAPI, collect raw user inputs, and display semantically neutral class predictions and probabilities.

The application must preserve the completed Stages 1–8 modelling work. It will not retrain, retune, replace, or modify the selected XGBoost pipeline.

## Verified project context

The current Git repository initially contained only `README.md` and the earlier design document. The following source artifacts were identified outside the repository:

- `C:\Users\ASUS\Downloads\final_loan_status_pipeline.pkl` — 746,857 bytes
- `C:\Users\ASUS\Downloads\fdm-miniproject.ipynb` — 1,478,316 bytes

Static pickle inspection confirms that the model artifact contains an sklearn `Pipeline` with `preprocessor` and `model` steps, a fitted `ColumnTransformer`, and an `xgboost.sklearn.XGBClassifier`. The pickle records scikit-learn version 1.6.1 and a binary classifier. Complete runtime inspection still requires installing XGBoost and compatible dependencies.

The supplied `.pkl` is the runtime source of truth and will be preserved byte-for-byte at `models/final_loan_status_pipeline.pkl`. The notebook will be preserved at `notebooks/fdm-miniproject.ipynb` without rewriting Stages 1–8.

## Architecture

The system contains two processes with a single model boundary:

```text
User
  |
  v
Streamlit frontend
  |  HTTP/JSON
  v
FastAPI backend
  |
  v
Metadata-driven input validation
  |
  v
One-row DataFrame containing ordered raw values
  |
  v
Saved sklearn Pipeline
  +-- trained numeric preprocessing
  +-- trained categorical preprocessing
  +-- final XGBoost classifier
  |
  v
Prediction and class-aware probabilities
  |
  v
FastAPI JSON response
  |
  v
Streamlit result display
```

The Streamlit process never imports or loads the model. It depends only on the backend's `/metadata` and `/predict` contracts.

## Repository components

- `backend/config.py` resolves project-relative model and schema paths.
- `backend/model_service.py` loads, checks, and invokes the fitted pipeline once.
- `backend/validation.py` validates raw payloads against the generated schema.
- `backend/main.py` owns FastAPI configuration, CORS, routes, and safe HTTP error mapping.
- `frontend/app.py` renders a dynamic Streamlit form and calls the backend.
- `models/input_schema.json` records the recovered raw input contract.
- `tests/` verifies artifacts, service behavior, endpoint contracts, validation, probability mapping, and repeatability.
- `docs/` explains the API, architecture, and manual test cases.

## Model inspection and compatibility

The artifact will first be loaded with the Python standard library's `pickle`. If that fails because the file uses joblib-specific serialization, `joblib.load` will be tried. Failure of both loaders will produce a clear developer-facing error; it will not trigger retraining or replacement.

Runtime checks will verify:

- the model file exists and is non-empty;
- the object is a fitted sklearn `Pipeline` or compatible pipeline object;
- preprocessing and classifier steps are present, using their actual discovered names;
- the classifier is an XGBoost classifier;
- `predict()` is callable;
- `predict_proba()` is used when available;
- the classifier's `classes_` is available; and
- raw expected feature names can be recovered.

Dependencies will be tested against the serialized artifact. Scikit-learn 1.6.1 is the initial compatibility pin. Any required XGBoost version will be established through successful loading and inference, documented, and pinned where necessary. Version errors will not be handled by retraining.

## Raw feature and schema recovery

The model service will recover raw input structure from `pipeline.feature_names_in_`, the fitted `ColumnTransformer.transformers_`, and its fitted child pipelines. Transformed one-hot column names will never become API inputs.

Static notebook evidence identifies these numeric inputs:

```text
loan_amount, term, property_value, income, Credit_Score, LTV, dtir1
```

It identifies these categorical inputs:

```text
loan_limit, Gender, approv_in_adv, loan_type, loan_purpose,
Credit_Worthiness, open_credit, business_or_commercial,
Neg_ammortization, interest_only, lump_sum_payment, construction_type,
occupancy_type, Secured_by, total_units, credit_type,
co-applicant_credit_type, age, submission_of_application, Region,
Security_Type
```

These lists are audit evidence, not hardcoded authority. The loaded fitted pipeline determines the final exact order and membership. Any mismatch between notebook and model will be reported.

`models/input_schema.json` will be generated from the fitted pipeline and contain:

- `selected_model`;
- ordered `expected_features`;
- `numeric_features`;
- `categorical_features`; and
- a `fields` entry for every expected feature.

Categorical `allowed_values` will come from the fitted `OneHotEncoder.categories_` and preserve exact spelling, including `Sex Not Available` when present.

Nullability will be derived only from reliable evidence. A fitted imputer demonstrates that the pipeline can technically process missing values, but it does not alone prove that every field is optional in the user contract. Notebook missing-value evidence and fitted preprocessing will be used together. Any still-uncertain nullability will be recorded conservatively and documented rather than guessed.

Schema generation is a development-time artifact step, not per-request work. Backend startup loads the generated schema and verifies that it agrees with the pipeline.

## Raw input and preprocessing boundary

The backend accepts raw numeric values, raw categorical values, and permitted nulls. It will not one-hot encode, scale, standardize, impute, or rename categories.

After validation, input is constructed with an explicit order:

```python
pd.DataFrame([cleaned_payload], columns=expected_features)
```

The saved pipeline performs every learned transformation exactly once before XGBoost inference.

## Input validation

`backend/validation.py` will:

1. require a JSON object;
2. reject empty payloads;
3. reject unexpected fields;
4. reject missing required fields;
5. permit `null` only for fields marked nullable;
6. reject booleans and non-numeric values for numeric fields;
7. reject categorical values outside fitted encoder categories; and
8. apply narrow domain checks to obvious financial quantities.

When present and non-null:

- `loan_amount` must be greater than zero;
- `term` must be greater than zero;
- `income`, `property_value`, `Credit_Score`, `LTV`, and `dtir1` must be non-negative.

No arbitrary maximum will be inferred from training data. Training extrema are historical observations, not deployment limits.

Validation will return clear structured HTTP 422 details, including invalid field names and allowed categorical values where useful. It will not alter accepted category labels or duplicate trained preprocessing.

## Backend service and API

`LoanPredictionService` loads the pipeline and schema once, exposes readiness and metadata, constructs ordered DataFrames, and invokes inference. Startup fails clearly if artifacts are missing, incompatible, unfitted, or inconsistent.

Prediction results use JSON-safe Python scalar values. Probabilities are paired with the actual classifier `classes_` sequence; array position 1 is never blindly assumed to represent label 1.

The API is titled `Loan Status Prediction API`, version `1.0.0`, with these endpoints:

- `GET /` returns service identity, running status, and model name.
- `GET /health` returns success only when the model and schema are loaded and verified.
- `GET /metadata` returns the frontend-safe schema contract.
- `POST /predict` validates a complete raw payload and returns the class and available per-class probabilities.

Expected user errors return HTTP 422. Unexpected model or server failures return HTTP 500 without internal tracebacks or machine-specific paths.

CORS will explicitly allow common local Streamlit development origins. Documentation will require narrowing origins in production.

## Target semantics

The business meaning of Status values 0 and 1 is not verified. All interfaces therefore use neutral output language:

- `predicted_class`
- `probability_class_0`
- `probability_class_1`
- “Predicted Class: 0/1”
- “Probability of Class 1”

The application will not label outputs approved, rejected, default, non-default, safe, or high risk without later documentary evidence.

## Streamlit frontend

`frontend/app.py` will use an environment-configurable backend URL with `http://127.0.0.1:8000` as the local default. At startup it fetches `/metadata`; failure displays a clear recovery message rather than crashing.

The page will contain:

- title: “Loan Status Prediction System”;
- subtitle: “Machine-learning decision support for loan assessment.”;
- a short description of the trained pipeline;
- an optional compact sidebar naming the system, XGBoost, and binary classification task;
- dynamically generated numeric and categorical controls;
- a “Predict Loan Status” action;
- neutral class and probability metrics; and
- the required financial-decision disclaimer.

Numeric fields use `st.number_input`. Categorical fields use `st.selectbox` populated only from `/metadata`. Nullable fields will provide an intentional missing-value control rather than silently inventing a value.

Fields will receive presentation-only friendly labels while outgoing JSON retains exact model feature names. Inputs will be grouped conservatively into applicant, loan, credit, property, and additional information; uncertain fields go under Additional Information.

Prediction calls will have a finite timeout and handle connection failures, backend validation details, non-success responses, and malformed success responses without terminating the Streamlit app.

## Testing and verification

Backend functionality will be developed test-first. Each production behavior will be preceded by a failing automated test that demonstrates the intended contract.

Automated coverage will include:

- model file existence and non-zero size;
- model loading and fitted pipeline structure;
- health and metadata responses;
- valid prediction with actual fitted categories;
- predicted class membership in `classes_`;
- probability bounds and correct class-to-probability mapping;
- missing, extra, empty, wrongly typed, negative, and invalid-category inputs;
- permitted null handling when supported; and
- consistency across repeated identical requests.

Test payloads will use encoder-derived categorical values and safe numeric values. No success category will be invented.

Frontend helpers that transform metadata, build payloads, or parse API results will be kept separable enough for focused tests. The full system will also be verified manually by running Uvicorn and Streamlit, calling `/health`, `/metadata`, and `/predict`, and confirming the frontend-to-backend path.

The final verification will run the complete `pytest -q` suite and check that preprocessing exists only inside the saved pipeline, raw columns arrive in exact order, class probabilities use `classes_`, and both supplied artifacts remain unchanged from their source bytes.

## Documentation and project presentation

The README will explain the academic context, dataset, four compared models, final XGBoost selection, architecture, installation, both run commands, tests, limitations, future work, and team members Shanaka, Praneesha, Dilki, and Dinuja.

Supporting documents will include:

- `docs/API.md` with actual schema-derived requests and responses;
- `docs/SYSTEM_ARCHITECTURE.md` explaining the process boundary and bundled preprocessing benefits; and
- `docs/TEST_CASES.md` with at least the 15 requested manual scenarios and blank manual result/status cells.

The final handoff will include the complete repository tree, artifact size and inspection results, exact feature groups, commands, automated test evidence, a real request/response example, unresolved issues, and a concise university presentation demo procedure.

## Dependencies and project hygiene

`requirements.txt` will contain FastAPI, Uvicorn, Streamlit, pandas, NumPy, scikit-learn, XGBoost, requests, HTTPX, pytest, and joblib, with compatibility pins only where the artifact requires them. Pickle will not be listed because it is part of Python.

`.gitignore` will exclude common Python environment and cache files but will not exclude the model. At approximately 0.75 MB, the current model is well below GitHub's normal 100 MB per-file limit.

## Constraints

- Do not retrain, retune, or replace XGBoost.
- Do not apply SMOTE.
- Do not change the Stage 1–8 notebook logic or reported metrics.
- Do not restore removed identifiers or suspected leakage features unless the fitted model demonstrably expects them.
- Do not manually reproduce or double-apply learned preprocessing.
- Do not change fitted categorical values.
- Do not fake predictions, probabilities, schema data, or test results.
- Do not make Streamlit load the model.
- Keep API, validation, inference, and user-interface responsibilities separate.

## Completion criteria

Stages 9 and 10 are complete when the original artifacts are preserved in the repository, the fitted pipeline is successfully inspected, the schema is derived and verified, FastAPI and Streamlit run from the repository root, the frontend dynamically consumes backend metadata and predictions, automated tests pass, live integration checks pass, documentation reflects actual recovered values, and all unresolved compatibility or semantic limitations are clearly reported.
