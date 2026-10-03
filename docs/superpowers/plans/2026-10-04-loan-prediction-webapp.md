# Loan Prediction Web Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the supplied fitted loan-status pipeline behind a validated FastAPI API and a metadata-driven Streamlit interface without changing the trained model or duplicating preprocessing.

**Architecture:** Preserve the supplied PKL and notebook, derive a JSON input contract from the fitted pipeline, and load both once in a focused model service. FastAPI owns validation and inference; Streamlit consumes only HTTP metadata and prediction responses.

**Tech Stack:** Python 3, FastAPI, Uvicorn, Streamlit, pandas, NumPy, scikit-learn 1.6.1, XGBoost, pickle/joblib, requests, HTTPX, pytest

**Spec:** `docs/superpowers/specs/2026-10-04-loan-prediction-webapp-design.md`

## Global Constraints

- Preserve `C:\Users\ASUS\Downloads\final_loan_status_pipeline.pkl` byte-for-byte at `models/final_loan_status_pipeline.pkl` and use that exact file at runtime.
- Preserve `C:\Users\ASUS\Downloads\fdm-miniproject.ipynb` at `notebooks/fdm-miniproject.ipynb`; do not alter Stages 1–8.
- Do not retrain, retune, replace, or regenerate XGBoost; do not add SMOTE.
- Do not manually encode, scale, standardize, or impute raw API inputs.
- Recover feature names, ordering, categories, step names, and classes from the fitted pipeline; notebook values are corroborating evidence only.
- Use neutral class terminology and map probabilities using the classifier's actual `classes_` ordering.
- Keep Streamlit model-free: all model access occurs through FastAPI.
- Run commands from the repository root on Windows.

## Review Focus

- Reject `NaN`, positive infinity, and negative infinity as numeric API values before they reach the pipeline; covered in Task 3.
- Reject booleans and numeric strings as numeric values instead of relying on Python's numeric coercion; covered in Task 3.
- Convert NumPy feature/category/class scalar values into stable JSON-native values; covered in Tasks 2 and 4.
- Detect a generated schema whose feature order or groups disagree with the fitted preprocessor; covered in Task 2.
- Render and transmit nullable frontend fields without inventing sentinel category strings; covered in Task 5.

---

### Task 1: Preserve artifacts and establish the compatible environment

**Files:**
- Create: `models/final_loan_status_pipeline.pkl`
- Create: `notebooks/fdm-miniproject.ipynb`
- Create: `backend/__init__.py`
- Create: `frontend/__init__.py`
- Create: `tests/__init__.py`
- Create: `requirements.txt`
- Create: `.gitignore`

**Interfaces:**
- Consumes: supplied PKL and notebook in `C:\Users\ASUS\Downloads`.
- Produces: preserved repository artifacts and a Python environment capable of loading the pickle.

- [ ] **Step 1: Record SHA-256 hashes and byte sizes of both source artifacts**

Run: `Get-FileHash C:\Users\ASUS\Downloads\final_loan_status_pipeline.pkl -Algorithm SHA256; Get-FileHash C:\Users\ASUS\Downloads\fdm-miniproject.ipynb -Algorithm SHA256; Get-Item ...`

Expected: both files exist and have non-zero sizes; retain the hashes for post-copy verification.

- [ ] **Step 2: Create project directories and copy both artifacts without modifying them**

Use PowerShell `Copy-Item -LiteralPath` with the exact source and destination paths. Add empty package markers with `apply_patch`.

- [ ] **Step 3: Verify copied hashes and sizes match the sources**

Run the same `Get-FileHash` and `Get-Item` checks against `models/` and `notebooks/`.

Expected: source and destination hashes and sizes match exactly.

- [ ] **Step 4: Add dependency and ignore configuration**

Pin `scikit-learn==1.6.1`; include only FastAPI, `uvicorn[standard]`, Streamlit, pandas, NumPy, XGBoost, requests, HTTPX, pytest, and joblib. Ignore the requested caches/environments without ignoring the PKL.

- [ ] **Step 5: Create `.venv`, install dependencies, and prove the artifact loads**

Run: `python -m venv .venv`, `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then a read-only pickle load/inspection command.

Expected: imports succeed; the loaded object, steps, classifier type, `classes_`, feature names, transformers, imputers, and encoder categories can be printed. If XGBoost compatibility fails, determine and pin the compatible version before continuing.

- [ ] **Step 6: Commit artifact and environment setup**

```bash
git add .gitignore requirements.txt backend/__init__.py frontend/__init__.py tests/__init__.py models/final_loan_status_pipeline.pkl notebooks/fdm-miniproject.ipynb
git commit -m "build: add trained model and project dependencies"
```

### Task 2: Recover and verify the model input contract

**Files:**
- Create: `backend/config.py`
- Create: `backend/model_service.py`
- Create: `models/input_schema.json`
- Create: `tests/test_model_service.py`

**Interfaces:**
- Consumes: fitted pipeline artifact from Task 1.
- Produces: `load_pipeline(path: Path) -> Pipeline`, `inspect_pipeline(pipeline: Pipeline) -> PipelineInspection`, `derive_input_schema(pipeline: Pipeline) -> dict[str, Any]`, and `LoanPredictionService(model_path: Path = MODEL_PATH, schema_path: Path = SCHEMA_PATH)` with `metadata`, `expected_features`, `classes`, `ready`, and `predict(payload)`.

- [ ] **Step 1: Write failing model-load and inspection tests**

Add tests proving the artifact is non-empty, loads as `Pipeline`, contains the actual preprocessing/model steps, uses `XGBClassifier`, exposes two classes, and yields exact raw numeric/categorical groups and fitted encoder categories. Add a test that NumPy scalar categories become JSON-native values.

- [ ] **Step 2: Run the model-service tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_model_service.py -v`

Expected: collection fails because `backend.config` and `backend.model_service` do not exist.

- [ ] **Step 3: Implement configuration, safe loading, and pipeline inspection**

Use `PROJECT_ROOT`, `MODEL_PATH`, and `SCHEMA_PATH` in `backend/config.py`. In `model_service.py`, implement pickle-first/joblib-fallback loading and inspection of actual step names, `ColumnTransformer.transformers_`, child pipelines, encoder `categories_`, imputers, `feature_names_in_`, and classifier `classes_`.

- [ ] **Step 4: Run inspection tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_model_service.py -v`

Expected: the model inspection tests pass and report the exact fitted contract.

- [ ] **Step 5: Write failing schema-agreement and ordered-inference tests**

Add tests that `derive_input_schema` covers every feature exactly once in pipeline order, uses only fitted encoder categories, emits JSON-native values, and rejects a schema with reordered/missing/group-mismatched features. Add a test proving `predict` constructs a DataFrame in `expected_features` order and maps probabilities by `classes_`.

- [ ] **Step 6: Run the new tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_model_service.py -v`

Expected: failures identify missing schema derivation, agreement checks, and service prediction behavior.

- [ ] **Step 7: Implement schema derivation and the service class**

Derive category values from the fitted encoder, numeric/categorical membership from the fitted transformer, and conservative nullability from fitted imputers plus notebook evidence. Implement schema consistency checks and class-aware prediction without manual preprocessing.

- [ ] **Step 8: Generate `models/input_schema.json` from the inspected fitted pipeline**

Create the JSON with `apply_patch` using the exact inspected values. Validate it by constructing `LoanPredictionService` against the file.

- [ ] **Step 9: Run Task 2 tests and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_model_service.py -v`

Expected: PASS.

```bash
git add backend/config.py backend/model_service.py models/input_schema.json tests/test_model_service.py
git commit -m "feat: recover fitted model input contract"
```

### Task 3: Implement metadata-driven raw-input validation

**Files:**
- Create: `backend/validation.py`
- Create: `tests/test_validation.py`

**Interfaces:**
- Consumes: schema dictionary generated in Task 2.
- Produces: `ValidationError(detail: dict[str, Any])` and `validate_payload(payload: Any, schema: Mapping[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write failing validation tests**

Cover a valid raw payload; non-object and empty payloads; missing and extra fields; null required fields; accepted nullable fields; booleans, numeric strings, NaN, and infinities in numeric fields; invalid categories with `allowed_values`; exact preservation of `Sex Not Available`; `loan_amount <= 0`; `term <= 0`; and negative values for the remaining named financial fields.

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_validation.py -v`

Expected: collection fails because `backend.validation` does not exist.

- [ ] **Step 3: Implement `ValidationError` and `validate_payload`**

Return a new cleaned dictionary without encoding, scaling, imputing, category renaming, or maximum-range enforcement. Preserve permitted nulls as `None` and require finite real numbers excluding booleans.

- [ ] **Step 4: Run validation tests and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_validation.py -v`

Expected: PASS.

```bash
git add backend/validation.py tests/test_validation.py
git commit -m "feat: validate raw loan inputs"
```

### Task 4: Expose the FastAPI service

**Files:**
- Create: `backend/main.py`
- Create: `tests/conftest.py`
- Create: `tests/test_health.py`
- Create: `tests/test_metadata.py`
- Create: `tests/test_prediction.py`

**Interfaces:**
- Consumes: `LoanPredictionService` and `validate_payload` from Tasks 2–3.
- Produces: FastAPI `app` with `GET /`, `GET /health`, `GET /metadata`, and `POST /predict`.

- [ ] **Step 1: Write failing root, health, and metadata endpoint tests**

Assert the API title/version, healthy loaded model response, model identity, required metadata keys, exact expected features, and local Streamlit CORS behavior.

- [ ] **Step 2: Run endpoint tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_health.py tests/test_metadata.py -v`

Expected: collection fails because `backend.main` does not exist.

- [ ] **Step 3: Implement app initialization, CORS, and GET routes**

Instantiate the service once during application initialization and fail fast on model/schema errors. Return only frontend-safe metadata.

- [ ] **Step 4: Run GET endpoint tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_health.py tests/test_metadata.py -v`

Expected: PASS.

- [ ] **Step 5: Write failing prediction and HTTP error tests**

Use encoder-derived categorical values and safe numeric values. Assert success, class membership, probability bounds, exact probability mapping against a direct pipeline call, repeated-request consistency, HTTP 422 details for validation cases, and sanitized HTTP 500 details for a forced inference failure.

- [ ] **Step 6: Run prediction tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_prediction.py -v`

Expected: failures identify the absent POST route and error mapping.

- [ ] **Step 7: Implement `POST /predict` and safe exception mapping**

Accept an arbitrary JSON object, pass it through `validate_payload`, invoke the service, translate `ValidationError` to HTTP 422, and return a generic HTTP 500 detail for unexpected failures.

- [ ] **Step 8: Run all backend tests and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_model_service.py tests/test_validation.py tests/test_health.py tests/test_metadata.py tests/test_prediction.py -v`

Expected: PASS with no unexpected warnings.

```bash
git add backend/main.py tests/conftest.py tests/test_health.py tests/test_metadata.py tests/test_prediction.py
git commit -m "feat: expose loan prediction API"
```

### Task 5: Build the metadata-driven Streamlit frontend

**Files:**
- Create: `frontend/app.py`
- Create: `tests/test_frontend.py`

**Interfaces:**
- Consumes: FastAPI `/metadata` and `/predict` JSON contracts.
- Produces: `friendly_label(name: str) -> str`, `group_for_field(name: str) -> str`, `fetch_metadata(base_url: str, timeout: float = 5.0) -> dict[str, Any]`, `submit_prediction(base_url: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]`, and the Streamlit application.

- [ ] **Step 1: Write failing frontend helper tests**

Test friendly labels, conservative grouping, URL construction, timeouts, backend-unavailable messages, structured HTTP 422 extraction, malformed success responses, neutral result parsing, and nullable payload preservation without sentinel strings.

- [ ] **Step 2: Run frontend tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_frontend.py -v`

Expected: collection fails because `frontend.app` does not exist.

- [ ] **Step 3: Implement HTTP and presentation helpers**

Use `requests` only; do not import backend model modules. Validate metadata/prediction response shape and raise concise user-facing errors.

- [ ] **Step 4: Run helper tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_frontend.py -v`

Expected: PASS.

- [ ] **Step 5: Implement the Streamlit page**

Fetch metadata from an environment-configurable base URL, dynamically render numeric/select controls in conservative groups, offer explicit null controls for nullable fields, send exact raw feature keys, show neutral metrics/progress/success output, and display the required disclaimer and connection/validation errors.

- [ ] **Step 6: Run the entire suite and commit**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

```bash
git add frontend/app.py tests/test_frontend.py
git commit -m "feat: add Streamlit prediction interface"
```

### Task 6: Verify live backend/frontend integration

**Files:**
- Modify only if a failing integration check exposes a tested defect.

**Interfaces:**
- Consumes: completed FastAPI and Streamlit applications.
- Produces: recorded real endpoint examples and verified run commands for documentation.

- [ ] **Step 1: Start Uvicorn in a hidden background process**

Run: `.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`

Expected: server starts without artifact or compatibility errors.

- [ ] **Step 2: Call `/`, `/health`, `/metadata`, and `/predict` with a schema-derived valid payload**

Expected: HTTP 200, healthy model, actual metadata, and a real neutral prediction with correctly mapped probabilities.

- [ ] **Step 3: Start Streamlit in a hidden background process and verify availability**

Run: `.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.headless true --server.port 8501`

Expected: `http://127.0.0.1:8501` responds and the Streamlit process can reach FastAPI.

- [ ] **Step 4: Stop only the two recorded verification processes**

Use their explicit process IDs; do not terminate unrelated Python processes.

- [ ] **Step 5: For any defect, add a failing regression test before applying the fix**

Run the targeted test to RED, implement the minimal fix, then rerun the full suite to GREEN.

### Task 7: Complete project and API documentation

**Files:**
- Create: `docs/API.md`
- Create: `docs/SYSTEM_ARCHITECTURE.md`
- Create: `docs/TEST_CASES.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: actual inspected schema and live responses from Tasks 2 and 6.
- Produces: reproducible setup, run, API, architecture, test, limitation, and demonstration guidance.

- [ ] **Step 1: Write API and architecture documentation using actual values**

Document all four endpoints, real request/response examples, CORS restrictions, the Streamlit-to-FastAPI boundary, and why preprocessing remains bundled with XGBoost.

- [ ] **Step 2: Write at least the 15 requested manual test cases**

Use columns `Test ID`, `Scenario`, `Input`, `Expected Result`, `Actual Result`, and `Status`; leave manual-only results/status blank.

- [ ] **Step 3: Replace the incomplete README with the requested 18 sections**

Include the actual model/schema facts, exact Windows commands, team members, limitations, future work, and a short presentation demo procedure without assigning unverified meanings to classes.

- [ ] **Step 4: Check documentation consistency and commit**

Run searches confirming every documented raw field exists in `models/input_schema.json`, every command uses repository-relative modules, and forbidden semantic labels are absent from result descriptions.

```bash
git add README.md docs/API.md docs/SYSTEM_ARCHITECTURE.md docs/TEST_CASES.md
git commit -m "docs: document loan prediction web application"
```

### Task 8: Final verification and handoff

**Files:**
- Modify only through regression-tested fixes.

**Interfaces:**
- Consumes: the complete repository.
- Produces: evidence required for the final user handoff.

- [ ] **Step 1: Run formatting and repository integrity checks**

Run: `git diff --check`, artifact SHA-256 comparisons, file-size checks, and searches for duplicated preprocessing or model loading in `frontend/`.

Expected: clean diff; copied artifact hashes match sources; only the backend loads the PKL; no application preprocessing duplicates the fitted pipeline.

- [ ] **Step 2: Run the complete automated test suite from a fresh process**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass with the final count recorded.

- [ ] **Step 3: Repeat live endpoint smoke tests**

Start Uvicorn, call `/health`, `/metadata`, and `/predict`, record the exact JSON, then stop the recorded process.

- [ ] **Step 4: Inspect final Git status and repository tree**

Run: `git status --short` and `tree /F` (excluding `.git` and `.venv` from the reported logical tree).

Expected: no accidental files or unrelated modifications.

- [ ] **Step 5: Prepare the final handoff**

Report changed files, tree, PKL confirmation/size/inspection, exact raw/numeric/categorical features, start/test commands, test evidence, real request/response example, unresolved issues, and the university demonstration procedure.
