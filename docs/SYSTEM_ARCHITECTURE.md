# System Architecture

## Runtime flow

```text
User
  |
  v
Streamlit Frontend (port 8501)
  |  GET /metadata
  |  POST /predict with raw HTTP/JSON values
  v
FastAPI Backend (port 8000)
  |
  v
Metadata-driven Validation
  |
  v
Ordered one-row pandas DataFrame
  |
  v
Saved PKL sklearn Pipeline
  +-- Numeric preprocessing
  |     +-- median imputation
  |     +-- standard scaling
  +-- Categorical preprocessing
  |     +-- most-frequent imputation
  |     +-- fitted one-hot encoding
  +-- Final XGBoost classifier
  |
  v
Predicted class + class-aware probabilities
  |
  v
FastAPI JSON response
  |
  v
Streamlit metrics and probability display
```

## Component responsibilities

### Streamlit frontend

`frontend/app.py` is an HTTP client. It retrieves `/metadata`, dynamically chooses numeric or categorical controls, preserves exact raw feature keys, submits `/predict`, and renders neutral results. It never imports or loads the PKL.

### FastAPI application

`backend/main.py` creates the API, configures local-development CORS, owns routes, and maps validation and inference failures to safe HTTP responses.

### Validation

`backend/validation.py` enforces the generated raw-input contract: exact fields, nullability, finite numeric types, fitted categories, and narrow financial-domain rules. It does not apply learned transformations.

### Model service

`backend/model_service.py` loads the supplied artifact once, verifies fitted structure, checks `input_schema.json` against the pipeline, reconstructs exact raw column order, calls prediction methods, and maps probabilities using `classes_`.

### Deployment artifacts

- `models/final_loan_status_pipeline.pkl` is the unchanged runtime model.
- `models/input_schema.json` is derived from the fitted preprocessor and encoder.
- `notebooks/fdm-miniproject.ipynb` preserves Stages 1–8.

## Why preprocessing and the model stay together

Saving preprocessing and XGBoost in one fitted sklearn pipeline provides:

- the same transformations used during training;
- fewer training/deployment mismatches;
- reproducible inference from raw input values;
- no duplicated encoding, scaling, or imputation logic; and
- less risk of applying preprocessing twice or in the wrong order.

The web application deliberately performs only basic validation. Learned imputation, scaling, and category encoding remain inside the saved pipeline.

## Startup safety

Backend import fails clearly unless the PKL and schema exist, the PKL is non-empty, loading succeeds, preprocessing and XGBoost are fitted, feature names and classes are recoverable, and the schema agrees with the fitted transformer. Therefore `/health` cannot report success for an unavailable model.

## Compatibility

The artifact was serialized with scikit-learn 1.6.1. It fails to load under scikit-learn 1.9.1 because a private serialized class changed. The verified runtime is Python 3.12 with scikit-learn 1.6.1 and XGBoost 3.4.1. The model itself was not changed or regenerated.

## Security and deployment note

The current CORS origins and local HTTP addresses are suitable for a university demonstration. A production deployment should use HTTPS, restrict CORS to its deployed frontend, add authentication where appropriate, avoid logging private applicant data, and manage model artifacts through a controlled release process.
