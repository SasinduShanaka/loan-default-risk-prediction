# Loan Default Risk Prediction

## 1. Project Overview

This repository contains a binary loan-status classification project and its end-to-end demonstration application. A Streamlit interface collects raw applicant and loan details, a FastAPI service validates them, and the supplied fitted pipeline applies its original preprocessing before producing an XGBoost class prediction and probabilities.

The interface displays a predicted repayment outcome using the project owner's supplied meanings: class 0 is “Repayments were handled normally” and class 1 is “The borrower failed to meet the repayment obligation.” Numeric classes remain in API responses.

## 2. Academic Context

- Course: **IT3051 – Fundamentals of Data Mining**
- Assessment: **Mini Project 2026**
- Application work: Stage 9 backend and Stage 10 frontend

The original notebook containing Stages 1–8 is preserved at `notebooks/fdm-miniproject.ipynb`.

## 3. Business Problem

Financial institutions evaluate many borrower, loan, credit, and property attributes when assessing lending outcomes. This project demonstrates how historical loan data can be used to train a classification pipeline and expose its result through a usable decision-support interface.

The output must not be used as the sole basis for a financial decision.

## 4. Dataset Summary

The original dataset contains approximately 148,670 rows and 34 columns. The target is `Status`, with an approximate distribution of 75% Class 0 and 25% Class 1.

Identifiers and unsuitable fields were removed during the completed modelling work. These include `ID`, `year`, and the conservatively excluded potential-leakage fields `Interest_rate_spread`, `rate_of_interest`, and `Upfront_charges`.

## 5. Machine Learning Workflow

Stages 1–8 include:

1. dataset inspection and cleaning;
2. exploratory data analysis;
3. target and leakage investigation;
4. train/test splitting;
5. median imputation and standardization for numeric features;
6. most-frequent imputation and one-hot encoding for categorical features;
7. model comparison and hyperparameter tuning; and
8. final model evaluation and pipeline serialization.

The web application does not reproduce these learned transformations. It submits raw values to the fitted pipeline.

## 6. Models Compared

The project compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## 7. Final Selected Model

XGBoost is the selected final classifier. The unchanged fitted artifact is:

```text
models/final_loan_status_pipeline.pkl
```

The 746,857-byte PKL contains an sklearn `Pipeline` with:

- `preprocessor`: fitted numeric and categorical transformations;
- `model`: fitted `XGBClassifier`; and
- classifier classes `[0, 1]`.

The artifact was serialized with scikit-learn 1.6.1. The verified runtime uses Python 3.12.15, scikit-learn 1.6.1, and XGBoost 3.4.1. Python 3.14 with scikit-learn 1.9.1 cannot load this pickle because a private scikit-learn serialization class changed.

## 8. Web Application

The Streamlit frontend retrieves the exact input schema from FastAPI and builds:

- six numeric inputs and an automatically calculated, read-only LTV percentage;
- 21 categorical dropdowns with readable labels where meanings are known; and
- blank initial fields, required entries, and sequential navigation.

Clicking **Predict Loan Status** sends raw JSON to FastAPI. The UI displays the predicted repayment outcome and both probabilities in responsive cards. Editing an input clears the previous prediction. Missing entries quietly disable Next; malformed numeric entries receive inline feedback.

Set `LOAN_CURRENCY` before starting Streamlit to label monetary fields and review values with the confirmed currency (for example, `$env:LOAN_CURRENCY = "USD"` in PowerShell). No currency is assumed when this setting is absent. This setting only labels values; it does not convert them. Loan amount, property value, and monthly income must use the same currency and units expected by the model.

## 9. System Architecture

```text
User
  -> Streamlit frontend
  -> HTTP/JSON
  -> FastAPI backend
  -> metadata-driven validation
  -> saved sklearn pipeline
       -> trained preprocessing
       -> XGBoost
  -> prediction JSON
  -> Streamlit result
```

See [System Architecture](docs/SYSTEM_ARCHITECTURE.md) for details.

## 10. Repository Structure

```text
loan-default-risk-prediction/
|-- backend/
|   |-- __init__.py
|   |-- config.py
|   |-- main.py
|   |-- model_service.py
|   `-- validation.py
|-- frontend/
|   |-- __init__.py
|   `-- app.py
|-- models/
|   |-- final_loan_status_pipeline.pkl
|   `-- input_schema.json
|-- notebooks/
|   `-- fdm-miniproject.ipynb
|-- tests/
|   |-- __init__.py
|   |-- conftest.py
|   |-- test_frontend.py
|   |-- test_health.py
|   |-- test_metadata.py
|   |-- test_model_service.py
|   |-- test_prediction.py
|   `-- test_validation.py
|-- docs/
|   |-- API.md
|   |-- SYSTEM_ARCHITECTURE.md
|   `-- TEST_CASES.md
|-- .gitignore
|-- pytest.ini
|-- requirements.txt
`-- README.md
```

## 11. Installation

Install Python 3.12 before creating the environment. The serialized model requires scikit-learn 1.6.1, which does not provide a Python 3.14 wheel.

From the repository root on Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If the `python` command already resolves to Python 3.12, the environment can also be created with:

```powershell
python -m venv .venv
```

## 12. Running Backend

From the repository root with the environment activated:

```powershell
uvicorn backend.main:app --reload
```

- API: <http://127.0.0.1:8000>
- Swagger: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>

The backend fails at startup if the model or schema is missing, empty, incompatible, unfitted, or inconsistent.

## 13. Running Frontend

Open a second activated terminal at the repository root:

```powershell
streamlit run frontend/app.py
```

Open <http://localhost:8501>.

The default API address is `http://127.0.0.1:8000`. Override it when necessary:

```powershell
$env:LOAN_API_URL = "http://127.0.0.1:8000"
streamlit run frontend/app.py
```

## 14. Running Tests

```powershell
pytest -q
```

The tests cover artifact integrity, fitted-pipeline inspection, schema agreement, validation, all API endpoints, real prediction/probability equivalence, consistent repeated predictions, CORS, frontend HTTP behavior, and error handling.

Manual cases are listed in [Test Cases](docs/TEST_CASES.md).

## 15. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service identity |
| GET | `/health` | Verified model readiness |
| GET | `/metadata` | Frontend-safe fitted input schema |
| POST | `/predict` | Validate raw values and return a prediction |

See [API Documentation](docs/API.md) for exact features, fitted categories, and verified examples.

## 16. Limitations

- The semantic meanings of `Status` 0 and 1 are not verified.
- Predictions reflect the supplied historical data and fitted model; they may inherit dataset bias or drift over time.
- The frontend is a local university demonstration and does not include authentication or persistent storage.
- Nulls are accepted only for the 11 fields that contain missing values in the supplied notebook (`loan_limit`, `approv_in_adv`, `loan_purpose`, `term`, `Neg_ammortization`, `property_value`, `income`, `age`, `submission_of_application`, `LTV`, and `dtir1`). Every request must still contain all 28 keys; accepted nulls are imputed by the fitted pipeline.
- The XGBoost artifact emits a compatibility advisory when loaded by a newer XGBoost runtime, although verified loading and prediction succeed with XGBoost 3.4.1.
- The model must only be loaded from a trusted source because pickle files can execute code during deserialization.

## 17. Future Work

- Verify and document the real-world meanings of target classes.
- Define business-level required-field rules with domain stakeholders.
- Add authenticated deployment with HTTPS and restricted production CORS.
- Monitor model performance, data drift, and fairness after deployment.
- Export future model releases using stable, versioned model formats alongside environment metadata.

## 18. Team Members

- Shanaka
- Praneesha
- Dilki
- Dinuja

## Presentation Demo

1. Start FastAPI and open `/health` to show that the fitted XGBoost pipeline is loaded.
2. Open `/metadata` or Swagger to show that raw inputs and dropdown values come from the fitted pipeline.
3. Start Streamlit and point out the dynamically generated grouped form.
4. Submit the defaults and show the neutral class result and both probabilities.
5. Select `Sex Not Available` to demonstrate exact category compatibility.
6. Briefly enter an invalid value through Swagger to demonstrate readable HTTP 422 validation.
7. Explain that Streamlit never loads the PKL and that preprocessing runs exactly once inside the saved pipeline.
