# Loan Status Prediction API

Base URL for local development: `http://127.0.0.1:8000`

Interactive Swagger documentation: `http://127.0.0.1:8000/docs`

The API accepts raw feature values. Do not encode, scale, or impute values before submitting them; the saved pipeline performs those fitted transformations.

## `GET /`

Returns basic service information.

```json
{
  "service": "Loan Status Prediction API",
  "status": "running",
  "model": "XGBoost"
}
```

## `GET /health`

Returns HTTP 200 only after application startup has successfully loaded and verified the model and schema.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "XGBoost"
}
```

The application fails during startup if the model or schema cannot be loaded, so it cannot incorrectly advertise a healthy unavailable model.

## `GET /metadata`

Returns the input contract used by the Streamlit frontend.

Summary of the current fitted model:

- Selected model: `XGBoost`
- Target: `Status`
- Raw features: 28
- Numeric features: 7
- Categorical features: 21
- Nulls are accepted for 11 fields with observed missing values in the supplied notebook: `loan_limit`, `approv_in_adv`, `loan_purpose`, `term`, `Neg_ammortization`, `property_value`, `income`, `age`, `submission_of_application`, `LTV`, and `dtir1`. All 28 keys remain mandatory; the fitted pipeline imputes accepted nulls.

### Expected raw feature order

```text
loan_limit
Gender
approv_in_adv
loan_type
loan_purpose
Credit_Worthiness
open_credit
business_or_commercial
loan_amount
term
Neg_ammortization
interest_only
lump_sum_payment
property_value
construction_type
occupancy_type
Secured_by
total_units
income
credit_type
Credit_Score
co-applicant_credit_type
age
submission_of_application
LTV
Region
Security_Type
dtir1
```

### Numeric features

```text
loan_amount, term, property_value, income, Credit_Score, LTV, dtir1
```

### Fitted categorical values

| Feature | Allowed values |
|---|---|
| `loan_limit` | `cf`, `ncf` |
| `Gender` | `Female`, `Joint`, `Male`, `Sex Not Available` |
| `approv_in_adv` | `nopre`, `pre` |
| `loan_type` | `type1`, `type2`, `type3` |
| `loan_purpose` | `p1`, `p2`, `p3`, `p4` |
| `Credit_Worthiness` | `l1`, `l2` |
| `open_credit` | `nopc`, `opc` |
| `business_or_commercial` | `b/c`, `nob/c` |
| `Neg_ammortization` | `neg_amm`, `not_neg` |
| `interest_only` | `int_only`, `not_int` |
| `lump_sum_payment` | `lpsm`, `not_lpsm` |
| `construction_type` | `mh`, `sb` |
| `occupancy_type` | `ir`, `pr`, `sr` |
| `Secured_by` | `home`, `land` |
| `total_units` | `1U`, `2U`, `3U`, `4U` |
| `credit_type` | `CIB`, `CRIF`, `EQUI`, `EXP` |
| `co-applicant_credit_type` | `CIB`, `EXP` |
| `age` | `25-34`, `35-44`, `45-54`, `55-64`, `65-74`, `<25`, `>74` |
| `submission_of_application` | `not_inst`, `to_inst` |
| `Region` | `North`, `North-East`, `central`, `south` |
| `Security_Type` | `Indriect`, `direct` |

Category spelling is preserved from the fitted encoder. In particular, `Sex Not Available` and `Indriect` must not be silently renamed.

## `POST /predict`

Submit one JSON object containing every raw feature. Keys may arrive in any JSON order; the backend explicitly reconstructs the fitted feature order before inference.

### Valid request

```json
{
  "loan_limit": "cf",
  "Gender": "Male",
  "approv_in_adv": "nopre",
  "loan_type": "type1",
  "loan_purpose": "p3",
  "Credit_Worthiness": "l1",
  "open_credit": "nopc",
  "business_or_commercial": "nob/c",
  "loan_amount": 296500.0,
  "term": 360.0,
  "Neg_ammortization": "not_neg",
  "interest_only": "not_int",
  "lump_sum_payment": "not_lpsm",
  "property_value": 418000.0,
  "construction_type": "sb",
  "occupancy_type": "pr",
  "Secured_by": "home",
  "total_units": "1U",
  "income": 5760.0,
  "credit_type": "CIB",
  "Credit_Score": 699.0,
  "co-applicant_credit_type": "CIB",
  "age": "45-54",
  "submission_of_application": "to_inst",
  "LTV": 75.12254902,
  "Region": "North",
  "Security_Type": "direct",
  "dtir1": 39.0
}
```

### Verified response

The supplied artifact returned this response during live verification:

```json
{
  "predicted_class": 0,
  "model": "XGBoost",
  "probability_class_0": 0.8956873416900635,
  "probability_class_1": 0.10431265830993652
}
```

The backend pairs probability columns with the classifier's actual `classes_` values (`[0, 1]`). It does not assume that array index 1 always represents class label 1.

### Validation error

Invalid input returns HTTP 422. For example:

```json
{
  "detail": {
    "message": "Invalid value for loan_type",
    "field": "loan_type",
    "allowed_values": ["type1", "type2", "type3"]
  }
}
```

Missing fields, unexpected fields, non-numeric values, non-finite numbers, invalid categories, and logically invalid negative values are rejected. Historical training minima and maxima are not treated as strict API limits.

### Unexpected server error

Unexpected inference failures return HTTP 500 without exposing tracebacks or local paths:

```json
{
  "detail": "Prediction failed unexpectedly"
}
```

## Target-label safety

The project documentation does not verify the business meaning of `Status` 0 and 1. The API therefore reports only neutral class identifiers and class probabilities. Consumers must not relabel them as approval, rejection, default, non-default, safe, or risky without verified target documentation.

## CORS

Local development permits `http://localhost:8501` and `http://127.0.0.1:8501`. Production deployments must replace these with the actual trusted frontend origin.
