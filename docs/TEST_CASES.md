# Manual Test Cases

Automated coverage is provided by `pytest -q`. The following cases support manual demonstration and acceptance testing. Fill in **Actual Result** and **Status** during manual execution.

| Test ID | Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| TC01 | Valid normal prediction | Submit all defaults shown by the Streamlit form | HTTP 200; UI displays a neutral predicted class and probabilities |  |  |
| TC02 | Class 0 prediction | Submit the complete default payload from `docs/API.md` | Predicted Class 0; class probabilities appear and sum to approximately 1 |  |  |
| TC03 | Class 1 prediction | Use numeric defaults and the first allowed encoder value for every categorical field | Predicted Class 1; class probabilities appear and sum to approximately 1 |  |  |
| TC04 | Missing required field | Remove `loan_amount` from the JSON request | HTTP 422 with `Missing required fields` and `loan_amount` |  |  |
| TC05 | Nullable field equals null | Send the complete payload with `dtir1: null` | HTTP 200; the fitted numeric imputer handles the null |  |  |
| TC06 | Negative numeric input | Send `loan_amount: -1` | HTTP 422 stating `loan_amount must be greater than zero` |  |  |
| TC07 | Wrong numeric data type | Send `loan_amount: "296500"` | HTTP 422 stating `Invalid numeric value for loan_amount` |  |  |
| TC08 | Invalid category | Send `loan_type: "type4"` | HTTP 422 listing the fitted allowed values |  |  |
| TC09 | Empty request | POST `{}` to `/predict` | HTTP 422 stating `Request body must not be empty` |  |  |
| TC10 | Unexpected extra field | Add `extra: 1` to a complete request | HTTP 422 listing `extra` as unexpected |  |  |
| TC11 | Extreme but valid numeric value | Send `income: 1000000000` with all other fields valid | Input passes validation; pipeline returns a prediction |  |  |
| TC12 | Backend unavailable | Stop Uvicorn and open/reload Streamlit | UI explains that it cannot connect and asks the user to start the backend |  |  |
| TC13 | Frontend/backend integration | Start both services, complete the form, and click **Predict Loan Status** | Streamlit sends raw JSON to FastAPI and displays the returned result |  |  |
| TC14 | Probability display | Complete any successful prediction | Probability of Class 1 is shown as a percentage and progress bar; Class 0 is optionally shown |  |  |
| TC15 | Repeated prediction consistency | Submit the identical valid payload twice | Both responses contain the same class and probabilities |  |  |

## Additional service checks

| Test ID | Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| TC16 | Health endpoint | `GET /health` | HTTP 200 with `status: ok` and `model_loaded: true` |  |  |
| TC17 | Metadata endpoint | `GET /metadata` | HTTP 200 with 28 raw features, 7 numeric features, and 21 categorical features |  |  |
| TC18 | Exact historical category | Choose `Sex Not Available` for Gender | Category is sent unchanged and prediction succeeds |  |  |
| TC19 | Zero term | Send `term: 0` | HTTP 422 stating `term must be greater than zero` |  |  |
| TC20 | Swagger availability | Open `http://127.0.0.1:8000/docs` | Interactive API documentation loads |  |  |
