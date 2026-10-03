"""Project-relative configuration for the prediction service."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_loan_status_pipeline.pkl"
SCHEMA_PATH = PROJECT_ROOT / "models" / "input_schema.json"
