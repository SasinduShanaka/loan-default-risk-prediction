"""Loading, inspection, and inference for the fitted loan pipeline."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from backend.config import MODEL_PATH, SCHEMA_PATH


@dataclass(frozen=True)
class PipelineInspection:
    """Raw-input contract recovered from a fitted pipeline."""

    preprocessor_step: str
    model_step: str
    expected_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    categories: dict[str, list[Any]]
    defaults: dict[str, Any]
    classes: list[Any]


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_pipeline(path: Path) -> Pipeline:
    """Load the supplied artifact without recreating or modifying it."""

    if not path.exists():
        raise RuntimeError(f"Model file not found: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Model file is empty: {path}")

    pickle_error: Exception | None = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"(?s).*If you are loading a serialized model.*",
                category=UserWarning,
            )
            with path.open("rb") as model_file:
                loaded = pickle.load(model_file)
    except Exception as exc:  # pragma: no cover - fallback depends on artifact format
        pickle_error = exc
        try:
            loaded = joblib.load(path)
        except Exception as joblib_error:
            raise RuntimeError(
                "Unable to load the model artifact with pickle or joblib"
            ) from joblib_error

    if not isinstance(loaded, Pipeline):
        loader = "joblib" if pickle_error else "pickle"
        raise RuntimeError(f"{loader} artifact is not an sklearn Pipeline")
    return loaded


def _find_steps(pipeline: Pipeline) -> tuple[str, str]:
    preprocessor_step = next(
        (
            name
            for name, step in pipeline.named_steps.items()
            if isinstance(step, ColumnTransformer)
        ),
        None,
    )
    model_step = next(
        (
            name
            for name, step in reversed(list(pipeline.named_steps.items()))
            if isinstance(step, XGBClassifier)
        ),
        None,
    )
    if preprocessor_step is None:
        raise RuntimeError("Pipeline does not contain a fitted ColumnTransformer")
    if model_step is None:
        raise RuntimeError("Pipeline does not contain an XGBoost classifier")
    return preprocessor_step, model_step


def _named_child(step: Any, child_type: type[Any]) -> Any | None:
    if isinstance(step, child_type):
        return step
    if isinstance(step, Pipeline):
        return next(
            (child for child in step.named_steps.values() if isinstance(child, child_type)),
            None,
        )
    return None


def inspect_pipeline(pipeline: Pipeline) -> PipelineInspection:
    """Recover fitted raw columns, categories, defaults, and model classes."""

    preprocessor_step, model_step = _find_steps(pipeline)
    preprocessor = pipeline.named_steps[preprocessor_step]
    model = pipeline.named_steps[model_step]

    if not hasattr(pipeline, "feature_names_in_"):
        raise RuntimeError("Fitted pipeline does not expose raw feature_names_in_")
    if not hasattr(preprocessor, "transformers_"):
        raise RuntimeError("Preprocessor is not fitted")
    if not hasattr(model, "classes_"):
        raise RuntimeError("Classifier is not fitted or does not expose classes_")

    expected_features = [str(value) for value in pipeline.feature_names_in_.tolist()]
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    categories: dict[str, list[Any]] = {}
    defaults: dict[str, Any] = {}

    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or not columns:
            continue
        feature_names = [str(column) for column in columns]
        encoder = _named_child(transformer, OneHotEncoder)
        imputer = _named_child(transformer, SimpleImputer)
        if imputer is None or not hasattr(imputer, "statistics_"):
            raise RuntimeError("Every deployed feature group must have a fitted imputer")

        statistics = [_python_scalar(value) for value in imputer.statistics_.tolist()]
        if len(statistics) != len(feature_names):
            raise RuntimeError("Imputer statistics do not align with feature names")

        if encoder is None:
            numeric_features.extend(feature_names)
            defaults.update(dict(zip(feature_names, statistics)))
            continue

        if not hasattr(encoder, "categories_"):
            raise RuntimeError("Categorical encoder is not fitted")
        if len(encoder.categories_) != len(feature_names):
            raise RuntimeError("Encoder categories do not align with feature names")
        categorical_features.extend(feature_names)
        defaults.update(dict(zip(feature_names, statistics)))
        for name, values in zip(feature_names, encoder.categories_):
            categories[name] = [_python_scalar(value) for value in values.tolist()]

    grouped = numeric_features + categorical_features
    if len(grouped) != len(set(grouped)) or set(grouped) != set(expected_features):
        raise RuntimeError("Preprocessor feature groups do not match pipeline inputs")

    return PipelineInspection(
        preprocessor_step=preprocessor_step,
        model_step=model_step,
        expected_features=expected_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categories=categories,
        defaults=defaults,
        classes=[_python_scalar(value) for value in model.classes_.tolist()],
    )


def derive_input_schema(pipeline: Pipeline) -> dict[str, Any]:
    """Build a JSON-safe raw-input schema from fitted preprocessing state."""

    inspection = inspect_pipeline(pipeline)
    numeric = set(inspection.numeric_features)
    fields: dict[str, dict[str, Any]] = {}
    for name in inspection.expected_features:
        if name in numeric:
            fields[name] = {
                "type": "number",
                "nullable": True,
                "default": _python_scalar(inspection.defaults[name]),
            }
        else:
            fields[name] = {
                "type": "category",
                "nullable": True,
                "allowed_values": inspection.categories[name],
                "default": _python_scalar(inspection.defaults[name]),
            }

    return {
        "selected_model": "XGBoost",
        "target": "Status",
        "expected_features": inspection.expected_features,
        "numeric_features": inspection.numeric_features,
        "categorical_features": inspection.categorical_features,
        "fields": fields,
    }


def _validate_schema(schema: Mapping[str, Any], inspection: PipelineInspection) -> None:
    if schema.get("expected_features") != inspection.expected_features:
        raise RuntimeError("Schema feature order does not match fitted pipeline")
    if schema.get("numeric_features") != inspection.numeric_features:
        raise RuntimeError("Schema numeric feature group does not match fitted pipeline")
    if schema.get("categorical_features") != inspection.categorical_features:
        raise RuntimeError("Schema categorical feature group does not match fitted pipeline")
    fields = schema.get("fields")
    if not isinstance(fields, dict) or list(fields) != inspection.expected_features:
        raise RuntimeError("Schema fields do not match fitted pipeline feature order")
    for name in inspection.categorical_features:
        if fields[name].get("allowed_values") != inspection.categories[name]:
            raise RuntimeError(f"Schema categories for {name} do not match fitted encoder")


def _class_key(value: Any) -> str:
    value = _python_scalar(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


class LoanPredictionService:
    """Single loaded pipeline plus its verified deployment schema."""

    def __init__(self, model_path: Path = MODEL_PATH, schema_path: Path = SCHEMA_PATH):
        self.pipeline = load_pipeline(model_path)
        self.inspection = inspect_pipeline(self.pipeline)
        if not schema_path.exists():
            raise RuntimeError(f"Input schema file not found: {schema_path}")
        try:
            self.metadata = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load input schema: {schema_path}") from exc
        _validate_schema(self.metadata, self.inspection)
        self.expected_features = list(self.inspection.expected_features)
        self.classes = list(self.inspection.classes)
        self.ready = True

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        input_frame = pd.DataFrame([dict(payload)], columns=self.expected_features)
        predicted_class = _python_scalar(self.pipeline.predict(input_frame)[0])
        result: dict[str, Any] = {
            "predicted_class": predicted_class,
            "model": self.metadata["selected_model"],
        }
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(input_frame)[0]
            for class_value, probability in zip(self.classes, probabilities):
                result[f"probability_class_{_class_key(class_value)}"] = float(probability)
        return result
