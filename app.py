from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from flask import Flask, render_template, request

from model_artifact import load_model as load_serialized_model

app = Flask(__name__)

import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lasso_cv_diabetes_model.pkl"

# Lista de campos esperados y sus etiquetas para el formulario.
FEATURE_LABELS: Sequence[Tuple[str, str]] = (
    ("age", "Edad"),
    ("sex", "Sexo (1 = Hombre, 0 = Mujer)"),
    ("bmi", "Índice de Masa Corporal (IMC)"),
    ("bp", "Presión Arterial"),
    ("s1", "Colesterol Total"),
    ("s2", "Colesterol LDL"),
    ("s3", "Colesterol HDL"),
    ("s4", "Triglicéridos"),
    ("s5", "Nivel de Glucosa"),
    ("s6", "Nivel de Insulina"),
)


expected_features: int = len(FEATURE_LABELS)

try:
    model = load_serialized_model()
    model_load_error: str | None = None
    try:
        expected_features = int(getattr(model, "n_features_in_", len(FEATURE_LABELS)))
    except Exception:  # pragma: no cover - atributo inesperado
        expected_features = len(FEATURE_LABELS)
except Exception as exc:  # pragma: no cover - fallo temprano
    model = None
    model_load_error = str(exc)
    expected_features = len(FEATURE_LABELS)
def load_model(model_path: Path) -> object:
    """Carga el modelo entrenado desde disco."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en '{model_path}'."
        )

    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


try:
    model = load_model(MODEL_PATH)
    model_load_error: str | None = None
except Exception as exc:  # pragma: no cover - fallo temprano
    model = None
    model_load_error = str(exc)


def _parse_features(form_data: "ImmutableMultiDict[str, str]") -> List[float]:
    """Convierte los valores del formulario a flotantes."""

    features: List[float] = []
    for field, _ in FEATURE_LABELS:
        raw_value = form_data.get(field)
        if raw_value in (None, ""):
            raise ValueError(f"El campo '{field}' es obligatorio.")
        try:
            features.append(float(raw_value))
        except ValueError as exc:  # pragma: no cover - validación básica
            raise ValueError(
                f"El valor de '{field}' debe ser numérico."
            ) from exc
    if len(features) != expected_features:
        raise ValueError(
            "El modelo fue entrenado con "
            f"{expected_features} características y recibió {len(features)}."
        )
    return features


@app.route("/", methods=["GET", "POST"])
def index():
    prediction: float | None = None
    error_message = model_load_error

    if request.method == "POST":
        if model is None:
            error_message = model_load_error or (
            error_message = (
                "El modelo no se pudo cargar, por lo que no es posible realizar predicciones."
            )
        else:
            try:
                features = _parse_features(request.form)
                prediction = float(model.predict([features])[0])
            except ValueError as exc:
                error_message = str(exc)
            except Exception as exc:  # pragma: no cover - errores inesperados
                error_message = f"No se pudo generar la predicción: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        error_message=error_message,
        feature_labels=FEATURE_LABELS,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
