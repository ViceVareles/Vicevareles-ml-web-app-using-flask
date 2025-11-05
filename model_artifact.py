"""Embedded regression coefficients for a lightweight predictor.

This module mirrors the behaviour of the original StandardScaler + LassoCV
pipeline but removes the runtime dependency on scikit-learn. The coefficients
were exported from the trained pipeline and are used by a small Python class to
produce identical predictions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

# Parámetros del StandardScaler entrenado
FEATURE_MEANS: Sequence[float] = (
    -1.44429466e-18,
    2.54321451e-18,
    -2.25592546e-16,
    -4.85408596e-17,
    -1.42859580e-17,
    3.89881064e-17,
    -6.02836031e-18,
    -1.78809958e-17,
    9.24348582e-17,
    1.35176953e-17,
)

FEATURE_SCALES: Sequence[float] = (
    0.04756515,
    0.04756515,
    0.04756515,
    0.04756515,
    0.04756515,
    0.04756515,
    0.04756515,
    0.04756515,
    0.04756515,
    0.04756515,
)

# Coeficientes del modelo LassoCV ya entrenado (sobre los datos escalados)
MODEL_COEFFICIENTS: Sequence[float] = (
    -0.30892106,
    -11.22504613,
    24.81684888,
    15.27130382,
    -27.08540992,
    14.38623131,
    0.0,
    6.83504132,
    31.86497214,
    3.17904105,
)

MODEL_INTERCEPT: float = 152.13348416289594


class EmbeddedRegressionModel:
    """Reimplementación ligera del pipeline StandardScaler + LassoCV."""

    def __init__(
        self,
        feature_means: Sequence[float],
        feature_scales: Sequence[float],
        coefficients: Sequence[float],
        intercept: float,
    ) -> None:
        self._feature_means = tuple(float(v) for v in feature_means)
        self._feature_scales = tuple(float(v) for v in feature_scales)
        self._coefficients = tuple(float(v) for v in coefficients)
        self._intercept = float(intercept)

        if not (
            len(self._feature_means)
            == len(self._feature_scales)
            == len(self._coefficients)
        ):
            raise ValueError("Los parámetros del modelo tienen longitudes incompatibles.")

    @property
    def n_features_in_(self) -> int:
        return len(self._coefficients)

    def _transform_row(self, row: Sequence[float]) -> Iterable[float]:
        for value, mean, scale in zip(row, self._feature_means, self._feature_scales):
            yield (float(value) - mean) / scale

    def predict(self, rows: Sequence[Sequence[float]]) -> List[float]:
        predictions: List[float] = []
        for row in rows:
            if len(row) != self.n_features_in_:
                raise ValueError(
                    "El modelo fue entrenado con "
                    f"{self.n_features_in_} características y recibió {len(row)}."
                )

            transformed = list(self._transform_row(row))
            prediction = self._intercept
            for value, coefficient in zip(transformed, self._coefficients):
                prediction += coefficient * value
            predictions.append(prediction)
        return predictions


@lru_cache(maxsize=1)
def load_model() -> EmbeddedRegressionModel:
    """Devuelve el modelo incrustado ya inicializado."""

    return EmbeddedRegressionModel(
        FEATURE_MEANS,
        FEATURE_SCALES,
        MODEL_COEFFICIENTS,
        MODEL_INTERCEPT,
    )
