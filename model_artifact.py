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
"""Auto-generated base64 encoding of the StandardScaler + LassoCV pipeline."""

from __future__ import annotations

import base64
import pickle
from functools import lru_cache
from typing import Any

MODEL_BASE64 = (
    "gASVLxgAAAAAAACMEHNrbGVhcm4ucGlwZWxpbmWUjAhQaXBlbGluZZSTlCmBlH2UKIwFc3RlcHOU"
    "XZQojAZzY2FsZXKUjBtza2xlYXJuLnByZXByb2Nlc3NpbmcuX2RhdGGUjA5TdGFuZGFyZFNjYWxl"
    "cpSTlCmBlH2UKIwJd2l0aF9tZWFulIiMCHdpdGhfc3RklIiMBGNvcHmUiIwObl9mZWF0dXJlc19p"
    "bl+USwqMD25fc2FtcGxlc19zZWVuX5SMFm51bXB5Ll9jb3JlLm11bHRpYXJyYXmUjAZzY2FsYXKU"
    "k5SMBW51bXB5lIwFZHR5cGWUk5SMAmk4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGJD"
    "CLoBAAAAAAAAlIaUUpSMBW1lYW5flGgSjAxfcmVjb25zdHJ1Y3SUk5RoFYwHbmRhcnJheZSTlEsA"
    "hZRDAWKUh5RSlChLAUsKhZRoF4wCZjiUiYiHlFKUKEsDaBtOTk5K/////0r/////SwB0lGKJQ1Ab"
    "faQafaQ6vNf+dNf+dEc8rnLBrXJBsLzUXfvTXfuLvHh4eHh4eHC8A6F5A6F5hjwcCM0bCM1bvNmJ"
    "ndiJnXS8G32kGn2kmjxdcKtccCtvPJR0lGKMBHZhcl+UaCJoJEsAhZRoJoeUUpQoSwFLCoWUaCyJ"
    "Q1APsIgSsIhiPzqwiBKwiGI/ELCIErCIYj8UsIgSsIhiPwOwiBKwiGI/F7CIErCIYj8LsIgSsIhi"
    "PwSwiBKwiGI/FbCIErCIYj8bsIgSsIhiP5R0lGKMBnNjYWxlX5RoImgkSwCFlGgmh5RSlChLAUsK"
    "hZRoLIlDUPtgVpJ1Wqg/F2FWknVaqD/8YFaSdVqoP/5gVpJ1Wqg/82BWknVaqD8AYVaSdVqoP/hg"
    "VpJ1Wqg/9GBWknVaqD//YFaSdVqoPwNhVpJ1Wqg/lHSUYowQX3NrbGVhcm5fdmVyc2lvbpSMBTEu"
    "Ny4xlHVihpSMBW1vZGVslIwoc2tsZWFybi5saW5lYXJfbW9kZWwuX2Nvb3JkaW5hdGVfZGVzY2Vu"
    "dJSMB0xhc3NvQ1aUk5QpgZR9lCiMA2Vwc5RHP1BiTdLxqfyMCG5fYWxwaGFzlIwKZGVwcmVjYXRl"
    "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"
)


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Decode and unpickle the bundled regression pipeline."""

    return pickle.loads(base64.b64decode(MODEL_BASE64))
