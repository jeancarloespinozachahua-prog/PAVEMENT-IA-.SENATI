"""Modelo de regresión lineal PCI = b0 + b1 * antigüedad y su evaluación."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

SEMILLA = 42


def entrenar(df: pd.DataFrame) -> LinearRegression:
    """Entrena LinearRegression con antigüedad (entrada) y PCI (salida)."""
    return LinearRegression().fit(df[["antiguedad"]].to_numpy(), df["pci"].to_numpy())


def entrenar_cuadratico(df: pd.DataFrame):
    """Modelo comparativo (grado 2) para discutir la curvatura del deterioro."""
    return make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(
        df[["antiguedad"]].to_numpy(), df["pci"].to_numpy()
    )


def metricas(y_real, y_pred) -> dict[str, float]:
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    return {
        "R2": float(r2_score(y_real, y_pred)) if len(y_real) > 1 else float("nan"),
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
    }


def predecir(modelo, edad, recortar: bool = True):
    """Predice PCI. Con recortar=True limita el resultado a 0-100 (rango físico)."""
    edad = np.atleast_1d(np.asarray(edad, float)).reshape(-1, 1)
    pred = modelo.predict(edad)
    pred = np.clip(pred, 0, 100) if recortar else pred
    return pred if pred.size > 1 else float(pred[0])


def dividir(df: pd.DataFrame, prueba: float = 0.3):
    """Separa entrenamiento y prueba.

    - Si existe la columna 'tramo' con varios tramos: separación por tramo (sin fuga de datos).
    - Si no: separación cronológica (los más antiguos al final se usan como prueba).
    Devuelve (train, test, descripcion).
    """
    if "tramo" in df.columns and df["tramo"].nunique() >= 4:
        gss = GroupShuffleSplit(n_splits=1, test_size=prueba, random_state=SEMILLA)
        i_tr, i_te = next(gss.split(df, groups=df["tramo"]))
        return df.iloc[i_tr], df.iloc[i_te], (
            f"Por tramo: {df.iloc[i_tr]['tramo'].nunique()} tramos entrenamiento / "
            f"{df.iloc[i_te]['tramo'].nunique()} tramos prueba"
        )
    d = df.sort_values("antiguedad")
    k = max(2, int(round(len(d) * prueba)))
    return d.iloc[:-k], d.iloc[-k:], (
        f"Cronológica: antigüedad ≤ {d.iloc[:-k]['antiguedad'].max():g} años para entrenar; "
        f"las {k} más antiguas para prueba (esto es extrapolar)"
    )


def evaluar(df: pd.DataFrame, prueba: float = 0.3) -> dict:
    """Evalúa el modelo: métricas en entrenamiento y prueba + validación cruzada LOO."""
    train, test, desc = dividir(df, prueba)
    mod = entrenar(train)
    pred_te = mod.predict(test[["antiguedad"]].to_numpy())
    pred_tr = mod.predict(train[["antiguedad"]].to_numpy())

    loo_pred = cross_val_predict(
        LinearRegression(), df[["antiguedad"]].to_numpy(), df["pci"].to_numpy(), cv=LeaveOneOut()
    )
    quad = entrenar_cuadratico(train)
    return {
        "division": desc,
        "n_train": len(train),
        "n_test": len(test),
        "train": metricas(train["pci"], pred_tr),
        "test": metricas(test["pci"], pred_te),
        "loocv": metricas(df["pci"], loo_pred),
        "cuadratico_test": metricas(test["pci"], quad.predict(test[["antiguedad"]].to_numpy())),
        "test_df": test.assign(pci_pred=pred_te),
    }


def ajuste_completo(df: pd.DataFrame) -> dict:
    """Ajuste con todos los datos (modelo final usado por la aplicación)."""
    mod = entrenar(df)
    pred = mod.predict(df[["antiguedad"]].to_numpy())
    return {
        "modelo": mod,
        "pendiente": float(mod.coef_[0]),
        "intercepto": float(mod.intercept_),
        "metricas": metricas(df["pci"], pred),
        "residuos": df["pci"].to_numpy() - pred,
        "edad_max_entrenamiento": float(df["antiguedad"].max()),
    }


def anios_hasta_umbral(modelo, edad_actual: float, umbral: float) -> float | None:
    """Años hasta que el PCI estimado cae al umbral (None si la pendiente no es negativa)."""
    m, b = float(modelo.coef_[0]), float(modelo.intercept_)
    if m >= 0:
        return None
    edad_umbral = (umbral - b) / m
    return max(0.0, edad_umbral - edad_actual)


def ecuacion(modelo) -> str:
    m, b = float(modelo.coef_[0]), float(modelo.intercept_)
    signo = "-" if m < 0 else "+"
    return f"PCI = {b:.2f} {signo} {abs(m):.2f} × antigüedad"
