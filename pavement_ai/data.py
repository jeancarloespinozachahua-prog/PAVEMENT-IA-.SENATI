"""Carga, limpieza y validación de datos (Pandas)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNAS_MINIMAS = ["antiguedad", "pci"]
EDAD_MAX = 60  # límite razonable de antigüedad (años)


class DatosInvalidos(ValueError):
    """El conjunto de datos no cumple los requisitos mínimos."""


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Limpia un DataFrame y devuelve (df_limpio, avisos).

    - exige columnas antiguedad y pci (se aceptan mayúsculas/espacios)
    - convierte a numérico, elimina nulos y duplicados exactos
    - descarta PCI fuera de 0-100 y antigüedades fuera de 0-EDAD_MAX
    """
    avisos: list[str] = []
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    faltan = [c for c in COLUMNAS_MINIMAS if c not in df.columns]
    if faltan:
        raise DatosInvalidos(f"Faltan columnas obligatorias: {', '.join(faltan)}")

    n0 = len(df)
    for c in COLUMNAS_MINIMAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nulos = df[COLUMNAS_MINIMAS].isna().any(axis=1).sum()
    if nulos:
        avisos.append(f"Se descartaron {nulos} fila(s) con valores vacíos o no numéricos.")
    df = df.dropna(subset=COLUMNAS_MINIMAS)

    fuera = ~df["pci"].between(0, 100) | ~df["antiguedad"].between(0, EDAD_MAX)
    if fuera.sum():
        avisos.append(f"Se descartaron {int(fuera.sum())} fila(s) fuera de rango (PCI 0-100, antigüedad 0-{EDAD_MAX}).")
    df = df[~fuera]

    dup = df.duplicated().sum()
    if dup:
        avisos.append(f"Se eliminaron {dup} fila(s) duplicadas.")
        df = df.drop_duplicates()

    if len(df) < 3:
        raise DatosInvalidos("Se necesitan al menos 3 registros válidos para entrenar el modelo.")
    if df["antiguedad"].nunique() < 2:
        raise DatosInvalidos("La antigüedad debe tener al menos 2 valores distintos.")
    if len(df) < 50:
        avisos.append(
            f"Solo hay {len(df)} registros; la guía del reto recomienda al menos 50. "
            "Las métricas de prueba son poco estables."
        )
    return df.reset_index(drop=True), avisos


def cargar_csv(origen: str | Path | object) -> tuple[pd.DataFrame, list[str]]:
    """Lee un CSV (ruta o archivo subido) y lo limpia."""
    try:
        df = pd.read_csv(origen)
    except Exception as exc:  # archivo ilegible
        raise DatosInvalidos(f"No se pudo leer el CSV: {exc}") from exc
    return limpiar(df)
