"""Clasificación visual local experimental a partir de fotos etiquetadas."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pavement_ai import pci

MIN_MUESTRAS_POR_CLASE = 5
TAMANO_IMAGEN = (128, 128)
CAMPOS = ("patologia", "severidad")


class ImagenInvalida(ValueError):
    """La imagen no se puede decodificar o no tiene un formato compatible."""


@dataclass
class ModeloVision:
    modelos: dict[str, Pipeline]
    conteos: dict[str, dict[str, int]]
    imagenes_invalidas: int


def _caracteristicas(contenido: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(contenido)) as imagen:
            gris = np.asarray(
                imagen.convert("RGB").resize(TAMANO_IMAGEN),
                dtype=np.float32,
            ) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagenInvalida("El archivo no contiene una imagen PNG o JPEG válida.") from exc

    return hog(
        rgb2gray(gris),
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def entrenar(
    ejemplos: Iterable[tuple[str, str, str, int]],
    directorio_fotos: str | Path,
) -> ModeloVision:
    """Entrena por patología y severidad usando (archivo, mtime, patología, severidad)."""
    nombres_validos = {
        "patologia": set(pci.PATOLOGIAS),
        "severidad": set(pci.SEVERIDADES),
    }
    caracteristicas: dict[str, list[np.ndarray]] = {campo: [] for campo in CAMPOS}
    etiquetas: dict[str, list[str]] = {campo: [] for campo in CAMPOS}
    invalidas = 0
    raiz = Path(directorio_fotos)

    for archivo, _, patologia, severidad in ejemplos:
        if not archivo or Path(archivo).name != archivo:
            invalidas += 1
            continue
        ruta = raiz / archivo
        if ruta.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            invalidas += 1
            continue
        try:
            vector = _caracteristicas(ruta.read_bytes())
        except (OSError, ImagenInvalida):
            invalidas += 1
            continue

        for campo, etiqueta in (("patologia", patologia), ("severidad", severidad)):
            if etiqueta in nombres_validos[campo]:
                caracteristicas[campo].append(vector)
                etiquetas[campo].append(etiqueta)

    modelos: dict[str, Pipeline] = {}
    conteos: dict[str, dict[str, int]] = {}
    for campo in CAMPOS:
        permitidas = pci.PATOLOGIAS if campo == "patologia" else pci.SEVERIDADES
        conteo = Counter(etiquetas[campo])
        conteos[campo] = {etiqueta: conteo.get(etiqueta, 0) for etiqueta in permitidas}
        if len(conteo) < 2 or min(conteo.values()) < MIN_MUESTRAS_POR_CLASE:
            continue
        clasificador = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
        )
        clasificador.fit(np.stack(caracteristicas[campo]), np.asarray(etiquetas[campo]))
        modelos[campo] = clasificador

    return ModeloVision(modelos, conteos, invalidas)


def predecir(modelo: ModeloVision, contenido: bytes) -> dict[str, tuple[str, float]]:
    """Devuelve etiquetas y probabilidades para los clasificadores que estén entrenados."""
    vector = _caracteristicas(contenido).reshape(1, -1)
    resultados: dict[str, tuple[str, float]] = {}
    for campo, clasificador in modelo.modelos.items():
        probabilidades = clasificador.predict_proba(vector)[0]
        indice = int(np.argmax(probabilidades))
        etiqueta = str(clasificador.classes_[indice])
        resultados[campo] = (etiqueta, float(probabilidades[indice]))
    return resultados
