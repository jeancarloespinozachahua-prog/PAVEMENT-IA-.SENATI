import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pavement_ai import vision as V  # noqa: E402


def _png_bytes(orientacion: str, desplazamiento: int = 0) -> bytes:
    imagen = np.zeros((128, 128, 3), dtype=np.uint8)
    if orientacion == "vertical":
        imagen[:, 30 + desplazamiento:35 + desplazamiento] = 255
    else:
        imagen[30 + desplazamiento:35 + desplazamiento, :] = 255
    buffer = io.BytesIO()
    Image.fromarray(imagen).save(buffer, format="PNG")
    return buffer.getvalue()


def test_no_entrena_con_muestras_insuficientes(tmp_path):
    for i in range(4):
        (tmp_path / f"foto{i}.png").write_bytes(_png_bytes("vertical", i))
    ejemplos = [
        (f"foto{i}.png", "0", "Grietas longitudinales", "Baja")
        for i in range(4)
    ]

    modelo = V.entrenar(ejemplos, tmp_path)

    assert modelo.modelos == {}
    assert modelo.conteos["patologia"]["Grietas longitudinales"] == 4


def test_entrena_y_predice_clases_con_suficientes_fotos(tmp_path):
    ejemplos = []
    for i in range(V.MIN_MUESTRAS_POR_CLASE):
        vertical = f"vertical{i}.png"
        horizontal = f"horizontal{i}.png"
        (tmp_path / vertical).write_bytes(_png_bytes("vertical", i))
        (tmp_path / horizontal).write_bytes(_png_bytes("horizontal", i))
        ejemplos.extend([
            (vertical, str(i), "Grietas longitudinales", "Baja"),
            (horizontal, str(i), "Grietas transversales", "Alta"),
        ])

    modelo = V.entrenar(ejemplos, tmp_path)
    prediccion = V.predecir(modelo, _png_bytes("vertical", 2))

    assert prediccion["patologia"][0] == "Grietas longitudinales"
    assert prediccion["severidad"][0] == "Baja"
    assert 0 <= prediccion["patologia"][1] <= 1


def test_rechaza_archivo_que_no_es_imagen():
    with pytest.raises(V.ImagenInvalida):
        V.predecir(V.ModeloVision({}, {}, 0), b"no es una imagen")
