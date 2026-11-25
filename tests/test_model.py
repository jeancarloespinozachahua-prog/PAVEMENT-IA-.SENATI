import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pavement_ai import data as D  # noqa: E402
from pavement_ai import model as M  # noqa: E402
from pavement_ai import pci as P  # noqa: E402

DATOS = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def df10():
    return D.cargar_csv(DATOS / "pavimento_simulado.csv")[0]


def test_ajuste_original(df10):
    aj = M.ajuste_completo(df10)
    assert aj["pendiente"] == pytest.approx(-8.0606, abs=1e-3)
    assert aj["intercepto"] == pytest.approx(111.1333, abs=1e-3)
    assert aj["metricas"]["R2"] == pytest.approx(0.9843, abs=1e-3)


def test_prediccion_recortada(df10):
    m = M.entrenar(df10)
    assert M.predecir(m, 0) == 100.0   # intercepto 111 -> recortado
    assert M.predecir(m, 20) == 0.0    # negativo -> recortado
    assert M.predecir(m, 20, recortar=False) < 0


def test_condicion():
    assert P.condicion(98) == "Bueno" and P.condicion(25) == "Grave" and P.condicion(5) == "Fallado"


def test_anios_hasta_umbral(df10):
    m = M.entrenar(df10)
    assert M.anios_hasta_umbral(m, 5, 55) == pytest.approx((55 - 111.1333) / -8.0606 - 5, abs=1e-3)


def test_validacion_columnas():
    with pytest.raises(D.DatosInvalidos):
        D.limpiar(pd.DataFrame({"a": [1, 2, 3]}))


def test_limpieza_descarta_basura():
    sucio = pd.DataFrame({"Antiguedad": [1, 2, 3, 4, "x", 5, 5], "PCI": [90, 80, 70, 60, 50, 150, 150]})
    limpio, avisos = D.limpiar(sucio)
    assert len(limpio) == 4 and avisos


def test_pocos_datos():
    with pytest.raises(D.DatosInvalidos):
        D.limpiar(pd.DataFrame({"antiguedad": [1, 2], "pci": [90, 80]}))


def test_division_por_tramo_sin_fuga():
    df = D.cargar_csv(DATOS / "inspecciones_simuladas.csv")[0]
    tr, te, _ = M.dividir(df)
    assert set(tr["tramo"]).isdisjoint(set(te["tramo"]))


def test_evaluar_claves(df10):
    ev = M.evaluar(df10)
    for k in ("train", "test", "loocv"):
        assert {"R2", "MAE", "RMSE"} <= set(ev[k])
    assert ev["test"]["MAE"] > ev["train"]["MAE"]  # extrapolar empeora
