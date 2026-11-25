"""Genera data/inspecciones_simuladas.csv: 10 registros reales del CSV del reto + 50 SIMULADOS.

Cumple la recomendación de la guía (>= 50 registros con tramo, fecha, antigüedad, tránsito,
tipo/severidad de falla y PCI). Todo lo marcado origen='simulado' es FICTICIO y se debe declarar así.
Uso:  python scripts/generar_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)

base = pd.read_csv(RAIZ / "data" / "pavimento_simulado.csv")
curva = lambda a: np.interp(a, base["antiguedad"], base["pci"])  # curva observada del CSV original

PATOLOGIAS = ["Grietas longitudinales", "Grietas transversales", "Grietas de esquina",
              "Losa dividida", "Despostillamiento", "Escalonamiento"]
EFECTO_TRANSITO = {"Bajo": 0.0, "Medio": -0.6, "Alto": -1.2}  # puntos PCI por año de antigüedad

filas = []
# Registros originales del CSV (un único tramo de referencia)
for _, r in base.iterrows():
    filas.append(dict(tramo="T00", fecha_inspeccion="", antiguedad=int(r.antiguedad), transito="No registrado",
                      tipo_falla="No registrado", severidad="", pci=int(r.pci), origen="original"))

# 10 tramos simulados x 5 inspecciones
for t in range(1, 11):
    transito = rng.choice(list(EFECTO_TRANSITO), p=[0.4, 0.35, 0.25])
    desfase = rng.normal(0, 3.5)  # calidad constructiva propia del tramo
    edades = np.sort(rng.choice(np.arange(1, 11), size=5, replace=False))
    for a in edades:
        pci = curva(a) + desfase + EFECTO_TRANSITO[transito] * a + rng.normal(0, 2.0)
        pci = int(round(np.clip(pci, 3, 100)))
        sev = "Baja" if pci > 70 else "Media" if pci > 40 else "Alta"
        fecha = pd.Timestamp("2026-06-30") - pd.Timedelta(days=int(rng.integers(0, 540)))
        filas.append(dict(tramo=f"HCO-T{t:02d}", fecha_inspeccion=fecha.date().isoformat(), antiguedad=int(a),
                          transito=transito, tipo_falla=rng.choice(PATOLOGIAS), severidad=sev, pci=pci,
                          origen="simulado"))

df = pd.DataFrame(filas)
destino = RAIZ / "data" / "inspecciones_simuladas.csv"
df.to_csv(destino, index=False)
print(f"{len(df)} registros -> {destino}")
