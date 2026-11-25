# PAVEMENT AI · Perú

Predicción del deterioro del pavimento rígido (índice PCI) mediante **regresión lineal**.
Proyecto SENATI – Inteligencia Artificial (reto individual, 20 puntos). Huánuco, Perú.

> ⚠️ Todos los datos incluidos son **ficticios** (educativos). El modelo no reemplaza la metodología PCI (ASTM D6433) ni una inspección estructural.

## Contenido

```
app.py                       Aplicación Streamlit (7 pestañas)
pavement_ai/                 Código del modelo
  data.py                    Carga, limpieza y validación (Pandas)
  model.py                   LinearRegression, métricas, división train/test, LOOCV
  pci.py                     Escala de condición y catálogo de patologías
  vision.py                  Clasificación visual local experimental
data/pavimento_simulado.csv  10 registros del reto (antigüedad, PCI)
data/inspecciones_simuladas.csv  60 registros: 10 originales + 50 SIMULADOS (columna `origen`)
scripts/generar_dataset.py   Genera el dataset ampliado (semilla fija)
scripts/generar_informe.py   Genera el informe PDF con los resultados reales
tests/test_model.py          Pruebas automáticas (pytest)
informe/informe_pavement_ai.pdf  Informe con evidencias y conclusiones
docs/guia_reto_original.html Guía original del reto (prototipo HTML)
```

## Instalación y uso

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Pruebas: `python -m pytest -q`  ·  Regenerar informe: `python scripts/generar_informe.py`

## Funcionalidades

- **Dashboard**: KPIs, gráfico antigüedad vs PCI con zona de extrapolación y umbral, condición de los tramos, patologías frecuentes.
- **Datos**: elegir dataset original, ampliado o **subir tu propio CSV** (`antiguedad`, `pci`; opcional `tramo`), con limpieza y avisos.
- **Modelo**: ecuación, pendiente/intercepto interpretados, R²/MAE/RMSE en entrenamiento, prueba y validación cruzada LOO, residuos, comparación con modelo de grado 2 y preguntas de sustentación.
- **Simulador**: PCI actual y futuro, condición, años hasta el umbral de intervención, recomendación de inspección, aviso de extrapolación y tres escenarios de referencia.
- **Análisis de imagen (experimental)**: clasificación visual local de patología y severidad usando fotos etiquetadas guardadas en el registro. Requiere al menos 5 fotos por categoría y 2 categorías para cada estimación. No calcula PCI ni reemplaza la inspección; mientras no haya suficientes ejemplos, indica que no puede predecir. Las fotos no se envían a servicios externos.
- **Registro de patologías**: 6 tipos de falla, severidad, área, PCI observado opcional y fotografía; validación de entradas y protección contra inyección de fórmulas en Excel. Se guarda en `data/patologias_registradas.csv` y `data/fotos/`.
- **Exportar**: predicciones en CSV y libro Excel (datos, métricas, modelo, patologías).

## Resultados (datos del reto, 10 registros)

`PCI = 111.13 − 8.06 × antigüedad` · R² = 0.984 · MAE = 2.44 · RMSE = 2.92 (ajuste completo).
Con separación cronológica (entrenar ≤ 7 años, probar 8–10 años) el MAE de prueba sube a 7.75 porque el deterioro real se acelera (curva, no recta); un modelo de grado 2 lo reduce a 1.28. Ver informe.

## Limitaciones

Solo 10 datos reales del reto (una sola serie); los 50 adicionales son sintéticos y sirven para validar el flujo, no para decisiones reales. Extrapolar fuera de 1–10 años no está respaldado por datos. Para uso real: inspecciones PCI reales, más tramos y variables (tránsito, clima, drenaje).

La clasificación visual es experimental y no incluye fotos de entrenamiento iniciales: aprende de las imágenes que se registran con su patología y severidad. No es una evaluación PCI certificada.
