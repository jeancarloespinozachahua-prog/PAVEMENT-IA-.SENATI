"""Genera informe/informe_pavement_ai.pdf (con figuras) a partir de los datos y el modelo reales.

Uso:  python scripts/generar_informe.py
"""
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from pavement_ai import data as D  # noqa: E402
from pavement_ai import model as M  # noqa: E402
from pavement_ai import pci as P  # noqa: E402

FIG = RAIZ / "informe" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

df10, _ = D.cargar_csv(RAIZ / "data" / "pavimento_simulado.csv")
df60, _ = D.cargar_csv(RAIZ / "data" / "inspecciones_simuladas.csv")
aj10, ev10 = M.ajuste_completo(df10), M.evaluar(df10)
aj60, ev60 = M.ajuste_completo(df60), M.evaluar(df60)
m10 = aj10["modelo"]
quad10 = M.entrenar_cuadratico(df10)

# ── figuras ──
AZUL, AMBAR, ROJO = "#0879c8", "#e7a62a", "#e0562b"
xs = np.linspace(0, 15, 100)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.scatter(df10["antiguedad"], df10["pci"], color=AZUL, zorder=3, label="Observado (CSV del reto)")
ax.plot(xs, M.predecir(m10, xs), "--", color=AMBAR, lw=2.5, label="Regresión lineal")
ax.plot(xs, np.clip(quad10.predict(xs.reshape(-1, 1)), 0, 100), ":", color="#2e9e5b", lw=2, label="Modelo grado 2 (comparativo)")
ax.axvspan(10, 15, color="#f0a0a0", alpha=.2, label="Extrapolación")
ax.set(xlabel="Antigüedad (años)", ylabel="PCI", ylim=(0, 105), title="Antigüedad vs PCI")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "regresion.png", dpi=170); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(df10["antiguedad"], aj10["residuos"], color=[AZUL if r >= 0 else ROJO for r in aj10["residuos"]])
ax.axhline(0, color="k", lw=.8)
ax.set(xlabel="Antigüedad (años)", ylabel="Residuo (obs − est.)", title="Residuos del modelo lineal (patrón en U = curvatura)")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "residuos.png", dpi=170); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.8))
sim = df60[df60["origen"] == "simulado"]
ax.scatter(sim["antiguedad"], sim["pci"], color="#8fb8d6", label="Simulado (50)", zorder=2)
ax.scatter(df10["antiguedad"], df10["pci"], color=AZUL, label="Original (10)", zorder=3)
ax.plot(xs, M.predecir(aj60["modelo"], xs), "--", color=AMBAR, lw=2.5, label="Regresión (60)")
ax.set(xlabel="Antigüedad (años)", ylabel="PCI", ylim=(0, 105), title="Dataset ampliado")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "ampliado.png", dpi=170); plt.close(fig)

# ── documento ──
# Fuente Unicode (para R², b₀, ≤...). Si no está disponible se usa Helvetica.
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FUENTE, FUENTE_B = "Helvetica", "Helvetica-Bold"
try:
    d = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DV", str(d / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-B", str(d / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-I", str(d / ("DejaVuSans-Oblique.ttf" if (d / "DejaVuSans-Oblique.ttf").exists() else "DejaVuSans.ttf"))))
    pdfmetrics.registerFont(TTFont("DV-BI", str(d / ("DejaVuSans-BoldOblique.ttf" if (d / "DejaVuSans-BoldOblique.ttf").exists() else "DejaVuSans-Bold.ttf"))))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
    FUENTE, FUENTE_B = "DV", "DV-B"
except Exception:
    pass

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor("#07569b"), fontSize=16, spaceBefore=12, fontName=FUENTE_B)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#07569b"), fontSize=12.5, fontName=FUENTE_B)
B = ParagraphStyle("B", parent=ss["BodyText"], alignment=TA_JUSTIFY, leading=14.5, fontName=FUENTE, fontSize=9.5)
SM = ParagraphStyle("SM", parent=B, fontSize=8.5, textColor=colors.grey)
TT = ParagraphStyle("TT", parent=ss["Title"], textColor=colors.HexColor("#082746"), fontSize=30, fontName=FUENTE_B)


def tabla(filas, anchos=None, cab=True):
    t = Table(filas, colWidths=anchos, hAlign="LEFT")
    est = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#c8d4df")), ("FONTSIZE", (0, 0), (-1, -1), 9), ("FONTNAME", (0, 0), (-1, -1), FUENTE),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if cab:
        est += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123e65")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(est))
    return t


f = lambda v: f"{v:.2f}"
S = []
S += [Spacer(1, 3 * cm), Paragraph("PAVEMENT AI", TT),
      Paragraph("Predicción del deterioro del pavimento rígido mediante regresión lineal", ParagraphStyle("c", parent=B, alignment=1, fontSize=13)),
      Spacer(1, .6 * cm),
      Paragraph("SENATI · Inteligencia Artificial · Reto individual (20 puntos)<br/>Huánuco, Perú · " + date.today().strftime("%d/%m/%Y"),
                ParagraphStyle("c2", parent=B, alignment=1)),
      Spacer(1, 1.2 * cm),
      Paragraph("Autor: ____________________________ &nbsp; Instructor: ____________________________",
                ParagraphStyle("c3", parent=B, alignment=1)),
      Spacer(1, 1.5 * cm),
      Paragraph("<b>Aviso:</b> todos los datos usados son ficticios y con fines educativos. El modelo no reemplaza la metodología PCI "
                "(ASTM D6433) ni una inspección estructural.", SM), PageBreak()]

S += [Paragraph("1. Introducción y objetivo", H1),
      Paragraph("Una municipalidad de Huánuco necesita analizar el deterioro de sus vías de concreto y planificar inspecciones. "
                "El objetivo es entrenar, evaluar e integrar un modelo de regresión lineal que estime el índice de condición del "
                "pavimento (PCI, 0–100) a partir de la antigüedad del pavimento, dentro de una aplicación web con dashboard, registro de "
                "patologías y simulador.", B),
      Paragraph("2. Datos", H1),
      Paragraph("Se trabajó con dos conjuntos: (a) el CSV del reto <i>pavimento_simulado.csv</i> con 10 registros ficticios "
                "(antigüedad 1–10 años, PCI de 98 a 25), y (b) un conjunto ampliado de 60 registros que añade 50 inspecciones "
                "<b>simuladas</b> en 10 tramos (con tránsito, tipo y severidad de falla), generadas con semilla fija por "
                "<i>scripts/generar_dataset.py</i>, para alcanzar el mínimo de 50 registros recomendado. Los registros simulados están "
                "marcados con <i>origen = simulado</i>.", B),
      tabla([["Antigüedad"] + [str(int(v)) for v in df10["antiguedad"]], ["PCI"] + [str(int(v)) for v in df10["pci"]]],
            [2.2 * cm] + [1.4 * cm] * 10),
      Spacer(1, 6),
      Paragraph("La limpieza con Pandas convierte a numérico, elimina nulos, duplicados y valores fuera de rango (PCI 0–100), y exige al "
                "menos 3 registros válidos con dos antigüedades distintas.", B),
      Image(str(FIG / "regresion.png"), 15 * cm, 8.1 * cm),
      Paragraph("3. Metodología", H1),
      Paragraph("Modelo: <b>LinearRegression</b> (scikit-learn) con la antigüedad como entrada y el PCI como salida: PCI = b₀ + b₁·antigüedad. "
                "Las predicciones se recortan a 0–100 solo para visualización. Evaluación con R², MAE y RMSE. En el conjunto de 10 datos "
                "no hay tramos, por lo que la separación es <b>cronológica</b> (entrenar con ≤ 7 años, probar con 8–10 años, es decir "
                "extrapolación); en el conjunto ampliado se separa <b>por tramo</b> para evitar fuga de información. Se añade validación "
                "cruzada leave-one-out y un modelo de grado 2 como comparación.", B),
      PageBreak(),
      Paragraph("4. Resultados", H1),
      Paragraph("4.1 Conjunto original (10 registros)", H2),
      Paragraph(f"Ecuación ajustada: <b>{M.ecuacion(m10)}</b>. Ajuste global: R² = {aj10['metricas']['R2']:.3f}, "
                f"MAE = {aj10['metricas']['MAE']:.2f}, RMSE = {aj10['metricas']['RMSE']:.2f}.", B),
      tabla([["Evaluación", "R²", "MAE", "RMSE"],
             ["Entrenamiento (≤7 años)"] + [f(ev10["train"][k]) for k in ("R2", "MAE", "RMSE")],
             ["Prueba (8–10 años)"] + [f(ev10["test"][k]) for k in ("R2", "MAE", "RMSE")],
             ["Validación cruzada LOO"] + [f(ev10["loocv"][k]) for k in ("R2", "MAE", "RMSE")],
             ["Modelo grado 2 (prueba)"] + [f(ev10["cuadratico_test"][k]) for k in ("R2", "MAE", "RMSE")]],
            [6 * cm, 3 * cm, 3 * cm, 3 * cm]),
      Spacer(1, 6),
      Image(str(FIG / "residuos.png"), 14 * cm, 6 * cm),
      Paragraph("4.2 Conjunto ampliado (60 registros, 50 simulados)", H2),
      Paragraph(f"Ecuación: <b>{M.ecuacion(aj60['modelo'])}</b>. División: {ev60['division']}.", B),
      tabla([["Evaluación", "R²", "MAE", "RMSE"],
             ["Entrenamiento"] + [f(ev60["train"][k]) for k in ("R2", "MAE", "RMSE")],
             ["Prueba (tramos no vistos)"] + [f(ev60["test"][k]) for k in ("R2", "MAE", "RMSE")],
             ["Validación cruzada LOO*"] + [f(ev60["loocv"][k]) for k in ("R2", "MAE", "RMSE")]],
            [6 * cm, 3 * cm, 3 * cm, 3 * cm]),
      Paragraph("*LOO no agrupa por tramo y es algo optimista; la métrica de prueba por tramo es la más fiable.", SM),
      Image(str(FIG / "ampliado.png"), 14 * cm, 7.6 * cm),
      PageBreak(),
      Paragraph("5. Interpretación", H1),
      Paragraph(f"<b>Pendiente.</b> Con los datos originales el PCI estimado cae unos {abs(aj10['pendiente']):.1f} puntos por año; con el conjunto "
                f"ampliado, {abs(aj60['pendiente']):.1f}. <b>Intercepto.</b> Es el valor extrapolado a antigüedad 0 (≈{aj10['intercepto']:.0f}), "
                "mayor que 100, por lo que se recorta. <b>MAE.</b> Error medio en puntos de PCI. En el conjunto de 10 datos el error de prueba "
                f"({ev10['test']['MAE']:.1f}) es mucho mayor que el de entrenamiento ({ev10['train']['MAE']:.1f}): los residuos tienen forma de U, "
                "el deterioro se acelera con la edad y una recta subestima la caída en los años finales (el modelo de grado 2 reduce el MAE de "
                f"prueba a {ev10['cuadratico_test']['MAE']:.1f}).", B),
      Paragraph("6. Tres escenarios simulados (modelo del CSV original)", H1)]
esc = [("A · Tramo joven", 2, 3), ("B · Tramo intermedio", 5, 3), ("C · Tramo envejecido", 8, 4)]
filas = [["Escenario", "Edad", "Horizonte", "PCI actual", "PCI futuro", "Condición", "¿Extrapola?"]]
for n, a, h in esc:
    pf = M.predecir(m10, a + h)
    filas.append([n, a, h, f"{M.predecir(m10, a):.1f}", f"{pf:.1f}", P.condicion(pf), "Sí" if a + h > 10 else "No"])
S += [tabla(filas, [4.2 * cm, 1.5 * cm, 2 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, 2.3 * cm]),
      Paragraph("El escenario C supera el máximo del historial (10 años): el resultado depende totalmente del supuesto de linealidad.", SM),
      Paragraph("7. Limitaciones y riesgos de extrapolar", H1),
      Paragraph("• La antigüedad por sí sola no determina el PCI: influyen tránsito, clima, drenaje, calidad constructiva y tipo y severidad de las "
                "fallas.<br/>• Con 10 datos (una sola serie) la prueba contiene solo 3 puntos; las métricas son inestables.<br/>"
                "• Los 50 registros añadidos son sintéticos: sirven para validar el flujo de trabajo, no para tomar decisiones reales.<br/>"
                "• Fuera del rango 1–10 años no hay evidencia de linealidad; el PCI está acotado a 0–100.<br/>"
                "• El modelo no reemplaza una evaluación PCI ni una inspección estructural.", B),
      Paragraph("8. Aplicación desarrollada", H1),
      Paragraph("Aplicación Streamlit con seis pestañas: Dashboard, Datos (carga/limpieza/subida de CSV), Modelo (métricas, residuos, "
                "sustentación), Simulador (PCI actual y futuro, años hasta el umbral, recomendación, tres escenarios), Registro de patologías "
                "(seis tipos, severidad, área, fotografía, validación y sanitización de texto) y Exportar (CSV y Excel). Incluye "
                "pruebas automáticas con pytest.", B),
      Paragraph("9. Conclusiones", H1),
      Paragraph(f"1) Una regresión lineal describe bien la tendencia general (R² = {aj10['metricas']['R2']:.2f}) pero subestima la aceleración "
                "del deterioro. 2) El modelo es útil para priorizar inspecciones dentro del rango de datos, no para extrapolar. 3) Para uso real se "
                "requieren inspecciones PCI reales, más tramos y variables adicionales (tránsito, clima, drenaje) y modelos no lineales.", B)]

SimpleDocTemplate(str(RAIZ / "informe" / "informe_pavement_ai.pdf"), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                  topMargin=2 * cm, bottomMargin=2 * cm, title="PAVEMENT AI – Informe").build(S)
print("OK -> informe/informe_pavement_ai.pdf")
