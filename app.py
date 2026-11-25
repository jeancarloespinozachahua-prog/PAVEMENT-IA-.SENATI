"""PAVEMENT AI · aplicación Streamlit.

Ejecutar:  python -m streamlit run app.py
"""
from __future__ import annotations

import io
import re
import uuid
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pavement_ai import data as D
from pavement_ai import model as M
from pavement_ai import pci as P
from pavement_ai import vision as V

RAIZ = Path(__file__).parent
DATOS = RAIZ / "data"
FOTOS = DATOS / "fotos"
REGISTRO = DATOS / "patologias_registradas.csv"
COLS_REGISTRO = ["id", "registrado_en", "tramo", "fecha_inspeccion", "antiguedad", "patologia",
                 "severidad", "area_m2", "pci_observado", "pci_estimado_modelo", "foto", "observaciones"]
MAX_FOTO_MB = 5

st.set_page_config(page_title="PAVEMENT AI · SENATI", page_icon="🛣️", layout="wide")

st.markdown(
    """
<style>
:root{--navy:#0a2540;--blue:#0b6bcb;--cyan:#22b8e8;--ink:#14283d;--mute:#5d7188;--line:#dce5ef;--card:#fff}
.block-container{padding-top:1.4rem;max-width:1280px}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none}
.hero{position:relative;overflow:hidden;background:linear-gradient(120deg,#0a2540 0%,#0d3b66 60%,#0b6bcb 120%);
 color:#fff;padding:30px 36px;border-radius:18px;margin-bottom:18px;box-shadow:0 10px 30px #0a254030}
.hero:after{content:"";position:absolute;left:0;right:0;bottom:22px;height:0;border-top:3px dashed #ffffff40}
.hero small{letter-spacing:3px;color:#7fdcff;font-weight:600;font-size:.72rem}
.hero h1{margin:6px 0 4px;font-size:2.5rem;font-weight:800;color:#fff;letter-spacing:-1px;padding:0}
.hero p{margin:0 0 26px;color:#cfe3f5;font-size:1rem}
.hero .tags span{display:inline-block;margin-right:8px;padding:3px 12px;border-radius:999px;
 background:#ffffff1c;color:#e6f3ff;font-size:.75rem;font-weight:600}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;
 box-shadow:0 2px 8px #14283d0a;height:100%}
.kpi .l{color:var(--mute);font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.6px}
.kpi .v{color:var(--navy);font-size:2rem;font-weight:800;line-height:1.25}
.kpi .s{color:var(--mute);font-size:.8rem}
.kpi.accent{border-left:5px solid var(--blue)}
.kpi .up{color:#2e9e5b}.kpi .down{color:#c0392b}
.badge{display:inline-block;padding:5px 16px;border-radius:999px;color:#fff;font-weight:700;font-size:1.05rem}
.sec{font-size:1.15rem;font-weight:700;color:var(--navy);margin:22px 0 6px;padding-left:10px;border-left:4px solid var(--cyan)}
.patgrid{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:6px}
.patgrid div{text-align:center;font-size:.8rem;width:118px;background:#fff;border:1px solid var(--line);
 border-radius:12px;padding:10px 6px;font-weight:600;color:var(--ink)}
div[data-testid="stTabs"] button[role="tab"]{font-weight:600;font-size:.95rem;padding:10px 18px}
div[data-testid="stPlotlyChart"],div[data-testid="stDataFrame"]{background:#fff;border:1px solid var(--line);
 border-radius:14px;padding:6px;box-shadow:0 2px 8px #14283d08}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
.brand{font-weight:800;font-size:1.25rem;color:var(--navy);letter-spacing:-.5px}
.brand span{color:var(--blue)}
.foot{color:var(--mute);font-size:.78rem;text-align:center;margin-top:28px;padding-top:12px;border-top:1px solid var(--line)}
</style>
<div class="hero"><small>SENATI · INTELIGENCIA ARTIFICIAL · RETO INDIVIDUAL</small>
<h1>PAVEMENT AI</h1>
<p>Predicción del deterioro del pavimento rígido mediante regresión lineal</p>
<div class="tags"><span>Huánuco · Perú</span><span>Machine Learning</span><span>Índice PCI</span></div></div>
""",
    unsafe_allow_html=True,
)

# ───────────────────────── utilidades ─────────────────────────
def seguro_csv(texto: str, maximo: int = 300) -> str:
    """Evita inyección de fórmulas al abrir el CSV en Excel y limita la longitud."""
    texto = re.sub(r"[\x00-\x1f]", " ", str(texto)).strip()[:maximo]
    return "'" + texto if texto[:1] in ("=", "+", "-", "@") else texto


def a_excel(hojas: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for nombre, df in hojas.items():
            df.to_excel(xw, sheet_name=nombre[:31], index=False)
    return buf.getvalue()


def cargar_registro() -> pd.DataFrame:
    if REGISTRO.exists() and REGISTRO.stat().st_size > 0:
        return pd.read_csv(REGISTRO)
    return pd.DataFrame(columns=COLS_REGISTRO)


def firma_fotos(registro: pd.DataFrame) -> tuple[tuple[str, str, str, int], ...]:
    """Firma los ejemplos etiquetados para invalidar la caché si cambian."""
    firma = []
    for _, fila in registro.iterrows():
        archivo = str(fila.get("foto", "")).strip()
        if not archivo or Path(archivo).name != archivo:
            continue
        ruta = FOTOS / archivo
        if ruta.is_file():
            firma.append((
                archivo,
                str(fila.get("patologia", "")),
                str(fila.get("severidad", "")),
                ruta.stat().st_mtime_ns,
            ))
    return tuple(firma)


@st.cache_resource(show_spinner=False)
def cargar_modelo_vision(firma: tuple[tuple[str, str, str, int], ...]) -> V.ModeloVision:
    return V.entrenar(firma, FOTOS)


def badge(pci_valor: float) -> str:
    c = P.condicion(pci_valor)
    return f'<span class="badge" style="background:{P.COLOR_CONDICION[c]}">{c}</span>'


def kpi(etiqueta: str, valor: str, sub: str = "", acento: bool = False, html_valor: bool = False) -> str:
    cls = "kpi accent" if acento else "kpi"
    return f'<div class="{cls}"><div class="l">{etiqueta}</div><div class="v">{valor}</div><div class="s">{sub}</div></div>'


def seccion(texto: str) -> None:
    st.markdown(f'<div class="sec">{texto}</div>', unsafe_allow_html=True)


def grafico(fig: go.Figure, contenedor=st) -> None:
    """Aplica un estilo unificado a todos los gráficos."""
    fig.update_layout(
        template="simple_white", paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Inter, Segoe UI, Arial", size=13, color="#14283d"),
        title=dict(font=dict(size=15, color="#0a2540"), x=0.01),
        legend=dict(orientation="h", y=-0.22, x=0), hovermode="closest", margin=dict(t=48, b=10, l=10, r=10),
    )
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    contenedor.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def medidor(valor: float) -> go.Figure:
    """Indicador tipo velocímetro del PCI con las bandas de condición."""
    bandas = [(0, 10, "Fallado"), (10, 25, "Grave"), (25, 40, "Muy pobre"), (40, 55, "Pobre"),
              (55, 70, "Regular"), (70, 85, "Satisfactorio"), (85, 100, "Bueno")]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=valor, number=dict(font=dict(size=44, color="#0a2540")),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#0a2540", thickness=0.28),
                   steps=[dict(range=[a, b], color=P.COLOR_CONDICION[n]) for a, b, n in bandas]),
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor="#fff")
    return fig


@st.cache_data(show_spinner=False)
def leer_fuente(nombre: str):
    return D.cargar_csv(DATOS / nombre)


# ───────────────────────── barra lateral: datos ─────────────────────────
st.sidebar.markdown('<div class="brand">PAVEMENT <span>AI</span></div>', unsafe_allow_html=True)
st.sidebar.caption("Centro de análisis vial · SENATI")
st.sidebar.divider()
st.sidebar.markdown("**Datos de entrenamiento**")
fuente = st.sidebar.radio(
    "Conjunto de datos",
    ["Original del reto (10 registros)", "Ampliado (60 registros, 50 simulados)", "Subir mi CSV"],
)
try:
    if fuente.startswith("Original"):
        df, avisos = leer_fuente("pavimento_simulado.csv")
    elif fuente.startswith("Ampliado"):
        df, avisos = leer_fuente("inspecciones_simuladas.csv")
    else:
        sub = st.sidebar.file_uploader("CSV con columnas antiguedad, pci", type="csv")
        if sub is None:
            st.info("Sube un CSV con las columnas `antiguedad` y `pci` (opcional: `tramo`) en la barra lateral.")
            st.stop()
        if sub.size > 5 * 1024 * 1024:
            st.error("El archivo supera 5 MB.")
            st.stop()
        df, avisos = D.cargar_csv(sub)
except D.DatosInvalidos as e:
    st.error(f"Datos no válidos: {e}")
    st.stop()

umbral = st.sidebar.slider("Umbral de intervención (PCI)", 10, 85, 55,
                           help="PCI a partir del cual se recomienda programar mantenimiento mayor.")
st.sidebar.info("Los datos incluidos son **ficticios**, con fines educativos.")

ajuste = M.ajuste_completo(df)
modelo = ajuste["modelo"]
ev = M.evaluar(df)
EDAD_MAX = ajuste["edad_max_entrenamiento"]

t_dash, t_datos, t_modelo, t_sim, t_vision, t_reg, t_exp = st.tabs(
    ["Dashboard", "Datos", "Modelo", "Simulador", "Análisis de imagen", "Registro de patologías", "Exportar"]
)

# ───────────────────────── Dashboard ─────────────────────────
with t_dash:
    c = st.columns(5)
    r2 = ajuste["metricas"]["R2"]
    c[0].markdown(kpi("Registros", str(len(df)), "inspecciones válidas", True), unsafe_allow_html=True)
    c[1].markdown(kpi("PCI promedio", f"{df['pci'].mean():.1f}", P.condicion(df['pci'].mean()), True), unsafe_allow_html=True)
    c[2].markdown(kpi("Pendiente", f"{ajuste['pendiente']:.2f}", "puntos PCI por año", True), unsafe_allow_html=True)
    c[3].markdown(kpi("R² (ajuste)", f"{r2:.3f}", "varianza explicada", True), unsafe_allow_html=True)
    c[4].markdown(kpi("MAE (ajuste)", f"{ajuste['metricas']['MAE']:.2f}", "error medio en puntos", True), unsafe_allow_html=True)
    st.write("")
    for a in avisos:
        st.warning(a)

    g1, g2 = st.columns([3, 2])
    xs = np.linspace(0, max(EDAD_MAX + 5, 15), 100)
    fig = go.Figure()
    fig.add_scatter(x=df["antiguedad"], y=df["pci"], mode="markers", name="Observado",
                    marker=dict(size=9, color="#0879c8"))
    fig.add_scatter(x=xs, y=M.predecir(modelo, xs), mode="lines", name="Regresión lineal",
                    line=dict(color="#e7a62a", dash="dash", width=3))
    fig.add_vrect(x0=EDAD_MAX, x1=xs.max(), fillcolor="#f0a0a0", opacity=0.15, line_width=0,
                  annotation_text="Extrapolación", annotation_position="top left")
    fig.add_hline(y=umbral, line_dash="dot", line_color="#b3261e", annotation_text=f"Umbral {umbral}")
    fig.update_layout(title="Antigüedad vs PCI", xaxis_title="Antigüedad (años)", yaxis_title="PCI",
                      yaxis_range=[0, 105], height=380, margin=dict(t=50, b=10))
    grafico(fig, g1)

    cond = df["pci"].map(P.condicion).value_counts().reindex(P.ORDEN_CONDICION).dropna()
    fig2 = go.Figure(go.Bar(x=cond.index, y=cond.values, marker_color=[P.COLOR_CONDICION[k] for k in cond.index]))
    fig2.update_layout(title="Registros por condición", height=380, margin=dict(t=50, b=10))
    grafico(fig2, g2)

    if "tipo_falla" in df.columns:
        fa = df[~df["tipo_falla"].isin(["No registrado"])]
        if len(fa):
            g3, g4 = st.columns(2)
            cnt = fa["tipo_falla"].value_counts()
            f3 = go.Figure(go.Bar(x=cnt.values, y=cnt.index, orientation="h", marker_color="#07569b"))
            f3.update_layout(title="Patologías más frecuentes", height=320, margin=dict(t=50, b=10),
                             yaxis=dict(autorange="reversed"))
            grafico(f3, g3)
            if "transito" in fa.columns:
                pt = fa.groupby("transito")["pci"].mean().reindex(P.TRANSITOS).dropna()
                f4 = go.Figure(go.Bar(x=pt.index, y=pt.values, marker_color="#0875bc"))
                f4.update_layout(title="PCI medio según nivel de tránsito (simulado)", height=320,
                                 margin=dict(t=50, b=10), yaxis_range=[0, 100])
                grafico(f4, g4)

# ───────────────────────── Datos ─────────────────────────
with t_datos:
    seccion("Conjunto de datos cargado")
    st.caption(f"{len(df)} registros válidos tras la limpieza.")
    st.dataframe(df, width="stretch", height=360)
    st.write("**Estadísticos descriptivos**")
    st.dataframe(df[["antiguedad", "pci"]].describe().T.round(2), width="stretch")
    st.download_button("Descargar dataset limpio (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "dataset_limpio.csv", "text/csv")
    st.info("El PCI observado debe provenir de una evaluación técnica (ASTM D6433). "
            "Si los datos son simulados, debe declararse explícitamente.")

# ───────────────────────── Modelo ─────────────────────────
with t_modelo:
    seccion("Regresión lineal: PCI = b₀ + b₁ · antigüedad")
    st.code(M.ecuacion(modelo))
    st.markdown(
        f"- **Pendiente (b₁ = {ajuste['pendiente']:.2f})**: el PCI estimado baja ≈ {abs(ajuste['pendiente']):.1f} puntos por cada año de antigüedad.\n"
        f"- **Intercepto (b₀ = {ajuste['intercepto']:.1f})**: valor extrapolado a antigüedad 0 (puede superar 100; se recorta al mostrar).\n"
        f"- **División de prueba**: {ev['division']} ({ev['n_train']} entrenamiento / {ev['n_test']} prueba)."
    )
    tabla = pd.DataFrame(
        {"Entrenamiento": ev["train"], "Prueba": ev["test"], "Validación cruzada LOO": ev["loocv"],
         "Modelo grado 2 (prueba)": ev["cuadratico_test"]}
    ).T.round(3)
    st.dataframe(tabla, width="stretch")
    st.caption("MAE = error absoluto medio (puntos PCI) · RMSE penaliza más los errores grandes · "
               "R² = proporción de variación explicada.")
    if ev["test"]["MAE"] > 2 * max(ev["train"]["MAE"], 0.5):
        st.warning("El error en prueba es mucho mayor que en entrenamiento: la relación no es perfectamente "
                   "lineal (el deterioro se acelera con la edad) o hay pocos datos. Ver el modelo de grado 2.")

    r1, r2 = st.columns(2)
    fr = go.Figure(go.Scatter(x=df["antiguedad"], y=ajuste["residuos"], mode="markers", marker=dict(size=9, color="#0879c8")))
    fr.add_hline(y=0, line_dash="dash")
    fr.update_layout(title="Residuos (observado − estimado)", xaxis_title="Antigüedad", yaxis_title="Residuo",
                     height=330, margin=dict(t=50, b=10))
    grafico(fr, r1)
    td = ev["test_df"]
    fp = go.Figure()
    fp.add_scatter(x=td["pci"], y=td["pci_pred"], mode="markers", marker=dict(size=9, color="#e7a62a"), name="Prueba")
    fp.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="#888"), name="Ideal")
    fp.update_layout(title="Real vs predicho (prueba)", xaxis_title="PCI real", yaxis_title="PCI predicho",
                     height=330, margin=dict(t=50, b=10))
    grafico(fp, r2)

    with st.expander("Preguntas de sustentación"):
        st.markdown(
            "**¿Qué representa la pendiente?** La pérdida media de PCI por año.\n\n"
            "**¿Qué significa MAE?** El error absoluto promedio entre PCI real y estimado, en puntos de PCI.\n\n"
            "**¿Por qué no basta la antigüedad?** El PCI depende del tránsito, clima, calidad constructiva, drenaje, "
            "tipo y severidad de fallas, etc. Dos vías de igual edad pueden tener PCI muy distintos.\n\n"
            "**¿Qué riesgos tiene extrapolar?** Fuera del rango de entrenamiento no hay evidencia de que la relación "
            "siga siendo lineal; el PCI está acotado a 0-100 y el deterioro real suele acelerarse."
        )

# ───────────────────────── Simulador ─────────────────────────
with t_sim:
    seccion("Simulador de PCI futuro")
    s1, s2 = st.columns(2)
    edad = s1.number_input("Antigüedad actual (años)", min_value=0.0, max_value=float(D.EDAD_MAX), value=5.0, step=0.5)
    horizonte = s2.number_input("Horizonte de predicción (años)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)

    pci_hoy = M.predecir(modelo, edad)
    pci_fut = M.predecir(modelo, edad + horizonte)
    anios = M.anios_hasta_umbral(modelo, edad, umbral)
    gcol, kcol = st.columns([2, 3])
    grafico(medidor(pci_fut), gcol)
    gcol.caption("PCI futuro estimado, sobre la escala de condición")
    delta = pci_fut - pci_hoy
    k1, k2 = kcol.columns(2)
    k1.markdown(kpi("PCI actual estimado", f"{pci_hoy:.1f}", P.condicion(pci_hoy)), unsafe_allow_html=True)
    k2.markdown(kpi("PCI futuro estimado", f"{pci_fut:.1f}",
                    f'<span class="{"down" if delta < 0 else "up"}">{delta:+.1f} puntos</span>', True), unsafe_allow_html=True)
    kcol.write("")
    k3, k4 = kcol.columns(2)
    k3.markdown(kpi("Condición futura", badge(pci_fut), f"a {edad + horizonte:g} años de antigüedad"), unsafe_allow_html=True)
    k4.markdown(kpi(f"Años hasta PCI {umbral}", "—" if anios is None else f"{anios:.1f}", "para inspección mayor"),
                unsafe_allow_html=True)
    st.write("")

    if edad + horizonte > EDAD_MAX:
        st.warning(f"Extrapolación: {edad + horizonte:g} años supera el máximo del historial ({EDAD_MAX:g}). "
                   "Interprete con cautela.")
    if M.predecir(modelo, edad + horizonte, recortar=False) < 0:
        st.warning("El modelo lineal produjo PCI < 0; se muestra recortado a 0.")
    if anios is not None and anios <= 1:
        st.error("Recomendación: programar inspección técnica prioritaria (PCI en/bajo el umbral en ≤ 1 año).")
    elif anios is not None and anios <= 3:
        st.warning(f"Recomendación: planificar inspección dentro de {max(0.5, anios - 1):.1f} año(s).")
    else:
        st.success("Sin urgencia según el modelo; mantener el ciclo normal de inspecciones.")

    xs = np.linspace(0, max(EDAD_MAX + 5, edad + horizonte + 2), 100)
    fs = go.Figure()
    fs.add_scatter(x=df["antiguedad"], y=df["pci"], mode="markers", name="Historial", marker=dict(color="#0879c8", size=8))
    fs.add_scatter(x=xs, y=M.predecir(modelo, xs), mode="lines", name="Modelo", line=dict(color="#e7a62a", dash="dash"))
    fs.add_scatter(x=[edad], y=[pci_hoy], mode="markers", name="Actual", marker=dict(color="#2e9e5b", size=13))
    fs.add_scatter(x=[edad + horizonte], y=[pci_fut], mode="markers", name="Predicción", marker=dict(color="#e0562b", size=13, symbol="diamond"))
    fs.add_hline(y=umbral, line_dash="dot", line_color="#b3261e")
    fs.update_layout(xaxis_title="Antigüedad (años)", yaxis_title="PCI", yaxis_range=[0, 105], height=380,
                     margin=dict(t=20, b=10))
    grafico(fs, st)

    seccion("Tres escenarios de referencia")
    esc = pd.DataFrame(
        [("A · Tramo joven", 2, 3), ("B · Tramo intermedio", 5, 3), ("C · Tramo envejecido", 8, 4)],
        columns=["Escenario", "Antigüedad actual", "Horizonte"],
    )
    esc["Antigüedad futura"] = esc["Antigüedad actual"] + esc["Horizonte"]
    esc["PCI actual"] = M.predecir(modelo, esc["Antigüedad actual"]).round(1)
    esc["PCI futuro"] = M.predecir(modelo, esc["Antigüedad futura"]).round(1)
    esc["Condición futura"] = esc["PCI futuro"].map(P.condicion)
    esc["¿Extrapola?"] = np.where(esc["Antigüedad futura"] > EDAD_MAX, "Sí", "No")
    st.dataframe(esc, width="stretch", hide_index=True)
    st.caption("El escenario C supera el rango del historial: ilustra el riesgo de extrapolar.")

# ───────────────────────── Análisis visual ─────────────────────────
with t_vision:
    seccion("Análisis visual local de pavimentos")
    st.warning(
        "Función experimental: el análisis solo aprende de fotos registradas y etiquetadas por una persona. "
        "No reemplaza una inspección técnica ni calcula el PCI a partir de la imagen."
    )
    registro_vision = cargar_registro()
    modelo_vision = cargar_modelo_vision(firma_fotos(registro_vision))

    c1, c2 = st.columns(2)
    c1.metric("Fotos válidas para entrenamiento", sum(
        modelo_vision.conteos["patologia"].values()
    ))
    c2.metric("Imágenes omitidas por formato o archivo inválido", modelo_vision.imagenes_invalidas)

    st.caption(
        f"Para activar cada estimación se requieren al menos {V.MIN_MUESTRAS_POR_CLASE} "
        "fotos etiquetadas por categoría en un mínimo de dos categorías. "
        "Las fotos se guardan y procesan localmente."
    )
    with st.expander("Ver ejemplos etiquetados disponibles"):
        tabla_conteos = pd.DataFrame([
            {"Tipo": "Patología", "Categoría": etiqueta, "Fotos": cantidad}
            for etiqueta, cantidad in modelo_vision.conteos["patologia"].items()
        ] + [
            {"Tipo": "Severidad", "Categoría": etiqueta, "Fotos": cantidad}
            for etiqueta, cantidad in modelo_vision.conteos["severidad"].items()
        ])
        st.dataframe(tabla_conteos, width="stretch", hide_index=True)

    imagen_subida = st.file_uploader(
        "Sube una imagen del pavimento (PNG/JPG, máximo 5 MB)",
        type=["png", "jpg", "jpeg"],
        key="analisis_visual_upload",
    )
    if imagen_subida is not None:
        if imagen_subida.size > MAX_FOTO_MB * 1024 * 1024:
            st.error(f"La imagen supera el límite de {MAX_FOTO_MB} MB.")
        else:
            contenido_imagen = imagen_subida.getvalue()
            izquierda, derecha = st.columns([1, 1])
            izquierda.image(contenido_imagen, caption="Imagen cargada", width="stretch")
            try:
                predicciones = V.predecir(modelo_vision, contenido_imagen)
            except V.ImagenInvalida as exc:
                derecha.error(str(exc))
            else:
                if "patologia" in predicciones:
                    etiqueta, confianza = predicciones["patologia"]
                    derecha.metric("Patología visual probable", etiqueta)
                    derecha.caption(f"Confianza estimada del clasificador: {confianza:.0%}")
                    if confianza < 0.55:
                        derecha.warning("Resultado de baja confianza; requiere revisión humana.")
                else:
                    derecha.info(
                        "Aún no hay suficientes fotos etiquetadas para estimar la patología. "
                        "Registra fotos con su categoría en la pestaña correspondiente."
                    )
                if "severidad" in predicciones:
                    etiqueta, confianza = predicciones["severidad"]
                    derecha.metric("Severidad visual probable", etiqueta)
                    derecha.caption(f"Confianza estimada del clasificador: {confianza:.0%}")
                else:
                    derecha.info("Aún se necesitan ejemplos etiquetados para estimar la severidad.")
                derecha.caption(
                    "La predicción visual no determina el estado estructural ni el PCI. "
                    "Confirma siempre el resultado con una inspección."
                )

# ───────────────────────── Registro de patologías ─────────────────────────
with t_reg:
    seccion("Catálogo y registro de patologías")
    st.markdown("<div class='patgrid'>" + "".join(
        f"<div>{P.svg_patologia(n)}<br><b>{n}</b></div>" for n in P.PATOLOGIAS) + "</div>", unsafe_allow_html=True)
    st.write("")

    with st.form("form_patologia", clear_on_submit=True):
        f1, f2, f3 = st.columns(3)
        tramo = f1.text_input("Tramo / vía (ej. Jr. 2 de Mayo, cuadra 3)", max_chars=60)
        fecha_insp = f2.date_input("Fecha de inspección", value=date.today(), max_value=date.today())
        edad_reg = f3.number_input("Antigüedad del pavimento (años)", 0.0, float(D.EDAD_MAX), 5.0, 0.5)
        f4, f5, f6 = st.columns(3)
        patol = f4.selectbox("Tipo de patología", list(P.PATOLOGIAS))
        sev = f5.selectbox("Severidad", P.SEVERIDADES, index=1)
        area = f6.number_input("Área afectada (m²)", min_value=0.0, max_value=100000.0, value=1.0, step=0.5)
        pci_obs = st.number_input("PCI observado (opcional; 0 = no disponible)", 0.0, 100.0, 0.0, 1.0,
                                  help="Debe provenir de una evaluación técnica.")
        foto = st.file_uploader("Fotografía (png/jpg, máx. 5 MB)", type=["png", "jpg", "jpeg"])
        obs = st.text_area("Observaciones", max_chars=300)
        enviar = st.form_submit_button("Guardar registro", type="primary")

    if enviar:
        errores = []
        if len(tramo.strip()) < 3:
            errores.append("Indique el tramo (mínimo 3 caracteres).")
        if area <= 0:
            errores.append("El área afectada debe ser mayor que 0.")
        if foto is not None and foto.size > MAX_FOTO_MB * 1024 * 1024:
            errores.append(f"La fotografía supera {MAX_FOTO_MB} MB.")
        if errores:
            for e in errores:
                st.error(e)
        else:
            nombre_foto = ""
            if foto is not None:
                FOTOS.mkdir(parents=True, exist_ok=True)
                ext = ".png" if foto.type == "image/png" else ".jpg"
                nombre_foto = f"{uuid.uuid4().hex[:12]}{ext}"
                (FOTOS / nombre_foto).write_bytes(foto.getvalue())
            est = M.predecir(modelo, edad_reg)
            nuevo = pd.DataFrame([{
                "id": uuid.uuid4().hex[:8], "registrado_en": datetime.now().isoformat(timespec="seconds"),
                "tramo": seguro_csv(tramo, 60), "fecha_inspeccion": fecha_insp.isoformat(), "antiguedad": edad_reg,
                "patologia": patol, "severidad": sev, "area_m2": area,
                "pci_observado": pci_obs if pci_obs > 0 else "", "pci_estimado_modelo": round(est, 1),
                "foto": nombre_foto, "observaciones": seguro_csv(obs),
            }])
            pd.concat([cargar_registro(), nuevo]).to_csv(REGISTRO, index=False)
            st.success(f"Registro guardado. PCI estimado por el modelo para {edad_reg:g} años: {est:.1f} "
                       f"({P.condicion(est)}).")

    reg = cargar_registro()
    st.write(f"**Registros guardados: {len(reg)}**")
    if len(reg):
        st.dataframe(reg.drop(columns=["id"]), width="stretch", hide_index=True)
        z1, z2 = st.columns(2)
        resumen = reg.groupby(["patologia", "severidad"]).agg(casos=("id", "count"), area_m2=("area_m2", "sum")).reset_index()
        z1.write("**Resumen por patología y severidad**")
        z1.dataframe(resumen, hide_index=True, width="stretch")
        fotos = [f for f in reg["foto"].dropna().astype(str) if f and (FOTOS / f).exists()]
        if fotos:
            z2.write("**Última fotografía**")
            z2.image(str(FOTOS / fotos[-1]), width="stretch")

# ───────────────────────── Exportar ─────────────────────────
with t_exp:
    seccion("Exportar resultados")
    pred = df.assign(pci_estimado=M.predecir(modelo, df["antiguedad"]).round(2))
    pred["error"] = (pred["pci"] - pred["pci_estimado"]).round(2)
    pred["condicion_observada"] = pred["pci"].map(P.condicion)
    metr = pd.DataFrame({"Entrenamiento": ev["train"], "Prueba": ev["test"], "LOOCV": ev["loocv"]}).T.reset_index(
        names="conjunto")
    parametros = pd.DataFrame({"parametro": ["intercepto", "pendiente", "umbral_intervencion", "n_registros"],
                               "valor": [ajuste["intercepto"], ajuste["pendiente"], umbral, len(df)]})
    hojas = {"Datos_y_predicciones": pred, "Metricas": metr, "Modelo": parametros, "Patologias": cargar_registro()}
    e1, e2 = st.columns(2)
    e1.download_button("Descargar predicciones (CSV)", pred.to_csv(index=False).encode("utf-8"),
                       "predicciones.csv", "text/csv")
    e2.download_button("Descargar todo (Excel)", a_excel(hojas), "pavement_ai_resultados.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.dataframe(pred, width="stretch", height=300)

st.markdown('<div class="foot">SENATI · Proyecto educativo · PAVEMENT AI — el modelo no reemplaza la metodología PCI '
            'ni una inspección estructural.</div>', unsafe_allow_html=True)
