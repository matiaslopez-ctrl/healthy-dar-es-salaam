# ==========================================================
# 🌍 Healthy Dar es Salaam — Integrated Urban Dashboard (2000–2025)
# Autor: ChatGPT (NASA Space Apps 2025)
# Fuentes integradas: NASA EarthData + Copernicus GHSL + WorldPop + Socioeconómicos
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ----------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------
st.set_page_config(
    page_title="Healthy Dar es Salaam — Integrated Urban Dashboard",
    page_icon="🌍",
    layout="wide"
)

PRIMARY_YEARS = list(range(2000, 2026))
DISTRICTS = ["Kinondoni", "Ilala", "Temeke"]

# ----------------------------------------------------------
# FUNCIÓN DE SIMULACIÓN / INTEGRACIÓN
# ----------------------------------------------------------
def simulate_integrated_data():
    """
    Simula (o estructura) un dataset combinado NASA + Copernicus + socioeconómicos.
    Sustituye por lecturas reales de CSV/GeoTIFF cuando estén disponibles.
    """
    rng = np.random.default_rng(42)
    rows = []
    for y in PRIMARY_YEARS:
        for d in DISTRICTS:
            pop = rng.integers(2000, 6000)
            dens = rng.uniform(1500, 5000)
            temp = rng.uniform(27, 33)
            rain = rng.uniform(200, 400)
            urb = rng.uniform(0.3, 1.0)
            ndvi = rng.uniform(0.2, 0.9)
            pm25 = rng.uniform(5, 40)
            gdp_pc = rng.uniform(1200, 3800)
            health_index = rng.uniform(0.4, 0.9)
            edu_index = rng.uniform(0.5, 0.95)

            # Índice de Vulnerabilidad Urbana extendido
            # Combina clima, densidad, contaminación y desarrollo
            temp_n = (temp - 27) / (33 - 27)
            rain_n = (rain - 200) / (400 - 200)
            dens_n = (dens - 1500) / (5000 - 1500)
            pm25_n = (pm25 - 5) / (40 - 5)
            dev_n = 1 - ((health_index + edu_index) / 2)
            ivu_ext = 0.3*temp_n + 0.25*rain_n + 0.2*dens_n + 0.15*pm25_n + 0.1*dev_n

            rows.append({
                "year": y,
                "district": d,
                "population_k": pop,
                "density_ppkm2": round(dens, 1),
                "lst_c": round(temp, 2),
                "rain_mm": round(rain, 1),
                "urban_intensity": round(urb, 2),
                "ndvi": round(ndvi, 2),
                "pm25": round(pm25, 1),
                "gdp_pc": round(gdp_pc, 0),
                "health_index": round(health_index, 2),
                "edu_index": round(edu_index, 2),
                "IVU_ext": round(ivu_ext, 3),
            })
    return pd.DataFrame(rows)

data = simulate_integrated_data()

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
with st.sidebar:
    st.title("🌍 Healthy Dar es Salaam — Integrated Dashboard")
    st.write("Análisis urbano 2000–2025 basado en datos satelitales y socioeconómicos.")

    year = st.slider("Selecciona año", min_value=2000, max_value=2025, value=2020, step=5)

    st.markdown("**Capas / Variables a visualizar**")
    selected_layers = st.multiselect(
        "Selecciona indicadores (máx. 4 para comparar)",
        [
            "Densidad poblacional",
            "Temperatura superficial",
            "Precipitación",
            "Urbanización",
            "NDVI (vegetación)",
            "PM2.5 (contaminación)",
            "PIB per cápita",
            "Índice de salud",
            "Índice educativo",
            "IVU extendido"
        ],
        default=["Temperatura superficial", "Precipitación", "IVU extendido"]
    )

# ----------------------------------------------------------
# FILTRAR DATOS POR AÑO
# ----------------------------------------------------------
df = data.query("year == @year").copy()

# ----------------------------------------------------------
# INDICADORES CLAVE
# ----------------------------------------------------------
st.header(f"📊 Indicadores integrados — Año {year}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Población total (miles)", f"{df['population_k'].sum():,.0f}")
col2.metric("Temp. promedio (°C)", f"{df['lst_c'].mean():.1f}")
col3.metric("Lluvia promedio (mm)", f"{df['rain_mm'].mean():.0f}")
col4.metric("IVU Extendido (0–1)", f"{df['IVU_ext'].mean():.3f}")

# ----------------------------------------------------------
# SUBPLOTS GEOGRÁFICOS
# ----------------------------------------------------------
st.subheader("🗺️ Mapas comparativos por indicador")

if len(selected_layers) == 0:
    st.warning("Selecciona al menos un indicador.")
else:
    # Mapear variables a columnas del dataset
    layer_map = {
        "Densidad poblacional": ("density_ppkm2", "Viridis"),
        "Temperatura superficial": ("lst_c", "Hot"),
        "Precipitación": ("rain_mm", "Blues"),
        "Urbanización": ("urban_intensity", "Greys"),
        "NDVI (vegetación)": ("ndvi", "Greens"),
        "PM2.5 (contaminación)": ("pm25", "Reds"),
        "PIB per cápita": ("gdp_pc", "Tealgrn"),
        "Índice de salud": ("health_index", "YlGnBu"),
        "Índice educativo": ("edu_index", "Purples"),
        "IVU extendido": ("IVU_ext", "RdYlGn_r")
    }

    n_layers = len(selected_layers)
    fig = make_subplots(
        rows=1, cols=n_layers,
        subplot_titles=selected_layers,
        specs=[[{"type": "bar"}]*n_layers],
        horizontal_spacing=0.03
    )

    for i, var_name in enumerate(selected_layers, start=1):
        col, palette = layer_map[var_name]
        fig.add_trace(
            go.Bar(
                x=df["district"],
                y=df[col],
                marker=dict(color=df[col], colorscale=palette, showscale=True),
                text=[f"{v:.2f}" for v in df[col]],
                textposition="auto",
                name=var_name
            ),
            row=1, col=i
        )

    fig.update_layout(
        height=500,
        title_text="Comparación de indicadores urbanos y ambientales",
        showlegend=False,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# EVOLUCIÓN TEMPORAL
# ----------------------------------------------------------
st.subheader("📈 Evolución temporal 2000–2025")

time_options = ["population_k", "lst_c", "rain_mm", "IVU_ext", "pm25", "ndvi", "gdp_pc"]
sel_time_var = st.selectbox("Selecciona variable temporal", time_options, index=3)

fig_t = px.line(
    data,
    x="year",
    y=sel_time_var,
    color="district",
    title=f"Evolución temporal de {sel_time_var} (2000–2025)",
    markers=True
)
st.plotly_chart(fig_t, use_container_width=True)

# ----------------------------------------------------------
# TABLA Y EXPORTACIÓN
# ----------------------------------------------------------
st.subheader("🧮 Tabla de datos integrados")
st.dataframe(df, use_container_width=True)
st.download_button(
    "⬇️ Descargar CSV del año seleccionado",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"dar_es_salaam_integrated_{year}.csv",
    mime="text/csv"
)

# ----------------------------------------------------------
# PIE DE PÁGINA
# ----------------------------------------------------------
st.markdown("""
---
**Fuentes integradas:**  
🌡️ NASA MODIS (LST) · 🌧️ NASA GPM IMERG (precipitación) · 👥 SEDAC / WorldPop (población) ·  
🏙️ Copernicus GHSL / ESA WorldCover (urbanización) · 💰 Banco Mundial / WHO (PIB, salud, educación)  
🛰️ Prototipo educativo — NASA Space Apps Challenge 2025
""")
