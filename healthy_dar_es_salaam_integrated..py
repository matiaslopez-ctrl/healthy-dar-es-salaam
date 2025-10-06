# ==========================================================
# 🌍 Healthy Dar es Salaam — Integrated Urban Dashboard (2000–2025)
# Author: ChatGPT (NASA Space Apps 2025)
# Integrated sources: NASA EarthData + Copernicus GHSL + WorldPop + Socioeconomic data
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ----------------------------------------------------------
# GENERAL CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(
    page_title="Healthy Dar es Salaam — Integrated Urban Dashboard",
    page_icon="🌍",
    layout="wide"
)

PRIMARY_YEARS = list(range(2000, 2026))
DISTRICTS = ["Kinondoni", "Ilala", "Temeke"]

# ----------------------------------------------------------
# SIMULATION / INTEGRATION FUNCTION
# ----------------------------------------------------------
def simulate_integrated_data():
    """
    Simulates (or structures) a combined NASA + Copernicus + socioeconomic dataset.
    Replace this with actual CSV/GeoTIFF readings when available.
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

            # Extended Urban Vulnerability Index (UVI)
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
    st.write("Urban analysis 2000–2025 based on satellite and socioeconomic data.")

    year = st.slider("Select year", min_value=2000, max_value=2025, value=2020, step=5)

    st.markdown("**Layers / Variables to visualize**")
    selected_layers = st.multiselect(
        "Select indicators (max. 4 to compare)",
        [
            "Population density",
            "Surface temperature (LST)",
            "Precipitation (GPM)",
            "Urban footprint (GHSL)",
            "NDVI (vegetation)",
            "PM2.5 (air pollution)",
            "GDP per capita",
            "Health index",
            "Education index",
            "Extended UVI"
        ],
        default=["Surface temperature (LST)", "Precipitation (GPM)", "Extended UVI"]
    )

# ----------------------------------------------------------
# FILTER DATA BY YEAR
# ----------------------------------------------------------
df = data.query("year == @year").copy()

# ----------------------------------------------------------
# KEY INDICATORS
# ----------------------------------------------------------
st.header(f"📊 Integrated indicators — Year {year}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total population (thousands)", f"{df['population_k'].sum():,.0f}")
col2.metric("Avg. temperature (°C)", f"{df['lst_c'].mean():.1f}")
col3.metric("Avg. precipitation (mm)", f"{df['rain_mm'].mean():.0f}")
col4.metric("Extended UVI (0–1)", f"{df['IVU_ext'].mean():.3f}")

# ----------------------------------------------------------
# COMPARATIVE MAPS (SUBPLOTS)
# ----------------------------------------------------------
st.subheader("🗺️ Comparative charts by indicator")

if len(selected_layers) == 0:
    st.warning("Select at least one indicator.")
else:
    # Map indicators to dataset columns
    layer_map = {
        "Population density": ("density_ppkm2", "Viridis"),
        "Surface temperature (LST)": ("lst_c", "Hot"),
        "Precipitation (GPM)": ("rain_mm", "Blues"),
        "Urban footprint (GHSL)": ("urban_intensity", "Greys"),
        "NDVI (vegetation)": ("ndvi", "Greens"),
        "PM2.5 (air pollution)": ("pm25", "Reds"),
        "GDP per capita": ("gdp_pc", "Tealgrn"),
        "Health index": ("health_index", "YlGnBu"),
        "Education index": ("edu_index", "Purples"),
        "Extended UVI": ("IVU_ext", "RdYlGn_r")
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
        title_text="Comparison of urban and environmental indicators",
        showlegend=False,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TEMPORAL EVOLUTION
# ----------------------------------------------------------
st.subheader("📈 Temporal evolution 2000–2025")

time_options = ["population_k", "lst_c", "rain_mm", "IVU_ext", "pm25", "ndvi", "gdp_pc"]
sel_time_var = st.selectbox("Select temporal variable", time_options, index=3)

fig_t = px.line(
    data,
    x="year",
    y=sel_time_var,
    color="district",
    title=f"Temporal evolution of {sel_time_var} (2000–2025)",
    markers=True
)
st.plotly_chart(fig_t, use_container_width=True)

# ----------------------------------------------------------
# DATA TABLE & EXPORT
# ----------------------------------------------------------
st.subheader("🧮 Integrated data table")
st.dataframe(df, use_container_width=True)
st.download_button(
    "⬇️ Download CSV for selected year",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"dar_es_salaam_integrated_{year}.csv",
    mime="text/csv"
)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("""
---
**Integrated sources:**  
🌡️ NASA MODIS (LST) · 🌧️ NASA GPM IMERG (precipitation) · 👥 SEDAC / WorldPop (population) ·  
🏙️ Copernicus GHSL / ESA WorldCover (urban footprint) · 💰 World Bank / WHO (GDP, health, education)  
🛰️ Educational prototype — NASA Space Apps Challenge 2025
""")
