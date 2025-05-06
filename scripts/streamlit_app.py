"""Streamlit dashboard prototype – London Residential Burglary
==============================================================

Run locally:

    streamlit run streamlit_app.py

Features
--------
* **Month selector** – pick any month in the dataset.
* Interactive **choropleth map** of burglary counts (actual or forecast).
* Simple **forecast option** (seasonal naïve: next‐month = same month last year).

Requirements: pandas, geopandas, streamlit, pydeck
"""
from __future__ import annotations
import pathlib
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# ── data paths --------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent  # Go up one level to project root
DATA = ROOT / "data_cache" / "processed"
LOOK = ROOT / "data_cache" / "lookups"

PANEL_FP = DATA / "ward_month_burglary.parquet"
GEO_JSON = LOOK / "wards_2024.geojson"
LOOKUP_CSV = LOOK / "LSOA21_WD24_Lookup.csv"

# ── load --------------------------------------------------------------------
@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL_FP)

@st.cache_data
def load_london_wards() -> set:
    lookup = pd.read_csv(LOOKUP_CSV)
    # Filter for London borough codes (E09) and get unique ward codes
    london_wards = lookup[lookup['LAD24CD'].str.startswith('E09', na=False)]['WD24CD'].unique()
    return set(london_wards)

@st.cache_resource
def load_geo() -> gpd.GeoDataFrame:
    london_wards = load_london_wards()
    gdf = gpd.read_file(GEO_JSON)[["WD24CD", "WD24NM", "geometry"]]
    # Filter for London wards
    gdf = gdf[gdf["WD24CD"].isin(london_wards)]
    gdf = gdf.to_crs(4326)  # lat/lon for web mapping
    return gdf

panel = load_panel()
geo = load_geo()

# ── sidebar controls --------------------------------------------------------
st.sidebar.title("London Burglary Dashboard")

months = panel["Month"].sort_values().unique()
def format_m(dt):
    return dt.strftime("%b %Y")
sel_month = st.sidebar.selectbox("Select month", months, format_func=format_m)

vis_mode = st.sidebar.radio("Show", ["Actual", "Forecast (seasonal naïve)"])

# ── prepare data ------------------------------------------------------------
if vis_mode.startswith("Forecast"):
    target_month = pd.Period(sel_month, freq="M").to_timestamp()
    hist_month   = target_month - pd.offsets.DateOffset(years=1)
    df_show = panel[panel["Month"] == hist_month].copy()
    df_show["Month"] = target_month
else:
    df_show = panel[panel["Month"] == sel_month].copy()

# merge geometry and handle missing values
chor = geo.merge(df_show[["WD24CD", "burglaries"]], on="WD24CD", how="left")
chor.loc[:, "burglaries"] = chor["burglaries"].fillna(0)

# Normalize burglary counts for colormap
norm = colors.Normalize(vmin=chor['burglaries'].min(), vmax=chor['burglaries'].max())
colormap = cm.get_cmap('Reds')
chor['fill_color'] = chor['burglaries'].apply(lambda x: colormap(norm(x))[:3])
chor['fill_color'] = chor['fill_color'].apply(lambda rgb: [int(c * 255) for c in rgb])

# ── main layout -------------------------------------------------------------
st.title("Residential Burglary in London")
subtitle = f"{vis_mode}: {format_m(sel_month)}"
st.markdown(f"## {subtitle}")

# pydeck map
mid_lon, mid_lat = -0.1275, 51.5072
geojson_layer = pdk.Layer(
    "GeoJsonLayer",
    data=chor.__geo_interface__,
    get_fill_color="[properties.fill_color[0], properties.fill_color[1], properties.fill_color[2]]",
    pickable=True,
    stroked=True,
    filled=True,
    get_line_color=[0, 0, 0],  # Black color for ward borders
    get_line_width=20,
    line_width_min_pixels=1,
    auto_highlight=True,
    highlight_color=[255, 255, 255, 100],
)

tooltip = {
    "html": "<b>Ward:</b> {WD24NM}<br/>"
            "<b>Burglaries:</b> {burglaries}",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}

deck = pdk.Deck(
    initial_view_state=pdk.ViewState(
        longitude=mid_lon,
        latitude=mid_lat,
        zoom=9
    ),
    layers=[geojson_layer],
    tooltip=tooltip
)

st.pydeck_chart(deck)

# Display wards with min and max burglary counts
min_burglary_ward = chor.loc[chor['burglaries'].idxmin()]
max_burglary_ward = chor.loc[chor['burglaries'].idxmax()]

st.markdown(f" Ward with Minimum Burglaries: {min_burglary_ward['WD24NM']} ({min_burglary_ward['burglaries']} burglaries)")
st.markdown(f" Ward with Maximum Burglaries: {max_burglary_ward['WD24NM']} ({max_burglary_ward['burglaries']} burglaries)")

# Add a dynamic legend with a colorbar
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=colormap),
    cax=ax, orientation='horizontal', label='Burglary Counts'
)

st.pyplot(fig)

# data table with search
with st.expander("Ward table"):
    search = st.text_input("Search wards", "")
    df_table = chor[["WD24CD", "WD24NM", "burglaries"]].copy()
    df_table.columns = ["Ward Code", "Ward Name", "Burglaries"]
    
    if search:
        mask = df_table["Ward Name"].str.contains(search, case=False)
        df_table = df_table[mask]
    
    st.dataframe(
        df_table.sort_values("Burglaries", ascending=False),
        use_container_width=True
    )

st.caption("Seasonal naïve forecast = same month, previous year.")
