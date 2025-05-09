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

WARD_PANEL_FP = DATA / "ward_month_burglary.parquet"
LSOA_PANEL_FP = DATA / "lsoa_month_burglary.parquet"
WARD_GEO_JSON = LOOK / "wards_2024.geojson"
LSOA_GEO_JSON = LOOK / "LSOA21_Boundaries.geojson"  # Corrected filename
LOOKUP_CSV = LOOK / "LSOA21_WD24_Lookup.csv"

# Configure Streamlit page settings
st.set_page_config(layout="wide")

# ── load --------------------------------------------------------------------
@st.cache_data
def load_ward_panel() -> pd.DataFrame:
    return pd.read_parquet(WARD_PANEL_FP)

@st.cache_data
def load_lsoa_panel() -> pd.DataFrame | None:
    try:
        return pd.read_parquet(LSOA_PANEL_FP)
    except FileNotFoundError:
        return None

@st.cache_data
def load_london_wards() -> set:
    lookup = pd.read_csv(LOOKUP_CSV)
    # Filter for London borough codes (E09) but exclude City of London (E09000001)
    london_wards = lookup[
        (lookup['LAD24CD'].str.startswith('E09', na=False)) & 
        (lookup['LAD24CD'] != 'E09000001')
    ]['WD24CD'].unique()
    return set(london_wards)

@st.cache_data
def load_london_lsoas() -> set | None:
    lookup = pd.read_csv(LOOKUP_CSV)
    # Filter for London borough codes (E09) but exclude City of London (E09000001)
    london_lsoas = lookup[
        (lookup['LAD24CD'].str.startswith('E09', na=False)) & 
        (lookup['LAD24CD'] != 'E09000001')
    ]['LSOA21CD'].unique()
    return set(london_lsoas)

@st.cache_resource
def load_ward_geo() -> gpd.GeoDataFrame:
    london_wards = load_london_wards()
    gdf = gpd.read_file(WARD_GEO_JSON)[["WD24CD", "WD24NM", "geometry"]]
    # Filter for London wards
    gdf = gdf[gdf["WD24CD"].isin(london_wards)]
    gdf = gdf.to_crs(4326)  # lat/lon for web mapping
    return gdf

@st.cache_resource
def load_lsoa_geo() -> gpd.GeoDataFrame | None:
    try:
        london_lsoas = load_london_lsoas()
        gdf = gpd.read_file(LSOA_GEO_JSON)[["LSOA21CD", "LSOA21NM", "geometry"]]
        # Filter for London LSOAs
        gdf = gdf[gdf["LSOA21CD"].isin(london_lsoas)]
        gdf = gdf.to_crs(4326)  # lat/lon for web mapping
        return gdf
    except FileNotFoundError:
        return None

# Load data
ward_panel = load_ward_panel()
lsoa_panel = load_lsoa_panel()
ward_geo = load_ward_geo()
lsoa_geo = load_lsoa_geo()

# ── sidebar controls --------------------------------------------------------
st.sidebar.title("London Burglary Dashboard")

view_level = st.sidebar.radio("View Level", 
    options=["Ward Level", "LSOA Level"],
    help="Ward Level shows data aggregated by electoral ward. LSOA (Lower Super Output Area) Level shows more detailed data at a smaller geographic level."
)

if view_level == "LSOA Level" and (lsoa_panel is None or lsoa_geo is None):
    st.sidebar.error("LSOA level data is not available yet. Please use Ward Level view.")
    view_level = "Ward Level"

# Use ward or LSOA data based on selection
panel = ward_panel if view_level == "Ward Level" else lsoa_panel
geo = ward_geo if view_level == "Ward Level" else lsoa_geo
id_col = "WD24CD" if view_level == "Ward Level" else "LSOA21CD"
name_col = "WD24NM" if view_level == "Ward Level" else "LSOA21NM"

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
chor = geo.merge(df_show[[id_col, "burglaries"]], left_on=id_col, right_on=id_col, how="left")
chor.loc[:, "burglaries"] = chor["burglaries"].fillna(0)

# Normalize burglary counts for colormap
norm = colors.Normalize(vmin=chor['burglaries'].min(), vmax=chor['burglaries'].max())
colormap = plt.colormaps.get_cmap('Reds')  # Changed from cm.get_cmap
chor['fill_color'] = chor['burglaries'].apply(lambda x: colormap(norm(x))[:3])
# Apply opacity: 0.75
chor['fill_color'] = chor['fill_color'].apply(lambda rgb: [int(c * 255) for c in rgb] + [0.75*255])

# ── main layout -------------------------------------------------------------
st.title("Residential Burglary in London")
subtitle = f"{vis_mode}: {format_m(sel_month)}"
st.markdown(f"## {subtitle}")

# pydeck map
mid_lon, mid_lat = -0.1275, 51.5072
geojson_layer = pdk.Layer(
    "GeoJsonLayer",
    data=chor.__geo_interface__,
    get_fill_color="properties.fill_color",  # Use the fill_color property directly
    pickable=True,
    stroked=True,
    filled=True,
    get_line_color=[0, 0, 0],  # Black color for ward borders
    get_line_width=0.01,
    line_width_min_pixels=1,
    auto_highlight=True,
    highlight_color=[255, 255, 255, 100],
)

tooltip = {
    "html": f"<b>{name_col}:</b> {{{name_col}}}<br/>"
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
min_burglaries_value = chor['burglaries'].min()
max_burglaries_value = chor['burglaries'].max()

min_burglary_areas = chor[chor['burglaries'] == min_burglaries_value]
max_burglary_areas = chor[chor['burglaries'] == max_burglaries_value]

area_type_display = "LSOAs" if name_col == "LSOA21NM" else "Wards"

min_area_names_list = min_burglary_areas[name_col].unique()
if len(min_area_names_list) > 5:
    min_area_names = ", ".join(min_area_names_list[:5]) + f", and {len(min_area_names_list) - 5} more"
else:
    min_area_names = ", ".join(min_area_names_list)

max_area_names_list = max_burglary_areas[name_col].unique()
if len(max_area_names_list) > 5:
    max_area_names = ", ".join(max_area_names_list[:5]) + f", and {len(max_area_names_list) - 5} more"
else:
    max_area_names = ", ".join(max_area_names_list)

st.markdown(f"{area_type_display} with Minimum Burglaries ({min_burglaries_value}): {min_area_names}")
st.markdown(f"{area_type_display} with Maximum Burglaries ({max_burglaries_value}): {max_area_names}")

# Add a dynamic legend with a colorbar
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=colormap),
    cax=ax, orientation='horizontal', label='Burglary Counts'
)

st.pyplot(fig)

# data table with search
table_title = "Ward table" if view_level == "Ward Level" else "LSOA table"
with st.expander(table_title):
    search_label = "Search by Ward name or code" if view_level == "Ward Level" else "Search by LSOA name or code"
    search = st.text_input(search_label, "")
    df_table = chor[[id_col, name_col, "burglaries"]].copy()
    df_table.columns = ["Code", "Name", "Burglaries"]
    
    if search:
        mask = (df_table["Name"].str.contains(search, case=False) | 
                df_table["Code"].str.contains(search, case=False))
        df_table = df_table[mask]
    
    st.dataframe(
        df_table.sort_values("Burglaries", ascending=False),
        use_container_width=True
    )

st.caption("Seasonal naïve forecast = same month, previous year.")
