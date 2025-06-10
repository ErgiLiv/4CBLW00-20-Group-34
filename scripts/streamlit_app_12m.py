"""Streamlit dashboard prototype - London Residential Burglary
==============================================================

Run locally:
    streamlit run scripts/streamlit_app_12m.py

"""
from __future__ import annotations
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
import branca.colormap as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

# ── data paths --------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent  # Go up one level to project root
DATA = ROOT / "data_cache" / "processed"
LOOK = ROOT / "data_cache" / "lookups"
PRED = ROOT / "predictions"

WARD_PANEL_FP = DATA / "ward_month_burglary.parquet"
LSOA_PANEL_FP = DATA / "lsoa_month_burglary.parquet"
WARD_GEO_JSON = LOOK / "wards_2024.geojson"
LSOA_GEO_JSON = LOOK / "LSOA21_Boundaries.geojson"  # Corrected filename
LOOKUP_CSV = LOOK / "LSOA21_WD24_Lookup.csv"
XGBOOST_PRED_CSV = PRED / "ward_burglary_predictions_12m.csv"  # Updated to 12-month predictions

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

# Individual burglary locations feature removed for performance reasons

@st.cache_data
def load_xgboost_predictions() -> pd.DataFrame:
    """Load XGBoost predictions for next 12 months."""
    df = pd.read_csv(XGBOOST_PRED_CSV, parse_dates=['Month'], infer_datetime_format=True)
    df["Predicted_Burglaries"] = pd.to_numeric(df["Predicted_Burglaries"], errors="coerce").fillna(0)
    return df

# Load data
ward_panel = load_ward_panel()
lsoa_panel = load_lsoa_panel()
ward_geo = load_ward_geo()
lsoa_geo = load_lsoa_geo()
xgboost_pred = load_xgboost_predictions()

# Get available months for both historical and next month forecast
historical_months = ward_panel["Month"].sort_values().unique()
forecast_month = xgboost_pred["Month"].min()  # Use earliest forecast month

# ── sidebar controls --------------------------------------------------------
st.sidebar.title("London Burglary Dashboard")

# Analysis mode selection
view_mode = st.sidebar.radio("Type of Analysis",
    options=["Historical Data", "Future Forecast"],
    help="Historical Data shows actual burglary counts. Future Forecast shows predicted burglaries for upcoming months using XGBoost model."
)

# NEW: Add forecast type selector when in Future Forecast mode
if view_mode == "Future Forecast":
	forecast_type = st.sidebar.radio("Forecast Type",
		options=["Ward Forecast", "LSOA Forecast"],
		help="Select forecast type for Future Forecast mode"
	)

# Add forecast month selector if in forecast mode
selected_forecast_date = None
if view_mode == "Future Forecast":
    forecast_dates = sorted(
		(
			# Use appropriate prediction dates based on forecast type.
			xgboost_pred["Month"].unique() 
			if forecast_type == "Ward Forecast" 
			else pd.read_csv(str(PRED / "lsoa_burglary_predictions_12m.csv"), parse_dates=['Month'])["Month"].unique()
		)
	)
    selected_forecast_date = st.sidebar.selectbox(
        "Select forecast month",
        options=sorted(forecast_dates),
        format_func=lambda x: x.strftime("%B %Y"),
        help="Select which month's forecast you want to view"
    )
    months_ahead = (selected_forecast_date.year - historical_months.max().year) * 12 + (selected_forecast_date.month - historical_months.max().month)
    
if view_mode == "Historical Data":
    view_level = st.sidebar.radio("View Level", 
        options=["Ward Level", "LSOA Level"],
        help="Ward Level shows data aggregated by electoral ward. LSOA (Lower Super Output Area) Level shows more detailed data at a smaller geographic level."
    )
# In forecast mode, set view_level based on forecast type
else:
    view_level = "Ward Level" if forecast_type == "Ward Forecast" else "LSOA Level"

if view_level == "LSOA Level" and (lsoa_panel is None or lsoa_geo is None):
    st.sidebar.error("LSOA level data is not available yet. Please use Ward Level view.")
    view_level = "Ward Level"

# Use ward or LSOA data based on selection
panel = ward_panel if view_level == "Ward Level" else lsoa_panel
geo = ward_geo if view_level == "Ward Level" else lsoa_geo
id_col = "WD24CD" if view_level == "Ward Level" else "LSOA21CD"
name_col = "WD24NM" if view_level == "Ward Level" else "LSOA21NM"

# Ensure panel is not None before proceeding
if panel is None:
    st.error("Selected data panel (Ward or LSOA) could not be loaded. Please check data availability.")
    st.stop()

def format_m(dt):
    return dt.strftime("%b %Y")

if view_mode == "Historical Data":
    sel_month = st.sidebar.selectbox("Select month", historical_months, format_func=format_m)
else:
    sel_month = forecast_month  # For forecast mode, we only have one month

# No longer showing individual burglary locations

# ── prepare data ------------------------------------------------------------
if view_mode == "Future Forecast":
    if forecast_type == "Ward Forecast":
        df_show = xgboost_pred[xgboost_pred['Month'] == selected_forecast_date].copy()
        
        # Create mapping for wards
        ward_id_mapping = geo.set_index(name_col)[id_col].to_dict()
        ward_name_mapping = geo.set_index(id_col)[name_col].to_dict()
        
        # Map ward IDs and names
        df_show[id_col] = df_show["Ward"].map(ward_id_mapping)
        df_show[name_col] = df_show["Ward"]  # Keep the ward name for later use
        df_show["burglaries"] = df_show["Predicted_Burglaries"]
    else:
        # LSOA Forecast: load LSOA predictions from file
        lsoa_pred_fp = PRED / "lsoa_burglary_predictions_12m.csv"
        df_lsoa_pred = pd.read_csv(lsoa_pred_fp, parse_dates=['Month'])
        df_show = df_lsoa_pred[df_lsoa_pred['Month'] == selected_forecast_date].copy()
        # Rename column so that it matches the geo merge key
        df_show = df_show.rename(columns={"LSOA": "LSOA21CD"})
        # Create mapping for LSOAs from geo file
        lsoa_name_mapping = geo.set_index("LSOA21CD")["LSOA21NM"].to_dict()
        df_show["LSOA21NM"] = df_show["LSOA21CD"].map(lsoa_name_mapping)
        # Remove rows with missing LSOA names
        df_show = df_show[df_show["LSOA21NM"].notna()]
        df_show["burglaries"] = df_show["Predicted_Burglaries"]
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
# Update title and subtitle
st.title("Residential Burglary in London")
if view_mode == "Future Forecast":
    subtitle = f"Forecast for {selected_forecast_date.strftime('%B %Y')}"
else:
    subtitle = f"{sel_month.strftime('%B %Y')}"
st.markdown(f"## {subtitle}")

# Calculate center coordinates of London
bounds = chor.total_bounds  # returns (minx, miny, maxx, maxy)
mid_lon = (bounds[0] + bounds[2]) / 2
mid_lat = (bounds[1] + bounds[3]) / 2

# Create Folium map - Dynamic sizing with dark mode
m = folium.Map(
    location=[mid_lat, mid_lon],
    zoom_start=10,
    tiles='cartodbpositron'
)

# Create colormap
min_value = chor['burglaries'].min()
max_value = chor['burglaries'].max()
colormap = cm.LinearColormap(
    colors=['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
    vmin=min_value,
    vmax=max_value,
    caption='Number of Burglaries'
)
colormap.add_to(m)

# Add choropleth layer
folium.GeoJson(
    chor.to_json(),
    name='Burglaries',
    style_function=lambda x: {
        'fillColor': colormap(x['properties']['burglaries']),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[name_col, 'burglaries'],
        aliases=[name_col.replace('NM', ' Name'), 'Burglaries'],
        style=('background-color: steelblue; color: white; font-family: arial; font-size: 12px; padding: 10px;')
    )
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Create a container for the map with dynamic sizing
map_container = st.container()
with map_container:
    # Custom CSS to make the map container responsive and full-width
    st.markdown(
        """
        <style>
        [data-testid="stHorizontalBlock"] > div {
            width: 100%;
        }
        .st-emotion-cache-16txtl3  {
            padding: 1rem;
        }
        iframe {
            width: 100% !important;
            min-height: 800px !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display map using st_folium with explicit height
    st_folium(m, width='100%', height=800)

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

# Historical Data section: Replace expander for data table with inline display
if view_mode == "Historical Data":
    # Removed st.expander("Data Table") wrapper
    search = st.text_input(
        "Search by " + ("Ward name or code" if view_level == "Ward Level" else "LSOA name or code"),
        key="historical_search"
    )
    df_table = chor[[id_col, name_col, "burglaries"]].copy()
    df_table.columns = ["Code", "Name", "Burglaries"]
    if search:
        mask = (df_table["Name"].str.contains(search, case=False)) | (df_table["Code"].str.contains(search, case=False))
        df_table = df_table[mask]
    st.dataframe(df_table.sort_values("Burglaries", ascending=False), use_container_width=True)

# ── Display forecast table and plots -------------------------------------------------------------
if view_mode == "Future Forecast":
    if forecast_type == "Ward Forecast":
        # Removed st.expander("Ward Forecasts") wrapper
        search = st.text_input("Search by Ward name or code", "")
        combined_df = df_show[[id_col, name_col, "Predicted_Burglaries"]].copy()
        if "Lower_CI" in df_show.columns and "Upper_CI" in df_show.columns:
            combined_df["Confidence Range"] = df_show.apply(
                lambda x: f"{x['Predicted_Burglaries']:.0f} ({x['Lower_CI']:.0f}-{x['Upper_CI']:.0f})",
                axis=1
            )
        combined_df.columns = ["Code", "Name", "Predicted Burglaries", "Confidence Range"]
        if search:
            mask = (combined_df["Name"].str.contains(search, case=False)) | (combined_df["Code"].str.contains(search, case=False))
            combined_df = combined_df[mask]
        st.dataframe(combined_df.sort_values("Predicted Burglaries", ascending=False), use_container_width=True)
    else:  # LSOA Forecast table
        # Removed st.expander("LSOA Forecasts") wrapper
        search = st.text_input("Search by LSOA name or code", "")
        combined_df = df_show[["LSOA21CD", "LSOA21NM", "Predicted_Burglaries"]].copy()
        if "Lower_CI" in df_show.columns and "Upper_CI" in df_show.columns:
            combined_df["Confidence Range"] = df_show.apply(
                lambda x: f"{x['Predicted_Burglaries']:.0f} ({x['Lower_CI']:.0f}-{x['Upper_CI']:.0f})",
                axis=1
            )
        combined_df.columns = ["Code", "Name", "Predicted Burglaries", "Confidence Range"]
        if search:
            mask = (combined_df["Name"].str.contains(search, case=False)) | (combined_df["Code"].str.contains(search, case=False))
            combined_df = combined_df[mask]
        st.dataframe(combined_df.sort_values("Predicted Burglaries", ascending=False), use_container_width=True)
        
    # Removed st.expander("Forecast Comparison Plots") wrapper; plots always shown
    if forecast_type == "Ward Forecast":
        available_wards = sorted(df_show["Ward"].unique())
        selected_ward_filter = st.selectbox("Select Ward for Plot", options=available_wards, index=0)
        
        import altair as alt
        
        st.markdown("#### Historical vs Forecast for " + selected_ward_filter)
        hist_data = panel[(panel["WD24NM"] == selected_ward_filter) &
                          (panel["Month"].dt.month == selected_forecast_date.month)].copy()
        if hist_data.empty:
            st.write("No historical data available for the selected ward and month.")
        else:
            hist_data["Year_Month"] = hist_data["Month"].dt.strftime("%b %Y")
            hist_chart = alt.Chart(hist_data).mark_line(point=True).transform_calculate(
                Type="'Historical Trend'"
            ).encode(
                x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("burglaries:Q", title="Burglaries"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Historical Trend", "Forecast Trend"],
                    range=["#ADD8E6", "#FF4B4B"]
                ), legend=alt.Legend(title="Trend"))
            ).properties(
                title=f"Historical Burglaries for {selected_ward_filter} in {selected_forecast_date.strftime('%B')} over the Years",
                height=400
            )
            forecast_subset = df_show[df_show["Ward"] == selected_ward_filter]
            if forecast_subset.empty:
                st.write("No forecast data available for the selected ward.")
            else:
                forecast_value = forecast_subset["Predicted_Burglaries"].iloc[0]
                forecast_month_str = selected_forecast_date.strftime("%b %Y")
                forecast_df = pd.DataFrame({"Year_Month": [forecast_month_str], "Forecast": [forecast_value]})
                forecast_point = alt.Chart(forecast_df).mark_point(color="#FF4B4B", size=100).encode(
                    x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Forecast:Q", title="Burglaries")
                )
                last_hist_value = hist_data.sort_values("Month")["burglaries"].iloc[-1]
                last_hist_label = hist_data.sort_values("Month")["Year_Month"].iloc[-1]
                connect_df = pd.DataFrame({
                    "Year_Month": [last_hist_label, forecast_month_str],
                    "Burglaries": [last_hist_value, forecast_value]
                })
                connect_line = alt.Chart(connect_df).mark_line(strokeDash=[5,3]).transform_calculate(
                    Type="'Forecast Trend'"
                ).encode(
                    x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Burglaries:Q", title="Burglaries"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=["Historical Trend", "Forecast Trend"],
                        range=["#ADD8E6", "#FF4B4B"]
                    ), legend=alt.Legend(title="Trend"))
                )
                combined_chart = hist_chart + forecast_point + connect_line
                st.altair_chart(combined_chart, use_container_width=True)

        st.markdown(" ")        
        st.markdown("#### Forecast Trend over Next 12 Months")
        trend_data = xgboost_pred[xgboost_pred["Ward"] == selected_ward_filter].copy()
        if trend_data.empty:
            st.write("No forecast trend data available for the selected ward.")
        else:
            trend_data.sort_values("Month", inplace=True)
            trend_chart = alt.Chart(trend_data).mark_line(point=True, strokeDash=[5,3]).transform_calculate(
                Type="'Forecast Trend'"
            ).encode(
                x=alt.X("Month:T", axis=alt.Axis(format="%b %Y", title="Month and Year", labelAngle=0, tickCount=12)),
                y=alt.Y("Predicted_Burglaries:Q", title="Predicted Burglaries"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Forecast Trend"], range=["#FF4B4B"]), legend=alt.Legend(title="Trend"))
            ).properties(
                title="Forecast Trend over Next 12 Months for " + selected_ward_filter,
                height=400
            )
            st.altair_chart(trend_chart, use_container_width=True)
    else:  # LSOA Forecast plots
        available_lsoas = sorted(df_show["LSOA21NM"].dropna().unique())
        selected_lsoa_filter = st.selectbox("Select LSOA for Plot", options=available_lsoas, index=0)
        
        import altair as alt
        
        st.markdown("#### Historical vs Forecast for " + selected_lsoa_filter)
        hist_data = panel[(panel["LSOA21NM"] == selected_lsoa_filter) &
                          (panel["Month"].dt.month == selected_forecast_date.month)].copy()
        if hist_data.empty:
            st.write("No historical data available for the selected LSOA and month.")
        else:
            hist_data["Year_Month"] = hist_data["Month"].dt.strftime("%b %Y")
            hist_chart = alt.Chart(hist_data).mark_line(point=True).transform_calculate(
                Type="'Historical Trend'"
            ).encode(
                x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("burglaries:Q", title="Burglaries"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Historical Trend", "Forecast Trend"],
                    range=["#ADD8E6", "#FF4B4B"]
                ), legend=alt.Legend(title="Trend"))
            ).properties(
                title=f"Historical Burglaries for {selected_lsoa_filter} in {selected_forecast_date.strftime('%B')} over the Years",
                height=400
            )
            forecast_subset = df_show[df_show["LSOA21NM"] == selected_lsoa_filter]
            if forecast_subset.empty:
                st.write("No forecast data available for the selected LSOA.")
            else:
                forecast_value = forecast_subset["Predicted_Burglaries"].iloc[0]
                forecast_month_str = selected_forecast_date.strftime("%b %Y")
                forecast_df = pd.DataFrame({"Year_Month": [forecast_month_str], "Forecast": [forecast_value]})
                forecast_point = alt.Chart(forecast_df).mark_point(color="#FF4B4B", size=100).encode(
                    x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Forecast:Q", title="Burglaries")
                )
                last_hist_value = hist_data.sort_values("Month")["burglaries"].iloc[-1]
                last_hist_label = hist_data.sort_values("Month")["Year_Month"].iloc[-1]
                connect_df = pd.DataFrame({
                    "Year_Month": [last_hist_label, forecast_month_str],
                    "Burglaries": [last_hist_value, forecast_value]
                })
                connect_line = alt.Chart(connect_df).mark_line(strokeDash=[5,3]).transform_calculate(
                    Type="'Forecast Trend'"
                ).encode(
                    x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Burglaries:Q", title="Burglaries"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=["Historical Trend", "Forecast Trend"],
                        range=["#ADD8E6", "#FF4B4B"]
                    ), legend=alt.Legend(title="Trend"))
                )
                combined_chart = hist_chart + forecast_point + connect_line
                st.altair_chart(combined_chart, use_container_width=True)

        st.markdown(" ")        
        st.markdown("#### Forecast Trend over Next 12 Months")
        import altair as alt
        trend_data = pd.read_csv(str(PRED / "lsoa_burglary_predictions_12m.csv"), parse_dates=['Month'])
        # Get the selected LSOA code based on the chosen LSOA name from the dropdown
        selected_lsoa_code = df_show.loc[df_show["LSOA21NM"] == selected_lsoa_filter, "LSOA21CD"].iloc[0]
        trend_data = trend_data[trend_data["LSOA"] == selected_lsoa_code].copy()
        if trend_data.empty:
            st.write("No forecast trend data available for the selected LSOA.")
        else:
            trend_data.sort_values("Month", inplace=True)
            trend_chart = alt.Chart(trend_data).mark_line(point=True, strokeDash=[5,3]).transform_calculate(
                Type="'Forecast Trend'"
            ).encode(
                x=alt.X("Month:T", axis=alt.Axis(format="%b %Y", title="Month and Year", labelAngle=0, tickCount=12)),
                y=alt.Y("Predicted_Burglaries:Q", title="Predicted Burglaries"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Forecast Trend"], range=["#FF4B4B"]), legend=alt.Legend(title="Trend"))
            ).properties(
                title="Forecast Trend over Next 12 Months for " + selected_lsoa_filter,
                height=400
            )
            st.altair_chart(trend_chart, use_container_width=True)
