# London Burglary Forecast Project

## Project Overview
This project analyzes and forecasts residential burglaries in London.

## Folder Structure
- `/data`: Dataset files.
- `/scripts`: Python scripts for data processing and analysis.
- `/docs`: Documentation and notes.

## Getting Started
1. Clone repository:
```bash
git clone <repo-url>

For the GeoJson:
1.I transform all the relevant files to the geojson file, so no need to download the raw file from the website, just download the ward_vis.zip, you can see "ward_vis\mygeodata\London_Ward.geojson".

2.Make sure folium library is in your python environment, open the "ward_vis\Bolder_vis.ipynb" file, replace the file path (London_Ward.geojson) on your PC, we only use "ward_vis\mygeodata\London_Ward.geojson" this file to visualize the map and border. 