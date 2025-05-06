"""
build_features.py – feature matrix generator
============================================

* Temporal lags : 1, 3, 12 months
* Spatial lag   : previous-month neighbour mean
* Static joins  :
      – IMD-2019 mean per ward  (imd2019_lsoa.csv)
      – Mid-2022 population per ward (ward_pop2022.csv; columns WD22CD, population_2022)

Output → data_cache/processed/features.parquet

Runs with one click – no CLI flags.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
CACHE         = ROOT / "data_cache"
PROCESSED_DIR = CACHE / "processed"
LOOKUP_DIR    = CACHE / "lookups"

PANEL_FP   = PROCESSED_DIR / "ward_month_burglary.parquet"
GEO_JSON   = LOOKUP_DIR   / "wards_2024.geojson"
LSOA_LOOK  = LOOKUP_DIR   / "LSOA21_WD24_Lookup.csv"
IMD_CSV    = LOOKUP_DIR   / "imd2019_lsoa.csv"
POP_CSV    = LOOKUP_DIR   / "ward_pop2022.csv"      # <-- ensure this file exists
FEATURE_FP = PROCESSED_DIR / "features.parquet"

LAGS = [1, 3, 12]   # months

# ── helpers ─────────────────────────────────────────────────────────────────
def add_temporal_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["WD24CD", "Month"])
    for k in LAGS:
        df[f"lag_{k}"] = (
            df.groupby("WD24CD")["burglaries"].shift(k).fillna(0)
        )
    return df


def make_neighbours(wards: gpd.GeoDataFrame) -> dict[str, list[str]]:
    sidx = wards.sindex
    nbrs = {c: [] for c in wards.WD24CD}
    for i, row in wards.iterrows():
        for j in sidx.intersection(row.geometry.bounds):
            if i == j:
                continue
            if row.geometry.touches(wards.at[j, "geometry"]):
                nbrs[row.WD24CD].append(wards.at[j, "WD24CD"])
    return nbrs


def add_spatial_lag(panel: pd.DataFrame,
                    neighbours: dict[str, list[str]]) -> pd.DataFrame:
    panel = panel.copy()
    panel["lag_spatial_avg"] = 0.0
    months = panel["Month"].sort_values().unique()
    for m in tqdm(months, desc="spatial lag"):
        prev = panel.loc[
            panel["Month"] == (m - pd.offsets.MonthBegin(1)),
            ["WD24CD", "burglaries"],
        ]
        prev_map = prev.set_index("WD24CD")["burglaries"].to_dict()
        mask = panel["Month"] == m
        panel.loc[mask, "lag_spatial_avg"] = panel.loc[mask, "WD24CD"].apply(
            lambda c: np.mean([prev_map.get(n, 0) for n in neighbours.get(c, [])])
            if neighbours.get(c) else 0
        )
    return panel


def load_imd_ward(imd_csv: Path, lookup_csv: Path) -> pd.DataFrame:
    imd = pd.read_csv(
        imd_csv,
        usecols=["LSOA code (2011)", "Index of Multiple Deprivation (IMD) Score"],
        dtype=str,
    ).rename(columns={
        "LSOA code (2011)": "LSOA21CD",
        "Index of Multiple Deprivation (IMD) Score": "imd_score",
    })
    imd["imd_score"] = pd.to_numeric(imd["imd_score"], errors="coerce")

    look = pd.read_csv(
        lookup_csv, usecols=["LSOA21CD", "WD24CD"], dtype=str
    )
    merged = imd.merge(look, on="LSOA21CD", how="left").dropna(subset=["WD24CD"])
    return merged.groupby("WD24CD", as_index=False)["imd_score"].mean()


def load_population(pop_csv: Path) -> pd.DataFrame:
    """
    ward_pop2022.csv provided by the GLA modelled back-series:
    columns include ward demographics with:
    - WD22CD: Ward code 
    - population_2022: Population value for 2022
    - Additional demographic data (births, deaths, migration flows etc)
    """
    pop = pd.read_csv(
        pop_csv,
        dtype={
            "WD22CD": str,
            "population_2022": float,  # Using float to handle any potential NA values
        }
    )
    
    # Group by ward code and sum population to get total ward population
    pop = pop.groupby("WD22CD")["population_2022"].sum().reset_index()
    
    pop = pop.rename(columns={
        "WD22CD": "WD24CD",
        "population_2022": "population",
    })
    pop["WD24CD"] = pop["WD24CD"].str.strip()
    return pop


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="help")
    _ = parser.parse_args([])

    # ensure required files exist
    for fp in (PANEL_FP, GEO_JSON, LSOA_LOOK, IMD_CSV, POP_CSV):
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp.relative_to(ROOT)}")

    panel = pd.read_parquet(PANEL_FP)

    # feature engineering
    panel = add_temporal_lags(panel)
    wards = gpd.read_file(GEO_JSON)[["WD24CD", "geometry"]]
    panel = add_spatial_lag(panel, make_neighbours(wards))

    # static joins
    imd = load_imd_ward(IMD_CSV, LSOA_LOOK)
    pop = load_population(POP_CSV)

    panel = panel.merge(imd, on="WD24CD", how="left")
    print(f"IMD coverage: {panel['imd_score'].notna().mean()*100:.1f}%")

    panel = panel.merge(pop, on="WD24CD", how="left")
    print(f"Pop coverage: {panel['population'].notna().mean()*100:.1f}%")

    FEATURE_FP.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FEATURE_FP, index=False)
    print(
        "✅  features saved →",
        FEATURE_FP.relative_to(ROOT),
        "rows:",
        len(panel),
    )


if __name__ == "__main__":
    main()
