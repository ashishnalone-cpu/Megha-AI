"""
districts.py

Loads the IND-DIS-732 district GeoJSON and builds a one-time mapping from
each NetCDF grid cell to the district it falls in. That mapping powers:
  - full per-district weather tables (rainfall, temp, wind, humidity)
  - district-level and state-level rainfall threshold queries
  - the district-wise (choropleth) and smoothed-pattern maps, for a
    single state OR for all of India (state_name=None)

The grid-to-district join is the expensive step, so it's meant to be built
once per app session (wrap the call in st.cache_resource in the app).
"""

import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nc_data import classify_rainfall, wind_dir_label


def load_district_gdf(geojson_path):
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.reset_index(drop=True)
    gdf["ST_NM_UP"] = gdf["ST_NM"].str.upper().str.strip()
    return gdf


def is_india_scope(state_name):
    return state_name is None or state_name.strip().upper() in ("INDIA", "ALL INDIA", "ALL-INDIA")


def build_grid_district_index(ds, gdf):
    """Returns a 2D int array (n_lat, n_lon): district row index, or -1."""
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    glon, glat = np.meshgrid(lons, lats)  # shape (n_lat, n_lon)

    pts = gpd.GeoDataFrame(
        {"grid_pos": np.arange(glon.size)},
        geometry=[Point(xy) for xy in zip(glon.ravel(), glat.ravel())],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, gdf[["geometry"]], how="left", predicate="within")
    joined = joined[~joined["grid_pos"].duplicated(keep="first")].sort_values("grid_pos")

    id_grid = joined["index_right"].values
    id_grid = np.where(pd.isna(id_grid), -1, id_grid).astype(int)
    return id_grid.reshape(glon.shape)


def _day_time_indices(ds, target_date):
    times = pd.to_datetime(ds["valid_time_ist"].values)
    mask = np.array([t.date() == target_date for t in times])
    if not mask.any():
        return None
    return np.where(mask)[0]


def day_total_grid(ds, target_date):
    """2D array (n_lat, n_lon) of total daily precip (mm), or None."""
    idx = _day_time_indices(ds, target_date)
    if idx is None:
        return None
    return ds["tp"].isel(time=idx).sum(dim="time").values


def compute_day_variable_grid(ds, target_date, variable):
    """
    2D array (n_lat, n_lon) for the requested variable, aggregated over the
    given IST calendar day. variable: 'rainfall', 'temperature_max',
    'temperature_min', 'wind', or 'humidity'.
    """
    idx = _day_time_indices(ds, target_date)
    if idx is None:
        return None

    if variable == "rainfall":
        return ds["tp"].isel(time=idx).sum(dim="time").values
    if variable == "temperature_max":
        return (ds["t2m"].isel(time=idx) - 273.15).max(dim="time").values
    if variable == "temperature_min":
        return (ds["t2m"].isel(time=idx) - 273.15).min(dim="time").values
    if variable == "wind":
        u = ds["u10"].isel(time=idx).mean(dim="time")
        v = ds["v10"].isel(time=idx).mean(dim="time")
        return (np.hypot(u, v) * 3.6).values
    if variable == "humidity":
        #return ds["q1000"].isel(time=idx).mean(dim="time").values
        return ((ds["t2m"].isel(time=idx)/0.622) * 100).min(dim="time").values
    raise ValueError(f"Unknown variable '{variable}'")


def _scope_rows(gdf, state_name):
    if is_india_scope(state_name):
        return gdf
    return gdf[gdf["ST_NM_UP"] == state_name.strip().upper()]


def district_rainfall_table(ds, gdf, id_grid, state_name, target_date):
    """Per-district average daily rainfall for one state, sorted descending."""
    day_grid = day_total_grid(ds, target_date)
    if day_grid is None:
        return None

    state_rows = _scope_rows(gdf, state_name)
    if state_rows.empty:
        return pd.DataFrame(columns=["district", "avg_rain_mm", "category"])

    results = []
    for row_idx, row in state_rows.iterrows():
        cell_mask = id_grid == row_idx
        if not cell_mask.any():
            continue
        avg_rain = float(np.nanmean(day_grid[cell_mask]))
        results.append({
            "district": row["DISTRICT"],
            "avg_rain_mm": round(avg_rain, 1),
            "category": classify_rainfall(avg_rain, hours=24),
        })
    return pd.DataFrame(results).sort_values("avg_rain_mm", ascending=False).reset_index(drop=True)


def state_rainfall_table(ds, gdf, id_grid, target_date):
    """Per-state average daily rainfall across all states, sorted descending."""
    day_grid = day_total_grid(ds, target_date)
    if day_grid is None:
        return None

    records = []
    for state_name, group in gdf.groupby("ST_NM_UP"):
        idxs = group.index.values
        cell_mask = np.isin(id_grid, idxs)
        if not cell_mask.any():
            continue
        avg_rain = float(np.nanmean(day_grid[cell_mask]))
        records.append({
            "state": state_name.title(),
            "avg_rain_mm": round(avg_rain, 1),
            "category": classify_rainfall(avg_rain, hours=24),
        })
    return pd.DataFrame(records).sort_values("avg_rain_mm", ascending=False).reset_index(drop=True)


def district_full_table(ds, gdf, id_grid, target_date, state_name=None):
    """
    Full per-district table: District, Rainfall (mm), Max/Min Temp, Wind,
    Humidity, Category. state_name=None -> all districts (India-wide).
    """
    rain_grid = compute_day_variable_grid(ds, target_date, "rainfall")
    if rain_grid is None:
        return None
    tmax_grid = compute_day_variable_grid(ds, target_date, "temperature_max")
    tmin_grid = compute_day_variable_grid(ds, target_date, "temperature_min")
    hum_grid = compute_day_variable_grid(ds, target_date, "humidity")

    idx = _day_time_indices(ds, target_date)
    u_grid = ds["u10"].isel(time=idx).mean(dim="time").values
    v_grid = ds["v10"].isel(time=idx).mean(dim="time").values

    rows_source = _scope_rows(gdf, state_name)
    if rows_source.empty:
        return pd.DataFrame()

    results = []
    for row_idx, row in rows_source.iterrows():
        cell_mask = id_grid == row_idx
        if not cell_mask.any():
            continue
        u_mean = float(np.nanmean(u_grid[cell_mask]))
        v_mean = float(np.nanmean(v_grid[cell_mask]))
        wind_label, _ = wind_dir_label(u_mean, v_mean)
        rain_val = float(np.nanmean(rain_grid[cell_mask]))
        results.append({
            "State": row["ST_NM"].title(),
            "District": row["DISTRICT"].title(),
            "Rainfall (mm)": round(rain_val, 1),
            "Max Temp (\u00b0C)": round(float(np.nanmean(tmax_grid[cell_mask])), 1),
            "Min Temp (\u00b0C)": round(float(np.nanmean(tmin_grid[cell_mask])), 1),
            "Wind": f"{wind_label} {np.hypot(u_mean, v_mean) * 3.6:.0f} km/h",
            "Humidity (%)": round(float(np.nanmean(hum_grid[cell_mask])), 1),
            "Category": classify_rainfall(rain_val, hours=24),
        })
    return pd.DataFrame(results).sort_values("Rainfall (mm)", ascending=False).reset_index(drop=True)


def summarize_district_table(table, scope_label, target_date):
    if table is None or table.empty:
        return f"No forecast data available for {scope_label} on {target_date}."
    avg_rain = table["Rainfall (mm)"].mean()
    avg_tmax = table["Max Temp (\u00b0C)"].mean()
    avg_tmin = table["Min Temp (\u00b0C)"].mean()
    wettest = table.iloc[0]
    category = classify_rainfall(avg_rain, hours=24)
    return (
        f"**{scope_label}** \u2014 {target_date}: average rainfall across districts is "
        f"~{avg_rain:.0f} mm ({category}), with temperatures ranging roughly "
        f"{avg_tmin:.0f}\u2013{avg_tmax:.0f}\u00b0C. Highest rainfall is expected in "
        f"**{wettest['District']}** (~{wettest['Rainfall (mm)']:.0f} mm)."
    )


def district_variable_series(ds, gdf, id_grid, target_date, variable, state_name=None):
    """GeoDataFrame subset with a 'value' column: mean of `variable` per
    district, for the choropleth (spatial) map. state_name=None -> all India."""
    grid = compute_day_variable_grid(ds, target_date, variable)
    if grid is None:
        return None

    rows_source = _scope_rows(gdf, state_name)
    if rows_source.empty:
        return rows_source

    values = []
    for row_idx in rows_source.index:
        cell_mask = id_grid == row_idx
        values.append(float(np.nanmean(grid[cell_mask])) if cell_mask.any() else np.nan)

    out = rows_source.copy()
    out["value"] = values
    return out


def get_state_geometry(gdf, state_name):
    """Dissolved (merged) polygon for one state (or all India), or None."""
    rows = _scope_rows(gdf, state_name)
    if rows.empty:
        return None
    try:
        return rows.union_all()
    except AttributeError:
        return rows.unary_union
